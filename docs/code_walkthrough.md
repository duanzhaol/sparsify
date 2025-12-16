# Sparsify 代码走读文档

本文档将带你深入理解 sparsify 代码库的结构和实现原理。

## 目录

1. [项目概览](#1-项目概览)
2. [代码结构](#2-代码结构)
3. [核心概念：稀疏自编码器 (SAE)](#3-核心概念稀疏自编码器-sae)
4. [入口点：__main__.py](#4-入口点__main__py)
5. [配置系统：config.py](#5-配置系统configpy)
6. [SAE模型：sparse_coder.py](#6-sae模型sparse_coderpy)
7. [训练器：trainer.py](#7-训练器trainerpy)
8. [数据处理：data.py](#8-数据处理datapy)
9. [优化器：sign_sgd.py 和 muon.py](#9-优化器sign_sgdpy-和-muonpy)
10. [性能优化：fused_encoder.py](#10-性能优化fused_encoderpy)
11. [工具函数：utils.py](#11-工具函数utilspy)
12. [完整训练流程](#12-完整训练流程)

---

## 1. 项目概览

Sparsify 是一个用于训练 **k-sparse autoencoders (SAEs)** 和 **transcoders** 的库。其核心思想来自论文 "Scaling and evaluating sparse autoencoders" (Gao et al. 2024)。

### 关键设计决策

1. **TopK 激活函数**：不使用 L1 正则化，而是直接使用 TopK 选择来强制稀疏性
2. **即时计算激活**：不缓存激活到磁盘，而是在训练时即时计算，节省存储空间
3. **分布式训练支持**：支持 DDP 和跨 GPU 分布模块

---

## 2. 代码结构

```
sparsify/
├── __init__.py          # 包入口，导出公共 API
├── __main__.py          # CLI 入口点
├── config.py            # 配置类定义
├── sparse_coder.py      # SAE/Transcoder 模型实现
├── trainer.py           # 训练循环
├── data.py              # 数据处理工具
├── fused_encoder.py     # 优化的编码器实现
├── sign_sgd.py          # SignSGD 优化器
├── muon.py              # Muon 优化器
├── utils.py             # 工具函数
└── xformers.py          # Triton 加速的解码器
```

---

## 3. 核心概念：稀疏自编码器 (SAE)

### 3.1 SAE 的数学原理

SAE 的目标是学习一个稀疏的、可解释的特征表示。给定输入 `x`：

```
编码: z = TopK(ReLU(W_enc @ (x - b_dec) + b_enc))
解码: x̂ = W_dec @ z + b_dec
损失: FVU = ||x - x̂||² / Var(x)  (Fraction of Variance Unexplained)
```

### 3.2 Transcoder

Transcoder 是 SAE 的变体，用于预测模块的输出（给定输入），而不是重建输入：

```
输入: 模块的输入 x_in
目标: 模块的输出 x_out
编码: z = TopK(ReLU(W_enc @ x_in + b_enc))
解码: x̂_out = W_dec @ z + b_dec
```

---

## 4. 入口点：__main__.py

文件位置：`sparsify/__main__.py`

### 4.1 RunConfig 类

```python
@dataclass
class RunConfig(TrainConfig):
    model: str = "HuggingFaceTB/SmolLM2-135M"      # 模型名称
    dataset: str = "EleutherAI/SmolLM2-135M-10B"  # 数据集
    split: str = "train"                           # 数据集分割
    ctx_len: int = 2048                            # 上下文长度
    load_in_8bit: bool = False                     # 8bit 量化加载
    # ... 更多配置
```

`RunConfig` 继承自 `TrainConfig`，添加了命令行特有的参数。

### 4.2 load_artifacts 函数

```python
def load_artifacts(args: RunConfig, rank: int) -> tuple[PreTrainedModel, Dataset]:
```

这个函数负责：
1. **加载模型**：根据 `loss_fn` 选择 `AutoModel` 或 `AutoModelForCausalLM`
2. **加载数据集**：支持 HuggingFace 数据集、本地数据集、或 memmap 格式
3. **数据预处理**：如果数据集未 tokenize，调用 `chunk_and_tokenize`

### 4.3 run 函数（主入口）

```python
def run():
    # 1. 初始化分布式环境
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None

    # 2. 解析命令行参数
    args = parse(RunConfig)

    # 3. 加载模型和数据
    model, dataset = load_artifacts(args, rank)

    # 4. 创建训练器并开始训练
    trainer = Trainer(args, dataset, model)
    trainer.fit()
```

### 4.4 分布式训练初始化

```python
if ddp:
    torch.cuda.set_device(int(local_rank))
    dist.init_process_group("nccl", timeout=timedelta(weeks=1))
```

注意：timeout 设置为一周，是为了处理大型数据集的预处理时间。

---

## 5. 配置系统：config.py

文件位置：`sparsify/config.py`

### 5.1 SparseCoderConfig（SAE 架构配置）

```python
@dataclass
class SparseCoderConfig(Serializable):
    activation: Literal["groupmax", "topk"] = "topk"  # 激活函数
    expansion_factor: int = 32    # 扩展因子（latent维度 = input维度 * expansion_factor）
    normalize_decoder: bool = True # 是否归一化解码器权重
    num_latents: int = 0          # latent 数量（0 表示使用 expansion_factor）
    k: int = 32                   # TopK 的 k 值
    multi_topk: bool = False      # 是否使用 Multi-TopK 损失
    skip_connection: bool = False # 是否包含跳跃连接
    transcode: bool = False       # 是否为 transcoder 模式
```

### 5.2 TrainConfig（训练配置）

```python
@dataclass
class TrainConfig(Serializable):
    sae: SparseCoderConfig

    batch_size: int = 32          # 批次大小（序列数）
    grad_acc_steps: int = 1       # 梯度累积步数
    micro_acc_steps: int = 1      # 微批次累积步数

    loss_fn: Literal["ce", "fvu", "kl"] = "fvu"  # 损失函数
    optimizer: Literal["adam", "muon", "signum"] = "signum"  # 优化器

    lr: float | None = None       # 学习率（None 表示自动选择）
    lr_warmup_steps: int = 1000   # 学习率预热步数

    hookpoints: list[str] = []    # 要训练 SAE 的 hookpoint 列表
    layers: list[int] = []        # 要训练 SAE 的层索引
    layer_stride: int = 1         # 层间隔

    distribute_modules: bool = False  # 是否将 SAE 分布到不同 GPU
```

### 5.3 损失函数选项

- **fvu**（默认）：局部重建损失，Fraction of Variance Unexplained
- **ce**：端到端交叉熵损失
- **kl**：端到端 KL 散度损失

---

## 6. SAE模型：sparse_coder.py

文件位置：`sparsify/sparse_coder.py`

### 6.1 SparseCoder 类结构

```python
class SparseCoder(nn.Module):
    def __init__(self, d_in: int, cfg: SparseCoderConfig, device, dtype):
        # 编码器：线性层
        self.encoder = nn.Linear(d_in, self.num_latents)

        # 解码器权重
        self.W_dec = nn.Parameter(...)  # shape: (num_latents, d_in)

        # 解码器偏置
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # 可选的跳跃连接
        self.W_skip = nn.Parameter(...) if cfg.skip_connection else None
```

### 6.2 编码过程

```python
def encode(self, x: Tensor) -> EncoderOutput:
    # 1. 减去解码器偏置（如果是 autoencoder）
    if not self.cfg.transcode:
        x = x - self.b_dec

    # 2. 调用融合编码器
    return fused_encoder(x, self.encoder.weight, self.encoder.bias,
                         self.cfg.k, self.cfg.activation)
```

### 6.3 解码过程

```python
def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
    # 使用稀疏矩阵乘法（只计算 top-k 个 latent 的贡献）
    y = decoder_impl(top_indices, top_acts, self.W_dec.mT)
    return y + self.b_dec
```

### 6.4 前向传播

```python
def forward(self, x: Tensor, y: Tensor | None = None,
            dead_mask: Tensor | None = None) -> ForwardOutput:
    # 1. 编码
    top_acts, top_indices, pre_acts = self.encode(x)

    # 2. 如果没有提供目标，则是自编码
    if y is None:
        y = x

    # 3. 解码
    sae_out = self.decode(top_acts, top_indices)

    # 4. 计算残差和损失
    e = y - sae_out
    total_variance = (y - y.mean(0)).pow(2).sum()
    fvu = e.pow(2).sum() / total_variance

    # 5. AuxK 损失（可选，用于激活死特征）
    if dead_mask is not None:
        auxk_loss = self._compute_auxk_loss(...)

    return ForwardOutput(sae_out, top_acts, top_indices, fvu, auxk_loss, ...)
```

### 6.5 解码器归一化

```python
def set_decoder_norm_to_unit_norm(self):
    """将解码器权重归一化为单位范数"""
    norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
    self.W_dec.data /= norm + eps

def remove_gradient_parallel_to_decoder_directions(self):
    """移除与解码器方向平行的梯度分量（保持单位范数约束）"""
    parallel_component = einsum(self.W_dec.grad, self.W_dec.data, "d_sae d_in, d_sae d_in -> d_sae")
    self.W_dec.grad -= einsum(parallel_component, self.W_dec.data, "d_sae, d_sae d_in -> d_sae d_in")
```

### 6.6 模型加载

```python
# 从 HuggingFace Hub 加载
sae = SparseCoder.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.10")

# 加载多个层
saes = SparseCoder.load_many("EleutherAI/sae-llama-3-8b-32x")
```

---

## 7. 训练器：trainer.py

文件位置：`sparsify/trainer.py`

### 7.1 Trainer 初始化

```python
class Trainer:
    def __init__(self, cfg: TrainConfig, dataset, model: PreTrainedModel):
        # 存储模型（冻结的，不训练）
        self.model = model

        # 解析 hookpoints
        if cfg.hookpoints:
            # 支持 glob 模式，如 "h.*.attn"
            raw_hookpoints = []
            for name, _ in model.base_model.named_modules():
                if any(fnmatchcase(name, pat) for pat in cfg.hookpoints):
                    raw_hookpoints.append(name)
            cfg.hookpoints = natsorted(raw_hookpoints)
        else:
            # 默认：所有层的残差流
            cfg.hookpoints = [f"{layers_name}.{i}" for i in range(N)]

        # 初始化 SAE
        for hook in self.local_hookpoints():
            self.saes[hook] = SparseCoder(input_widths[hook], cfg.sae, device)

        # 初始化优化器
        self._init_optimizer(cfg)
```

### 7.2 优化器初始化

```python
match cfg.optimizer:
    case "adam":
        # 学习率与 latent 数量成反比
        lr = cfg.lr or 2e-4 / (sae.num_latents / (2**14)) ** 0.5

    case "muon":
        # Muon 优化器，使用 Newton-Schulz 正交化
        self.optimizers = [Muon(muon_params, lr=cfg.lr or 2e-3)]

    case "signum":
        # SignSGD + ScheduleFree
        lr = cfg.lr or 5e-3 / (sae.num_latents / (2**14)) ** 0.5
        opt = ScheduleFreeWrapper(SignSGD(pgs), momentum=0.95)
```

### 7.3 核心训练循环 fit()

```python
def fit(self):
    # 冻结模型
    self.model.requires_grad_(False)

    # 定义 hook 函数（核心！）
    def hook(module: nn.Module, inputs, outputs):
        # 1. 获取激活
        name = module_to_name[module]
        outputs = outputs.flatten(0, 1)  # (batch * seq, dim)

        # 2. 分布式：收集所有 rank 的激活
        if self.cfg.distribute_modules:
            dist.all_gather_into_tensor(world_outputs, outputs)

        # 3. 初始化偏置（第一次迭代）
        if self.global_step == 0:
            raw.b_dec.data = outputs.mean(0)

        # 4. 前向传播 SAE
        out = wrapped(x=inputs, y=outputs, dead_mask=dead_mask)

        # 5. 计算损失并反向传播
        loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss
        loss.backward()

        # 6. 对于端到端训练，返回修改后的激活
        if self.cfg.loss_fn in ("ce", "kl"):
            return out.sae_out

    # 注册 hooks
    handles = [mod.register_forward_hook(hook) for mod in name_to_module.values()]

    # 主训练循环
    for batch in dl:
        x = batch["input_ids"].to(device)

        # 前向传播（触发 hooks）
        self.model(x)

        # 梯度累积完成后更新
        if substep == 0:
            # 移除与解码器方向平行的梯度
            for sae in self.saes.values():
                sae.remove_gradient_parallel_to_decoder_directions()

            # 优化器步骤
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad()
```

### 7.4 Hook 机制详解

PyTorch 的 forward hook 允许我们拦截模块的输出：

```python
def hook(module, inputs, outputs):
    # inputs: 模块的输入
    # outputs: 模块的输出

    # 对于端到端训练，可以返回修改后的输出
    return modified_outputs
```

Hook 在 `model.forward()` 过程中被自动调用，这使得我们可以：
1. 捕获任意层的激活
2. 用 SAE 重建的激活替换原始激活（端到端训练）

### 7.5 分布式训练

```python
def distribute_modules(self):
    """将模块分布到不同 rank"""
    if not self.cfg.distribute_modules:
        return

    layers_per_rank = len(self.cfg.hookpoints) // dist.get_world_size()
    self.module_plan = [
        self.cfg.hookpoints[start : start + layers_per_rank]
        for start in range(0, len(self.cfg.hookpoints), layers_per_rank)
    ]
```

两种分布式模式：
- **DDP**（默认）：每个 GPU 复制所有 SAE，梯度同步
- **distribute_modules**：每个 GPU 只训练部分层的 SAE，激活通过 all_gather 共享

---

## 8. 数据处理：data.py

文件位置：`sparsify/data.py`

### 8.1 chunk_and_tokenize

```python
def chunk_and_tokenize(
    data: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int = 2048,
) -> Dataset:
    """GPT 风格的分块和 tokenize"""

    def _tokenize_fn(x: dict[str, list]):
        # 1. 用 EOS token 连接所有文本
        sep = tokenizer.eos_token
        joined_text = sep.join([""] + x[text_key])

        # 2. Tokenize 并分块
        output = tokenizer(
            joined_text,
            max_length=chunk_size,
            return_overflowing_tokens=True,
            truncation=True,
        )

        # 3. 丢弃最后一个不完整的块
        output = {k: v[:-1] for k, v in output.items()}
        return output

    return data.map(_tokenize_fn, batched=True, batch_size=2048)
```

### 8.2 MemmapDataset

```python
class MemmapDataset(TorchDataset):
    """基于内存映射的数据集，用于大规模预 tokenize 数据"""

    def __init__(self, data_path: str, ctx_len: int):
        # 加载为只读内存映射
        self.mmap = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.mmap = self.mmap.reshape(-1, ctx_len)

    def __getitem__(self, idx):
        return {"input_ids": torch.from_numpy(self.mmap[idx].astype(np.int64))}
```

---

## 9. 优化器：sign_sgd.py 和 muon.py

### 9.1 SignSGD

文件位置：`sparsify/sign_sgd.py`

```python
class SignSGD(Optimizer):
    """L∞ 范数下的最陡下降"""

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    # 只使用梯度的符号
                    p.add_(p.grad.sign(), alpha=-lr)
```

SignSGD 的优势：
- 对梯度大小不敏感，只关注方向
- 更新大小一致，适合大规模稀疏参数

### 9.2 Muon 优化器

文件位置：`sparsify/muon.py`

Muon (MomentUm Orthogonalized by Newton-schulz) 是一种使用谱范数的广义最陡下降优化器。

```python
def quintic_newtonschulz(G: Tensor, steps: int) -> Tensor:
    """Newton-Schulz 迭代计算 G 的正交化"""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / X.norm(dim=(-2, -1), keepdim=True)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    return X
```

Muon 的核心思想：
1. 对梯度应用动量
2. 使用 Newton-Schulz 迭代正交化梯度
3. 按谱范数约束更新步长

---

## 10. 性能优化：fused_encoder.py

文件位置：`sparsify/fused_encoder.py`

### 10.1 FusedEncoder 自定义 autograd 函数

```python
class FusedEncoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, k, activation):
        # 1. 线性层 + ReLU
        preacts = F.relu(F.linear(input, weight, bias))

        # 2. TopK 选择
        if activation == "topk":
            values, indices = torch.topk(preacts, k, dim=-1)
        elif activation == "groupmax":
            values, indices = preacts.unflatten(-1, (k, -1)).max(dim=-1)

        # 保存用于反向传播
        ctx.save_for_backward(input, weight, bias, indices)
        return values, indices, preacts
```

### 10.2 优化的反向传播

```python
@staticmethod
def backward(ctx, grad_values, grad_indices, grad_preacts):
    input, weight, bias, indices = ctx.saved_tensors

    # 对输入的梯度：使用 embedding_bag（稀疏操作）
    grad_input = F.embedding_bag(
        indices, weight, mode="sum",
        per_sample_weights=grad_values
    )

    # 对权重的梯度：使用 index_add（稀疏操作）
    contributions = grad_values.unsqueeze(2) * input.unsqueeze(1)
    grad_weight.index_add_(0, indices.flatten(), contributions)
```

关键优化点：
- 只计算被选中的 top-k 个 latent 的梯度
- 使用 `embedding_bag` 和 `index_add_` 进行稀疏操作
- 避免完整的 (N, M) 矩阵运算

---

## 11. 工具函数：utils.py

文件位置：`sparsify/utils.py`

### 11.1 get_layer_list

```python
def get_layer_list(model: PreTrainedModel) -> tuple[str, nn.ModuleList]:
    """自动检测模型的层列表"""
    N = model.config.num_hidden_layers
    # 找到长度为 N 的 ModuleList
    for name, mod in model.base_model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == N:
            return name, mod
```

### 11.2 resolve_widths

```python
def resolve_widths(model, module_names) -> dict[str, int]:
    """通过一次前向传播确定各模块的输出维度"""

    def hook(module, _, output):
        shapes[module_to_name[module]] = output.shape[-1]

    # 注册 hooks
    handles = [mod.register_forward_hook(hook) for mod in modules]

    # 用 dummy 输入做一次前向传播
    model(**model.dummy_inputs)

    return shapes
```

### 11.3 解码器实现

```python
# 默认实现：使用 embedding_bag
def eager_decode(top_indices, top_acts, W_dec):
    return F.embedding_bag(top_indices, W_dec.mT,
                           per_sample_weights=top_acts, mode="sum")

# Triton 加速实现（如果可用）
def triton_decode(top_indices, top_acts, W_dec):
    return xformers_embedding_bag(top_indices, W_dec.mT, top_acts)
```

---

## 12. 完整训练流程

### 12.1 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                         训练流程                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 初始化                                                       │
│     ├── 加载预训练模型（冻结）                                     │
│     ├── 加载数据集并 tokenize                                     │
│     └── 初始化 SAE 和优化器                                       │
│                                                                 │
│  2. 训练循环                                                      │
│     ┌──────────────────────────────────────────────────────┐    │
│     │  for batch in dataloader:                            │    │
│     │      │                                               │    │
│     │      ▼                                               │    │
│     │  ┌─────────────────────────────────────────────┐     │    │
│     │  │ model.forward(input_ids)                    │     │    │
│     │  │     │                                       │     │    │
│     │  │     ├── Layer 0 ──> Hook 捕获激活           │     │    │
│     │  │     │                  │                    │     │    │
│     │  │     │           ┌──────▼──────┐             │     │    │
│     │  │     │           │ SAE.forward │             │     │    │
│     │  │     │           │  - encode   │             │     │    │
│     │  │     │           │  - decode   │             │     │    │
│     │  │     │           │  - loss     │             │     │    │
│     │  │     │           │  - backward │             │     │    │
│     │  │     │           └─────────────┘             │     │    │
│     │  │     │                                       │     │    │
│     │  │     ├── Layer 1 ──> Hook ...               │     │    │
│     │  │     ├── ...                                │     │    │
│     │  │     └── Layer N ──> Hook ...               │     │    │
│     │  └─────────────────────────────────────────────┘     │    │
│     │      │                                               │    │
│     │      ▼                                               │    │
│     │  optimizer.step() (每 grad_acc_steps 步)             │    │
│     │      │                                               │    │
│     │      ▼                                               │    │
│     │  save_checkpoint() (每 save_every 步)                │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 单个 Batch 的详细流程

```python
# 1. 输入数据
input_ids: (batch_size, seq_len)  # e.g., (32, 2048)

# 2. 模型前向传播（触发 hooks）
for layer in model.layers:
    hidden_states = layer(hidden_states)
    # Hook 被调用:
    #   - 捕获 hidden_states: (32, 2048, hidden_dim)
    #   - 展平: (32 * 2048, hidden_dim) = (65536, hidden_dim)

# 3. SAE 前向传播（在 hook 内）
# 编码
pre_acts = ReLU(hidden_states @ W_enc.T + b_enc)  # (65536, num_latents)
top_acts, top_indices = topk(pre_acts, k=32)       # (65536, 32)

# 解码（稀疏）
sae_out = sparse_decode(top_acts, top_indices, W_dec) + b_dec  # (65536, hidden_dim)

# 4. 计算损失
residual = hidden_states - sae_out
fvu = residual.pow(2).sum() / hidden_states.var() * hidden_dim

# 5. 反向传播（只通过 SAE）
fvu.backward()

# 6. 优化器更新（累积后）
optimizer.step()
```

### 12.3 关键数据形状

| 变量 | 形状 | 说明 |
|------|------|------|
| input_ids | (B, S) | 输入 token IDs |
| hidden_states | (B, S, D) | 模型隐藏状态 |
| hidden_states_flat | (B*S, D) | 展平后的隐藏状态 |
| W_enc | (M, D) | 编码器权重 |
| W_dec | (M, D) | 解码器权重 |
| pre_acts | (B*S, M) | 预激活值 |
| top_acts | (B*S, k) | Top-k 激活值 |
| top_indices | (B*S, k) | Top-k 索引 |
| sae_out | (B*S, D) | SAE 重建输出 |

其中：
- B = batch_size
- S = seq_len
- D = hidden_dim
- M = num_latents (= D * expansion_factor)
- k = 稀疏度参数

---

## 附录：常用命令速查

```bash
# 基础训练
python -m sparsify EleutherAI/pythia-160m

# 指定数据集
python -m sparsify gpt2 togethercomputer/RedPajama-Data-1T-Sample

# 训练 transcoder
python -m sparsify gpt2 --transcode

# 自定义 hookpoints
python -m sparsify gpt2 --hookpoints "h.*.attn" "h.*.mlp.act"

# 分布式训练
torchrun --nproc_per_node 8 -m sparsify meta-llama/Meta-Llama-3-8B \
    --distribute_modules --batch_size 1 --grad_acc_steps 8

# 查看所有选项
python -m sparsify --help
```
