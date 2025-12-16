# Sparsify Trainer 代码走读

本文档详细分析 `sparsify/trainer.py` 的代码结构和训练流程。

---

## 1. 文件概览

```
trainer.py
├── Trainer 类
│   ├── __init__()        # 初始化：解析 hookpoints、创建 SAE、设置优化器
│   ├── load_state()      # 从检查点恢复训练状态
│   ├── get_current_k()   # 获取当前 k 值（支持 k 衰减）
│   ├── fit()             # 主训练循环
│   ├── local_hookpoints() # 获取当前 rank 负责的 hookpoints
│   ├── maybe_all_cat()   # 分布式：跨进程拼接张量
│   ├── maybe_all_reduce() # 分布式：跨进程归约
│   ├── distribute_modules() # 分布式：规划模块分配
│   ├── _checkpoint()     # 保存检查点
│   ├── save()            # 保存当前状态
│   └── save_best()       # 保存最佳模型
└── SaeTrainer = Trainer  # 别名（向后兼容）
```

---

## 2. Trainer.__init__() 初始化流程

**代码位置**: `trainer.py:28-184`

```
__init__(cfg, dataset, model)
│
├─► [1] 保存模型引用
│       self.model = model
│
├─► [2] 解析 hookpoints
│       │
│       ├─► 如果指定了 hookpoints（支持 glob 模式）
│       │       for name in model.named_modules():
│       │           if fnmatchcase(name, pattern):
│       │               hookpoints.append(name)
│       │
│       └─► 否则使用 layers 参数
│               hookpoints = [f"{layers_name}.{i}" for i in layers]
│
├─► [3] 应用 layer_stride
│       cfg.hookpoints = cfg.hookpoints[::cfg.layer_stride]
│
├─► [4] 分布式模块规划
│       self.distribute_modules()
│
├─► [5] 解析各 hookpoint 的输入维度
│       input_widths = resolve_widths(model, cfg.hookpoints)
│
├─► [6] 初始化 SAE
│       for hook in local_hookpoints():
│           for seed in init_seeds:
│               self.saes[name] = SparseCoder(input_width, cfg.sae, ...)
│
├─► [7] 设置优化器
│       │
│       ├─► adam: Adam8bit + 线性 warmup scheduler
│       ├─► muon: Muon + Adam (bias) + warmup scheduler
│       └─► signum: SignSGD + ScheduleFree (无 scheduler)
│
├─► [8] 初始化训练状态
│       self.global_step = 0
│       self.num_tokens_since_fired = {...}  # 死特征检测
│       self.exclude_tokens = tensor([...])
│       self.initial_k, self.final_k = ...   # k 衰减
│       self.best_loss = {...}
│
└─► 初始化完成
```

### 2.1 Hookpoint 解析示例

```python
# 输入 glob 模式
cfg.hookpoints = ["layers.*.mlp", "layers.*.self_attn"]

# 匹配结果（假设 36 层模型）
cfg.hookpoints = [
    "layers.0.mlp", "layers.0.self_attn",
    "layers.1.mlp", "layers.1.self_attn",
    ...
    "layers.35.mlp", "layers.35.self_attn"
]

# 应用 layer_stride=4
cfg.hookpoints = [
    "layers.0.mlp", "layers.0.self_attn",
    "layers.4.mlp", "layers.4.self_attn",
    ...
]
```

### 2.2 优化器配置对比

| 优化器 | 学习率公式 | Scheduler | 特点 |
|--------|-----------|-----------|------|
| adam | `2e-4 / sqrt(num_latents / 2^14)` | 线性 warmup | 标准，需要更多显存 |
| muon | `2e-3`（固定） | 线性 warmup | 二阶优化，更快收敛 |
| signum | `5e-3 / sqrt(num_latents / 2^14)` | ScheduleFree | 默认，显存友好 |

---

## 3. Trainer.fit() 主训练循环

**代码位置**: `trainer.py:237-596`

### 3.1 训练前准备

```
fit()
│
├─► [1] 设置精度
│       torch.set_float32_matmul_precision("high")  # 启用 TF32
│
├─► [2] 冻结模型
│       self.model.requires_grad_(False)
│
├─► [3] 判断分布式模式
│       rank_zero = not dist.is_initialized() or dist.get_rank() == 0
│       ddp = dist.is_initialized() and not self.cfg.distribute_modules
│
├─► [4] 初始化 WandB（仅 rank 0）
│
├─► [5] 创建 DataLoader
│       dl = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
│
├─► [6] 初始化统计变量
│       did_fire = {name: zeros(num_latents)}  # 特征激活标记
│       avg_fvu = defaultdict(float)           # 平均 FVU
│       avg_auxk_loss = defaultdict(float)     # 平均 AuxK 损失
│
├─► [7] CE 损失特殊处理
│       if loss_fn == "ce":
│           clean_loss = model(x, labels=x).loss  # 计算初始 CE
│           if transcode:
│               # 端到端 transcoder：替换模块为 Identity
│               set_submodule(model, hookpoint, nn.Identity())
│
├─► [8] 构建模块映射
│       name_to_module = {name: model.get_submodule(name) for ...}
│       module_to_name = {module: name for ...}
│
└─► 进入主循环
```

### 3.2 主训练循环

```
for batch in dl:
    │
    ├─► [1] 准备输入
    │       x = batch["input_ids"].to(device)
    │       tokens_mask = ~isin(x, exclude_tokens)  # 排除特定 token
    │
    ├─► [2] 首次迭代：DDP 包装 SAE
    │       if not maybe_wrapped:
    │           maybe_wrapped = {name: DDP(sae) for ...} if ddp else self.saes
    │
    ├─► [3] 统计 token 数量
    │       num_tokens_in_step += tokens_mask.sum()
    │
    ├─► [4] KL 损失：计算 clean logits
    │       if loss_fn == "kl":
    │           clean_probs = model(x).logits.softmax(dim=-1)
    │
    ├─► [5] 注册 hooks 并前向传播
    │       handles = [mod.register_forward_hook(hook) for mod in ...]
    │       try:
    │           match loss_fn:
    │               case "ce": model(x, labels=x); backward()
    │               case "kl": compute_kl(); backward()
    │               case "fvu": model(x)  # hook 内部 backward
    │       finally:
    │           for handle in handles: handle.remove()
    │
    ├─► [6] 检查是否需要优化器更新
    │       step, substep = divmod(global_step + 1, grad_acc_steps)
    │       if substep == 0:
    │           │
    │           ├─► 移除平行于 decoder 方向的梯度
    │           │       sae.remove_gradient_parallel_to_decoder_directions()
    │           │
    │           ├─► 优化器更新
    │           │       optimizer.step()
    │           │       optimizer.zero_grad()
    │           │       scheduler.step()
    │           │
    │           ├─► 更新 k 值
    │           │       k = get_current_k()
    │           │       for sae in saes: sae.cfg.k = k
    │           │
    │           ├─► 更新死特征统计
    │           │       for name, counts in num_tokens_since_fired.items():
    │           │           counts += num_tokens_in_step
    │           │           counts[did_fire[name]] = 0
    │           │
    │           ├─► 保存检查点（每 save_every 步）
    │           │
    │           └─► 记录到 WandB（每 wandb_log_frequency 步）
    │
    ├─► [7] 更新全局步数
    │       global_step += 1
    │       pbar.update()
    │
    └─► 继续下一个 batch

训练结束
├─► 保存最终检查点
└─► 关闭进度条
```

### 3.3 Hook 函数（核心）

**代码位置**: `trainer.py:340-447`

```
def hook(module, inputs, outputs):
    │
    ├─► [1] 解包 tuple
    │       inputs = inputs[0]
    │       outputs, *aux_out = outputs
    │
    ├─► [2] 分布式：all_gather 收集数据
    │
    ├─► [3] 展平维度
    │       outputs = outputs.flatten(0, 1)
    │       inputs = inputs.flatten(0, 1) if transcode else outputs
    │           │
    │           └─► 【关键】决定 SAE 输入来源
    │
    ├─► [4] 应用 token mask
    │       outputs = outputs[mask]
    │       inputs = inputs[mask]
    │
    ├─► [5] 首次迭代：初始化偏置
    │       if global_step == 0:
    │           if transcode:
    │               encoder.bias = -mean(inputs) @ encoder.weight.T
    │           b_dec = mean(outputs)
    │
    ├─► [6] 归一化 decoder
    │       if normalize_decoder and not transcode:
    │           sae.set_decoder_norm_to_unit_norm()
    │
    ├─► [7] SAE 前向传播
    │       out = wrapped_sae(x=inputs, y=outputs, dead_mask=...)
    │
    ├─► [8] 更新激活统计
    │       did_fire[name][out.latent_indices] = True
    │
    └─► [9] 损失处理
            │
            ├─► ce/kl 模式：返回修改后的输出
            │       output[mask] = out.sae_out
            │       return output
            │
            └─► fvu 模式：局部反向传播
                    loss = fvu + auxk_alpha * auxk_loss + multi_topk_fvu / 8
                    loss.backward()
```

---

## 4. 关键数据结构

### 4.1 self.saes

```python
self.saes: dict[str, SparseCoder] = {
    "layers.0.mlp": SparseCoder(...),
    "layers.4.mlp": SparseCoder(...),
    ...
}
# 如果使用多个 seed
self.saes = {
    "layers.0.mlp/seed0": SparseCoder(...),
    "layers.0.mlp/seed1": SparseCoder(...),
    ...
}
```

### 4.2 maybe_wrapped

```python
# DDP 模式
maybe_wrapped: dict[str, DDP] = {
    "layers.0.mlp": DDP(sae),
    ...
}

# 非 DDP 模式
maybe_wrapped: dict[str, SparseCoder] = self.saes
```

### 4.3 num_tokens_since_fired

```python
# 死特征检测：记录每个 latent 距离上次激活的 token 数
self.num_tokens_since_fired: dict[str, Tensor] = {
    "layers.0.mlp": tensor([0, 1523, 0, 892341, ...]),  # shape: [num_latents]
    ...
}
# 如果 count > dead_feature_threshold，则认为该特征"死亡"
```

### 4.4 did_fire

```python
# 当前 step 内每个 latent 是否被激活
did_fire: dict[str, Tensor] = {
    "layers.0.mlp": tensor([True, False, True, False, ...]),  # shape: [num_latents]
    ...
}
```

---

## 5. 分布式训练

### 5.1 两种模式对比

| 模式 | 条件 | SAE 分布 | 数据分布 |
|------|------|----------|----------|
| DDP | `dist.is_initialized() and not distribute_modules` | 每个 GPU 有完整副本 | 数据并行 |
| distribute_modules | `distribute_modules=True` | 每个 GPU 负责部分层 | 需要 all_gather |

### 5.2 distribute_modules 流程

```python
# trainer.py:630-646
def distribute_modules(self):
    # 计算每个 rank 负责多少层
    layers_per_rank = len(hookpoints) // world_size

    # 规划分配
    self.module_plan = [
        hookpoints[start:start+layers_per_rank]
        for start in range(0, len(hookpoints), layers_per_rank)
    ]
    # 例如 8 GPU, 32 层:
    # Rank 0: layers.0, layers.1, layers.2, layers.3
    # Rank 1: layers.4, layers.5, layers.6, layers.7
    # ...
```

### 5.3 Hook 中的分布式处理

```python
# trainer.py:358-380
if self.cfg.distribute_modules:
    # 收集所有 GPU 的 outputs
    world_outputs = outputs.new_empty(batch * world_size, ...)
    dist.all_gather_into_tensor(world_outputs, outputs)
    outputs = world_outputs

    # 检查当前 rank 是否负责这个模块
    if name not in self.module_plan[rank]:
        return  # 跳过
```

---

## 6. 检查点保存

### 6.1 保存内容

```
checkpoints/{run_name}/
├── config.json                    # 训练配置
├── state.pt                       # {global_step: int}
├── optimizer_0.pt                 # 优化器状态
├── lr_scheduler_0.pt              # 学习率调度器状态
├── rank_0_state.pt                # {num_tokens_since_fired, best_loss}
├── layers.0.mlp/
│   ├── sae.safetensors           # SAE 权重
│   └── cfg.json                   # SAE 配置
├── layers.4.mlp/
│   └── ...
└── best/                          # 最佳检查点（如果 save_best=True）
    └── ...
```

### 6.2 恢复训练

```python
# 命令行
python -m sparsify ... --resume

# 代码
trainer = Trainer(cfg, dataset, model)
trainer.load_state(checkpoint_path)
trainer.fit()
```

---

## 7. 损失函数

### 7.1 FVU (默认)

**Fraction of Variance Unexplained** - 局部重建损失

```python
# 在 hook 内部计算
loss = out.fvu + auxk_alpha * out.auxk_loss + out.multi_topk_fvu / 8
loss.backward()  # 局部反向传播，梯度只流向 SAE
```

### 7.2 CE (端到端)

**Cross-Entropy Loss** - 端到端训练

```python
# hook 返回修改后的激活
output[mask] = out.sae_out
return output

# 主循环中计算 CE 损失
ce = model(x, labels=x).loss
ce.backward()  # 梯度流经整个模型到 SAE
```

### 7.3 KL (端到端)

**KL Divergence** - 端到端训练

```python
# 先计算 clean logits
clean_probs = model(x).logits.softmax(dim=-1)

# hook 修改激活后，计算 dirty logits
dirty_lps = model(x).logits.log_softmax(dim=-1)

# KL 散度
kl = -torch.sum(clean_probs * dirty_lps, dim=-1).mean()
kl.backward()
```

---

## 8. 完整训练流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Trainer.__init__()                          │
├─────────────────────────────────────────────────────────────────────┤
│  1. 解析 hookpoints (支持 glob 模式)                                  │
│  2. 初始化 SAE (SparseCoder)                                         │
│  3. 设置优化器 (adam/muon/signum)                                     │
│  4. 初始化训练状态                                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           Trainer.fit()                             │
├─────────────────────────────────────────────────────────────────────┤
│  准备阶段:                                                           │
│  1. 冻结模型                                                         │
│  2. 初始化 WandB                                                     │
│  3. 创建 DataLoader                                                  │
│  4. 构建模块映射 (name_to_module, module_to_name)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         主训练循环                                    │
├─────────────────────────────────────────────────────────────────────┤
│  for batch in dataloader:                                           │
│      │                                                              │
│      ├─► 注册 forward hooks                                          │
│      │                                                              │
│      ├─► 模型前向传播 model(x)                                        │
│      │       │                                                      │
│      │       └─► 触发 hook() ──────────────────────────┐             │
│      │                                                 │             │
│      │   ┌─────────────────────────────────────────────┴───────┐    │
│      │   │                    hook()                           │    │
│      │   ├─────────────────────────────────────────────────────┤    │
│      │   │  1. 解包 inputs/outputs                              │    │
│      │   │  2. 展平维度 [B,S,H] → [B*S,H]                        │    │
│      │   │  3. 选择 SAE 输入:                                   │    │
│      │   │     inputs = inputs if transcode else outputs        │    │
│      │   │  4. 首次迭代初始化偏置                                 │    │
│      │   │  5. SAE 前向: out = sae(x=inputs, y=outputs)         │    │
│      │   │  6. 更新激活统计                                      │    │
│      │   │  7. 损失处理:                                        │    │
│      │   │     - fvu: loss.backward() 局部反向传播               │    │
│      │   │     - ce/kl: return 修改后的 outputs                 │    │
│      │   └─────────────────────────────────────────────────────┘    │
│      │                                                              │
│      ├─► 移除 hooks                                                  │
│      │                                                              │
│      ├─► 每 grad_acc_steps 步:                                       │
│      │       optimizer.step()                                       │
│      │       更新死特征统计                                           │
│      │       保存检查点 (每 save_every 步)                            │
│      │       记录 WandB                                              │
│      │                                                              │
│      └─► global_step += 1                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           训练结束                                   │
├─────────────────────────────────────────────────────────────────────┤
│  1. 保存最终检查点                                                    │
│  2. 保存最佳模型 (如果 save_best=True)                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. 关键配置参数速查

| 参数 | 默认值 | 影响 |
|------|--------|------|
| `batch_size` | 32 | 每次前向传播的序列数 |
| `grad_acc_steps` | 1 | 梯度累积步数 |
| `micro_acc_steps` | 1 | SAE 内部微批次（节省显存）|
| `loss_fn` | "fvu" | 损失函数类型 |
| `optimizer` | "signum" | 优化器类型 |
| `lr` | None (自动) | 学习率 |
| `auxk_alpha` | 0.0 | AuxK 损失权重 |
| `dead_feature_threshold` | 10M | 死特征判定阈值 |
| `save_every` | 1000 | 保存间隔 |
| `distribute_modules` | False | 是否跨 GPU 分布 SAE |
