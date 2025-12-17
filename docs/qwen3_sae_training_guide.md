# Qwen3-8B SAE 训练完整配置指南

## 1. Qwen3-8B 模型结构

### 1.1 模型基本参数

| 参数 | 值 |
|------|-----|
| num_hidden_layers | 36 |
| hidden_size | 4096 |
| intermediate_size | 12288 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 (GQA) |

### 1.2 单层结构 (Qwen2DecoderLayer)

```
layers.X/
├── input_layernorm          # RMSNorm, 输出维度: 4096
├── self_attn/               # Qwen2Attention
│   ├── q_proj               # Linear(4096 -> 4096)
│   ├── k_proj               # Linear(4096 -> 1024)  # GQA
│   ├── v_proj               # Linear(4096 -> 1024)  # GQA
│   └── o_proj               # Linear(4096 -> 4096)
├── post_attention_layernorm # RMSNorm, 输出维度: 4096
└── mlp/                     # Qwen2MLP
    ├── gate_proj            # Linear(4096 -> 12288)
    ├── up_proj              # Linear(4096 -> 12288)
    └── down_proj            # Linear(12288 -> 4096)
```

### 1.3 数据流

```
hidden_states (4096)
    │
    ▼
input_layernorm ──────────────────────────┐
    │                                     │
    ▼                                     │
┌─────────────────────────────────────┐   │
│ self_attn                           │   │
│   q_proj ─┐                         │   │
│   k_proj ─┼─► Attention ─► o_proj ──┼───┼─► + (残差)
│   v_proj ─┘       ▲                 │   │
│                   │                 │   │
│            [o_proj 的输入]          │   │
└─────────────────────────────────────┘   │
    │                                     │
    ▼                                     │
post_attention_layernorm ─────────────────┤
    │                                     │
    ▼ [gate_up 的输入]                    │
┌─────────────────────────────────────┐   │
│ mlp                                 │   │
│   gate_proj ─► SiLU ─┐              │   │
│                      ├─► * ─► down_proj ─► + (残差)
│   up_proj ───────────┘              │   │
└─────────────────────────────────────┘   │
    │                                     │
    ▼                                     │
output hidden_states (4096)
```

---

## 2. 你的训练目标分析

### 2.1 gate_up 矩阵乘输入

**目标位置**: MLP 的输入，即 `gate_proj` 和 `up_proj` 的共同输入

**推荐 hookpoint**: `layers.X.post_attention_layernorm`
- 原因：`post_attention_layernorm` 的输出就是 MLP 的输入
- 模式：`--hook_mode output`（默认）
- 输入维度：4096

### 2.2 o 矩阵乘输入

**目标位置**: `o_proj` 的输入（attention 计算后、投影前的激活）

**推荐 hookpoint**: `layers.X.self_attn.o_proj`
- 原因：直接 hook 到 o_proj，可以获取其输入
- 模式：`--hook_mode input`
- 输入维度：4096

**重要说明**：使用 `--hook_mode input` 可以在模块输入上训练 autoencoder。

---

## 3. 完整配置参数说明

### 3.1 所有可用参数

```bash
python -m sparsify --help
```

以下是所有参数的详细说明：

#### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | str | `HuggingFaceTB/SmolLM2-135M` | 模型名称或路径 |
| `dataset` | str | `EleutherAI/SmolLM2-135M-10B` | 数据集名称或路径 |
| `split` | str | `train` | 数据集分割 |
| `ctx_len` | int | `2048` | 上下文长度 |
| `hf_token` | str | `None` | HuggingFace API token |
| `revision` | str | `None` | 模型版本 |
| `load_in_8bit` | bool | `False` | 是否 8-bit 量化加载 |
| `max_examples` | int | `None` | 最大样本数 |
| `resume` | bool | `False` | 是否从检查点恢复 |
| `text_column` | str | `text` | 文本列名 |
| `shuffle_seed` | int | `42` | 数据集打乱种子 |
| `data_preprocessing_num_proc` | int | `cpu_count()//2` | 数据预处理进程数 |
| `data_args` | str | `""` | 传给 HF datasets 的额外参数 |

#### SAE 架构参数 (`sae.*`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sae.activation` | str | `topk` | 激活函数：`topk` 或 `groupmax` |
| `sae.expansion_factor` | int | `32` | 扩展因子 (latent 维度 = input × factor) |
| `sae.normalize_decoder` | bool | `True` | 是否归一化解码器权重为单位范数 |
| `sae.num_latents` | int | `0` | latent 数量 (0 表示使用 expansion_factor) |
| `sae.k` | int | `32` | TopK 的 k 值（激活的 latent 数量）|
| `sae.multi_topk` | bool | `False` | 是否使用 Multi-TopK 损失 |
| `sae.skip_connection` | bool | `False` | 是否添加线性跳跃连接 |

#### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_size` | int | `32` | 批次大小（序列数）|
| `grad_acc_steps` | int | `1` | 梯度累积步数 |
| `micro_acc_steps` | int | `1` | 微批次累积（节省显存）|
| `loss_fn` | str | `fvu` | 损失函数：`fvu`, `ce`, `kl` |
| `optimizer` | str | `signum` | 优化器：`signum`, `adam`, `muon` |
| `lr` | float | `None` | 学习率（None 表示自动计算）|
| `lr_warmup_steps` | int | `1000` | 学习率预热步数（仅 adam）|
| `k_decay_steps` | int | `0` | k 值衰减步数（实验性）|
| `auxk_alpha` | float | `0.0` | AuxK 损失权重（激活死特征）|
| `dead_feature_threshold` | int | `10000000` | 死特征阈值（token 数）|
| `exclude_tokens` | list | `[]` | 排除的 token ID 列表 |

#### Hookpoint 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hookpoints` | list | `[]` | hookpoint 列表（支持 glob 模式）|
| `hook_mode` | str | `output` | hook 模式：`output`（默认）, `input`, `transcode` |
| `layers` | list | `[]` | 层索引列表 |
| `layer_stride` | int | `1` | 层间隔 |
| `init_seeds` | list | `[0]` | 初始化种子列表 |

#### 分布式训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `distribute_modules` | bool | `False` | 是否将 SAE 分布到不同 GPU |

#### 保存和日志参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_every` | int | `1000` | 保存间隔（步数）|
| `save_best` | bool | `False` | 是否保存最佳检查点 |
| `save_dir` | str | `checkpoints` | 保存目录 |
| `finetune` | str | `None` | 微调的预训练 SAE 路径 |
| `log_to_wandb` | bool | `True` | 是否记录到 WandB |
| `run_name` | str | `None` | 运行名称 |
| `wandb_log_frequency` | int | `1` | WandB 日志频率 |

---

## 4. 训练脚本示例

### 4.1 训练 gate_up 输入的 SAE (单层示例：第 16 层)

```bash
python -m sparsify \
    Qwen/Qwen3-8B \
    HuggingFaceFW/fineweb \
    \
    --split "train" \
    --ctx_len 2048 \
    --max_examples 1000000 \
    --text_column "text" \
    --shuffle_seed 42 \
    --data_preprocessing_num_proc 8 \
    --data_args "name=sample-10BT" \
    \
    --sae.activation "topk" \
    --sae.expansion_factor 32 \
    --sae.normalize_decoder True \
    --sae.num_latents 0 \
    --sae.k 32 \
    --sae.multi_topk False \
    --sae.skip_connection False \
    \
    --hookpoints "layers.16.post_attention_layernorm" \
    --hook_mode output \
    --init_seeds 0 \
    \
    --batch_size 8 \
    --grad_acc_steps 4 \
    --micro_acc_steps 2 \
    --loss_fn "fvu" \
    --optimizer "signum" \
    --lr 5e-3 \
    --auxk_alpha 0.03125 \
    --dead_feature_threshold 10000000 \
    \
    --save_every 1000 \
    --save_best True \
    --save_dir "checkpoints" \
    --run_name "qwen3-8b-layer16-mlp-input-sae" \
    --log_to_wandb True \
    --wandb_log_frequency 1
```

### 4.2 训练 o_proj 输入的 SAE (单层示例：第 16 层)

```bash
python -m sparsify \
    /model-weights/Qwen3-8BB \
    /mnt/data/fineweb-edu/sample/10BT \
    \
    --split "train" \
    --ctx_len 2048 \
    --max_examples 1000000 \
    --text_column "text" \
    --shuffle_seed 42 \
    --data_preprocessing_num_proc 8 \
    \
    --sae.activation "topk" \
    --sae.expansion_factor 32 \
    --sae.normalize_decoder True \
    --sae.num_latents 0 \
    --sae.k 32 \
    --sae.multi_topk False \
    --sae.skip_connection False \
    \
    --hookpoints "layers.16.self_attn.o_proj" \
    --hook_mode input \
    --init_seeds 0 \
    \
    --batch_size 8 \
    --grad_acc_steps 4 \
    --micro_acc_steps 2 \
    --loss_fn "fvu" \
    --optimizer "signum" \
    --lr 5e-3 \
    --auxk_alpha 0.03125 \
    --dead_feature_threshold 10000000 \
    \
    --save_every 1000 \
    --save_best True \
    --save_dir "checkpoints" \
    --run_name "qwen3-8b-layer16-o-proj-input-sae" \
    --log_to_wandb True \
    --wandb_log_frequency 1
```

### 4.3 同时训练两个位置 (多 hookpoints)

```bash
python -m sparsify \
    Qwen/Qwen3-8B \
    HuggingFaceFW/fineweb \
    \
    --split "train" \
    --ctx_len 2048 \
    --max_examples 1000000 \
    --text_column "text" \
    --shuffle_seed 42 \
    --data_preprocessing_num_proc 8 \
    --data_args "name=sample-10BT" \
    \
    --sae.activation "topk" \
    --sae.expansion_factor 32 \
    --sae.normalize_decoder True \
    --sae.num_latents 0 \
    --sae.k 32 \
    --sae.multi_topk False \
    --sae.skip_connection False \
    \
    --hookpoints "layers.16.post_attention_layernorm" "layers.16.self_attn" \
    --hook_mode output \
    --init_seeds 0 \
    \
    --batch_size 8 \
    --grad_acc_steps 4 \
    --micro_acc_steps 2 \
    --loss_fn "fvu" \
    --optimizer "signum" \
    --lr 5e-3 \
    --auxk_alpha 0.03125 \
    --dead_feature_threshold 10000000 \
    \
    --save_every 1000 \
    --save_best True \
    --save_dir "checkpoints" \
    --run_name "qwen3-8b-layer16-multi-hookpoints" \
    --log_to_wandb True \
    --wandb_log_frequency 1
```

### 4.4 多层训练 (使用 glob 模式)

```bash
# 训练所有层的 post_attention_layernorm (MLP 输入)
python -m sparsify \
    Qwen/Qwen3-8B \
    HuggingFaceFW/fineweb \
    \
    --split "train" \
    --ctx_len 2048 \
    --max_examples 1000000 \
    --text_column "text" \
    --shuffle_seed 42 \
    --data_preprocessing_num_proc 8 \
    --data_args "name=sample-10BT" \
    \
    --sae.activation "topk" \
    --sae.expansion_factor 32 \
    --sae.normalize_decoder True \
    --sae.num_latents 0 \
    --sae.k 32 \
    --sae.multi_topk False \
    --sae.skip_connection False \
    \
    --hookpoints "layers.*.post_attention_layernorm" \
    --hook_mode output \
    --layer_stride 4 \
    --init_seeds 0 \
    \
    --batch_size 4 \
    --grad_acc_steps 8 \
    --micro_acc_steps 4 \
    --loss_fn "fvu" \
    --optimizer "signum" \
    --lr 5e-3 \
    --auxk_alpha 0.03125 \
    --dead_feature_threshold 10000000 \
    \
    --save_every 1000 \
    --save_best True \
    --save_dir "checkpoints" \
    --run_name "qwen3-8b-all-layers-mlp-input" \
    --log_to_wandb True \
    --wandb_log_frequency 1
```

---

## 5. 分布式训练

### 5.1 多 GPU DDP 训练 (权重复制)

```bash
torchrun --nproc_per_node 8 -m sparsify \
    Qwen/Qwen3-8B \
    HuggingFaceFW/fineweb \
    \
    --split "train" \
    --ctx_len 2048 \
    --max_examples 1000000 \
    --text_column "text" \
    --shuffle_seed 42 \
    --data_preprocessing_num_proc 8 \
    --data_args "name=sample-10BT" \
    \
    --sae.activation "topk" \
    --sae.expansion_factor 32 \
    --sae.normalize_decoder True \
    --sae.num_latents 0 \
    --sae.k 32 \
    --sae.multi_topk False \
    --sae.skip_connection False \
    \
    --hookpoints "layers.16.post_attention_layernorm" \
    --hook_mode output \
    --init_seeds 0 \
    \
    --batch_size 4 \
    --grad_acc_steps 8 \
    --micro_acc_steps 2 \
    --loss_fn "fvu" \
    --optimizer "signum" \
    --lr 5e-3 \
    --auxk_alpha 0.03125 \
    --dead_feature_threshold 10000000 \
    \
    --load_in_8bit True \
    \
    --save_every 1000 \
    --save_best True \
    --save_dir "checkpoints" \
    --run_name "qwen3-8b-layer16-mlp-input-ddp" \
    --log_to_wandb True \
    --wandb_log_frequency 1
```

### 5.2 多 GPU 分布模块训练 (SAE 分布到不同 GPU)

```bash
# 注意：hookpoints 数量必须能被 GPU 数量整除
torchrun --nproc_per_node 8 -m sparsify \
    Qwen/Qwen3-8B \
    HuggingFaceFW/fineweb \
    \
    --split "train" \
    --ctx_len 2048 \
    --max_examples 1000000 \
    --text_column "text" \
    --shuffle_seed 42 \
    --data_preprocessing_num_proc 8 \
    --data_args "name=sample-10BT" \
    \
    --sae.activation "topk" \
    --sae.expansion_factor 32 \
    --sae.normalize_decoder True \
    --sae.num_latents 0 \
    --sae.k 32 \
    --sae.multi_topk False \
    --sae.skip_connection False \
    \
    --hookpoints "layers.*.post_attention_layernorm" \
    --hook_mode output \
    --layer_stride 1 \
    --init_seeds 0 \
    \
    --distribute_modules True \
    \
    --batch_size 2 \
    --grad_acc_steps 16 \
    --micro_acc_steps 4 \
    --loss_fn "fvu" \
    --optimizer "signum" \
    --lr 5e-3 \
    --auxk_alpha 0.03125 \
    --dead_feature_threshold 10000000 \
    \
    --load_in_8bit True \
    \
    --save_every 1000 \
    --save_best True \
    --save_dir "checkpoints" \
    --run_name "qwen3-8b-all-layers-mlp-input-distributed" \
    --log_to_wandb True \
    --wandb_log_frequency 1
```

---

## 6. Python API 使用

如果你需要更灵活的控制，可以使用 Python API：

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

# ============ 配置 ============
MODEL_NAME = "Qwen/Qwen3-8B"
DATASET_NAME = "HuggingFaceFW/fineweb"

# SAE 配置
sae_config = SaeConfig(
    activation="topk",           # 激活函数
    expansion_factor=32,         # 扩展因子
    normalize_decoder=True,      # 归一化解码器
    num_latents=0,               # 0 表示使用 expansion_factor
    k=32,                        # TopK 的 k 值
    multi_topk=False,            # Multi-TopK 损失
    skip_connection=False,       # 跳跃连接
)

# 训练配置
train_config = TrainConfig(
    sae=sae_config,

    # 训练参数
    batch_size=8,
    grad_acc_steps=4,
    micro_acc_steps=2,

    # 损失和优化
    loss_fn="fvu",
    optimizer="signum",
    lr=5e-3,
    lr_warmup_steps=1000,

    # 稀疏性相关
    k_decay_steps=0,
    auxk_alpha=0.03125,
    dead_feature_threshold=10_000_000,
    exclude_tokens=[],

    # Hookpoints
    hookpoints=["layers.16.post_attention_layernorm"],
    hook_mode="output",  # "output", "input", 或 "transcode"
    layers=[],
    layer_stride=1,
    init_seeds=[0],

    # 分布式
    distribute_modules=False,

    # 保存和日志
    save_every=1000,
    save_best=True,
    save_dir="checkpoints",
    finetune=None,
    log_to_wandb=True,
    run_name="qwen3-8b-layer16-mlp-input-python",
    wandb_log_frequency=1,
)

# ============ 加载数据 ============
dataset = load_dataset(
    DATASET_NAME,
    name="sample-10BT",
    split="train",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenized = chunk_and_tokenize(
    dataset,
    tokenizer,
    max_seq_len=2048,
)

# 打乱并限制样本数
tokenized = tokenized.shuffle(seed=42)
tokenized = tokenized.select(range(min(1_000_000, len(tokenized))))
tokenized = tokenized.with_format("torch")

# ============ 加载模型 ============
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# ============ 训练 ============
trainer = Trainer(train_config, tokenized, model)
trainer.fit()
```

---

## 7. Hookpoint 快速参考

### 7.1 常用 Hookpoint 对照表

| 你想要的位置 | Hookpoint | hook_mode | 说明 |
|-------------|-----------|-----------|------|
| 层输入 (残差流) | `layers.X` | `output` | 层的完整输出 |
| Attention 输入 | `layers.X.input_layernorm` | `output` | LayerNorm 后 |
| Attention 输出 | `layers.X.self_attn` | `output` | 含 o_proj |
| **o_proj 输入** | `layers.X.self_attn.o_proj` | **`input`** | Attention 内部 |
| **MLP/gate_up 输入** | `layers.X.post_attention_layernorm` | **`output`** | LayerNorm 后 |
| MLP 输出 | `layers.X.mlp` | `output` | 含 down_proj |
| down_proj 输入 | `layers.X.mlp.down_proj` | `input` | MLP 内部 |

### 7.2 Glob 模式示例

```bash
# 所有层的某个模块
--hookpoints "layers.*.post_attention_layernorm"

# 特定层范围 (0-9)
--hookpoints "layers.[0-9].post_attention_layernorm"

# 多个模块
--hookpoints "layers.16.post_attention_layernorm" "layers.16.self_attn"

# 所有 attention
--hookpoints "layers.*.self_attn"

# 所有 MLP
--hookpoints "layers.*.mlp"
```

---

## 8. 显存优化建议

对于 Qwen3-8B (约 16GB 权重)：

| 配置 | 预估显存 | 建议 |
|------|----------|------|
| 单 GPU, fp16 | ~40GB | 减小 batch_size, 增加 micro_acc_steps |
| 单 GPU, 8-bit | ~24GB | 使用 `--load_in_8bit` |
| 多 GPU, DDP | ~40GB/GPU | 增加 grad_acc_steps |
| 多 GPU, distribute | ~20GB/GPU | 使用 `--distribute_modules` |

推荐配置组合：
```bash
# 单 40GB GPU
--batch_size 4 --grad_acc_steps 8 --micro_acc_steps 4

# 单 24GB GPU (8-bit)
--load_in_8bit --batch_size 2 --grad_acc_steps 16 --micro_acc_steps 4

# 8x 40GB GPU (distributed)
--distribute_modules --batch_size 2 --grad_acc_steps 8 --micro_acc_steps 2
```

---

## 9. 常见问题

### Q1: hook_mode 的三种模式有什么区别？

| hook_mode | SAE 输入 | SAE 目标 | 用途 |
|-----------|----------|----------|------|
| `output` | 模块输出 | 模块输出 | 默认，在模块输出上训练 autoencoder |
| `input` | 模块输入 | 模块输入 | 在模块输入上训练 autoencoder |
| `transcode` | 模块输入 | 模块输出 | 从输入预测输出的 transcoder |

### Q2: 什么时候使用 hook_mode=input？

当你想要训练的激活是某个模块的**输入**而不是输出时使用。例如：
- `o_proj` 的输入（attention 计算后、投影前的激活）
- `down_proj` 的输入（MLP 中间激活）

### Q3: 如何选择 k 值和 expansion_factor？

- `expansion_factor=32`: 标准设置，latent 维度 = 4096 × 32 = 131072
- `k=32`: 每个 token 激活 32 个 latent (稀疏度 32/131072 ≈ 0.02%)

更大的 k 值会降低重建误差但减少稀疏性；更大的 expansion_factor 会增加表达能力但需要更多计算。

### Q4: auxk_alpha 有什么作用？

`auxk_alpha` 控制 AuxK 损失的权重，用于激活"死特征"（长时间不被激活的 latent）。推荐值为 `1/32 = 0.03125`。
