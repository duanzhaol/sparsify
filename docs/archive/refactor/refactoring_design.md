> Archived document: this file is kept for historical reference and may not match the current codebase.
> For current guidance, start from `docs/README.md` and the active docs under `docs/`.

# Sparsify 重构设计文档

## Context

当前代码库从上游 fork 后经历了大量实验性开发，积累了 21 个功能模块，其中只有约一半在实战脚本中实际使用。`trainer.py` 膨胀到 1439 行，条件分支复杂。现在需要重构为一个干净、可读、同时支持 NVIDIA CUDA 和华为 Ascend NPU 的精简基础设施。

**目标**：只保留实际使用的功能，降低代码复杂度，提高可读性和可维护性。

---

## 一、保留与移除清单

### 保留的功能

| 功能 | 说明 |
|------|------|
| TopK 稀疏编码 | 核心算法 |
| FVU 损失 | 唯一损失函数 |
| AuxK 死特征修复 | 所有脚本都用 |
| Decoder 范数约束 | 所有脚本都用 |
| Hookpoint 系统 | glob 模式选择模块，但 hook 行为硬编码为 input 模式 |
| DDP 数据并行 | 标准分布式训练 |
| Signum 优化器 | 唯一优化器 |
| Exceed 评估指标 | 自定义评估体系 |
| Tiling | TiledSparseCoder |
| Hadamard 旋转 | 有实战验证 |
| Resume / Finetune | 训练生命周期 |
| CUDA/NPU 双后端 | device.py 抽象层 |

### 移除的功能

| 功能 | 原因 | 涉及代码 |
|------|------|----------|
| CE/KL 端到端损失 | 无脚本，约 80 行分支 | trainer.py 多处 |
| Transcode 模式 | 无脚本 | trainer.py ~15 处分支 |
| hook_mode 概念 | 只用 input，删除选择逻辑 | config.py, trainer.py |
| GroupMax 激活 | 无脚本 | fused_encoder.py |
| Multi-TopK 损失 | 所有脚本设为 False | trainer.py, sparse_coder.py |
| K-Decay 退火 | 无脚本 | trainer.py |
| Skip Connection | 所有脚本设为 False | sparse_coder.py |
| Outlier Clipping | git log 标记效果差 | outlier_clip.py（整个文件）, trainer.py |
| Distillation + LowRank | 可以后续再加回来 | lowrank_encoder/（整个目录）, trainer.py |
| distribute_modules | 无脚本，约 30 处条件判断 | trainer.py |
| Adam / Muon 优化器 | 无脚本 | muon.py（整个文件）, trainer.py ~70 行 |
| BitsAndBytes 8bit | 仅 CUDA，NPU 不支持 | \_\_main\_\_.py |

---

## 二、重构后的文件结构

```
sparsify/
├── __init__.py              # 公共 API（简化导出）
├── __main__.py              # CLI 入口（简化）
├── config.py                # 配置（大幅精简）
├── data.py                  # 数据加载（不变）
├── device.py                # CUDA/NPU 抽象（修 bug）
├── sparse_coder.py          # SAE 模型（精简）
├── tiled_sparse_coder.py    # Tiled SAE（去除 LowRank 依赖）
├── fused_encoder.py         # TopK 编码器（去除 GroupMax）
├── hadamard.py              # Hadamard 旋转（不变）
├── sign_sgd.py              # SignSGD 优化器（不变）
├── utils.py                 # 工具函数（不变）
├── trainer.py               # 训练主循环（精简 + 拆分）
├── checkpoint.py            # 新文件：检查点保存/加载
└── eval/                    # 评估工具（不变）
    ├── __init__.py
    ├── encoders.py
    ├── pca.py
    └── two_stage.py

删除：
├── outlier_clip.py          # 整个文件
├── muon.py                  # 整个文件
└── lowrank_encoder/         # 整个目录
```

---

## 三、核心设计思路

### 3.1 trainer.py 拆分策略

**现状**：trainer.py 1439 行，包含初始化、训练循环、hook 函数、检查点、日志、分布式工具，全部在一个文件中。

**拆分方案**：拆为 2 个文件，不做深层抽象。

#### trainer.py（约 700 行）

保留 `Trainer` 类的核心逻辑：

```
Trainer
├── __init__()          # 初始化 SAE、优化器、状态
├── fit()               # 训练主循环
├── _create_hook()      # 返回 forward hook 闭包（内联在 fit 中或独立方法）
├── maybe_all_cat()     # DDP 辅助
└── maybe_all_reduce()  # DDP 辅助
```

**关键简化点**：
- `__init__` 中不再有 optimizer 的 match/case 三分支，只有 Signum 一条路径
- `fit()` 中不再有 `match self.cfg.loss_fn` 三分支，只有 FVU 的 partial forward
- hook 函数中不再有 `match self.cfg.hook_mode` 三分支，直接取 input
- 不再有 `if self.cfg.distribute_modules` 的 all_gather 和 module_plan 逻辑
- 不再有 `if self.cfg.loss_fn in ("ce", "kl")` 的提前 return 和模块替换

#### checkpoint.py（约 250 行）

抽出所有检查点相关的独立函数和 Trainer 的 mixin 方法：

```
# 模块级工具函数（从 trainer.py 迁移过来）
is_tiled_checkpoint()
get_checkpoint_num_tiles()
load_sae_checkpoint()
expand_range_pattern()          # hookpoint 模式扩展也可放在 utils.py

# Trainer 的检查点方法（通过 mixin 注入）
class CheckpointMixin:
    load_state()                # 恢复训练状态
    save()                      # 保存当前检查点
    save_best()                 # 保存最优检查点
    _checkpoint()               # 内部保存实现
    _load_elbow_thresholds()    # 加载 exceed 阈值
```

Trainer 通过多继承获得这些方法：

```python
# trainer.py
from .checkpoint import CheckpointMixin

class Trainer(CheckpointMixin):
    def __init__(self, ...): ...
    def fit(self, ...): ...
```

这不是"复杂抽象"——只是把一组功能内聚的方法放到另一个文件里，减少单文件长度。Trainer 的主逻辑和检查点逻辑没有交叉耦合，通过 self 访问共享状态即可。

### 3.2 config.py 精简

**移除的字段**（约 15 个）：

```python
# SparseCoderConfig 中移除：
# - activation（硬编码为 "topk"，不再需要字段）
# - multi_topk
# - skip_connection
# - encoder_rank

# TrainConfig 中移除：
# - hook_mode（硬编码为 input）
# - loss_fn（硬编码为 FVU）
# - optimizer（硬编码为 signum）
# - k_decay_steps
# - dead_feature_threshold（改用固定常量或留一个简单字段）
# - exclude_tokens
# - distribute_modules
# - distill_from, distill_lambda_decode, distill_lambda_acts, freeze_decoder
# - use_outlier_clip, outlier_k, outlier_ema_alpha, outlier_warmup_steps, outlier_loss_mode
```

**保留的字段**（约 20+ 个）：

```python
@dataclass
class SparseCoderConfig:
    expansion_factor: int = 32
    normalize_decoder: bool = True
    num_latents: int = 0
    k: int = 32

@dataclass
class TrainConfig:
    sae: SparseCoderConfig
    # 训练超参
    batch_size: int = 32
    grad_acc_steps: int = 1
    micro_acc_steps: int = 1
    max_tokens: int | None = None
    lr: float | None = None
    auxk_alpha: float = 0.0
    dead_feature_threshold: int = 10_000_000
    # Hookpoint
    hookpoints: list[str]
    layers: list[int]
    layer_stride: int = 1
    init_seeds: list[int] = [0]
    # Tiling
    num_tiles: int = 1
    global_topk: bool = False
    input_mixing: bool = False
    # Hadamard
    use_hadamard: bool = False
    hadamard_block_size: int = 128
    hadamard_seed: int = 0
    hadamard_use_perm: bool = True
    # Exceed
    exceed_alphas: list[float]
    elbow_threshold_path: str | None = None
    # 保存 & 日志
    save_every: int = 1000
    save_best: bool = False
    save_dir: str = "checkpoints"
    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_project: str | None = None
    wandb_log_frequency: int = 1
    # 生命周期
    finetune: str | None = None
    resume: bool = False
```

**`__post_init__` 大幅简化**：移除所有已删功能的互斥校验（CE/KL、outlier、distill、distribute_modules），只保留：
- hadamard_block_size 必须是 2 的幂
- layers 与 layer_stride 不能同时指定
- num_tiles > 1 时的约束
- init_seeds 至少 1 个
- exceed_alphas 正数校验

### 3.3 sparse_coder.py 精简

**移除**：
- `ForwardOutput.multi_topk_fvu` 字段
- `forward()` 中的 multi_topk 计算（约 10 行）
- `skip_connection` 相关逻辑
- `from_pretrained_lowrank()` 方法
- `transcoder` 参数（构造函数和 forward 中）

**结果**：SparseCoder 变成纯粹的自编码器——encode → topk → decode → FVU。

### 3.4 tiled_sparse_coder.py 精简

**移除**：
- 对 `LowRankSparseCoder` 的依赖和 import
- `_init_lowrank_saes()` 相关逻辑（如果存在）

**保留**：global_topk、input_mixing、per-tile encoding/decoding 全部保留。

### 3.5 fused_encoder.py 精简

**移除**：GroupMax 激活分支（`activation == "groupmax"` 的 if/else）。

**结果**：FusedEncoder.forward 只有 TopK 一条路径，`activation` 参数不再需要。

### 3.6 \_\_main\_\_.py 精简

**移除**：
- `hook_mode` CLI 参数
- `loss_fn` CLI 参数
- `optimizer` CLI 参数
- `load_in_8bit` 和 BitsAndBytesConfig
- `distill_from` 相关参数
- `exclude_tokens` 相关参数

**简化**：
- `load_artifacts()` 不再需要 8bit 分支
- `run()` 不再需要 finetune/distill 的条件逻辑（finetune 保留，distill 移除）

### 3.7 device.py Bug 修复

两个已知问题（来自代码审核）在重构中一并修复：

1. **`get_dist_backend()`** 增加 CPU fallback：
   - 非 NPU 非 CUDA 时返回 `"gloo"` 而非 `"nccl"`

2. **DDP device_ids** 使用 LOCAL_RANK：
   - `trainer.py` 中 `DDP(sae, device_ids=[dist.get_rank()])` 改为使用 `LOCAL_RANK`

### 3.8 Hook 函数简化

当前 hook 函数是 trainer.py 中最复杂的部分（约 276 行）。移除功能后的简化效果：

**移除的条件分支**：
```
- match self.cfg.hook_mode（3 分支 → 1 行直接取 input）
- if self.cfg.distribute_modules（all_gather 逻辑 → 删除）
- if self.cfg.loss_fn in ("ce", "kl")（hook return 逻辑 → 删除）
- if self.cfg.use_outlier_clip（预处理 + 损失修改 → 删除）
- if self.cfg.sae.multi_topk（额外损失项 → 删除）
- distillation loss 计算 → 删除
```

**简化后的 hook 数据流**（核心路径变得清晰）：

```
input activations
    │
    ▼
flatten batch × seq → [N, D]
    │
    ▼
apply token mask（排除 padding 等）
    │
    ▼
[可选] Hadamard 旋转
    │
    ▼
SAE forward（encode → topk → decode → FVU + AuxK）
    │
    ▼
Exceed 指标计算（如果有 elbow 阈值）
    │
    ▼
loss = FVU + auxk_alpha * auxk_loss
    │
    ▼
loss.backward()
```

### 3.9 训练主循环简化

**当前 fit() 中的 match/case**：
```python
match self.cfg.loss_fn:
    case "ce":   # ~25 行
    case "kl":   # ~25 行
    case "fvu":  # ~20 行
```

**重构后**：只保留 FVU 路径，用 `partial_forward_to_layer` 做部分前向传播，不再需要 match/case。

### 3.10 print → logging

在重构过程中顺带完成：
- `trainer.py` 顶部添加 `logger = logging.getLogger(__name__)`
- 所有 `print()` 替换为 `logger.info()` / `logger.warning()`
- `__main__.py` 中配置 `logging.basicConfig(level=logging.INFO)`

---

## 四、文件级改动清单

| 文件 | 操作 | 预估行数 | 说明 |
|------|------|----------|------|
| `sparsify/config.py` | 重写 | ~120 行 | 移除 15+ 个字段，简化校验 |
| `sparsify/sparse_coder.py` | 修改 | ~260 行 | 移除 multi_topk/skip/transcode/lowrank |
| `sparsify/tiled_sparse_coder.py` | 修改 | ~370 行 | 移除 LowRank 依赖 |
| `sparsify/fused_encoder.py` | 修改 | ~90 行 | 移除 groupmax 分支 |
| `sparsify/trainer.py` | 重写 | ~700 行 | 精简 + 拆出 checkpoint |
| `sparsify/checkpoint.py` | 新建 | ~250 行 | 从 trainer.py 拆出 |
| `sparsify/__main__.py` | 修改 | ~180 行 | 移除已删功能的 CLI 参数 |
| `sparsify/__init__.py` | 修改 | ~15 行 | 移除 LowRank 导出 |
| `sparsify/device.py` | 修改 | ~120 行 | 修 get_dist_backend + DDP bug |
| `sparsify/outlier_clip.py` | 删除 | - | 整个文件 |
| `sparsify/muon.py` | 删除 | - | 整个文件 |
| `lowrank_encoder/` | 删除 | - | 整个目录 |

**不变的文件**：`data.py`, `hadamard.py`, `sign_sgd.py`, `utils.py`, `eval/`

---

## 五、重构顺序

建议按以下顺序实施，每步可独立验证：

1. **先删外围模块**：删除 `outlier_clip.py`、`muon.py`、`lowrank_encoder/`，更新 import
2. **精简 config.py**：移除字段，简化校验
3. **精简模型层**：`sparse_coder.py`、`tiled_sparse_coder.py`、`fused_encoder.py`
4. **重写 trainer.py**：移除所有已删功能分支，拆出 `checkpoint.py`
5. **精简入口**：`__main__.py`、`__init__.py`
6. **修 bug**：`device.py` 的两个已知问题
7. **print → logging**：全局替换
8. **更新测试**：确保 tests/ 和 tests/ascend/ 通过

---

## 六、验证方式

```bash
# 1. 重构后运行全部现有测试
pytest tests/ -v

# 2. Ascend NPU 测试
pytest tests/ascend/ -v

# 3. 用实战脚本验证端到端训练（小规模）
python -m sparsify HuggingFaceTB/SmolLM2-135M \
    --hookpoints "layers.[0-3].self_attn.q_proj" \
    --expansion_factor 4 -k 32 \
    --auxk_alpha 0.03125 \
    --batch_size 2 --max_tokens 10000 \
    --save_dir /tmp/test_refactor

# 4. 验证 Tiling
python -m sparsify HuggingFaceTB/SmolLM2-135M \
    --hookpoints "layers.[0-3].self_attn.q_proj" \
    --num_tiles 4 \
    --batch_size 2 --max_tokens 10000

# 5. 验证 Hadamard
python -m sparsify HuggingFaceTB/SmolLM2-135M \
    --hookpoints "layers.[0-3].self_attn.q_proj" \
    --use_hadamard --hadamard_block_size 128 \
    --batch_size 2 --max_tokens 10000

# 6. 验证 Resume
python -m sparsify HuggingFaceTB/SmolLM2-135M \
    --hookpoints "layers.0.self_attn.q_proj" \
    --max_tokens 5000 --run_name test_resume
python -m sparsify HuggingFaceTB/SmolLM2-135M \
    --hookpoints "layers.0.self_attn.q_proj" \
    --max_tokens 10000 --run_name test_resume --resume

# 7. 验证 DDP（多卡环境）
torchrun --nproc_per_node 2 -m sparsify HuggingFaceTB/SmolLM2-135M \
    --hookpoints "layers.[0-3].self_attn.q_proj" \
    --batch_size 1 --max_tokens 10000
```
