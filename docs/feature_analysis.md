# 项目功能全景分析

## 一、SAE 模型变体（三选一）

单次训练时按优先级选择一种模型类（`trainer.py:240-266`）：

| 模型类 | 触发条件 | 源文件 |
|--------|----------|--------|
| **TiledSparseCoder** | `num_tiles > 1` | `tiled_sparse_coder.py` |
| **LowRankSparseCoder** | `encoder_rank > 0`（且 `num_tiles=1`） | `lowrank_encoder/lowrank_encoder.py` |
| **SparseCoder** | 以上均不满足（默认） | `sparse_coder.py` |

注意：这是**构造分支三选一**，不是概念上的全局互斥——同一个项目中三种模型类都可用于不同的训练实验。

---

## 二、训练目标与模式

**损失函数**（三选一）：

- `fvu`（默认）：局部重建损失，Fraction of Variance Unexplained。见 `trainer.py:845-853,910-933`
- `ce`：端到端交叉熵损失，需完整模型前向。见 `trainer.py:646-653,999-1021`
- `kl`：端到端 KL 散度损失。见 `trainer.py:979-984,1023-1046`

**Hook 模式**（三选一）：

- `output`（默认）：用模块输出做自编码
- `input`：用模块输入做自编码
- `transcode`：从模块输入预测模块输出

**稀疏激活**（二选一）：

- `topk`（默认）：标准 Top-K 选择
- `groupmax`：分组取最大值

---

## 三、训练增强功能

| 功能 | 配置项 | 代码位置 | 说明 |
|------|--------|----------|------|
| AuxK 死特征修复 | `auxk_alpha` | `trainer.py:826-830,846-848` | 辅助损失激活死特征 |
| Multi-TopK 损失 | `multi_topk` | `trainer.py:850-852,929,932` | 额外损失项 |
| K-Decay 退火 | `k_decay_steps` | `trainer.py:507-513` | 从大 k 线性退到小 k |
| Skip Connection | `skip_connection` | `sparse_coder.py` | 线性残差连接 |
| 排除特定 token | `exclude_tokens` | `trainer.py:730-733` | 训练时忽略指定 token |
| Exceed 评估指标 | `elbow_threshold_path` + `exceed_alphas` | `trainer.py:857-908` | 自定义误差超标率指标 |

---

## 四、预处理增强（二选一或都不开）

| 功能 | 配置项 | 源文件 | 说明 |
|------|--------|--------|------|
| Hadamard 旋转 | `use_hadamard` + 3 个子参数 | `hadamard.py`, `trainer.py:735-750` | 块对角 Hadamard 变换减少离群值 |
| Outlier Clipping | `use_outlier_clip` + 4 个子参数 | `outlier_clip.py`, `trainer.py:752-774` | EMA 统计量裁剪离群维度 |

两者**互斥**，且都不支持 `ce`/`kl` 损失。

---

## 五、生命周期功能

| 功能 | 配置项 | 触发位置 | 说明 |
|------|--------|----------|------|
| Resume | `--resume` | `__main__.py:211-232`, `trainer.py:442-505`(load_state) | 从 checkpoint 恢复训练 |
| Finetune | `--finetune path` | `__main__.py:233-235` | 加载已有 SAE 权重继续训练 |
| Distillation | `--distill_from path` | `trainer.py:201-230,935-947` | 教师 SAE 蒸馏到低秩学生 |

---

## 六、分布式与设备

| 功能 | 配置项 | 说明 |
|------|--------|------|
| DDP 数据并行 | `torchrun` | 标准模式，SAE 在所有 GPU 上复制 |
| Module 分布式 | `distribute_modules=True` | 每 GPU 只训练部分层的 SAE，省显存 |
| 设备抽象 | `device.py` | CUDA/NPU/CPU 统一接口，NPU 用 HCCL |
| 8bit 加载 | `load_in_8bit` | 仅 CUDA 生效，NPU 自动回退 bf16 |

---

## 七、数据与评估工具

| 组件 | 文件 | 说明 |
|------|------|------|
| 数据加载 | `data.py` | HF Dataset + MemmapDataset（.bin） |
| Two-Stage 评估 | `sparsify/eval/two_stage.py` | 两阶段编码器评估 |
| PCA 评估 | `sparsify/eval/pca.py` | PCA 基线对比 |
| Elbow 阈值计算 | `compute_elbow_thresholds.py` | 预计算各层误差阈值 |

---

## 八、功能兼容/互斥关系（完整硬规则）

### 明确互斥

| 规则 | 代码位置 |
|------|----------|
| `use_hadamard` ✕ `use_outlier_clip` | `config.py:256-260` |
| `use_hadamard` ✕ `loss_fn ∈ {ce, kl}` | `config.py:246-250` |
| `use_outlier_clip` ✕ `loss_fn ∈ {ce, kl}` | `config.py:261-265` |
| `num_tiles > 1` ✕ `hook_mode="transcode"` | `config.py:230-231` |
| `num_tiles > 1` ✕ `distill_from` | `config.py:233-237` |
| `distribute_modules` ✕ `loss_fn ∈ {ce, kl}` | `config.py:207-211` |
| `layers` ✕ `layer_stride != 1` | `config.py:204-205` |

### 明确依赖

| 规则 | 代码位置 |
|------|----------|
| `distill_from` → 要求 `encoder_rank > 0` | `config.py:224-227` |
| `global_topk` / `input_mixing` → 仅 `num_tiles > 1` 时有意义 | `tiled_sparse_coder.py` |

### 值域/运行时约束

| 规则 | 代码位置 |
|------|----------|
| `hadamard_block_size` 必须是正的 2 的幂 | `config.py:241-244` |
| `outlier_k` 必须 > 0 | `config.py:254-255` |
| `exceed_alphas` 所有值必须为正 | `config.py:217-218` |
| `elbow_threshold_path` 指定时文件必须存在 | `config.py:220-221` |
| `init_seeds` 至少 1 个 | `config.py:213-214` |
| `distribute_modules` 时所有 hook 输出宽度必须一致 | `trainer.py:187-191` |
| `distribute_modules` 时模块数需被 world_size 整除 | `trainer.py:1288` |
| `load_in_8bit` 仅 CUDA，NPU 自动回退 | `__main__.py` |

### 关系图

```
                        loss_fn = "fvu"
          ┌────────────────┼────────────────┐
          │                │                │
     ┌────▼─────┐   ┌─────▼──────┐   ┌─────▼────────────┐
     │ Hadamard │   │  Outlier   │   │  distribute      │
     │          │✕──│  Clip      │   │  _modules        │
     └──────────┘   └────────────┘   └──────────────────┘
          ✕                ✕                   ✕
     loss_fn∈{ce,kl}  loss_fn∈{ce,kl}    loss_fn∈{ce,kl}


     ┌──────────┐        ✕          ┌──────────────┐
     │ Tiling   │───────────────────│ Transcoding  │
     │num_tiles │───────────────────│              │
     └───┬──────┘        ✕          └──────────────┘
         │
         │           ✕              ┌──────────────┐
         └──────────────────────────│ Distillation │
                                    └──────┬───────┘
                                           │ 依赖
                                           ▼
                                    encoder_rank > 0
```

---

## 九、脚本实战验证情况

| 分类 | 功能 | 证据 |
|------|------|------|
| **频繁实战** | topk + fvu + signum + auxk + normalize_decoder + hook_mode=input + exceed | 所有 `scripts/first_time_train/` 脚本 |
| **有脚本，验证过** | Tiling（`num_tiles`） | `scripts/tiling_train/` |
| **有脚本，验证过** | Hadamard | `scripts/hadmard/` |
| **有脚本，验证过** | Distillation（`distill_from` + `encoder_rank`） | `scripts/distill/` |
| **脚本中出现但始终关闭** | `multi_topk=False`, `skip_connection=False` | 所有脚本中显式设为 False |
| **git log 标记效果差** | Outlier Clip | 提交 "add CLIP OUTLIERS, BAD" |
| **无脚本入口** | ce/kl 损失、transcode、groupmax、k_decay、distribute_modules、muon/adam | `scripts/` 中未检索到 |

---

## 十、重构取舍建议

### P0：保留最小核心

- TopK 编码 + FVU 损失 + AuxK + Decoder 范数约束 + Hook 系统 + DDP + Signum + Exceed 评估
- 这是所有训练脚本的公共基础

### P1：选择性保留（按当前实验需要决定）

- Hadamard（有脚本验证）
- Tiling（有脚本验证）
- Distillation + LowRank（有脚本验证）
- Resume / Finetune（训练生命周期基础设施）

### P2：建议下线（显著降低 trainer.py 复杂度）

- **CE/KL 端到端损失** — 代码已实现可运行，但无实战脚本，且贡献了 `trainer.py` 中约 80 行独立分支（`trainer.py:646-659, 979-1046, 1154-1157`）以及 hook 返回逻辑中的 `if loss_fn in ("ce","kl")` 条件
- **Transcoding** — 无脚本，在 hook 选择、宽度计算、decoder 范数等多处引入条件分支
- **GroupMax** — 无脚本
- **Multi-TopK** — 所有脚本显式设为 False
- **K-Decay** — 无脚本
- **Skip Connection** — 所有脚本显式设为 False
- **distribute_modules** — 无脚本，且在 `trainer.py` 中贡献了约 30 处条件判断
- **Muon / Adam** — 无脚本
- **Outlier Clip** — git 记录标记效果不好

经逐行统计：CE/KL 路径约 80 行独立代码 + 多处条件判断，distribute_modules 约 30 处条件判断，transcode 约 15 处条件分支，其余功能各有少量分支。`trainer.py` 共 1439 行，这些功能合计涉及约 200-250 行直接代码和条件分支，占比约 15-18%。加上移除后可简化的间接逻辑（如 hook 返回路径统一、loss 计算路径统一），合理估计实际可简化幅度为 **20-25%**。
