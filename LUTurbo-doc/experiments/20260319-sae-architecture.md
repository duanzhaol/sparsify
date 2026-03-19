# Experiment: 20260319-sae-architecture

> 关联 idea：[ideas/sae-improvement.md](../ideas/sae-improvement.md)（§3.2-3.3, §3.9）
> 决策树节点：B1-gated, B1-jump, B1-residual
> Worktree 分支：exp/sae-architecture
> 状态：planning
> 依赖：[20260319-sae-diagnostic](20260319-sae-diagnostic.md)（最优超参配置 + K-FVU 基线）
> 代码可用性：**全部 requires infra change**（需修改 sparse_coder.py / trainer.py）

## 1. 目标

对比 SAE 架构改进在小 K（32, 64）下的重构质量，确定哪种方案最能降低 K 需求（Goal A）。

## 2. 背景

实验 1（diagnostic）将建立标准 SAE 在最优超参下的 K-FVU 基线，同时 Matryoshka 和正交性正则已在实验 1 Phase 1 中测试（低成本方案前移）。本实验聚焦需要代码修改的架构级改进。

**优先级分层**（不强制全部并行，按信息价值/成本排序）：
- **第一层**：Gated SAE + JumpReLU SAE — 两者都改变编码方式，对字典质量的影响最直接
- **第二层**：残差 SAE — 实现成本较高，在 OMP 曲线差时提升优先级

## 3. 实现方案

### 3.1 子实验 A: Gated SAE（requires infra change）

**需要修改的文件**：`sparsify/sparse_coder.py`, `sparsify/config.py`

**核心改动**：在 `SparseCoder` 中增加 gated 模式：

```python
# config.py 新增
class SparseCoderConfig:
    gated: bool = False

# sparse_coder.py encode 方法改动
if self.cfg.gated:
    # gate 分支：决定选哪些特征
    gate_logits = x_centered @ self.W_gate.T + self.b_gate
    gate_topk = topk(sigmoid(gate_logits), k)  # 二值选择

    # magnitude 分支：决定系数大小
    mag = x_centered @ self.W_mag.T + self.b_mag
    top_acts = mag[gate_topk.indices]
    top_indices = gate_topk.indices
else:
    # 现有逻辑不变
```

**新增参数**：`W_gate` (N×h), `b_gate` (N), `W_mag` (N×h), `b_mag` (N)

**关于参数翻倍的说明**：W_gate + W_mag 确实让编码器参数翻倍。但 LUTurbo 推理时不用编码器（太大，8h²），而是用 C1h+C1i 等方法做选择。编码器只在训练时用。所以 Gated SAE 的核心评估标准是**它训练出的解码器（字典）质量是否更好**。

**次要观测**：gate 网络（W_gate）是否比标准编码器更容易低秩化——如果 gate 的有效秩 << h，未来可能用低秩 gate 做低成本选择（与 C1f 协同）。

**训练配置**：K = {32, 64, 128}，expansion_factor 用实验 1 最优值。

### 3.2 子实验 B: JumpReLU SAE（requires infra change）

**需要修改的文件**：`sparsify/sparse_coder.py`, `sparsify/config.py`

**核心改动**：新增 activation='jumprelu' 模式：

```python
class SparseCoderConfig:
    activation: str = "topk"  # "topk" | "jumprelu"

# encode 中
if self.cfg.activation == "jumprelu":
    preacts = x_centered @ self.encoder.weight.T + self.encoder.bias
    # 每特征独立阈值 θ_i（可学习参数）
    mask = (preacts > self.threshold).float()  # self.threshold: [N]
    # STE：前向用阶跃，反向用 sigmoid 近似
    bandwidth = 100.0  # 控制 STE 近似的锐度
    mask_ste = mask + (torch.sigmoid(bandwidth * (preacts - self.threshold)) - mask).detach()
    z = torch.relu(preacts) * mask_ste
```

**新增参数**：`self.threshold` (N,) 可学习阈值

**评估方式**（重点测 train-deploy mismatch）：
1. JumpReLU 训练后，观察 K 的分布（mean, std, P99）
2. **用 JumpReLU 训练的字典 + TopK=32/64/128 部署**：测 FVU（mismatch 代价）
3. 对比基线：用 TopK=32/64/128 训练的标准 SAE 的 FVU
4. 如果 JumpReLU 字典 + TopK=32 的 FVU < 标准 TopK=32 的 FVU → mismatch 可接受

**训练配置**：通过调整初始阈值控制平均 K ≈ 128。

### 3.3 子实验 C: 残差 SAE（requires infra change，优先级：中）

**需要修改的文件**：新增训练脚本或修改 `trainer.py`

**方案：级联训练（简单优先）**：

```python
# Phase 1: 训练 Level 1 SAE（标准训练，K1=32/64）
# python -m sparsify ... --sae.k 32 --save_dir checkpoints/residual_L1

# Phase 2: 冻结 Level 1，用残差训练 Level 2
def train_level2(sae_L1, data):
    for x in data:
        x_hat_1 = sae_L1(x).sae_out.detach()  # 冻结
        residual = x - x_hat_1
        # 用残差训练 Level 2 SAE
        out_L2 = sae_L2(residual)
        loss = fvu(residual, out_L2.sae_out)
```

**训练配置**：
- (K1, K2) = {(16, 16), (32, 32), (16, 48), (48, 16)}
- Level 2 的 expansion_factor 可以更小（残差可能更低维）
- 对比基线：单级 K=K1+K2 的标准 SAE

**评估重点**：
- 两级 (K1=32, K2=32) vs 单级 K=64 的 FVU
- Level 1 的激活模式：热集是否能直接覆盖？（C1 友好度）
- 残差的有效维度（PCA 分析，复用 0A-3 脚本）

## 4. 运行方法

```bash
MODEL=/root/models/Qwen3-0.6B
DATASET=/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048
THRESHOLD_DIR=/root/sparsify-ascend/thresholds/Qwen3-0.6B

# 子实验 A: Gated SAE
for K in 32 64 128; do
    python -m sparsify $MODEL $DATASET \
        --sae.k $K --sae.gated \
        <实验1最优超参> --save_dir checkpoints/gated_k${K}
done

# 子实验 B: JumpReLU
python -m sparsify $MODEL $DATASET \
    --sae.activation jumprelu \
    <实验1最优超参> --save_dir checkpoints/jumprelu

# 统一评估（所有 checkpoint 都用同一个 eval 脚本）
python scripts/eval_exceed.py \
    --checkpoint checkpoints/gated_k32 \
    --model $MODEL --dataset $DATASET \
    --elbow_threshold_path $THRESHOLD_DIR/thresholds_up.json
```

## 5. 观测指标

所有子实验统一报告以下指标：

| 指标 | 含义 | 对比对象 |
|------|------|----------|
| FVU@K=32 | 小 K 重构质量 | 实验 1 基线 |
| FVU@K=64 | 中 K 重构质量 | 实验 1 基线 |
| FVU@K=128 | 大 K 重构质量 | 实验 1 基线 |
| p@τ=0.3 | 严格阈值下的补偿比例 | 实验 1 基线 |
| p@τ=0.5 | 宽松阈值下的补偿比例 | 实验 1 基线 |
| OMP(32) FVU | 字典质量（oracle 上限） | 实验 1 oracle 曲线 |
| dead_ratio | 死特征比例 | 实验 1 基线 |

子实验特有指标：
- **Gated**：gate 网络 W_gate 的有效秩（SVD 奇异值分布）
- **JumpReLU**：K 的分布（mean, std, P99）；TopK 部署时 FVU vs JumpReLU 原生 FVU（mismatch 代价）
- **残差**：Level 1/2 各自的 FVU；Level 1 热集 recall；查表存储倍率（两套 LUT vs 一套）；两轮选择的总访存开销

## 6. 预期结果与判定标准

### 整体判定

| 子实验 | 成功标准 | 失败标准 |
|--------|---------|---------|
| Gated SAE | FVU@K=32 < 基线 FVU@K=64 | FVU@K=32 ≈ 基线 FVU@K=32（无改善） |
| JumpReLU | JumpReLU 字典 + TopK=32 的 FVU < 基线 TopK=32 FVU | mismatch 损失 >20%，或 K 方差太大（P99 > 3×mean） |
| 残差 SAE | (K1=32,K2=32) FVU < 基线 K=64 FVU | 总 K 一样时两级不如单级 |

### 汇总对比

最终产出：**FVU@K 曲线图**，含基线 + 各子实验最优配置，在同一张图上直观对比哪种架构在小 K 下最优。

## 7. 实际结果

（实验完成后填写）

## 8. 结论与影响

（实验完成后填写）
- 更新 decision-tree.md：B1-gated, B1-jump, B1-residual 的状态
- 确定最优架构 → 与实验 3（结构化 SAE）的最优方案组合测试
- 如果某个架构在 K=32 下达到当前 K=128 的精度 → 根本性改变 C1 可行性格局
