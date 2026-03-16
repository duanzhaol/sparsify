# Experiment: 20260315-cg-coefficients-phase1

> 关联 idea：[ideas/cg-coefficients.md](../ideas/cg-coefficients.md)
> 决策树节点：C2c（CG 迭代求解系数）
> Worktree 分支：exp/cg-coefficients-phase1
> 状态：completed

## 1. 目标

在不改变训练流程的前提下，验证 CG 求解的最优系数比 SAE 编码器内积系数能带来多大的重构误差改善。

## 2. 背景

CG 方案的完整分析见 [ideas/cg-coefficients.md](../ideas/cg-coefficients.md)。以下仅说明本次实验的定位。

当前方案 (C2a) 中，SAE 编码器同时承担"选择基向量"和"计算系数"两个职责。CG 方案将二者解耦：编码器只负责选择，系数由 CG 迭代求解最优值。

**本次实验是阶段 1（最小验证）**：使用已训练好的 SAE，保持编码器的选择不变，仅将系数计算替换为 CG 求解，对比重构质量。不需要修改训练代码。三个阶段的完整路线见 idea 文档 §6。

**关于 C1（选择问题）**：本实验中选择仍使用 SAE 编码器（C1a），这在推理时不可行（访存 8h² > 原始 matmul 最大 4h²）。但阶段 1 的目的是控制变量，隔离 CG 系数本身的精度收益。CG 对 C1 的间接作用（编码器可简化、容错性提升、K 可能降低）详见 idea 文档 §3.2、§5.4。

## 3. 实现方案

新增一个独立的评估脚本（不修改现有训练代码），流程如下：

```python
# 伪代码
def evaluate_cg_vs_inner_product(sae, calibration_data):
    for x in calibration_data:
        # 1. 编码器选择（两种方法共用同一选择结果）
        top_acts, top_indices, _ = sae.encode(x)
        S = top_indices  # 选中的 K 个基向量索引

        # 2. 方法 A：内积系数（当前方案）
        recon_inner = sae.decode(top_acts, top_indices)

        # 3. 方法 B：CG 最优系数
        B_S = sae.W_dec[S]              # K × h，选中的基向量
        alpha_init = top_acts            # 用编码器输出作为 CG 初始值
        alpha_cg = cg_solve(B_S, x - sae.b_dec, alpha_init, max_iter=10)
        recon_cg = alpha_cg @ B_S + sae.b_dec

        # 4. 记录指标
        record_metrics(x, recon_inner, recon_cg)
```

CG 求解器实现要点：
- 约定 B_S ∈ R^{K×h}（K 个 h 维基向量作为行向量）
- 求解 B_S B_S^T α = B_S (x - b_dec)（K×K Gram 矩阵形式的正规方程）
- 测试两种初始值：(a) 编码器内积输出 top_acts，(b) 零初始化（作为对照）
- 最大迭代次数 t=10，同时记录每次迭代后的残差以观察收敛速度
- 需要处理 batch 维度（每个样本独立求解）

增加零初始化对照的原因：区分"问题本身简单（零初始值也快收敛）"和"编码器提供了有价值的先验（零初始值慢，编码器初始值快）"。

需要修改的文件：
- 新增 `scripts/eval_cg_coefficients.py`（或类似路径）

依赖的现有代码：
- `sparsify/sparse_coder.py`：`SparseCoder.encode()` 用于获取选择结果，`W_dec` 获取基向量
- `sparsify/sparse_coder.py`：`SparseCoder.load_from_disk()` 加载 checkpoint

## 4. 运行方法

```bash
# 需要一个已训练好的 SAE checkpoint
python scripts/eval_cg_coefficients.py \
    --checkpoint <path_to_sae_checkpoint> \
    --model <model_name>  \
    --num_samples 1024 \
    --cg_max_iter 10 \
    --hookpoint <target_hookpoint>
```

校准数据：从训练同分布的数据集中采样激活（复用训练时的 forward hook 机制）。

## 5. 观测指标

| 指标 | 含义 |
|------|------|
| MSE_inner | 内积系数的逐样本重构 MSE |
| MSE_cg | CG 系数（编码器初始值）的逐样本重构 MSE |
| MSE_reduction | (MSE_inner - MSE_cg) / MSE_inner |
| FVU_inner vs FVU_cg | 归一化版本的重构误差对比 |
| exceed_ratio(τ) | 在 τ = 0.25, 0.3, 0.5 下的超阈值维度比例 p，**与 MSE_reduction 同等重要** |
| p_reduction(τ) | (p_inner - p_cg) / p_inner，各 τ 下 p 的相对变化 |
| cg_iters_encoder_init | 编码器初始值时 CG 收敛所需迭代次数（残差 < 1e-6） |
| cg_iters_zero_init | 零初始值时 CG 收敛所需迭代次数（对照组） |
| condition_number | B_S B_S^T 的条件数（反映选中基向量的正交性） |

## 6. 预期结果

**主要判定指标**：MSE_reduction 和 p_reduction(τ=0.3) 综合判断。

注意：Phase 1 使用为内积系数训练的 D，CG 改善量可能低估真实潜力（D 未为 CG 优化），也可能与下游任务收益不完全对应。因此条件数的诊断价值可能高于 MSE 本身。

| 条件 | 判定 | 后续行动 |
|------|------|----------|
| MSE_reduction > 10% 或 p_reduction(τ=0.3) > 15% | 成功 | 进入 CG 阶段 2（训练时加入 CG） |
| MSE_reduction 5%~10%，且条件数 > 20 | 有潜力 | 条件数大说明训练时加入 CG + 正交化约束有优化空间，值得尝试阶段 2 |
| MSE_reduction 5%~10%，且条件数 < 10 | 天花板低 | 基向量已近似正交，CG 在此 SAE 架构下提升有限 |
| MSE_reduction < 5% 且条件数 < 10 | 收益不足 | CG 方向暂停，转向 B 分支（Gated SAE / 其他架构） |

附加观察：
- 如果零初始值和编码器初始值的收敛次数接近（差 ≤ 2 次），说明问题本身简单，编码器先验价值不大
- 如果差距显著（> 3 次），说明编码器即使不提供最优系数，仍给出了有价值的方向信息
- 如果条件数普遍 > 100，说明基向量间相关性高，CG 收敛慢，训练时需引入正交化约束

## 7. 实际结果

实验代码位于 `sparsify-cg` worktree 的 `experiments/cg_coefficients/` 目录，使用 LUT 部署权重（非原始 SAE 训练 checkpoint）。

### 7.1 实验设置

- **模型**：Qwen3-0.6B（28 层, h=1024, N=8192, K=128）、Qwen3-4B（36 层, h=2560, N=20480, K=256）
- **数据**：FineWeb-EDU 10BT 预分词数据集，4096 samples/层，80 shards 全加载
- **CG 设置**：max_iter=10, tol=1e-6, 全部在 float32 下求解（编码器选择在 bf16 下执行）
- **阈值**：使用 Kneedle 离线校准的 per-operator 绝对阈值 θ，τ 从 0.1~1.0 扫描
- **基线**：额外加入 `torch.linalg.lstsq` exact LS 作为理论上界
- **方法论修正**（响应外部评审）：
  1. 所有 CG/exact 求解强制 float32（避免 bf16 精度问题）
  2. 使用 GlobalAccumulator 累积原始 SSE/variance/exceed counts（非 batch 平均）
  3. 加载全部 80 个 arrow shard（非单 shard）
  4. 增加 exact LS 基线确认 CG 收敛性

### 7.2 核心结果

**CG 完全收敛**：所有 34 个实验中 CG(10 iter) = exact LS，gap = 0%。说明 10 次迭代在条件数 5~25 的范围内完全足够。

**Qwen3-0.6B**（6 层 × 3 算子 = 18 实验）：

| 层 | 算子 | MSE reduction | κ | p_inner@τ=0.5 | p_cg@τ=0.5 | p_reduction |
|----|------|-------------|---|--------------|-----------|-------------|
| 0 | mlp | 18.2% | 8.9 | 11.5% | 9.2% | 19.9% |
| 0 | qkv | 12.3% | 5.7 | 5.8% | 5.0% | 13.9% |
| 0 | o_proj | 26.6% | 17.4 | 3.2% | 3.1% | 2.5% |
| 5 | mlp | 12.5% | 5.7 | 7.1% | 6.3% | 11.1% |
| 5 | qkv | 14.5% | 8.7 | 6.4% | 5.3% | 17.8% |
| 5 | o_proj | 27.4% | 19.1 | 3.9% | 3.8% | 3.5% |
| 10 | mlp | 15.3% | 7.8 | 8.8% | 7.3% | 16.9% |
| 10 | qkv | 14.9% | 8.9 | 4.6% | 3.8% | 16.7% |
| 10 | o_proj | 21.3% | 12.1 | 4.3% | 4.1% | 4.2% |
| 20 | qkv | 15.3% | 23.1 | 5.7% | 4.6% | 19.5% |
| 27 | mlp | 16.2% | 11.1 | 10.2% | 8.3% | 18.3% |

**Qwen3-4B**（8 层 × 2 算子 = 16 实验，无 o_proj LUT）：

| 层 | 算子 | MSE reduction | κ | p_inner@τ=0.5 | p_cg@τ=0.5 | p_reduction |
|----|------|-------------|---|--------------|-----------|-------------|
| 0 | mlp | 17.4% | 9.0 | 8.2% | 6.6% | 19.3% |
| 0 | qkv | 14.1% | 7.3 | 1.9% | 1.8% | 7.8% |
| 5 | mlp | 13.1% | 9.8 | 13.9% | 11.6% | 16.0% |
| 5 | qkv | 15.0% | 12.1 | 12.6% | 10.4% | 17.6% |
| 10 | mlp | 9.8% | 6.3 | 29.7% | 27.3% | 7.9% |
| 25 | qkv | 16.0% | 25.0 | 25.6% | 22.7% | 11.3% |
| 30 | qkv | 14.5% | 19.3 | 23.2% | 20.6% | 11.3% |
| 35 | mlp | 10.4% | 15.4 | 21.2% | 19.3% | 9.2% |

### 7.3 关键发现

1. **MSE reduction 范围**：0.6B 为 12~27%，4B 为 9.8~17.4%。均满足"MSE_reduction > 10%"的成功判据
2. **p_reduction@τ=0.5 范围**：mlp 7.9~19.9%，qkv 7.8~19.5%，o_proj 仅 2.5~4.2%
3. **o_proj 悖论**：MSE reduction 最高（21~27%），但 p_reduction 最低（2~4%）。原因是 o_proj 的误差分布呈极端形态——大部分维度误差极小（远低于阈值），少数维度误差极大（远超阈值），CG 改善的"中间区域"误差很少
4. **条件数与收益相关**：κ 越高，MSE reduction 越大（κ=25 时 16%，κ=6 时 9.8%），符合预期
5. **编码器初始值 vs 零初始值**：两者最终收敛到相同解，但编码器初始值收敛更快（残差降 1~2 个数量级），说明编码器提供了有价值的方向先验
6. **跨模型一致性**：0.6B 和 4B 的 MSE reduction 范围和条件数范围高度一致，说明结论可泛化

### 7.4 详细数据

- JSON 结果：`sparsify-cg/experiments/cg_coefficients/results/{Qwen3-0.6B,Qwen3-4B}/`
- CSV 汇总：`sparsify-cg/experiments/cg_coefficients/results/{Qwen3-0.6B,Qwen3-4B}/summary.csv`
- 可视化：`sparsify-cg/experiments/cg_coefficients/results/{Qwen3-0.6B,Qwen3-4B}/exceed_ratio.png`

## 8. 结论与影响

### 判定

按 §6 的判定标准：**MSE_reduction > 10%（大部分实验满足）→ 成功**。

但需要加上实际意义的补充判定：**CG 在推理时的性价比存疑**。

### 核心结论

1. **CG 系数确实优于内积系数**：在相同选择下，CG 将 MSE 降低 10~17%，证明内积系数并非最优
2. **CG 10 次迭代 = exact LS**：在当前条件数范围（5~25）下，10 次迭代完全足够，无需更多
3. **p_reduction 收益有限**：MSE 降低 10~17% 仅转化为 p 降低 8~20%，绝对值来看在线计算比例仅减少 1~3 个百分点
4. **CG 自身的计算开销**：10 次迭代需要 10 次 (K×K) 矩阵向量乘，这本身就是额外的在线计算。当 p 减少量 < CG 开销时，CG 是净亏损

### 对决策树的影响

| 节点 | 状态变更 | 说明 |
|------|----------|------|
| C2c | 待验证 → **已验证（精度有效，但性价比待定）** | CG 确实改善系数质量，但推理时引入的额外计算可能抵消收益 |
| C2a | 保持当前 | 内积系数虽不最优，但足够接近且零额外开销 |

### 对耦合路径 4 的影响

decision-tree.md §4.2 路径 4 预期 CG 可以解锁 B/C1/D 三个分支。Phase 1 实验显示：

- **D（p↓）**：确认有效但幅度有限（p 减少 1~3 个百分点）
- **B（基向量库质量↑）**：Phase 1 未涉及训练，但条件数的层间变异（5~25）表明不同层的优化空间不同
- **C1（编码器可简化）**：Phase 1 确认编码器提供了有价值的初始值先验，但并非不可替代（零初始值也能收敛到相同解）
- **K 下界降低**：未观察到支持此假设的证据

### 建议

1. **不建议进入 Phase 2（CG 融入训练）**：Phase 1 已用 exact LS 确认了理论天花板，Phase 2 中训练会让基向量更正交（κ↓），CG 收益反而更小，存在自我抵消效应
2. **CG 作为备选保留**：如果未来 K 显著降低（如 K=32），CG 的相对开销下降，可能变得有价值
3. **建议优先探索其他分支**：
   - C1（选择策略）：已有 [activation-patterns 实验](20260316-activation-patterns.md) 计划，用 oracle 基线评估不同 C1 方案的理论上限
   - B（基向量构建）：尝试 Gated SAE、结构约束 SAE 等
   - D1-sub（阈值策略）：per-operator 自适应 τ 可能比 CG 更直接有效
