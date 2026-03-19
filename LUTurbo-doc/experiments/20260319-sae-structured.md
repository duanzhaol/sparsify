# Experiment: 20260319-sae-structured

> 关联 idea：[ideas/sae-improvement.md](../ideas/sae-improvement.md)（§4）
> 决策树节点：B1-constrained, C1c, A2b
> Worktree 分支：exp/sae-structured
> 状态：planning
> 依赖：
>   - [20260319-sae-diagnostic](20260319-sae-diagnostic.md) Phase 0B（latent grouping oracle 结果）
>   - [20260319-sae-diagnostic](20260319-sae-diagnostic.md) Phase 1（最优超参配置）
> 代码可用性：**requires infra change**（需新增 Group TopK 路由器和训练逻辑）
> 注：Tiling 实验已前移到 sae-diagnostic Phase 1d（available now）

## 1. 目标

验证 Group TopK 训练能否让激活具备分组结构，使 C1c 条件子库从当前的 90-100%N 缩减到 <50%N，同时重构质量损失 <20%。

## 2. 背景

激活模式实验（[20260316-activation-patterns](20260316-activation-patterns.md)）的核心发现：

- C1c 条件子库完全不可行：全子库 90-100%N，聚类高度重叠
- 根因：训练目标（FVU + auxk）不包含分组信号，基向量学成了通用表示
- C1h+C1i 对 MLP/QKV 仅达到 74% recall@16%N，不够用

**与 C1c 的区别**：C1c 测的是"token 聚类 → 子库"失败了。但 Group TopK 是"latent 分组 + router 选组"——latent 是否能形成可路由的 group 是不同的问题。实验 1 Phase 0B 的 latent grouping oracle 将回答这个前置问题。

**Tiling 前移说明**：Tiling（T=2/4/8，含 global_topk 和 input_mixing 变体）已前移到 sae-diagnostic Phase 1d（available now，零代码改动）。本实验只包含需要代码修改的 Group TopK。

## 3. 实现方案

### 3.1 Group TopK（requires infra change）

**需要修改的文件**：`sparsify/sparse_coder.py`, `sparsify/config.py`, `sparsify/trainer.py`

**核心改动**：

```python
# config.py 新增
class SparseCoderConfig:
    num_groups: int = 0           # G，0 表示不分组（标准 TopK）
    active_groups: int = 0        # g，每个输入选多少个组

# sparse_coder.py 新增
class SparseCoder:
    def __init__(self, ...):
        if cfg.num_groups > 0:
            # 独立路由器（G×h），不从组内编码器聚合
            self.group_router = nn.Linear(d_in, cfg.num_groups)

    def encode(self, x):
        if self.cfg.num_groups > 0:
            G = self.cfg.num_groups
            g = self.cfg.active_groups
            group_size = self.num_latents // G

            # Step 1: 路由器选组
            group_scores = self.group_router(x_centered)  # [batch, G]
            # Gumbel-Softmax 使选组可微
            _, top_groups = group_scores.topk(g, dim=-1)  # [batch, g]

            # Step 2: 在选中组的 latent 并集上做全局 top-K
            # 注意：不强制 k_per_group = K/g，而是全局 top-K
            candidate_mask = build_candidate_mask(top_groups, group_size, G)
            preacts = x_centered @ self.encoder.weight.T + self.encoder.bias
            preacts[~candidate_mask] = -inf  # 只在候选集中选
            top_acts, top_indices = topk(relu(preacts), K)

            return top_acts, top_indices

# 注意：以上是训练期原型实现（先算全量 preacts 再 mask），方便验证正确性。
# 最终推理路径应只计算被选中 groups 的 local preacts，
# 即 O(h × g×N/G) 而非 O(h × N)。
```

**设计要点**：
- 路由器 R ∈ R^{G×h} 是独立小网络，不从组内编码器聚合（否则仍需计算全部 N 个 logits）
- 选中 g 个组后，在 g×N/G 个候选上做**全局 top-K**（不强制每组固定配额），让强组自然贡献更多特征
- 组选择用 Gumbel-Softmax（训练可微）或 straight-through estimator
- **路由器训练信号**：主要靠端到端 FVU loss 反传（Gumbel-Softmax 保证梯度流过路由器）。可选加 load balancing 辅助损失（类似 MoE 的 auxiliary loss），防止少数组吃掉所有流量导致退化。是否需要 balance loss 作为超参探索的一部分

**参数扫描**：

| G | g | K | 候选集大小 g×N/G | 搜索空间压缩比 |
|---|---|---|:---:|:---:|
| 16 | 2 | 128 | 2×N/16 = N/8 | 8x |
| 16 | 4 | 128 | 4×N/16 = N/4 | 4x |
| 32 | 4 | 128 | 4×N/32 = N/8 | 8x |
| 64 | 4 | 128 | 4×N/64 = N/16 | 16x |
| 64 | 8 | 128 | 8×N/64 = N/8 | 8x |

### 3.2 Sub-phase B: C1c/C1h 复测

用 Sub-phase A 中最优的 Group TopK SAE checkpoint，重跑激活模式分析 pipeline：

```bash
python scripts/analyze_activation_patterns.py \
    --checkpoint <最优Group TopK SAE> \
    --model $MODEL \
    --analyses hotset sublibrary seed_expand \
    --output_dir experiments/sae_structured/results/Qwen3-0.6B/c1_retest/
```

## 4. 运行方法

```bash
MODEL=/root/models/Qwen3-0.6B
DATASET=/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048
THRESHOLD_DIR=/root/sparsify-ascend/thresholds/Qwen3-0.6B

# Group TopK 训练
for G in 16 32 64; do
    for g in 2 4 8; do
        python -m sparsify $MODEL $DATASET \
            --sae.k 128 --sae.num_groups $G --sae.active_groups $g \
            <实验1最优超参> \
            --save_dir checkpoints/group_topk_G${G}_g${g}
    done
done

# 评估
for ckpt in checkpoints/group_topk_*; do
    python scripts/eval_exceed.py \
        --checkpoint $ckpt --model $MODEL --dataset $DATASET \
        --elbow_threshold_path $THRESHOLD_DIR/thresholds_up.json
done

# C1c 复测（用最优 checkpoint）
python scripts/analyze_activation_patterns.py \
    --checkpoint <best_group_topk> --model $MODEL
```

## 5. 观测指标

### 训练评估

| 指标 | 含义 | 判定标准 |
|------|------|----------|
| FVU@K=32 | 小 K 重构质量 | 对比实验 1 基线（Goal A 耦合） |
| FVU@K=64 | 中 K 重构质量 | 对比实验 1 基线 |
| FVU@K=128 | 大 K 重构质量 | 相对标准 SAE 损失 <20% |

**FVU@K=32/64 的评估方式**：训练时固定 K=128，评估时对编码器输出按激活值降序截断到 top-32/top-64 后重构。这样可以用一个 checkpoint 同时观测不同 K 下的表现，且不需要为每个 K 单独训练。
| p@τ=0.3, 0.5 | 补偿比例 | 不大幅上升 |
| dead_ratio | 死特征比例 | 无异常增加 |
| group_balance | 各组被选中频率的均匀度（entropy/max_entropy） | > 0.7（避免退化到少数组吃掉所有流量） |
| route_stability | 相邻 token 选同一组的概率 | 参考指标（与 C1e 关联） |

### C1c 复测

| 指标 | 原始 SAE | 目标 |
|------|:---:|:---:|
| 全子库/N | 90-100% | <50%（理想 <25%） |
| 25%N 截断 recall@G=32 | 57-84% | >90% |
| C1h+C1i mlp recall@16%N | 66-74% | >85% |
| C1h+C1i o_proj recall@12.5%N | 88% | ≥88%（不退化） |
| cross-route gap | 极小（0.001-0.06） | >5%（分组有区分度） |

**关于 cross-route gap**：gap 大说明分组有区分度（好），但也意味着路由错误代价更高。所以需要同时看 group_balance——如果某些组极少被选中，路由器的学习信号不足。

## 6. 预期结果与判定标准

### 整体 go/no-go

| 结果 | 判定 | 后续 |
|------|------|------|
| FVU@K=128 ↑ <20% 且 全子库 <50%N | **Goal B 成功** | 与实验 2 最优方案组合测试 |
| FVU@K=128 ↑ <20% 但 全子库仍 >70%N | 分组不够强 | 增大 G、增加共激活正则辅助 |
| FVU@K=128 ↑ >20% 且 全子库 <50%N | 重构代价太大 | 调低分组约束强度 |
| FVU@K=128 ↑ >20% 且 全子库 >70%N | **Goal B 失败** | 激活空间可能本身无可利用分组结构 |

### Goal A × Goal B 耦合观测

关键问题：分组是否反而让小 K 更有利？

| 观测 | 含义 |
|------|------|
| Group TopK FVU@K=32 < 基线 FVU@K=32 | 分组对小 K 有帮助（协同） |
| Group TopK FVU@K=32 > 基线 FVU@K=32 | 分组约束损害了小 K（冲突） |
| Group TopK FVU@K=128 ≈ 基线但 FVU@K=32 差很多 | 分组只在大 K 下不损失，小 K 下约束太强 |

## 7. 实际结果

（实验完成后填写）

## 8. 结论与影响

（实验完成后填写）
- 更新 decision-tree.md：B1-constrained, C1c, A2b 的状态
- 如果 Goal B 成功 → MLP/QKV 的 C1 问题可能被根本解决
- 与实验 2 结果联合分析：Goal A + Goal B 是否协同
- 如果协同 → 组合最优架构 + Group TopK 做端到端验证
