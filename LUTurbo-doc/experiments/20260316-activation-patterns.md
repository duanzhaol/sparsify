# Experiment: 20260316-activation-patterns

> 关联 idea：[ideas/activation-patterns.md](../ideas/activation-patterns.md)
> 决策树节点：C1（选择算法）、A2b（条件分解）
> Worktree 分支：exp/activation-patterns
> 状态：planning

## 1. 目标

通过 oracle baseline 实验，测量四种 C1 候选方案的理论上限效果，用统一的"候选集大小 → recall → 重构误差"框架筛选值得投入实现的方案。

## 2. 背景

C1 有多个候选方案，但缺少数据判断哪些值得投入。本实验不实现任何在线选择算法，而是模拟每种方案在**完美先知条件**下的效果上限。如果 oracle 上限不够好，方案可以直接排除。

详细的假设推导见 [ideas/activation-patterns.md](../ideas/activation-patterns.md)。

## 3. 实现方案

新增 `scripts/analyze_activation_patterns.py`，包含数据采集和四个 oracle baseline。

### 3.1 数据采集

```python
def collect_activations(model, sae_dict, dataset, num_samples, seq_len):
    """
    收集各层 SAE 激活的 top-K 索引和值。

    返回：
      per_layer[layer_name] = {
          'indices': Tensor[total_tokens, K],  # int64
          'values':  Tensor[total_tokens, K],  # float32
          'seq_boundaries': List[int],          # 序列边界（用于重叠率分析）
      }
    """
    # batch 化 encode，一次处理 [batch, seq_len, h] → [batch*seq_len, K]
    # 记录序列边界以区分同一序列内的 token 对 vs 跨序列的 token 对
```

**实现要点**：
- batch 化 `sae.encode()` 处理整个序列，不逐 token 循环
- 内存估计：2048 samples × 512 seq_len × 128 K × 2 (indices + values) × 4 bytes ≈ 1GB，可接受
- 多层在一次 model forward 中同时采集
- 记录每个 token 的序列内位置（用于区分 prefill/decode 阶段分析）

### 3.2 Oracle Baseline A：增量选择（C1e 上限）

```python
def oracle_incremental(indices, values, seq_boundaries):
    """
    模拟：保留前一 token 的 top-K，只允许替换 m 个位置。
    替换策略：oracle（从真实 top-K 中选不在旧集合里的、激活值最大的 m 个）。

    参数扫描：m = 4, 8, 16, 32, 64, K

    输出（每个 m 值）：
    - recall: 命中真实 top-K 的比例（均值 + P50/P90/P99 分布）
    - recall_weighted: 按激活值加权的 recall
    - mse_c2a: 用命中的基向量 + C2a 内积系数重构的 MSE
    - mse_c2c: 用命中的基向量 + C2c CG 系数（t=5）重构的 MSE
    注意：不使用"完全 oracle 最优系数"（完整最小二乘），因为实际推理中只会用 C2a 或 C2c。
    用完全最优系数会高估 C1 方案的实际价值。
    - replacement_count_dist: 实际需要替换多少个才能 100% recall 的分布
    """
    for each consecutive token pair (t, t+1) within same sequence:
        S_old = set(indices[t])
        S_true = set(indices[t+1])
        overlap = S_old & S_true
        need_replace = S_true - S_old  # 真正需要替换的

        for m in [4, 8, 16, 32, 64, K]:
            # oracle：从 need_replace 中取激活值最大的 min(m, |need_replace|) 个
            # 候选集 = overlap ∪ oracle_new
            # recall = |候选集 ∩ S_true| / K
```

**关键输出**：
- replacement_count 的**分布直方图**（不只是均值），尤其 P90/P99
- m vs recall 曲线（这就是"候选集大小 → recall"曲线的增量版本）
- value-aware recall：被漏掉的基向量的激活值有多大

### 3.3 Oracle Baseline B：热集选择（C1h 上限）

```python
def oracle_hotset(indices, values):
    """
    模拟：固定全局热集 H，测每个 token 的命中情况。

    参数扫描：|H| = N*1%, N*5%, N*10%, N*20%

    输出（每个 |H| 值）：
    - per_token_recall: 每个 token 的 |S_t ∩ H| / K 的分布
    - per_token_recall_weighted: 按激活值加权
    - residual_search_space: 未命中部分需要在多大范围搜索
    - hot_value_ratio: 热集命中的基向量的激活值总和占比
    """
    # 1. 统计全局频率
    freq = count_frequencies(indices)  # [N]

    for pct in [0.01, 0.05, 0.10, 0.20]:
        H = top_pct_indices(freq, pct)  # 全局最热的 pct*N 个

        for each token t:
            hit = S_t & H
            miss = S_t - H
            recall = len(hit) / K
            recall_w = sum(values[hit]) / sum(values[S_t])
```

**关键输出**：
- per-token recall 的**分布**，尤其关注最低 10% 的 token（这些是层次化方案的薄弱点）
- |H| vs mean recall 曲线
- 热集命中 vs 未命中基向量的激活值大小对比——如果热集基向量的系数也大，重构贡献集中在热集

### 3.4 Oracle Baseline C：条件子库（A2b/C1c 上限）

```python
def oracle_sublibrary(indices, values):
    """
    模拟：离线聚类后为每个簇构建子库，oracle routing 到正确簇。

    参数扫描：G = 8, 16, 32, 64

    输出（每个 G 值）：
    - sublibrary_size: 每个簇的子库大小（均值、最大值）
    - oracle_recall: 假设完美路由时的 top-K recall
    - cross_route_recall: 路由到错误簇时的 recall（评估路由错误代价）
    - size_vs_recall_curve: 截断子库大小后 recall 的变化
    """
    # 1. 构建特征：value-aware 稀疏向量
    features = sparse_weighted_vectors(indices, values)  # [num_tokens, N]

    # 2. 聚类（用随机投影降到 ~128 维后做 K-means）
    projected = random_projection(features, dim=128)
    labels = KMeans(G).fit_predict(projected)

    for g in range(G):
        cluster_tokens = tokens_in_cluster(g)
        # 子库 = 该簇所有 token 的 top-K 索引并集
        sublibrary = union_of_topk(cluster_tokens)
        # 但子库可能很大，需要截断
        for N_sub in [N//G, N//(G//2), ...]:
            # 取频率最高的 N_sub 个
            truncated = top_freq_in_cluster(sublibrary, N_sub)
            recall = mean([|S_t ∩ truncated| / K for t in cluster_tokens])

    # 3. 路由错误分析
    for each token t:
        correct_cluster = labels[t]
        for wrong_cluster in sample_wrong_clusters(3):
            cross_recall = |S_t ∩ sublibrary[wrong_cluster]| / K
```

**关键输出**：
- **子库大小 vs recall 曲线**（核心判据：如果 N_sub = N/G 时 recall > 90%，A2b 值得做）
- 路由错误时 recall 掉多少（如果掉得少，路由器可以简单；如果掉得多，路由器必须精确）
- 不同 G 值的 pareto frontier（子库大小越小越好，recall 越高越好）

### 3.5 Oracle Baseline D：种子扩展（C1i 上限）

```python
def oracle_seed_expand(indices, values):
    """
    模拟：从少量强激活种子出发，利用共激活关系扩展候选集。

    参数扫描：
    - 种子数 s = 4, 8, 16, 32
    - 每个种子的近邻数 n = 8, 16, 32, 64

    输出：
    - candidate_size: 扩展后的候选集大小
    - recall: 候选集对真实 top-K 的覆盖率
    - candidate_size_vs_recall_curve: 核心曲线
    """
    # 1. 构建共激活近邻表（用 PMI 而非原始条件概率，控制基频效应）
    pmi_matrix = compute_pmi(indices)  # PMI(i,j) = log P(i,j) / (P(i)*P(j))
    neighbor_table = {}  # i → top-n PMI 近邻

    for each token t:
        S_true = set(indices[t])
        sorted_by_value = sort(S_true, key=values, descending=True)

        for s in [4, 8, 16, 32]:
            seeds = sorted_by_value[:s]

            for n in [8, 16, 32, 64]:
                candidates = set(seeds)
                for seed in seeds:
                    candidates |= set(neighbor_table[seed][:n])

                recall = len(candidates & S_true) / K
```

**关键输出**：
- (s, n) → (candidate_size, recall) 的 2D 表格
- 候选集大小 vs recall 曲线（和其他三个 baseline 画在同一张图上）
- PMI 近邻表的稳定性：用一半数据构建表、另一半数据测试，看 recall 是否一致

**使用 PMI 而非条件概率的原因**：
- 条件概率 P(j|i) 被高频基向量主导——高频 j 对所有 i 的条件概率都高，造成虚假的"近邻"
- PMI 控制了基频：PMI(i,j) = log P(i,j)/(P(i)P(j))，只有真正超出独立假设的共激活才会得到高 PMI

### 3.6 层间对比汇总

```python
def layer_comparison(all_layer_results):
    """
    将四个 oracle baseline 的核心指标按层汇总。

    输出表格：
    | 层 | 增量 recall@m=16 | 热集 recall@10% | 子库 recall@N/16 | 种子 recall@s=16 | 最优方案 |
    """
```

### 3.7 统一可视化

**核心图表：候选集大小 → recall 统一曲线**

四个 oracle baseline 在同一张图上：
- X 轴（主）：候选集大小（log scale）
- X 轴（辅）：等效访存量（bytes）。不同方案的在线开销不只是候选集大小：
  - C1e（增量）：候选集访存 + 状态维护（K×sizeof(int64)）+ 替换判断开销
  - C1h（热集）：热集访存（|H|×h×sizeof(float)）+ 冷集搜索
  - A2b（子库）：路由代价（h×G×sizeof(float)）+ 子库搜索
  - C1i（种子扩展）：种子获取开销 + 近邻表访问（s×n×sizeof(int64)）
  等效访存 = 候选集选择访存 + 方案特有的额外访存。精确值依赖实现细节，此处取 proxy 估计。
- Y 轴：top-K recall（或 value-weighted recall）
- 四条曲线 + 一条"全搜索"基线（recall=100%, 候选集=N）
- 每条曲线上标注关键点（如 recall=90%/95%/99% 时的候选集大小）

**每个 baseline 的分布图**：
- 直方图展示 recall 分布（不只是均值）
- 按层分面展示

### 3.8 输出格式

1. **JSON 数据文件**：完整数值结果，供后续程序读取
2. **PNG 图表**：统一曲线 + 各 baseline 分布图
3. **控制台汇总报告**：关键判定表格

## 4. 运行方法

```bash
python scripts/analyze_activation_patterns.py \
    --checkpoint <path_to_sae_checkpoint_dir> \
    --model <model_name> \
    --num_samples 2048 \
    --seq_len 512 \
    --layers "all" \
    --output_dir results/activation_patterns/
```

校准数据：训练同分布数据集。鲁棒性检查时可通过 `--dataset` 参数指定不同数据集。

## 5. 观测指标

### 5.1 Oracle A：增量选择

| 指标 | 含义 | 关键阈值 |
|------|------|----------|
| recall@m=16 | 允许替换 16 个时的 recall | >90% 说明 C1e 高度可行 |
| recall_weighted@m=16 | value-aware 版本 | >95% 说明漏掉的都是弱激活 |
| replacement_count_P90 | 90% 的 token 需要替换多少个 | <32 可操作 |
| replacement_count_P99 | 99% 的 token 需要替换多少个 | <64 可操作 |

### 5.2 Oracle B：热集选择

| 指标 | 含义 | 关键阈值 |
|------|------|----------|
| per_token_recall@10% | |H|=10%N 时每 token 的 recall | 均值 >40%，P10 >20% |
| per_token_recall_weighted@10% | value-aware 版本 | 均值 >50% |
| hot_value_ratio@10% | 热集命中的激活值占总激活值比例 | >50% 说明热集重构贡献大 |

### 5.3 Oracle C：条件子库

| 指标 | 含义 | 关键阈值 |
|------|------|----------|
| oracle_recall@N_sub=N/G | 子库大小 N/G 时的 recall | >90% 说明 A2b 可行 |
| cross_route_mse_degradation | 路由错误时 MSE 相对正确路由的恶化比例 | <50% 说明路由可以不精确 |
| cross_route_p_increase | 路由错误时 p(τ=0.3) 的绝对增量 | <10pp 说明可容忍 |
| min_N_sub_for_95recall | 达到 95% recall 所需的最小子库大小 | <N/4 有意义 |

### 5.4 Oracle D：种子扩展

| 指标 | 含义 | 关键阈值 |
|------|------|----------|
| recall@s=16,n=32 | 16 个种子各取 32 个近邻时的 recall | >85% 说明有价值 |
| candidate_size@90recall | 达到 90% recall 的候选集大小 | <N/4 有意义 |
| cross_validation_gap | 训练集/测试集上 recall 差距 | <5% 说明近邻表稳定 |

### 5.5 层间对比

| 指标 | 含义 |
|------|------|
| best_method_per_layer | 每层在相同候选集大小下 recall 最高的方案 |
| layer_variance | 各指标跨层的方差（方差大 → 分层策略有价值） |

## 6. 预期结果与判定标准

### 6.1 综合判定（基于统一曲线）

| 结果 | 判定 | 后续行动 |
|------|------|----------|
| 某方案在候选集 < N/4 时 recall > 95% | 该方案理论可行 | 进入算法设计：如何低开销近似 oracle |
| 某方案最好也需要 > N/2 的候选集 | 该方案无价值 | 从 C1 候选中排除 |
| 多个方案都在 N/4 处 recall > 90% | 比较它们的访存量 | 选择等效访存最低的方案 |
| 所有方案在 N/4 处 recall < 80% | 激活模式无可利用结构 | 退回 B 分支（换 SAE 架构）或转向 C1f（低秩编码器） |

### 6.2 组合发现的解读

- **增量 + 热集都好**：考虑混合方案——热集作为常驻集、增量更新冷集部分。
- **子库好 + 种子扩展好**：考虑两阶段——先路由到子库缩小范围，再用种子扩展精选。
- **层间差异大**：不同层用不同 C1 方案，不增加算法复杂度（只是配置不同）。
- **全部 oracle 上限都不好**：说明 top-K 选择本身就是"硬问题"，当前 SAE 架构（B1）产生的激活模式不利于快速选择。应优先转向 B 分支（换架构）或 D2（微调消除选择需求）。

### 6.3 Decode vs Prefill 差异

对所有指标分别报告 decode proxy 和 prefill 阶段的结果。

**关于 decode proxy 的说明**：本实验中无法真正模拟自回归 decode（需要逐 token 生成），因此用序列后半段 token（position > seq_len/2）作为 proxy。这是一个近似——它能反映深位置 token 的激活模式，但不等同于真实 decode 中逐步生成的行为。具体局限：
- Prefill 中所有 token 同时可见完整上文，而真实 decode 是增量生成
- 序列后半段的位置效应可能与真实 decode 位置不同
- 如果 oracle baseline 在这个 proxy 上效果好，真实 decode 效果**不一定更好**（但也不太可能更差，因为 decode 通常变化更缓慢）

对 LUTurbo 的 CPU 低延迟推理场景，decode 阶段的指标权重更高。后续如需精确验证，可用自回归生成的真实 decode trace 复测。

## 7. 实际结果

（实验完成后填写）

## 8. 结论与影响

（实验完成后填写）
- 对 decision-tree.md 哪些节点的状态有影响
- 哪些 C1 方案的 oracle 上限通过/未通过判定
- 是否需要新增 C1h（层次化选择）、C1i（共激活引导选择）节点
- 统一曲线上各方案的排名
- 下一步：通过判定的方案进入"低开销近似实现"阶段
