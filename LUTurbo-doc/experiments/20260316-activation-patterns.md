# Experiment: 20260316-activation-patterns

> 关联 idea：[ideas/activation-patterns.md](../ideas/activation-patterns.md)
> 决策树节点：C1（选择算法）、A2b（条件分解）
> Worktree 分支：exp/activation-patterns
> 状态：Phase 2 completed（C1e + C1h + C1i + C1c 全部完成）

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

    返回（逐层落盘，内存峰值 = 单层缓存）：
      per_layer[layer_name] = {
          'top_k_indices': Tensor[total_tokens, K],    # int32, top-K 索引
          'top_k_values':  Tensor[total_tokens, K],    # float32, top-K 激活值
          'top_2k_indices': Tensor[total_tokens, 2K],  # int32, top-2K 索引（用于 topL 变体）
          'top_2k_values':  Tensor[total_tokens, 2K],  # float32, top-2K 激活值
          'token_pos':      Tensor[total_tokens],      # int32, 每个 token 在序列中的位置
          'seq_boundaries': List[int],                  # 序列边界
      }
    """
    # batch 化 encode，一次处理 [batch, seq_len, h] → [batch*seq_len, 2K]
    # 用 forward hook 流式捕获单层 hidden states：拿到一层就立刻做 SAE encode，
    # 然后将 top-K / top-2K 结果落盘（mmap/npz）并释放该层缓存
    # 记录序列边界以区分同一序列内的 token 对 vs 跨序列的 token 对
```

**实现要点**：
- batch 化 `sae.encode()` 处理整个序列，不逐 token 循环
- **内存策略**：逐层处理，每层 encode 后立即落盘（mmap/npz），索引使用 int32、激活值使用 float32，内存峰值 ≈ 单层缓存 ~2GB（2048×512×256×(4+4) bytes）。不同时驻留所有层，避免总内存随层数线性增长。
- **hidden state 采集方式**：用 forward hook 流式处理单层输出，捕获到某层 hidden states 后立即 SAE encode 并释放，不先把所有层的 hidden states 全部驻留内存
- 记录每个 token 的序列内位置 `token_pos`（用于区分 prefill/decode 阶段、union2 边界条件过滤）
- 增量选择需要 top-L（L=2K）候选：encode 时取 top-2K 而非 top-K，前 K 个作为 top-K，前 1.5K/2K 作为 top-L 保留集

### 3.2 Oracle Baseline A：增量选择（C1e 上限）

```python
def oracle_incremental(layer_data):
    """
    模拟：保留前一 token 的选择结果，只允许替换 m 个位置。
    替换策略：oracle（从真实 top-K 中选不在旧集合里的、激活值最大的 m 个）。

    输入：
    - layer_data['top_k_indices'], layer_data['top_k_values']
    - layer_data['top_2k_indices'], layer_data['top_2k_values']
    - layer_data['token_pos'], layer_data['seq_boundaries']

    参数扫描：m = 0, 4, 8, 16, 32, 64, K

    三种保留集变体：
    - variant='topK':   保留 S_{t-1}（前一 token 的 top-K，K 个元素）
    - variant='topL':   保留前一 token 的 top-L 候选池（L = 1.5K, 2K）
    - variant='union2':  保留 S_{t-1} ∪ S_{t-2}（前两步 top-K 并集）

    注意：不同变体的保留集大小不同（topK=K, topL=1.5K/2K, union2≤2K），
    因此同一个 m 下的 recall 不可直接比较。跨变体对比时应使用"总候选集大小"
    （= |retained| + m）作为 X 轴，在相同总候选预算下比较 recall。
    跨变体比较统一限制到 pos≥2 的 token 子集，避免边界退化行为污染统计。

    边界条件（union2）：
    - pos=0：无前序 token，退化为全搜索基线，不参与增量统计
    - pos=1：S_{t-2} 不存在，退化为 topK 变体
    - pos≥2：正常使用 S_{t-1} ∪ S_{t-2}
    - 不跨序列边界：不使用上一序列的激活

    输出（每个 (variant, m) 组合）：
    - recall: 命中真实 top-K 的比例（均值 + P50/P90/P99 分布）
    - recall_weighted: 按激活值加权的 recall
    - new_mass_ratio: Σ_{i ∈ S_t \\ retained} |α_i| / Σ_{i ∈ S_t} |α_i|
      （新进入基向量的激活质量占比，比 overlap count 更直接反映漏选代价）
    - mse_c2a: 用命中的基向量 + C2a 内积系数重构的 MSE
    - mse_c2c: 用命中的基向量 + C2c CG 系数（t=5）重构的 MSE
    注意：不使用"完全 oracle 最优系数"（完整最小二乘），因为实际推理中只会用 C2a 或 C2c。
    用完全最优系数会高估 C1 方案的实际价值。
    - replacement_count_dist: 实际需要替换多少个才能 100% recall 的分布
    - burstiness: 替换数 > 阈值的连续 token 段长度分布（run length distribution）
    """
    top_k_indices = layer_data['top_k_indices']
    top_k_values = layer_data['top_k_values']
    top_2k_indices = layer_data['top_2k_indices']
    token_pos = layer_data['token_pos']
    seq_boundaries = layer_data['seq_boundaries']

    for variant in ['topK', 'topL_1.5', 'topL_2.0', 'union2']:
        for each consecutive token pair (t, t+1) within same sequence:
            if variant == 'topK':
                S_retained = set(top_k_indices[t])                 # |S| = K
            elif variant == 'topL_1.5':
                S_retained = set(top_2k_indices[t][:int(1.5*K)])   # |S| = 1.5K
            elif variant == 'topL_2.0':
                S_retained = set(top_2k_indices[t])                # |S| = 2K
            elif variant == 'union2':
                if token_pos[t] == 0: continue  # 跳过序列首 token
                if token_pos[t] == 1:
                    S_retained = set(top_k_indices[t])             # 退化为 topK
                else:
                    S_retained = set(top_k_indices[t]) | set(top_k_indices[t-1])  # |S| ≤ 2K

            S_true = set(top_k_indices[t+1])
            overlap = S_retained & S_true
            need_replace = S_true - S_retained

            for m in [0, 4, 8, 16, 32, 64, K]:
                # oracle：从 need_replace 中取激活值最大的 min(m, |need_replace|) 个
                # 候选集 = overlap ∪ oracle_new
                # recall = |候选集 ∩ S_true| / K
```

**关键输出**：
- replacement_count 的**分布直方图**（不只是均值），尤其 P90/P99
- m vs recall 曲线（含 m=0 和 m=K 两个锚点，完整展示从纯复用到全搜索的趋势）
- **new-mass ratio 分布**：new_mass = Σ_{i ∈ S_t \ retained} |α_i| / Σ_{i ∈ S_t} |α_i|，作为主指标之一
- value-aware recall：被漏掉的基向量的激活值有多大
- **三种保留集的对比**：以"总候选集大小 = |retained| + m"为 X 轴，在相同总预算下比较 topK vs topL vs union2 的 recall，判断更大记忆深度的边际收益
- **burstiness**：run length 分布直方图，关注连续坏段。使用两类阈值：
  - 固定阈值：replacement_count > K/4, K/2（跨层跨变体可比，对接工程预算）
  - 相对阈值：replacement_count > median+1σ（单方案内部异常检测）

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

> go/no-go 阈值基于 **topK 基线变体**（最保守场景）。topL 和 union2 是"额外记忆预算能换多少收益"的探索，不影响 go/no-go 判定。

| 指标 | 变体 | 含义 | 关键阈值 |
|------|------|------|----------|
| recall@m=0 | topK | 纯复用（不替换任何基向量）的 recall | 基线锚点 |
| recall@m=16 | topK | 允许替换 16 个时的 recall | >90% 说明 C1e 高度可行 |
| recall_weighted@m=16 | topK | value-aware 版本 | >95% 说明漏掉的都是弱激活 |
| new_mass_ratio | topK | 新进入基向量的激活质量占比 | <10% 说明漏选代价低 |
| replacement_count_P90 | topK | 90% 的 token 需要替换多少个 | <32 可操作 |
| replacement_count_P99 | topK | 99% 的 token 需要替换多少个 | <64 可操作 |
| burstiness_max_run | topK | 替换数超 K/4 的最长连续段 | <5 tokens 可操作 |
| recall@budget=1.5K | 跨变体(pos≥2) | 总候选预算=1.5K 时各变体的 recall | topL/union2 是否优于 topK+m |
| recall@budget=2K | 跨变体(pos≥2) | 总候选预算=2K 时各变体的 recall | topL/union2 是否优于 topK+m |

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

> 模型：Qwen3-0.6B（28 层，h=1024，N=8192/16384，K=128）
> 数据：FineWeb-Edu，2560 seq × 512 tokens = 1,310,720 tokens
> Hotset：5 层（0, 7, 14, 21, 27）× 3 算子（mlp, qkv, o_proj）
> Incremental：2 层（7, 14）× 3 算子

### 7.1 Hotset（C1h）

**Gini 系数**（基向量使用频率的不均匀度，0=均匀，1=极度集中）：

| 层 | mlp | qkv | o_proj |
|----|-----|-----|--------|
| 0  | 0.46 | 0.40 | 0.42 |
| 7  | 0.45 | 0.43 | 0.66 |
| 14 | 0.35 | 0.37 | 0.58 |
| 21 | 0.49 | 0.56 | 0.88 |
| 27 | 0.46 | 0.43 | 0.77 |

**20% 热集 recall**（|H| = N×20%，per-token recall 均值 / weighted recall 均值）：

| 层 | mlp | qkv | o_proj |
|----|-----|-----|--------|
| 0  | 56% / 50% | 49% / 45% | 51% / 51% |
| 7  | 52% / 53% | 50% / 53% | 72% / 76% |
| 14 | 43% / 48% | 45% / 55% | 63% / 73% |
| 21 | 54% / 58% | 60% / 69% | 94% / 96% |
| 27 | 51% / 62% | 49% / 57% | 81% / 84% |

**关键发现**：
- **o_proj 是热集选择的最佳目标**。Layer 21 o_proj Gini=0.88，20% 热集 recall 94%（P10=89%，方差极小），10% 热集就能达到 84% recall。
- **MLP/QKV 中等**。20% 热集 recall 43-60%，不足以单独用热集覆盖，但作为混合方案的"常驻集"仍有价值。
- **Layer 0 是例外**。所有算子 Gini 都低（0.40-0.46），频率分布接近均匀，热集效果差。
- **Weighted recall ≥ unweighted recall**：高频基向量倾向于有更大的激活值，实际重构贡献比索引命中数暗示的更好。

### 7.2 Incremental（C1e）

**m=0 overlap（topK 变体，相邻 token 自然重叠率）**：

| 层 | mlp | qkv | o_proj |
|----|-----|-----|--------|
| 7  | 24% | 20% | 37% |
| 14 | 25% | 26% | 37% |

**replacement_count（topK，每步需要替换的基向量数）**：

| 层 | mlp (mean/P90) | qkv (mean/P90) | o_proj (mean/P90) |
|----|----------------|-----------------|---------------------|
| 7  | 97.3 / 116 | 102.0 / 120 | 80.5 / 105 |
| 14 | 95.5 / 115 | 94.8 / 113 | 80.5 / 102 |

**Oracle m=64 recall（topK 变体）**：

| 层 | mlp | qkv | o_proj |
|----|-----|-----|--------|
| 7  | 74% | 70% | 86% |
| 14 | 75% | 76% | 86% |

**跨变体对比（m=0 recall）**：

| 变体 | Layer 7 mlp | Layer 7 o_proj | Layer 14 o_proj |
|------|-------------|----------------|-----------------|
| topK | 24.0% | 37.1% | 37.1% |
| topL_1.5 | 28.9% | 44.1% | 44.0% |
| topL_2.0 | 32.6% | 47.6% | 47.8% |
| union2 | ≈topL_1.5 水平 | ≈topL_1.5 水平 | ≈topL_1.5 水平 |

**关键发现**：
- **C1e 单独不可行**。即使 o_proj（最好的算子），自然重叠率也只有 37%，平均每步要替换 80/128 个基向量。
- **Oracle m=64 仍不够**：最好情况（o_proj）86% recall，MLP/QKV 仅 70-76%。需要替换一半基向量才能达到不够用的 recall。
- **topL 变体有 +5~10pp 边际收益**：topL_2.0 > topL_1.5 > topK，但差距不够大。保留更多候选能提高覆盖，但每步替换量依然太大。
- **Burstiness 极严重**：replacement > K/4 的连续段长度 ≈ 整个序列（mean_run ≈ 510），说明几乎每个 token 都需要大量替换，没有"稳定区间"。
- **o_proj 一致最好**：所有指标中 o_proj 都显著优于 MLP 和 QKV。

### 7.3 Seed Expansion（C1i）— Phase 2

> 数据：2560 seq × 512 tokens = 1,310,720 tokens，layers 7/14 × 3 算子
> 代码：`experiments/activation_patterns/seed_expand/run.py`

**PMI 近邻表统计**：

| 算子 | avg valid neighbors (max=64) | coverage (有任意邻居的比例) |
|------|:---:|:---:|
| mlp | 64.0 | 100% |
| qkv | 64.0 | 100% |
| o_proj | 63.5-64.0 | 90-100% |

**Oracle seeds（完美种子，取 top-K 中最强的 s 个）**：

| 配置 | mlp recall | qkv recall | o_proj recall |
|------|:---:|:---:|:---:|
| s=8, n=32 | 21-23% | 18-24% | 12-18% |
| s=16, n=32 | 31-32% | 28-34% | 22-28% |
| s=16, n=64 | 40-41% | 35-42% | 26-34% |
| s=32, n=64 | 53-54% | 50-55% | 44-49% |

**Hotset-as-seeds（C1h+C1i 组合，20% 热集命中作为种子）**：

| 配置 | mlp | qkv | o_proj (L7) | o_proj (L14) |
|------|:---:|:---:|:---:|:---:|
| 平均种子数 | 55-67 | 57-65 | 91.6 | 80.7 |
| H=20%, n=8: recall | 54-63% | 54-62% | 80% | 68% |
| H=20%, n=16: recall | 59-68% | 59-67% | 84% | 70% |
| H=20%, n=32: recall | 66-74% | 66-73% | **88%** | 74% |
| H=20%, n=64: recall | 74-81% | 73-80% | **92%** | 78% |
| H=20%, n=32: recall_w | 74-80% | 75-80% | **92%** | 82% |
| H=20%, n=32: cand/N | 15-16% | 15-16% | 12.5% | 10% |
| H=20%, n=32: P10 | 57-59% | 56-58% | 81% | 65% |

**Cross-validation（训练集建表 vs 全量建表，s=16, n=32）**：

| 算子 | recall (full table) | recall (train-only) | gap |
|------|:---:|:---:|:---:|
| mlp | 0.3108-0.3233 | 0.3104-0.3228 | <0.001 |
| qkv | 0.2811-0.3353 | 0.2808-0.3346 | <0.001 |
| o_proj | 0.2210-0.2755 | 0.2210-0.2750 | <0.001 |

**关键发现**：

1. **Oracle seeds 单独不够**：最好配置 s=32,n=64 也只有 44-55% recall。共激活结构存在但不够紧密——128 个活跃基向量之间不都是"亲密朋友"，无法从少量种子高精度恢复完整 top-K。C1i 作为独立方案不可行。

2. **C1h+C1i 组合效果出色**：
   - **o_proj 最好**：L7 在 12.5%N 候选集下达到 88% recall（recall_w=92%），21%N 下达到 92% recall。原因是 o_proj 的 hotset recall 本身就高（72%），平均产生 ~92 个种子（vs mlp ~67 个），种子数量优势驱动扩展覆盖。
   - **mlp/qkv 中等**：n=32 时 66-74% recall（cand 15-16%N），n=64 时 74-81%（cand 25-27%N）。与 Phase 1 的 hotset-only 43-60% 相比有显著提升，但距离 90%+ recall 还有差距。
   - **层间一致性好**：Layer 7 与 14 趋势一致，o_proj > mlp ≈ qkv。

3. **PMI 近邻表极其稳定**：CV gap < 0.001（甚至接近 0），邻居关系是模型固有结构，不依赖特定数据。近邻表可离线预计算一次，部署时直接查表。

4. **n=32 可能是工程甜点**：n=32 → n=64 recall 增加 ~8pp 但候选集几乎翻倍。n=32 在 recall/candidate 比值上更优。

### 7.4 Sublibrary（C1c）— Phase 2

> 数据：2560 seq × 512 tokens = 1,310,720 tokens，layers 7/14 × 3 算子
> 代码：`experiments/activation_patterns/sublibrary/run.py`

**全子库大小**（不截断，各簇 top-K 索引并集占 N 的比例）：

| G | mlp (L7/L14) | qkv (L7/L14) | o_proj (L7/L14) |
|---|:---:|:---:|:---:|
| 8 | 91%/96% | 96%/100% | 100%/97% |
| 16 | 98%/94% | 100%/100% | 93%/95% |
| 32 | 95%/96% | 97%/97% | 97%/95% |
| 64 | 90%/94% | 91%/98% | 91%/97% |

全子库大小普遍在 90-100%N，说明不同簇的 token 使用的基向量几乎完全重叠。

**截断子库 recall（oracle routing，取簇内频率最高的 N_sub 个基向量）**：

25%N 截断（mlp/qkv: N_sub=2048, o_proj: N_sub=4096）：

| G | mlp (L7/L14) | qkv (L7/L14) | o_proj (L7/L14) |
|---|:---:|:---:|:---:|
| 8 | 66%/57% | 64%/57% | 80%/71% |
| 16 | 68%/59% | 68%/59% | 81%/72% |
| 32 | 70%/62% | 71%/61% | 82%/73% |
| 64 | 73%/65% | 73%/63% | 84%/74% |

50%N 截断（mlp/qkv: N_sub=4096, o_proj: N_sub=8192）：

| G | mlp (L7/L14) | qkv (L7/L14) | o_proj (L7/L14) |
|---|:---:|:---:|:---:|
| 8 | 85%/80% | 83%/80% | 92%/87% |
| 16 | 86%/82% | 86%/81% | 93%/88% |
| 32 | 87%/84% | 88%/82% | 94%/89% |
| 64 | 89%/86% | 89%/84% | 95%/89% |

**Cross-route gap（路由到错误簇时 recall 下降幅度）**：

| G | mlp (L7/L14) | qkv (L7/L14) | o_proj (L7/L14) |
|---|:---:|:---:|:---:|
| 8 | 0.153/0.034 | 0.039/0.002 | 0.000/0.013 |
| 16 | 0.021/0.050 | 0.002/0.001 | 0.020/0.039 |
| 32 | 0.014/0.035 | 0.017/0.041 | 0.011/0.037 |
| 64 | 0.044/0.059 | 0.058/0.009 | 0.031/0.014 |

**关键发现**：

1. **全子库 ≈ 全库，聚类分离度极差**：所有配置的全子库大小都在 90-100%N。不同簇的 token 使用的基向量高度重叠，没有实现真正的"分区"效果。SAE 基向量的使用模式不存在明显的类型分区。

2. **截断子库效果中等**：25%N 时 recall 57-84%，50%N 时 80-95%。没有任何配置在 N/G 时达到 90% recall（§5 的 go/no-go 阈值）。

3. **增加 G 收益递减**：G 从 8 到 64（8 倍），recall 仅增加 5-8pp。更多簇没有带来更好的分区。

4. **路由错误代价小但原因不好**：cross-route gap 普遍很小（0.001-0.06），路由到错误簇 recall 几乎不掉。但这是因为簇间子库高度重叠（"路由不重要因为哪个簇都差不多"），而非路由鲁棒性好。

5. **C1c 单独不如 C1h+C1i**：C1c 在 25%N 时 mlp recall 57-73%，而 C1h+C1i 在 16%N 时就达到 74%。C1h+C1i 用更小的候选集达到更高的 recall。

6. **根因假设**：缺乏聚类结构最可能的解释是当前训练目标（FVU + auxk_loss）没有显式鼓励分组结构，基向量倾向于学成通用表示。这暗示从训练端（B1-constrained）引入分组约束可能有效，但尚未排除激活空间本身不具备可利用分组结构的可能性。详见 [ideas/structured-sae.md](../ideas/structured-sae.md)。

## 8. 结论与影响

### 8.1 C1 方案判定

| 方案 | Oracle 上限 | 判定 | 备注 |
|------|------------|------|------|
| C1e（增量选择） | 自然重叠 20-37%，m=64 recall 70-86% | **单独不可行** | 每步替换量太大，burstiness 极严重 |
| C1h（热集选择） | o_proj 中深层: 20% 热集 → 72-94% recall；浅层/MLP/QKV: 43-63% | **o_proj 中深层高度可行，其余部分可行** | o_proj L7/21/27 表现突出；L0/L14 及 MLP/QKV 需配合其他方案 |
| C1i（种子扩展） | oracle seeds s=32,n=64 仅 44-55% recall | **独立不可行** | 共激活结构不够紧密，少量种子无法恢复完整 top-K |
| C1h+C1i（热集+扩展） | o_proj: 88% recall@12.5%N；mlp/qkv: 66-74%@15-16%N | **o_proj 高度可行，mlp/qkv 中等** | 热集提供大量种子（67-92 个），PMI 扩展补全低频部分 |
| C1c（条件子库） | 全子库 90-100%N，25%N 截断 recall 57-84% | **单独不可行** | 聚类高度重叠，子库无法有效缩减搜索空间；需 B1-constrained 配合 |

### 8.2 对 decision-tree.md 的影响

- **C1e**：状态更新为"oracle 上限不足，单独不可行"。作为混合方案的补充组件保留（用于热集未覆盖的动态部分），但不作为独立 C1 方案。
- **C1h（新增节点）**：状态为"oracle 验证通过（o_proj），待实现"。需要在决策树 C1 分支下新增此节点。
- **C1i（新增节点）**：状态为"独立不可行，C1h+C1i 组合 oracle 验证通过（o_proj 最佳）"。
- **C1c**：状态更新为"oracle 上限不足，单独不可行"。全子库 90-100%N 说明当前 SAE 激活缺乏可用于条件分解的聚类结构。条件子库需配合结构化 SAE 训练（B1-constrained）才可能有效。
- **算子异质性发现**：o_proj 在所有分析中都显著优于 MLP 和 QKV。不同算子需要不同 C1 策略。o_proj 用 C1h+C1i 可达 88% recall；MLP/QKV 在候选集 ≤25%N 的预算下最高仅 74% recall（C1h+C1i@n=32, 15-16%N），C1c@50%N 可达 80-89% 但候选集过大。需要从 SAE 训练端（B1-constrained）引入结构化改进以在小候选集下提升 MLP/QKV recall。

### 8.3 下一步建议

1. **o_proj 组合方案原型**：o_proj 的 C1h+C1i 组合在 12.5%N 候选下达到 88% recall（recall_w=92%），已达到实用水平。可进入在线算法实现阶段：(1) 离线预计算热集表+PMI 近邻表；(2) 在线：热集打分→提取种子→查近邻表扩展→候选精排。
2. **MLP/QKV 方案探索**：Phase 2 全部完成后，MLP/QKV 在候选集 ≤25%N 预算下最高仅 74% recall（C1h+C1i@n=32, 15-16%N）；C1c@50%N 可达 80-89% 但候选集过大。剩余可行路径：(a) 扩大热集到 30%N；(b) **结构化 SAE 训练（B1-constrained）**，从 SAE 训练端引入分组约束使激活具备聚类结构，让 C1c 在更小候选集下变得可行。详见 [ideas/structured-sae.md](../ideas/structured-sae.md)
3. **结构化 SAE 探索**（新方向）：C1c 失败的根本原因是 SAE 激活缺乏分组结构（全子库 90-100%N），这是纯重构训练目标的自然结果。训练时加入分组约束（Group TopK、共激活正则、分块对角编码器）可能同时解决 B（基向量库质量）和 C1（选择开销）两个瓶颈。
4. **扩展模型验证**：在 Qwen3-4B 或更大模型上复测，验证 C1h+C1i 的算子差异模式及 C1c 的聚类缺失是否跨模型成立。
