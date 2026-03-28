# Prior Research History: LUTurbo-Compatible SAE Search

## 文档状态

这份文档用于指导**当前这条 LUTurbo AutoResearch run 在新约束下继续搜索**。
它既不是简单沿用旧时代结论继续刷高成本主线，也不是完全无视既有历史从零开始重开世界。

- 统计来源：`research/history/round_summaries/round_0001.json` 到 `round_0171.json`、`research/history/frontier.json`、`research/history/results.tsv`、`sparsify/sparse_coder.py`
- 当前状态：已完成 171 轮，`round 172` 已启动
- 新管理逻辑已加入 LUTurbo 兼容性约束与 total_cost 预算约束
- 旧 history 仍然有价值，但其中大量结论来自高 EF / 高成本 regime，不能直接当成当前轮次的默认策略
- 旧 frontier 主要用来提供结构灵感、兼容性判断与高质量参考点，不应继续直接主导当前搜索注意力
- 当前阶段的默认目标是补全 `total_cost < 0.5x` 的低开销区域；如果该区域没有可用点，可先从最简单 baseline（如 `TopK`）重新建立对照

请这样使用本文件：

- 用它来总结旧探索中哪些结构思路值得继承，哪些问题已经暴露
- 用它来指导当前 run 在新约束下的搜索方向，而不是机械复用旧实验结论
- 用它来帮助判断某个 family 是否与 LUTurbo/Lottable 兼容

不要这样使用本文件：

- 不要把跨不同 EF、不同 recipe、不同训练阶段的 FVU 直接当成绝对排名
- 不要把旧 frontier 当成当前默认主线答案
- 不要忽略新的架构约束，只按旧结果继续搜索

---

## 1. 本次 AutoResearch 搜索目标

这次 AutoResearch 的目标，是在新兼容性与成本约束下，继续搜索**能够替代 LUTurbo 现有 SAE 方案**的可部署架构。
这不是单纯沿旧高成本主线做局部清洗，但也不是完全抛弃历史、从零开始重开。

要求如下：

- 优先找结构上可落地、可导出、可替换的方案
- 旧 frontier 只能作为参考，不是当前默认目标答案
- 当前阶段优先补 `total_cost < 0.5x` 区域的 Pareto 前沿，在这个区域尽量压低 FVU
- `0.5x-1.0x` 区间主要作为质量锚点或过渡对照
- `>1.0x` 区间只保留少量高质量参考，不应继续作为默认主战场
- 如果低开销区域暂时没有可用点，可以从 `TopK`、`factorized_topk`、`lowrank_residual` 这类简单可导出 baseline 重新建立起点

## 2. LUTurbo/Lottable 兼容性约束

新 proposal 进入主线前，先按下面的标准判断。

硬约束：

1. 最终重构必须能表示成一个或多个固定向量库上的有限加权和。
2. 每个重构分支都必须能静态导出成固定向量库。

补充说明：

- 允许固定层数的多阶段、多分支、残差式分解。
- 不要求必须是单库、单阶段、单步重构。
- 只要每个阶段都对应静态可导出的向量库，且最终重构仍然是这些库项的有限加权和，就满足约束。

兼容性标签：

### `直接兼容`

最终形式接近：

`x_hat = Σ alpha_i b_i`

动作：可直接进入主线搜索。

### `扩展兼容`

最终形式是多个静态库或多个静态分支的有限加权和，例如：

`x_hat = Σ_u beta_u u_u + Σ_m pi_m c_m + Σ_i alpha_i b_i + ...`

动作：写清导出路径后，可以进入主线搜索。

### `不兼容`

典型情况：

- 表示依赖整个 batch，而不是单样本
- 向量库本身依赖输入动态生成
- 很难定义成单样本的静态导出表示

动作：只能作为灵感参考，不能作为主线候选。

这里不把”在线选择成本大”当作兼容性判据；成本问题单独评估。

### 2.1 选择成本维度

兼容性标签只判断”decoder 端能不能导出”，不判断”encoder 端的在线选择代价能不能接受”。

LUTurbo 部署的总访存预算约束为：

`total_cost = selection_cost + deployment_cost`

其中 `selection_cost` 是 encoder 端的访存量（为每个 token 选择 top-K latent），`deployment_cost` 是部署端的查表访存量（K × n_output + trunk/codebook 等额外库）。

**AutoResearch 硬约束：total_cost ≤ 1.5 × h × n。** 超过此阈值的配置将被 policy 拦截。

当前各架构的典型 encoder selection_cost / (h×n) 比率（部署成本额外叠加）：

| 架构 | encoder 比率 | 是否通过 (≤1.5x, encoder-only) |
|---|---|---|
| topk (EF=12) | 3.0x | 不通过 |
| gated (EF=12) | 6.0x | 不通过 |
| lowrank_two_stage_soft_codebook_residual (EF=12) | 6.25x | 不通过 |
| bucketed 系 (EF=12) | 12.25x | 不通过 |
| factorized_topk (EF=12) | 1.62x | 不通过（但接近） |

注意：以上仅为 encoder 选择成本。实际 budget 以 total_cost = selection + deployment 为准，部署查表成本额外贡献约 K×n（K=32 约 +3%，K=128 约 +12%）。

降低 total_cost 的已知方向：

降低 encoder 成本：
- 减小 `EXPANSION_FACTOR`（直接减少字典大小 N = d_in × EF）
- 减小 `TRUNK_RANK`（低秩近似更激进）
- 减少 `NUM_CODES`（更少的 codebook 条目）
- 减小 `STAGE1_RATIO`（调整两阶段预算分配）
- 使用低秩 scorer 代替全字典评分（如 `FACTORIZED_HIDDEN_DIM`）
- 探索不依赖 `h → h×EF` 全连接层的选择机制
- 分级/粗到细的选择策略（先用小 scorer 缩小候选集）

降低 deployment 成本：
- 减小 `K`（直接减少查表次数）
- 减小 `TRUNK_RANK`（减少 trunk 部署库大小）
- 减少 `NUM_CODES`（减少 codebook 部署库大小）

每个 SparseCoder 子类都实现了 `selection_cost_estimate()` 方法，返回 encoder / deployment / combined 三套指标。

---

## 3. 历史总览

### 3.1 四个阶段

| 阶段 | 轮次 | 主导思路 | 主要结果 |
|---|---:|---|---|
| P1 | 1-39 | 先把 `lowrank_two_stage_residual` 低 K frontier 跑明白，再做 K/LR/AUX 清洗 | 建立了最早的低 K 主线，确认 plain `lowrank_residual` 只在极低 K 有价值 |
| P2 | 40-69 | `K=24, EF=12` 严重停滞后，大量扫“机制型” family | 大多数 selector gimmick 全部失败；第一次发现 soft-codebook 是有效方向 |
| P3 | 70-116 | 围绕 soft-codebook 展开家族化扩张 | `lowrank_two_stage_soft_codebook_residual` 接管 EF=12 主线，`routed_*` 在部分 K 上更强 |
| P4 | 117-171 | 固定到 EF=16 后，围绕 bucketed 主干做 K 扫描、架构对照、优化器/预处理/损失清洗 | `bucketed_lowrank_two_stage_soft_codebook_residual` 成为 EF=16 主线；大多数后续清洗没有超越它 |

### 3.2 Agent 思考方式的演化

从 history 看，agent 的思维大致经历了四次转折：

1. 先验证“低秩 trunk + 残差稀疏 refinement”是否成立。  
   这一步是正确的，因为它先把一个有效 inheritance line 建起来，而不是直接乱扫。

2. 当 `K=24, EF=12` 长时间不进步时，agent 转向“机制穷举”。  
   这一阶段 architecture churn 很重，但也正是在这里试出了 `lowrank_soft_codebook_residual` 这个真正有价值的父节点。

3. 发现 soft-codebook 有用后，agent 的主线判断变成了：  
   “粗糙结构需要先被 coarse branch 吸收，再把稀疏预算留给 residual refinement。”  
   这一步是整个历史里最重要的结构性进展。

4. 到 EF=16 阶段后，agent 基本认定 `bucketed_lowrank_two_stage_soft_codebook_residual` 是当前最强 backbone，于是大量轮次在它周围做 K 扫描、家族对照和 recipe 清洗。  
   问题在于后半段 architecture round 仍然很多，非架构轴尝试偏少，而且不少结论带有 regime shift 干扰。

### 3.3 结构上真正有价值的 inheritance line

最值得继承的不是某一个单点，而是下面这条 family 演化链：

1. `lowrank_residual`
2. `lowrank_soft_codebook_residual`
3. `lowrank_two_stage_soft_codebook_residual`
4. `routed_lowrank_two_stage_soft_codebook_residual`
5. `bucketed_lowrank_two_stage_soft_codebook_residual`

换句话说，历史不是“随机试了很多方法”，而是逐渐学到：

- 单纯 residual sparse coding 不够
- coarse codebook 分支确实有用
- 两阶段 residual refinement 比单阶段更稳
- routing 有时能改善 support allocation
- bucketing 在强 backbone 上有效，在弱 backbone 上无效

---

## 4. 当前 frontier 与主线结论

**注意：frontier 已从 (K, EF) 离散格子改为 (total_cost, FVU) 2D Pareto front。**

K 和 EF 不再是固定约束，而是可自由调整的参数。frontier 中每个点代表一个 (total_cost, FVU) 的非支配权衡。total_cost = encoder 选择成本 + 部署查表成本。

以下是旧格式下的历史最优，供新一轮搜索参考：

| 架构 | 历史最好 FVU | 备注 |
|---|---:|---|
| `lowrank_two_stage_soft_codebook_residual` | 0.0826 | EF=12 主线 |
| `routed_lowrank_two_stage_soft_codebook_residual` | 0.0565 | EF=12 高 K 主线 |
| `bucketed_lowrank_two_stage_soft_codebook_residual` | 0.0451 | EF=16 主线 |

可以直接得出三个结论：

1. 旧的 `lowrank_two_stage_residual` 已经不是 frontier 主人，只是早期关键过渡节点。
2. soft-codebook 两阶段家族是最强主线。
3. `bucketed_lowrank_two_stage_soft_codebook_residual` 在大参数预算下最强。

---

## 5. 架构台账

说明：

- 表中“次数”按有训练结果的 round 统计，不含纯 `policy_reject`
- “最好点”格式为 `r轮次 K/EF FVU`
- “兼容性”使用上面的三档定义

### 5.1 单库 selector 家族

| 架构 | 次数 | 最好点 | 核心思路 | 兼容性 | 结论与可改造方向 |
|---|---:|---|---|---|---|
| `topk` | 1 | `r56 K24/EF12 0.2736` | 最朴素单库稀疏编码 | 直接兼容 | 作为基线有意义，但在目标 regime 明显弱；不应重新作为主线扫，只保留作 sanity baseline。 |
| `gated` | 1 | `r53 K24/EF12 0.2679` | 在 top-k 前加 sigmoid gate | 直接兼容 | 单独 gating 很弱；如果重用，只应挂到强 backbone 上，而不是 standalone。 |
| `routed` | 1 | `r54 K24/EF12 0.2878` | 用 router 扰动 support 排名 | 直接兼容 | 裸 routed 不成立；routing 只有接到强 soft-codebook/two-stage 主干上才有意义。 |
| `jumprelu` | 2 | `r167 K32/EF16 0.2467` | 可学习阈值平滑激活 | 直接兼容 | 历史上始终偏弱；若重试，必须有明确阈值控制或 deployment 截断方案，不要再直接裸试。 |
| `group_topk` | 1 | `r48 K24/EF12 0.2737` | 先组内竞争，再全局取 top-k | 直接兼容 | 组约束在裸模型上太强；如果要重用，只应作为强 coarse branch 后的 residual selector。 |
| `factorized_topk` | 1 | `r47 K24/EF12 0.3714` | 稀疏前加低秩 bottleneck scorer | 直接兼容 | 在 EF=12 时 scorer factorization 明显伤害表达（FVU=0.3714）。但在成本约束（≤1.5x）下，factorized_topk 是少数能在 EF≤10 保持可行的架构之一，在低 EF regime 下值得重新评估。 |
| `bucketed_topk` | 1 | `r55 K24/EF12 0.2725` | 根据范数在两套 scorer 间混合 | 直接兼容 | bucketing 本身不是答案；后面真正有效的是把 bucketing 放到强 soft-codebook 两阶段 backbone 上。 |
| `batch_topk` | 2 | `r166 K32/EF16 0.2637` | 在 batch 级别分配总预算 | 不兼容 | 当前定义依赖 batch 共同决策，不适合作为 LUTurbo/Lottable 主线。若重构，必须改成单样本可导出的变长 K 方案。 |
| `adaptive_budget_topk` | 1 | `r108 K24/EF12 0.3593` | 按样本难度动态分配 quota，但仍基于 batch 总预算 | 不兼容 | 和 `batch_topk` 同类问题；若保留思想，只能改成单样本 capped variable-K，而不是 batch-coupled 预算。 |
| `whitened_topk` | 1 | `r57 K24/EF12 165315.8` | 归一化/混合作为预条件再 top-k | 直接兼容 | 训练严重爆炸。当前 trainable whitening 路线直接冻结，除非先有明确数学修正。 |
| `multi_branch_gated` | 1 | `r42 K24/EF12 0.2645` | 多个 gated 分支加权混合后再 top-k | 直接兼容 | branch specialization 没有转成效果；如果重用，应让分支对应明确静态子库，而不是只混 scorer。 |

### 5.2 残差与 low-rank 主干家族

| 架构 | 次数 | 最好点 | 核心思路 | 兼容性 | 结论与可改造方向 |
|---|---:|---|---|---|---|
| `two_stage_residual` | 1 | `r44 K24/EF12 0.2579` | 不带 low-rank trunk 的两阶段 residual sparse | 扩展兼容 | 说明“分阶段”本身不够，必须配合更强 coarse branch。 |
| `lowrank_residual` | 4 | `r160 K24/EF16 0.1007` | 低秩 trunk + 单阶段 residual sparse | 扩展兼容 | 早期是重要父节点，超低 K 有价值；但后续被 soft-codebook 分支整体超越。 |
| `lowrank_two_stage_residual` | 26 | `r16 K32/EF12 0.0893` | 低秩 trunk + 两阶段 residual sparse | 扩展兼容 | 这是早期主线 workhorse，证明 staged residual 是有效的；现在不应重开大扫，但应作为对照父节点保留。 |
| `routed_lowrank_two_stage_residual` | 1 | `r162 K24/EF16 0.0943` | 给两阶段 residual 加 router | 扩展兼容 | 历史上还经历过 code-fix 才跑通，但最终仍不敌 soft-codebook 主线；routing 只有挂到更强 backbone 上才值得。 |
| `lowrank_gated_residual` | 7 | `r171 K64/EF16 0.0776` | residual sparse 改为 gated selector | 扩展兼容 | 绝对数值不算极差，但从未真正赢过当前主线；如果重试，只应配合 LR retune 或更强 coarse 分支。 |
| `lowrank_jumprelu_residual` | 5 | `r148 K32/EF16 0.0954` | residual sparse 改为 JumpReLU | 扩展兼容 | learned threshold 没转成收益；目前应降级为次要灵感而非主线。 |
| `lowrank_grouped_residual` | 2 | `r161 K24/EF16 0.1007` | residual sparse 改为组内竞争 | 扩展兼容 | grouped 约束太强；如果保留，只能放在已经有 coarse branch 的后段。 |
| `lowrank_factorized_residual` | 1 | `r61 K24/EF12 0.1298` | factorized residual scorer | 扩展兼容 | 比 plain residual 差，说明 bottleneck scorer 没解决关键问题。 |
| `lowrank_multi_branch_residual` | 3 | `r157 K24/EF16 0.0998` | 多个 residual scorer 分支混合 | 扩展兼容 | 分支混合只作用在 scorer 上不够；若重做，应让每个分支对应独立可导出的子库。 |
| `bucketed_lowrank_residual` | 3 | `r163 K24/EF16 0.0985` | residual scorer 做 low/high bucketing | 扩展兼容 | 证明 bucketing 裸用不够，真正有效的是 bucketed + soft-codebook + two-stage 的组合。 |
| `lowrank_adaptive_budget_residual` | 1 | `r109 K24/EF12 0.1288` | low-rank trunk + batch-coupled adaptive budget | 不兼容 | 与 `adaptive_budget_topk` 同类，当前不应进入主线。 |
| `whitened_lowrank_residual` | 1 | `r114 K24/EF12 8473.23` | 先预条件 residual 再 sparse 选择 | 扩展兼容 | 训练直接爆炸，说明当前 whitening/preconditioner 设计不稳；冻结。 |
| `whitened_lowrank_gated_residual` | 1 | `r41 K16/EF12 7.8932` | whitened residual + gated selector | 扩展兼容 | 同样灾难性，不值得继续。 |

### 5.3 Codebook / VQ 家族

| 架构 | 次数 | 最好点 | 核心思路 | 兼容性 | 结论与可改造方向 |
|---|---:|---|---|---|---|
| `codebook_topk` | 1 | `r49 K24/EF12 0.2688` | 先 coarse codebook，再 sparse residual | 扩展兼容 | coarse codebook 方向本身是对的，但没有 low-rank trunk 时太弱；后续已经被 `lowrank_soft_codebook_residual` 吸收。 |
| `residual_vq` | 2 | `r168 K32/EF16 0.2429` | hard codebook + sparse residual | 扩展兼容 | 离散分配太硬，整体偏脆；后续应优先 soft-codebook，不要回到硬 VQ 主线。 |
| `two_code_residual_vq` | 1 | `r51 K24/EF12 0.2633` | 两级 hard codebook 后再 sparse | 扩展兼容 | 更深的离散 coarse path 没救回来；说明“再加一个 VQ 层”不是关键。 |
| `lowrank_residual_vq` | 5 | `r147 K32/EF16 0.0944` | low-rank trunk + hard codebook + sparse residual | 扩展兼容 | 比纯 VQ 好很多，但仍 consistently 输给 soft-codebook 系；如果保留，只适合作为“soft -> hard”蒸馏灵感。 |

### 5.4 Soft-codebook 主线家族

| 架构 | 次数 | 最好点 | 核心思路 | 兼容性 | 结论与可改造方向 |
|---|---:|---|---|---|---|
| `lowrank_soft_codebook_residual` | 9 | `r64 K32/EF12 0.0909` | low-rank trunk + soft codebook + 单阶段 sparse residual | 扩展兼容 | 这是第一条真正成功的新父节点；如果重新起步，它应该是最简单的 soft-codebook 对照组。 |
| `lowrank_grouped_soft_codebook_residual` | 5 | `r146 K32/EF16 0.0895` | soft-codebook 后接 grouped sparse | 扩展兼容 | grouped 版始终没打过 plain/two-stage/bucketed；不建议作为主线。 |
| `lowrank_gated_soft_codebook_residual` | 3 | `r139 K32/EF16 0.0872` | soft-codebook 后接 gated sparse | 扩展兼容 | 比 grouped 稍好，但仍没形成主线优势；如果重用，只能作为 selector head 实验，不要单独长期占轮次。 |
| `lowrank_two_stage_soft_codebook_residual` | 21 | `r137 K32/EF16 0.0826` | low-rank trunk + soft codebook + 两阶段 residual sparse | 扩展兼容 | 整个 history 最重要 family 之一；EF=12 大部分 frontier 由它接管，应该继续作为核心父节点。 |
| `lowrank_asymmetric_two_stage_soft_codebook_residual` | 3 | `r140 K32/EF16 0.0885` | 把两阶段预算前置，75/25 分配 | 扩展兼容 | 固定非对称预算没有证明有效；若再试，应改成可学习或数据依赖的 stage budget，而不是手写比例。 |
| `routed_lowrank_two_stage_soft_codebook_residual` | 13 | `r97 K128/EF12 0.0565` | 在两阶段 soft-codebook 主干上显式做 routing | 扩展兼容 | 这是第二条真正成功主线；EF=12 的 K=8/16/64/96/128 都由它占据，说明 routing 在强 backbone 上确实有价值。 |
| `routed_lowrank_asymmetric_two_stage_soft_codebook_residual` | 1 | `r116 K24/EF12 0.1546` | routing + 非对称预算同时叠加 | 扩展兼容 | 复杂度加太快，效果没有支撑；说明同时叠多个变化不如沿主线逐步改。 |
| `bucketed_lowrank_two_stage_soft_codebook_residual` | 22 | `r130 K128/EF16 0.0451` | 在两阶段 soft-codebook 主干上做 bucketed stage scorer | 扩展兼容 | 当前 EF=16 主线霸主。真正证明了 bucketing 要依附强 backbone 才有收益。下一轮应优先从它及其父节点继续改。 |
| `whitened_lowrank_two_stage_soft_codebook_residual` | 1 | `r118 K16/EF12 14872.87` | 在 soft-codebook 两阶段主干上加 residual whitening | 扩展兼容 | 严重爆炸，当前 whitening 路线冻结。 |

---

## 6. 关键 family 的结构公式

这一节不是为了数学完整，而是为了让下一轮 agent 明白哪些结构是可以直接继承的。

### 6.1 `lowrank_residual`

`x_hat = U(Vx) + Σ_i alpha_i b_i`

- `U(Vx)` 是低秩 trunk
- `b_i` 是 residual sparse dictionary

结论：这是第一代“扩展兼容”结构。

### 6.2 `lowrank_soft_codebook_residual`

`x_hat = U(Vx) + Σ_m pi_m c_m + Σ_i alpha_i b_i`

- 在 low-rank trunk 后，加一层 soft codebook coarse branch
- 然后再做 residual sparse coding

结论：这是整个 soft-codebook 主线的起点。

### 6.3 `lowrank_two_stage_soft_codebook_residual`

`x_hat = U(Vx) + Σ_m pi_m c_m + Σ_i alpha_i^(1) b_i + Σ_j alpha_j^(2) b_j`

- 先 low-rank trunk
- 再 soft codebook coarse branch
- 再做两阶段 residual sparse refinement

结论：这是当前最值得继续继承的基础公式之一。

### 6.4 `routed_lowrank_two_stage_soft_codebook_residual`

最终重构公式与上一条相同；变化只在 support selection：

- 路由器只改变选择哪些原子进入 stage1/stage2
- 最终仍落在固定库上

结论：routing 在这里是“selector 改造”，不是“表示改造”。

### 6.5 `bucketed_lowrank_two_stage_soft_codebook_residual`

最终重构公式仍然与 5.3 相同；变化在 stage scorer：

- 每个 stage 都不是单一 scorer，而是 low/high 两套 scorer 由 norm gate 混合
- 但最终 decode 仍是固定库

结论：bucketing 在这里是有效的，因为 backbone 本身已经足够强。

---

## 7. 从 history 提炼出的硬结论


这部分内容和结论都是在之前的约束下得到的。在新的约束下，也许之前不行的方案，现在可能变得可行了。

所以，请你不要直接复用这部分的结论。


### 7.1 哪些方向已经基本证伪

- 所有 batch-coupled budget family：`batch_topk`、`adaptive_budget_topk`、`lowrank_adaptive_budget_residual`
- 所有当前实现形态的 whitening/preconditioning family：`whitened_topk`、`whitened_lowrank_residual`、`whitened_lowrank_gated_residual`、`whitened_lowrank_two_stage_soft_codebook_residual`
- 裸 selector gimmick：`topk`、`gated`、`routed`、`group_topk`、`factorized_topk`、`multi_branch_gated`
- 硬 VQ 主线：`residual_vq`、`two_code_residual_vq`

这些方向不是“永远不许碰”，而是在没有新理由时，不应重新占据**高成本主线**轮次。
尤其对裸 selector family，上述负结论主要成立于高 EF / 高成本 regime；在 `total_cost < 0.5x` 的极低开销区，这些简单架构仍可能是合理起点，不应因旧 regime 的失败被一刀切排斥。

### 7.2 哪些方向应作为真正的可继承父节点

- `lowrank_soft_codebook_residual`
- `lowrank_two_stage_soft_codebook_residual`
- `routed_lowrank_two_stage_soft_codebook_residual`
- `bucketed_lowrank_two_stage_soft_codebook_residual`

这四个 family 构成下一轮最值得复用的核心父系。

### 7.3 哪些负面结果不应该简单归罪于 architecture

history 里有几个常见误判来源：

1. EF regime shift。  
   很多 family 早期主要在 `EF=12` 下试，后面又被拿到 `EF=16` 下做现代对照。输掉并不总是 architecture 本身的锅。

2. Optimizer/LR 失配。  
   尤其在 EF=16 阶段，`adam` 的历史表现更像“逐步靠近可用区间但仍未调好”，而不是“一次失败就可完全判死”。

3. 只看最后 F，不看曲线形状。  
   历史后段已经出现明显模式：某些 recipe 最终 F 差一点，但曲线形状显示训练仍在改善或者只是 LR 偏大/偏小。

4. code-fix 和 scientific result 混在一起。
   `routed_*` 家族至少出现过”先修实现，再重新比较”的情况，不能把第一次失败直接当作结构负结论。

### 7.4 新成本约束下的 regime shift

以上 §7.1-7.3 的结论均在 EF=12 或 EF=16 下得出。当前 AutoResearch 已引入选择成本硬约束（≤1.5×h×n），所有 EF=12 配置均超出预算。这意味着：

1. 之前在 EF=12/16 下的 FVU 排名不直接适用于低 EF regime
2. “简单”架构（topk、factorized_topk）在低 EF 下可能是最佳 cost-FVU 权衡
3. 复杂架构（bucketed_lowrank_two_stage_*）在 EF≤2 才能满足成本约束，复杂度优势未必能在如此低的 EF 下体现
4. 应将”低 EF 下的 FVU”作为新的独立搜索维度
5. 在 `total_cost < 0.5x` 区域，应积极探索更简单、更小、更稀疏的结构，而不是默认沿高质量主线做邻域微调
6. 对多专家/多子库方案，只有在 router 很轻、expert 更小、且最终仍能导出为静态子库有限加权和时才值得优先尝试

---

## 8. 非架构因素的复盘

### 8.1 这个系统历史上太偏 architecture churn

`171` 个 round 里：

- 以 `architecture` 为主的轮次：`85`
- 以 `k` 为主的轮次：`46`
- `auxk_alpha`：`13`
- `lr`：`8`
- `optimizer`：`5`
- 其他训练轴总和：很少

这说明：

- 历史经验很丰富，但主要集中在“换 family”
- recipe 空间其实并没有被对称地探索

### 8.2 已经得到的 recipe 层经验

- `signum` 仍然是目前最稳的主线优化器
- `adam` 不应因一次失败被判死，但它的有效区间显然和 `signum` 不同，必须配 LR retune 一起看
- `Hadamard / whitening` 在当前实现上几乎一律负面
- `AUXK_ALPHA` 能做局部清洗，但从历史看它不是决定性主效应
- `Matryoshka` 和 loss shaping 目前只试了很少轮，不足以下结论，但现阶段也没有显示出能压过主线 backbone

### 8.3 下一轮对训练轴的要求

下一轮如果再试 optimizer / loss / preprocessing，必须遵守：

1. 改 optimizer 时允许联动调 LR。  
   不要再用“换了 optimizer 但沿用旧 LR 就输了”来下结论。

2. 必须看训练曲线形状，不只看最终 F。  
   至少要区分：
   - 直接崩坏
   - 明显收敛到差点
   - 在改善但步子不对

3. 训练轴探索只能挂在强兼容 backbone 上。  
   不要在已经弱势的 family 上做大量 recipe 清洗。

---

## 9. 下一轮 AutoResearch 应如何修改流程

### 9.1 不要用单一叙事理解“下一轮”

下一轮既不是“完全从零开始重搜 SAE”，也不是“继续沿旧高成本主线一路微调”。

更准确地说，应分成两个 regime：

- 在 `total_cost < 0.5x` 区域，目标是补低开销前沿，允许从简单架构重新出发
- 在 `total_cost >= 0.5x` 区域，高质量 soft-codebook 主系仍可作为对照、锚点和继承参考

### 9.2 新 family 进入候选集前，必须先写出这五项

1. 它的最终重构公式
2. 其中有哪些固定向量库
3. 它是 `直接兼容`、`扩展兼容` 还是 `不兼容`
4. 它是从哪个历史父节点改出来的
5. 它相对于父节点只改了哪一个结构点

如果这五项写不清，就不应该进入 round。

### 9.3 允许优先探索的结构原语

在当前阶段，优先考虑下面这些仍可导出、且有望兼顾低成本的原语：

- low-rank trunk
- soft codebook coarse branch
- 1-2 个 residual sparse refinement branch
- routing / gating / bucketing 这类 selector-side 改造
- 多个静态子库的显式分支
- factorized / grouped / coarse-to-fine scorer
- 轻量多专家 / 多子库路由，但前提是 router 很轻、expert 更小、最终仍能导出为静态子库有限加权和

### 9.4 仍应默认冻结的结构原语

- batch-coupled budget
- trainable whitening / preconditioning family
- 没有明确理由的硬 VQ 深化
- 没有导出公式的在线复杂 trunk

### 9.5 搜索策略应按成本区间分层

第一层：低开销区结构层  
在 `<0.5x` 区域优先补前沿，可从简单架构重新出发，重点关注极低 EF、极低 K、更短选择链路、factorized scorer、轻量多专家/多子库方案。

第二层：中高成本对照层  
在 `0.5x-1.0x` 或更高区域，只保留少量强 backbone 作为质量锚点和解释性对照，必要时沿 soft-codebook 主系做小步结构改造。

第三层：recipe 层  
optimizer / LR / loss / aux 清洗应只挂在当前最有希望的 1-2 个 family 上，不要在明显弱势 family 上做大量 recipe 清洗。

### 9.6 建议的搜索顺序

1. 先补 `<0.5x` 区域：`topk`、`factorized_topk`、`lowrank_residual`、更轻的 factorized/grouped/coarse-to-fine 变体
2. 再看 `0.5x-1.0x` 区域：寻找更强但仍不过度复杂的过渡结构
3. 仅保留少量高质量 soft-codebook 主系点作为质量锚点：`lowrank_two_stage_soft_codebook_residual`、`routed_*`、`bucketed_*`
4. 如果新结构来自高质量主系，应写清自己继承自哪个父节点；如果新结构是低开销新起点，也要写清为何它更适合 `<0.5x` regime

---

## 10. 下一轮最值得探索的修改方向

这些方向都建立在“复用旧经验，但按新成本 regime 重新排序优先级”之上。

### 10.1 在高质量参考线上，只改 selector

目标 family：

- `lowrank_two_stage_soft_codebook_residual`
- `bucketed_lowrank_two_stage_soft_codebook_residual`

可改点：

- 更稳定的 routing / bucketing 方式
- stage1 / stage2 的可学习预算，而不是固定 50/50 或 75/25
- 让 selector 改造不改变最终可导出公式

### 10.2 在高质量参考线里，默认保留 coarse branch

history 已经很清楚：

- soft-codebook 是真正解决问题的关键增量
- 退回到纯 residual family，通常只会得到更差对照

所以如果你是在 `>=0.5x` 的高质量参考线上继续改，新家族最好默认保留 coarse branch。
但在 `<0.5x` 的低开销区，不应把“必须保留 coarse branch”当成硬约束。

### 10.3 在低开销区，优先尝试简单且更短链路的结构

更合理的候选包括：

- 极低 EF + 小 K 的 `topk`
- factorized / grouped / coarse-to-fine scorer
- 更激进的 `lowrank_residual`
- 轻量多专家 / 多子库方案，但必须满足 LUTurbo 可导出约束

这些结构未必能赢下全局最低 FVU，但它们更有机会先把 total_cost 压进 `<0.5x` 区域。

### 10.4 如果要做多分支，就让每个分支真的对应静态子库

过去失败的一个典型模式是：

- 分支只在 scorer 上混合
- 最终 decode 仍然太像单库

更合理的方向是：

- 如果做 branch specialization，就让每个 branch 都对应明确可导出的静态子库
- 否则只是让 encoder 更复杂，不一定给部署带来任何有意义的结构

### 10.5 把“兼容性”写进 prompt，而不是靠事后人工筛选

下一轮 prompt 应该显式告诉 agent：

- 目标不是任意提高 FVU
- 目标是找到可替代 LUTurbo/Lottable 现有 SE 模型的兼容结构
- 每个 proposal 必须附带导出公式和兼容性标签

---

## 11. 给下一轮 agent 的起始先验

可以直接把下面这几条作为 prompt 中的硬性提醒：

1. 当前默认目标不是继续刷新全局最低 FVU，而是补全 `total_cost < 0.5x` 区域的 Pareto 前沿；在这个区域允许从简单架构（如 `topk`、`factorized_topk`、`lowrank_residual`）重新出发。
2. 任何新架构都必须先写出最终重构公式，并判断是 `直接兼容`、`扩展兼容` 还是 `不兼容`。
3. `batch_topk`、`adaptive_budget_*`、所有 `whitened_*`、硬 VQ 主线，默认冻结，不要无理由重开。
4. 如果改优化器，不要只换优化器名；允许同时调 LR，并且必须看训练曲线形状，而不是只看最后的 F。
5. 在 `<0.5x` 区域，简单架构不是被默认排除的对象；在 `>=0.5x` 区域，真正值得继续的高质量父节点只有少数几个。
6. 如果一个 proposal 不能说明它相对于历史父节点到底改了什么，就说明它还不够成熟，不应进入 round。

---

## 12. 一句话总结

过去 171 轮最重要的收获，不是“试过很多架构”，而是：

- 在高成本高质量 regime，已经基本找到了有效主系：
  `low-rank trunk -> soft codebook coarse branch -> staged sparse refinement -> selector-side routing/bucketing`
- 在新 total_cost 约束下，当前更重要的问题不是继续刷这条主系，而是补全 `<0.5x` 低开销区域，寻找更简单、更短链路、仍保持 LUTurbo 可导出的结构

所以，下一轮不该再被“只沿高质量主系推进”这一个叙事绑住；正确做法是：
低开销区优先探索简单结构，高质量主系只保留为参考锚点，并且所有改动都必须保持 LUTurbo/Lottable 可导出性。
