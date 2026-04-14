# LUTurbo 在线补偿热点维度实验设计

## 1. 背景与目标

当前 LUTurbo 在线补偿机制采用“按百分比选择输入维度做精确补偿”的策略。该策略在部署上是动态的，但目前还不清楚：

- 被选中补偿的输入维度是否高度集中在少量热点维度上
- 这些热点维度是否在长时间推理过程中保持稳定
- 热点现象是少数层特有，还是多数启用补偿的层都存在
- 后续是否值得把“纯动态百分比补偿”演进为“静态热点 shortlist + 动态长尾补偿”

本实验的目标是：

- 在真实 LUTurbo 运行时中，对完整推理链路里所有启用 SAE + 在线补偿的层进行打点
- 统计“输入侧被选中做在线补偿的维度”的热度分布
- 判断是否存在特别热、特别稳定的补偿维度
- 为后续补偿策略优化提供证据，而不是依赖离线猜测


## 2. 核心结论问题

实验最终需要明确回答以下问题：

1. 是否存在少量输入维度覆盖了大部分补偿事件
2. 若存在热点，这些热点是否跨时间窗口稳定
3. 热点是 layer-specific，还是可以抽象出全局规律
4. 当前百分比补偿是否值得演进为：
   - 静态热点 shortlist
   - 静态热点 + 动态长尾补偿
   - 继续维持纯动态百分比补偿


## 3. 范围与非目标

### 3.1 实验范围

- 统计对象固定为：`输入侧被选中做在线补偿的维度`
- 统计环境固定为：`真实 LUTurbo runtime`
- 统计范围固定为：`完整推理链路中所有启用 SAE + 在线补偿的层`
- 统计口径固定为：`运行时真实选中的 selected_input_dims`

### 3.2 非目标

首版实验不追求以下内容：

- 不在离线脚本里重写一套补偿选择逻辑
- 不做输出侧维度热度统计
- 不做全量 per-token 详细日志落盘
- 不做复杂 counterfactual replay
- 不在第一版就联动修改当前补偿算法
- 不允许为了实验显著影响“关闭统计时”的正常推理性能


## 4. 设计原则

### 4.1 真实运行时优先

实验必须在真实 LUTurbo runtime 中打点，避免离线复现实验与真实部署逻辑不一致。

### 4.2 默认关闭，关闭时近似零扰动

统计功能默认关闭。关闭时应满足：

- 不注册额外 collector
- 不分配统计缓冲
- 不打开文件
- 不修改补偿 kernel 本身
- 热路径中只有一个极薄的条件分支，最好在初始化阶段完成旁路裁剪

### 4.3 分层统计，不做全局混合

不同层的输入维度空间、补偿比例、误差结构都可能不同，因此必须以层为单位分别建模与分析，不能把所有层混成一个全局直方图。

### 4.4 聚合优先，trace 只做抽样

主统计以聚合计数为主，尽量避免 I/O 放大。详细 trace 只保留极低比例抽样用于 sanity check。

### 4.5 分析代码独立存放

实验相关解析与可视化代码单独放在独立目录，避免污染现有训练/导出主线。


## 5. 推荐方案

推荐采用“分层画像版”方案：

- 在 runtime 中做轻量聚合统计
- 按固定 token 窗口输出快照
- 抽样保留极少量详细 trace
- 离线分析阶段输出表格、热图、coverage 曲线和稳定性曲线

不推荐：

- 只做最简总次数统计，因为无法判断稳定性
- 做全量 per-token dump，因为 I/O 与存储开销过大，且容易显著扰动 runtime


## 6. 打点位置

打点位置固定在：

- 补偿维度已经根据当前百分比规则选出之后
- 真正执行在线补偿 GEMV / matmul 之前

这样能保证统计到的正是部署路径最终使用的 `selected_input_dims`，而不是某个中间候选集合。

首版不在 selector 内部多处埋点，不打散补偿主链逻辑。


## 7. 运行时统计对象

每个启用补偿的层维护一个独立统计器。

### 7.1 每层常驻聚合统计

- `layer_name`
- `d_in`
- `token_count`
- `total_selected_events`
- `selected_count[d_in]`
- `selected_k_hist`

如果运行时已有可直接复用的 selector score / abs residual / ranking score，则额外记录：

- `selected_score_sum[d_in]`
- `selected_score_max[d_in]`

若该分数在当前实现中获取代价较高，则首版可以只实现 count 版。

### 7.2 窗口快照

每处理固定数量 token，输出一个窗口级 snapshot。推荐窗口大小：

- `window_tokens = 100_000`

每个 snapshot 至少包含：

- `window_index`
- `token_count_cumulative`
- `top_dims_by_count`
- `top_1_4_16_64_128_coverage`
- `gini`
- `entropy`
- `effective_dims`

如果实现了 score 版统计，则补充：

- `top_dims_by_score`
- `score_share_top_1_4_16_64_128`

### 7.3 抽样 trace

仅用于 sanity check，不参与主指标。建议：

- `trace_sample_rate = 0.001`

每条 trace 可包含：

- `global_token_index`
- `sequence_index`
- `layer_name`
- `selected_input_dims`
- `selected_scores`（如可得）
- `selected_dim_count`


## 8. 热点定义

对每一层定义：

- `D = d_in`
- `T = token_count`
- `E = total_selected_events`

对每个维度 `d`：

- `selection_freq[d] = selected_count[d] / T`
- `selection_share[d] = selected_count[d] / E`
- `uniform_share = 1 / D`
- `hotness_ratio[d] = selection_share[d] / uniform_share`

解释如下：

- `hotness_ratio ≈ 1`：接近随机均匀选中，不热
- `hotness_ratio >= 4`：明显热
- `hotness_ratio >= 10`：非常热，值得重点关注

实验主结论不只看单个热点值，还看整体集中度。


## 9. 主分析指标

### 9.1 Count 维度

- top-1 / 4 / 16 / 64 / 128 coverage
- Gini coefficient
- Shannon entropy
- effective dims（可按 `exp(entropy)` 定义）
- hotness ratio 排名前若干维度

### 9.2 Score 维度（可选增强）

如果记录了 score，则再看：

- `selected_score_share[d]`
- score 版 top-N coverage
- count 热点与 score 热点的一致性

这样可以区分：

- “经常出现但不关键”的维度
- “出现次数不算很多，但一旦出现就是高误差”的维度

### 9.3 稳定性维度

对相邻窗口比较：

- top-128 Jaccard overlap
- top-64 coverage 变化量
- Gini 变化量


## 10. 收敛与停止条件

实验不建议一开始固定跑一整晚不判断，而是采用窗口式收敛判定。

### 10.1 最低运行预算

- 至少先跑到 `1M tokens`

### 10.2 推荐停止条件

对单层而言，当连续 3 个窗口同时满足：

- `top-128 Jaccard > 0.95`
- `top-64 coverage` 变化 `< 0.5%`
- `gini` 变化 `< 0.01`

即可认为该层热点结构基本收敛。

全局停止条件可定义为：

- 绝大多数启用补偿的层已收敛

### 10.3 硬上限

- 正式版建议上限：`10M tokens`
- 若要更稳，可以扩展到：`20M tokens`


## 11. 实验分阶段流程

### 11.1 Stage 0：Smoke

先跑 `100k ~ 200k tokens`，目标是验证：

- 所有启用补偿的层都能正确产生日志
- `selected_count` 求和与实际补偿维度总数一致
- `selected_k_hist` 与当前百分比补偿配置一致
- 开启统计后没有显著破坏 runtime 正常行为

同时必须做一次最小性能 A/B：

- `stats off`
- `stats on`

### 11.2 Stage 1：Formal

- 全链路开启统计
- 按 `100k tokens/window` 输出窗口快照
- 至少跑到 `1M tokens`
- 根据稳定性规则决定是否继续到 `10M` 或更高

### 11.3 Stage 2：Optional Follow-up

如果 Stage 1 发现明显稳定热点，再做小范围后续验证：

- 比较不同补偿比例下热点是否更集中
- 评估“静态热点 shortlist + 动态长尾”是否具备实现价值

该阶段不属于首版实验必做项。


## 12. 输出产物

### 12.1 运行时原始输出

建议输出到独立目录，例如：

- `comp_hotdims/`

每层：

- `comp_hotdims/<layer_name>/summary.json`
- `comp_hotdims/<layer_name>/windows.jsonl`

全局：

- `comp_hotdims/global_summary.json`
- `comp_hotdims/sampled_traces.jsonl`

### 12.2 离线分析结果

建议生成：

- `layer_summary.tsv`
- `global_layer_ranking.tsv`
- 每层 top-32 热点条形图
- 每层 `coverage vs top-N` 曲线
- 每层稳定性曲线
- 跨层概览热图


## 13. 结果解释规则

### 13.1 强热点

当某层满足：

- top-64 或 top-128 coverage 很高
- 且跨窗口热点集合稳定

则判定为“强热点层”。

这意味着后续值得探索：

- 静态热点 shortlist
- 静态热点 + 动态长尾补偿

### 13.2 弱热点

当某层存在一定集中度，但：

- coverage 不够高
- 或跨窗口波动较大

则判定为“弱热点层”。

这更适合做 per-layer 差异化策略，而不是全局静态化。

### 13.3 无明显热点

当分布较散，且 top 集合不稳定，则说明：

- 当前百分比动态补偿本质上高度依赖 token 内容
- 静态热点化价值可能有限


## 14. 性能与工程约束

### 14.1 关闭统计时的性能要求

关闭统计时应尽量做到：

- 不影响正常推理性能
- 不增加额外显存/内存长期占用
- 不改动正常补偿结果

实现后应给出一次 `stats off` 与 `stats on` 的吞吐对比，确认统计开销在可接受范围内。

### 14.2 代码存放位置

本实验的解析与分析代码建议独立放在：

- `experiments/compensation_hotdims/`

建议包含：

- `README.md`
- `schema.py`
- `analyze_runtime_dump.py`
- `plot.py`
- `summarize.py`

运行时主链路里只保留一个轻量接入点，不把分析逻辑散落到现有训练/导出主线。


## 15. 后续决策分支

本实验完成后，建议按以下分支决策：

- 若多数层存在强热点：
  - 优先设计静态热点 shortlist 原型
- 若只有少数层存在强热点：
  - 优先做 per-layer 热点化
- 若多数层无明显热点：
  - 不优先投入热点固化
  - 转而优化 selector、补偿比例或补偿计算实现


## 16. 推荐的一句话结论模板

实验最终应能产出类似如下的明确结论，而不是只给图表：

- “在线补偿输入维度在多数层呈现明显长尾热点，top-64 已覆盖 X% 补偿事件，且跨窗口稳定，值得继续做静态 shortlist。”
- “在线补偿输入维度仅在少数层呈现稳定热点，不建议做全局统一热点化，更适合 per-layer 策略。”
- “在线补偿输入维度整体较散且不稳定，当前百分比动态补偿仍是更合理的主线，不建议优先做静态热点固化。”
