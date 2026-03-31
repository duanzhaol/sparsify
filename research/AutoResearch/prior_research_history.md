# Prior Research History: Objective-Restart Brief

## 1. 文档用途

这份文档的作用不是告诉 Agent “当前最优主线是什么”，而是提供一份低偏置背景说明。

应这样使用它：

- 把它当成兼容性约束、工程不变量、和未决问题清单
- 不要把旧位置、旧轮次、旧 leaderboard 的 family 排名当成当前 target 的默认真理
- 不要把这里出现的方向理解成固定 opening order
- 当前 target 上的结论，必须依赖当前 target 自己的新结果

当前固定事实：

- 训练 hookpoint 固定为 `layers.[3].self_attn.q_proj`
- 成本 proxy 固定按 fused QKV 部署矩阵 `1024 x 4096` 统计
- 成本硬约束固定为 `total_cost <= 1.5 x (1024 x 4096)`
- 当前单目标固定为 `objective_score = total_cost_ratio + exceed_alpha_0.50`
- 其中 `total_cost_ratio = total_cost / (1024 x 4096)`
- `FVU` 继续保留，但只作诊断与 tie-break，不再是主优化轴

一句话：现在要优化的是“总部署代理成本 + 在线补偿需求”，不是单独的 low-cost，也不是单独的 low-FVU。

## 1.1 当前 target 的阅读方式

下面这些是当前 target 的阅读方式，不是结论：

- 比较两个候选时，先拆开看 `total_cost_ratio` 和 `exceed_alpha_0.50`，再看它们合成后的 `objective_score`
- 如果一个结构让 cost 略升，但 exceed 明显下降，只要 objective 更低且仍在预算内，就值得保留
- 如果一个结构只让 FVU 更低，但 objective 没降，就不应当作主要成功
- 不要把“更靠左”或“更低 FVU”误当成新的全局主线；真正的 incumbent 只由 objective 决定
- 连续多轮只在同一 family 上做 `K / LATENTS_PER_EXPERT / rank / lr` 小插值、而 objective 没有扩展时，应优先换结构槽位

## 2. LUTurbo/Lottable 兼容性约束

硬约束：

1. 最终重构必须能写成一个或多个静态向量库上的有限加权和。
2. 在线侧允许选择、路由、低秩系数、分阶段 residual，但静态库本身不能依赖当前样本动态生成。
3. 如果一个方案依赖 batch 级联动、跨样本共享预算、或输入依赖的动态生成字典，则不能进入主线。

当前 run 的成本定义：

- `selection_cost`: encoder 端为选出在线原子所需的访存
- `deployment_cost`: trunk / sparse lookup / codebook lookup 等部署端访存
- `total_cost = selection_cost + deployment_cost`
- `total_cost_ratio = total_cost / (1024 x 4096)`
- `objective_score = total_cost_ratio + exceed_alpha_0.50`
- 成本硬约束：`total_cost <= 1.5 x (1024 x 4096)`

兼容性注册表：

| family | final form | search note | structural note | compatibility |
|---|---|---|---|---|
| `topk` | `x_hat = Σ_i α_i b_i` | candidate | 单库稀疏和，最直接的可导出形式 | 直接兼容 |
| `factorized_topk` | `x_hat = Σ_i α_i b_i` | candidate | factorized scorer 只改变在线打分链路 | 直接兼容 |
| `jumprelu` | `x_hat = Σ_i α_i b_i` | candidate | threshold 只改变支持形成方式 | 直接兼容 |
| `group_topk` | `x_hat = Σ_i α_i b_i` | candidate | 组内竞争不改变 decoder 形式 | 直接兼容 |
| `gated` | `x_hat = Σ_i α_i b_i` | candidate | 门控仍导出到同一静态库和 | 直接兼容 |
| `routed` | `x_hat = Σ_i α_i b_i` | candidate | route 只改 support ranking | 直接兼容 |
| `expert_topk` | `x_hat = Σ_e Σ_{i∈S_e(x)} α_{e,i} b_{e,i}` | candidate | 轻量 router 只选择静态子库；激活路径仍是子库上的有限加权和 | 直接兼容 |
| `shared_routed_expert_topk` | `x_hat = Σ_{i∈S_shared(x)} α_i^sh b_i^sh + Σ_{e∈E(x)} Σ_{j∈S_e(x)} α_{e,j} b_{e,j}` | candidate | shared 子库始终激活，routed experts 只在静态子库间选择；最终仍是静态库有限加权和 | 直接兼容 |
| `shared_routed_factorized_expert_topk` | `x_hat = Σ_{i∈S_shared(h(x))} α_i^sh b_i^sh + Σ_{e∈E(h(x))} Σ_{j∈S_e(h(x))} α_{e,j} b_{e,j}` | candidate | factorized / low-rank basis 只改变 shared+routed 的打分链路，decoder 仍是静态子库有限加权和 | 直接兼容 |
| `bucketed_topk` | `x_hat = Σ_i α_i b_i` | candidate | bucket 决策不改变导出形式 | 直接兼容 |
| `whitened_topk` | `x_hat = Σ_i α_i b_i` | candidate | 预处理路径需要在当前 target 单独验证 | 直接兼容 |
| `batch_topk` | batch-coupled | blocked | 依赖 batch 共享预算 | 不兼容 |
| `adaptive_budget_topk` | batch-coupled / sample-coupled quota | blocked | 当前实现不是单样本静态导出 | 不兼容 |
| `lowrank_expert_topk` | `x_hat = Σ_r β_r u_r + Σ_{i∈S_{e(x)}(x)} α_{e,i} b_{e,i}` | candidate | low-rank trunk 吃掉平滑主干，router 只选择静态 expert 子库，最终仍是静态库上的有限加权和 | 扩展兼容 |
| `lowrank_residual` | `x_hat = Σ_r β_r u_r + Σ_i α_i b_i` | candidate | trunk + sparse residual | 扩展兼容 |
| `lowrank_expert_residual` | `x_hat = Σ_r β_r u_r + Σ_{i∈S_{e(x)}(x)} α_{e,i} b_{e,i} + Σ_j γ_j c_j` | candidate | trunk + expert 子库 + 全局 residual 补偿都落在静态向量库上 | 扩展兼容 |
| `lowrank_factorized_residual` | `x_hat = Σ_r β_r u_r + Σ_i α_i b_i` | candidate | 低秩 trunk + factorized residual scorer | 扩展兼容 |
| `lowrank_gated_residual` | `x_hat = Σ_r β_r u_r + Σ_i α_i b_i` | candidate | residual selector head 改为 gated | 扩展兼容 |
| `lowrank_jumprelu_residual` | `x_hat = Σ_r β_r u_r + Σ_i α_i b_i` | candidate | residual selector head 改为 JumpReLU | 扩展兼容 |
| `lowrank_grouped_residual` | `x_hat = Σ_r β_r u_r + Σ_i α_i b_i` | candidate | grouped residual selector | 扩展兼容 |
| `lowrank_multi_branch_residual` | `x_hat = Σ_r β_r u_r + Σ_i α_i b_i` | candidate | scorer mixture 仍可导出为静态库和 | 扩展兼容 |
| `whitened_lowrank_residual` | `x_hat = Σ_r β_r u_r + Σ_i α_i b_i` | candidate | 预处理路径需要在当前 target 单独验证 | 扩展兼容 |
| `whitened_lowrank_gated_residual` | `x_hat = Σ_r β_r u_r + Σ_i α_i b_i` | candidate | 预处理 + gated residual 需要 fresh validation | 扩展兼容 |
| `two_stage_residual` | multi-stage sparse sum | candidate | staged residual 仍是静态库和 | 扩展兼容 |
| `two_stage_residual_expert` | `x_hat = Σ_{i∈S_1(x)} α_i b_i + Σ_{j∈S_{e(r_1)}(r_1)} γ_{e,j} c_{e,j}` | candidate | coarse 静态库 + expert residual 子库的两段式修正，仍是静态库有限和 | 扩展兼容 |
| `lowrank_two_stage_residual` | trunk + two-stage sparse sum | candidate | staged residual path | 扩展兼容 |
| `routed_lowrank_two_stage_residual` | trunk + routed two-stage sparse sum | candidate | route 只在选择链路 | 扩展兼容 |
| `codebook_topk` | `x_hat = Σ_m π_m c_m + Σ_i α_i b_i` | candidate | coarse codebook + sparse residual | 扩展兼容 |
| `residual_vq` | `x_hat = Σ_m π_m c_m + Σ_i α_i b_i` | candidate | hard codebook coarse branch | 扩展兼容 |
| `two_code_residual_vq` | multi-codebook + sparse residual | candidate | 更深 coarse path，结构更复杂 | 扩展兼容 |
| `lowrank_residual_vq` | trunk + codebook + sparse residual | candidate | trunk + codebook + sparse residual | 扩展兼容 |
| `lowrank_soft_codebook_residual` | trunk + soft codebook + sparse residual | candidate | 结构参数更多，部署侧额外查表更重 | 扩展兼容 |
| `lowrank_two_stage_soft_codebook_residual` | trunk + soft codebook + two-stage sparse residual | candidate | staged + codebook + trunk，结构更复杂 | 扩展兼容 |
| `lowrank_asymmetric_two_stage_soft_codebook_residual` | trunk + soft codebook + asymmetric staged residual | candidate | asymmetric staged residual | 扩展兼容 |
| `routed_lowrank_two_stage_soft_codebook_residual` | trunk + soft codebook + routed staged residual | candidate | routed staged residual | 扩展兼容 |
| `bucketed_lowrank_two_stage_soft_codebook_residual` | trunk + soft codebook + bucketed staged residual | candidate | bucketed staged residual | 扩展兼容 |
| `whitened_lowrank_two_stage_soft_codebook_residual` | trunk + codebook + staged residual | candidate | 预处理 + staged/codebook 组合需 fresh validation | 扩展兼容 |
| `lowrank_adaptive_budget_residual` | batch-coupled residual budget | blocked | 当前实现仍不是单样本固定导出 | 不兼容 |

说明：

- `直接兼容` / `扩展兼容` family 都可以进入搜索
- 这张表只表达“结构上能否导出到 LUTurbo/Lottable”，不表达效果排名
- `candidate` 只表示“可以测”，不表示“应该优先”或“已经被证明更好”

## 3. 可信的继承项

下面这些可以从旧 run 继承，因为它们是工程不变量，不依赖旧位置的性能结论：

- 单变量原则必须继续执行；`param_only` 一次只改一个 env 参数
- 新增 tunable 参数必须打通 `sparsify/ + override_registry + config_resolution + runner + scripts/autoresearch_test.sh`
- 第一轮引入新参数后，必须检查 round config 与 checkpoint config 中该参数真的生效
- invalid round、silent fallback、schema mismatch、launcher 未透传，这些问题会直接污染结论
- `total_cost` 必须拆成 encoder 与 deployment 两部分看
- `objective_score` 必须拆成 `total_cost_ratio` 与 `exceed_alpha_0.50` 两部分看
- compatibility 与 performance 是两回事；结构兼容不代表一定值得继续

## 4. 不可信的旧结论

下面这些都不应直接继承到当前 target：

- 旧位置上的 family 排名
- 旧位置上的最优点或历史“objective 分布形状”
- “复杂 backbone 一定优于简单结构”这类总括判断
- “简单结构已经证伪”这类总括判断
- “当前主战场一定是某个固定 low-cost 区间”这类结论
- 对 whitening / codebook / lowrank / two-stage / routed / MoE-like 的好坏预判
- 某个 family 在旧位置失败，就默认它在新位置也失败
- 某个 family 在旧位置表现强，就默认它在新位置仍是 opening 主线
- 固定 opening order、固定主线 family、固定 deferred family 列表

一句话：旧 run 只能提供工程经验和搜索空间，不能提供当前 target 的性能真理。

## 5. 当前 target 上真正未知的事

这些问题都还没有在当前 target 上被证明：

- 哪类结构最能降低当前 `objective_score`
- 当前 incumbent 的主要瓶颈更常见地来自 cost，还是来自 exceed
- 允许适度增 cost 来换更低 exceed 的最佳交换率大致在哪
- `shared bottleneck + expert-specific head` 是否比现有 shared+routed recipe 更高效
- expert 内部 low-rank、或 `shared low-rank trunk + routed residual experts` 是否能给出更好的 objective
- 哪些结构能在更小 `K` 下仍控制住 `exceed_alpha_0.50`
- preprocessing / loss 切换会不会比结构切换更重要
- asymmetric MoE-like / routed 子库方案能否在保持导出约束的前提下形成更低 objective 的新点

Agent 应把这些当成未决问题，而不是已有答案。

## 6. 当前高优先级方向

当前优先级最高的方向不是“默认继续压某个固定成本区间”，而是明确围绕单目标做结构性探索：

- 新的 matched-objective architecture probe，而不是继续沿同一 family 做线性插值
- 不对称分工的 sparse MoE-like 结构：shared/coarse 分支吃平滑主干，routed experts 只做 residual cleanup，active path 尽量短
- `shared bottleneck + expert-specific head`
- expert 内部 low-rank
- `shared low-rank trunk + routed residual experts`
- 更轻的 scorer / router
- 能在较小 `K` 下仍把 `exceed_alpha_0.50` 控住的结构
- 宁可做中等风险但高信息增量的新原型，也不要把预算继续花在 incumbent 邻域的低信息细扫上

这些方向优先级高，不是因为它们已经被证明最好，而是因为：

- 它们更直接回答“怎样同时控制 cost 与 exceed”
- 它们更可能带来新的结构信息，而不是同一 recipe 上的局部插值
- 它们能测试“容量是否能增大，但 active path 仍保持很短”这个核心问题
- 当 objective 长时间不动时，它们比继续补 `K / rank / lr / LATENTS_PER_EXPERT` 插值更有信息价值

如果要实现新的原型，应优先满足：

- `ACTIVE_EXPERTS` 很小，例如 1 或 2
- router 足够轻，不要把省下来的选择成本重新吃掉
- 总激活路径仍短，不要因为多 expert 让总 `K` 无限制膨胀
- shared / coarse 分支若存在，应主要负责平滑主干，不要和 routed experts 对称地共同承担主表示
- 最终形式仍能写成若干静态子库上的有限加权和

## 7. 当前默认做法

如果当前 target 还没有足够本地证据，默认做法应当是：

- 先建立当前 target 自己的 objective anchors
- 再根据当前 target 自己的 leaderboard 形状决定下一步往哪里扩
- incumbent 只当 matched baseline / objective anchor，不要把它自动理解成默认 continuation plan
- 优先选择归因清晰、能补充新结构信息的实验；在工程可控的前提下，允许选择中等接线风险的新结构原型
- 如果最近 2 轮都属于同一 family 且只是局部参数插值，同时 objective 没有明显扩展，则下一轮默认应优先换结构槽位
- 继续留在同一 family 的前提是：你能明确指出还有哪个高价值、未验证、且足以改变 objective 结论的结构假设
- 如果一个方向只让 FVU 下降、却没有让 objective 下降，不应继续主导预算
- 如果一个方向让 cost 上升，但 exceed 明显下降且 objective 更低，则可以继续
- 不要让旧位置的主线故事替代当前 target 的新证据
