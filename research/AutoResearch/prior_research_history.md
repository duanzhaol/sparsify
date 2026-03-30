# Prior Research History: Target-Restart Brief

## 1. 文档用途

这份文档的作用不是给出“当前最优主线”，而是给 Agent 一份更低偏置的背景说明。

应当这样使用它：

- 把它当成兼容性约束、工程不变量、和未决问题清单
- 不要把旧位置、旧轮次、旧 frontier 的 family 排名当成当前 target 的默认真理
- 不要把这里出现的候选方向理解成固定 opening order
- 当前 target 上的结论，必须依赖当前 target 自己的新结果

当前固定事实：

- 训练 hookpoint 固定为 `layers.[3].self_attn.q_proj`
- 成本 proxy 固定按 fused QKV 部署矩阵 `1024 x 4096` 统计
- 训练代理目标是维护 `(total_cost, FVU)` 的 2D Pareto frontier

## 1.1 当前 target 的弱先验更新

下面这些只算当前 target 的弱先验，不是结论：

- 当前主战场应放在 `<0.25x total_cost` 区域；`0.25x-0.35x` 只作辅助对照，`>0.4x` 只保留少量质量锚点
- 当前 `<0.25x` 区域最强兼容前沿主要由 `shared_routed_expert_topk` 构成；它现在更适合作为 matched low-cost baseline，而不是被直接视为“旧 family”
- `expert_topk` 仍有极低成本 anchor 价值，但在 `<0.25x` 的质量明显弱于当前 shared+routed 主线
- `lowrank_expert_residual` 在 `0.49x-0.54x` 一带给出了当前更强的质量锚点，但这不意味着后续应让中成本结构继续主导搜索预算

一句话：当前 target 上，优先问题不是“哪个中成本结构最好”，而是“谁能在 `<0.25x` 区域打败现有 low-cost anchors”。

## 2. LUTurbo/Lottable 兼容性约束

硬约束：

1. 最终重构必须能写成一个或多个静态向量库上的有限加权和。
2. 在线侧允许选择、路由、低秩系数、分阶段 residual，但静态库本身不能依赖当前样本动态生成。
3. 如果一个方案依赖 batch 级联动、跨样本共享预算、或输入依赖的动态生成字典，则不能进入主线。

当前 run 的成本定义：

- `selection_cost`: encoder 端为选出在线原子所需的访存
- `deployment_cost`: trunk / sparse lookup / codebook lookup 等部署端访存
- `total_cost = selection_cost + deployment_cost`
- 当前成本 proxy 以 fused QKV 原始矩阵 `1024 x 4096` 为分母
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
- total_cost 必须拆成 encoder 与 deployment 两部分看
- compatibility 与 performance 是两回事；结构兼容不代表一定值得继续

## 4. 不可信的旧结论

下面这些都不应直接继承到当前 target：

- 旧位置上的 family 排名
- 旧位置上的最优 frontier 点
- “复杂 backbone 一定优于简单结构”这类总括判断
- “简单结构已经证伪”这类总括判断
- 对 whitening / codebook / lowrank / two-stage / routed / MoE-like 的好坏预判
- 某个 family 在旧位置失败，就默认它在新位置也失败
- 某个 family 在旧位置表现强，就默认它在新位置仍是 opening 主线
- 固定 opening order、固定主线 family、固定 deferred family 列表

一句话：旧 run 只能提供工程经验和搜索空间，不能提供当前 target 的性能真理。

## 5. 当前 target 上真正未知的事

这些问题都还没有在当前 target 上被证明：

- 简单结构和复杂结构，谁在当前位置的低成本区更优
- low-rank trunk 是否能显著降低当前 target 的所需 K
- factorized scorer 在当前位置是收益还是约束
- codebook / VQ / staged residual 的额外结构是否值得其部署开销
- whitening / preprocessing 在当前位置是稳定增益还是副作用
- loss/preprocess 切换是否会比结构切换更重要
- MoE-like / multi-branch / routed 子库方案是否能在保持导出约束的前提下带来新 Pareto 点

Agent 应把这些当成未决问题，而不是已有答案。

## 6. 当前高优先级方向

当前优先级最高的方向不是泛泛的“结构扩展”，而是明确围绕 `<0.25x` 主战场补点：

- `shared_routed_expert_topk` 的 matched-cost 轻量变体
- 更小 `K`
- 更小 `LATENTS_PER_EXPERT`
- 更轻的 shared scorer / router
- 只有在仍能保持 low-cost 带的前提下，才考虑 very-light residual / factorized 变体

这些方向优先级高，不是因为它们已经被证明最好，而是因为：

- 当前 `<0.25x` 前沿已经有可用点，但质量仍明显不够好
- 这一区间目前的最强点仍主要来自轻量 shared+routed 路线
- 很多会自然把成本抬到 `>0.35x` 的结构，即使质量更高，也不直接回答当前主问题

如果要实现新的 low-cost 原型，应优先满足：

- `ACTIVE_EXPERTS` 很小，例如 1 或 2
- router 足够轻，不要把省下来的选择成本重新吃掉
- 总激活路径仍短，不要因为多 expert 让总 `K` 无限制膨胀
- shared 分支若存在，不能自然把 total_cost 抬出 low-cost 带
- 最终形式仍能写成若干静态子库上的有限加权和

一句话：当前优先级最高的是“更轻的 low-cost MoE-like 结构”，不是一般性的中成本结构扩展。

## 7. 当前默认做法

如果当前 target 还没有足够本地证据，默认做法应当是：

- 先建立当前 target 自己的可解释 baseline 和 cost anchors
- 再根据当前 target 的 frontier 形状决定下一步往哪里扩
- 优先选择归因清晰、接线风险低、能补充新信息的实验
- 当 `<0.25x` 区域已经有 anchor 后，默认应优先继续改善这一区域，而不是把主要预算转回 `0.5x` 左右的结构打磨
- 如果 low-cost family 持续不能给出满意结果，应优先尝试新的轻量 routed / shared+routed 变体，而不是直接跳回更重的 trunk/residual 结构
- 不要让旧位置的主线故事替代当前 target 的新证据
