# Qwen3-4B Deploy Objective AutoResearch Design

## 1. 背景与目标

本设计用于构建一个全新的、轻量专用版 AutoResearch 框架，专门服务于如下单任务目标：

- 模型固定为 `Qwen3-4B`
- hookpoint 固定为 `layers.[17].self_attn.q_proj`
- SAE family 固定为 `product_key_expert_jumprelu`
- 优化目标固定为：
  - `objective = total_cost_ratio + best_exceed_alpha_0.50`

其中：

- `total_cost_ratio` 来自训练启动前的静态 cost proxy 输出
- `best_exceed_alpha_0.50` 指训练从开始到当前 checkpoint 为止，历史最优的 `exceed_alpha_0.50`

该框架不复用现有 `research/AutoResearch/` 的代码、状态目录、frontier、controller、prompt、history，也不共享其默认目标定义与 continuation 逻辑。

本次设计的核心思想是：

- 保持训练代码与训练脚本不被自动修改
- 仅进行 `param-only` 自动搜索
- 使用 checkpoint 恢复式训练
- 由程序负责状态机和硬规则
- 由 agent 在边界情况下决策是否继续当前 trial，以及下一个新 trial 的参数


## 2. 非目标

首版不追求以下能力：

- 不替代现有通用 `research/AutoResearch/`
- 不支持任意模型 / 任意 hookpoint / 任意 family 的泛化研究
- 不自动编辑 SAE Python 代码或训练 shell 脚本
- 不同时并发运行多个 trial
- 不引入复杂的贝叶斯优化器、bandit、MCTS 等搜索器
- 不以内建 Web 搜索为必要组件


## 3. 设计原则

### 3.1 彻底隔离

新框架与旧框架在以下层面隔离：

- 代码目录隔离
- 状态目录隔离
- trial 日志隔离
- leaderboard 隔离
- prompt / agent 决策上下文隔离

### 3.2 单目标优先

框架只优化一个主目标：

- `best_objective_so_far = total_cost_ratio + best_exceed_alpha_0.50_so_far`

FVU 仅作为诊断项与并列时的次级参考，不参与主排序。

### 3.3 深度优先 + 保守继续

调度策略固定为：

- 同一时刻只跑一个 trial
- 每个 trial 固定占用两张卡
- 对于已有潜力的 trial，优先继续吃下一段训练预算
- 只有当当前 trial 被判定“不值得继续”时，才生成并启动下一个新 trial

### 3.4 Agent 决策受约束

agent 不能任意改代码，也不能自由生成无限参数空间中的配置。它只能：

- 判断当前 trial 是否继续
- 在允许的参数空间内提出下一个新 trial

所有参数都必须经过程序侧的合法性校验和硬约束过滤。


## 4. 首版固定边界

### 4.1 固定训练上下文

首版框架的运行上下文固定为：

- `MODEL_PATH=$HOME/models/Qwen3-4B`
- `HOOKPOINTS='layers.[17].self_attn.q_proj'`
- `ELBOW_THRESHOLD_PATH=$HOME/sparsify-ascend/thresholds/Qwen3-4B/thresholds_q.json`
- `ARCHITECTURE=product_key_expert_jumprelu`
- 训练命令基于现有 `scripts/autoresearch_test.sh`

### 4.2 固定资源模型

- 同时只运行 1 个 trial
- 每个 trial 固定使用 2 卡
- 总新 trial 上限为 200

注意：由于每个新 trial 的第一次 checkpoint 就是 15M tokens，因此该系统的默认定位不是“一晚上跑完 200 个新 trial”，而是“长周期连续研究框架，可跨多晚恢复续跑”。

### 4.3 只做 param-only 搜索

首版允许自动搜索的参数为：

- 结构参数：
  - `K`
  - `NUM_EXPERTS`
  - `LATENTS_PER_EXPERT`
  - `ACTIVE_EXPERTS`
- 训练参数：
  - `LR`
  - `AUXK_ALPHA`

其中首版主搜索重心放在结构参数；训练参数只做少量微调。


## 5. 目录与状态组织

### 5.1 新框架代码目录

建议新增独立代码目录：

- `research/deploy_objective_autoresearch/`

建议模块边界如下：

- `config.py`
  - 固定运行上下文
  - 动态搜索边界定义
  - checkpoint 梯度
- `state_store.py`
  - 读写全局状态
  - 维护 trial 列表与 leaderboard
- `trial_runner.py`
  - 启动训练
  - 恢复 checkpoint
  - 跑到下一个 checkpoint 后返回
- `metrics_extractor.py`
  - 从 checkpoint / metrics / log 中提取结构化摘要
- `scheduler.py`
  - 负责深度优先调度
  - 负责 continue / stop / spawn next
- `agent_interface.py`
  - 负责向 agent 构造摘要
  - 解析 agent 决策输出
- `proposal_policy.py`
  - 动态提议动作类型
  - 参数边界与提议规范化规则
- `main.py`
  - CLI 入口

### 5.2 本次 run 的独立状态目录

本次 run 的状态目录建议固定在：

- `research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective/`

建议文件布局：

- `README.md`
  - 当前 run 目的、约束、说明
- `run_config.json`
  - 固定运行上下文与动态搜索边界
- `state.json`
  - 全局状态机
- `leaderboard.json`
  - 当前最优 trial 列表
- `trials.jsonl`
  - 全量 trial 生命周期记录
- `agent_decisions.jsonl`
  - 每次 checkpoint 的 agent 决策
- `trials/<trial_id>/`
  - 单个 trial 的本地摘要、checkpoint 指针、日志、解析结果


## 6. 主指标定义

### 6.1 静态量

每个 trial 在启动前即可计算：

- `total_cost_ratio`

这是该 trial 的静态部署代价项。

### 6.2 动态量

每个 trial 在训练过程中持续更新：

- `best_exceed_alpha_0.50_so_far`
- `best_objective_so_far = total_cost_ratio + best_exceed_alpha_0.50_so_far`
- `latest_exceed_alpha_0.50`
- `best_fvu_so_far`
- `latest_fvu`

### 6.3 最近窗口改善量

为了做 continuation 决策，每个 checkpoint 还需要计算最近一个 15M 窗口内的变化：

- `delta_best_exceed`
- `delta_best_objective`
- `delta_best_fvu`


## 7. Checkpoint 机制

### 7.1 固定阶梯

新 trial 的第一次 checkpoint 固定为 15M tokens。

之后固定每 15M 决策一次：

- `15M -> 30M -> 45M -> 60M -> 75M -> ...`

### 7.2 Checkpoint 驱动的生命周期

每个 trial 只能处于下列状态之一：

- `pending`
- `running`
- `checkpoint_ready`
- `continue_scheduled`
- `stopped`
- `finished`

状态变化规则：

- 新 trial 创建后进入 `pending`
- 启动训练后进入 `running`
- 跑到下一个 checkpoint 后进入 `checkpoint_ready`
- 若决定继续则进入 `continue_scheduled`
- 若恢复启动则重新回到 `running`
- 若确定不再继续则进入 `stopped`
- 若达到框架设定的自然结束条件则进入 `finished`


## 8. 调度策略

### 8.1 深度优先

首版采用深度优先策略：

- 当前 trial 若仍有潜力，则优先继续，不急于切换新 trial
- 只有当当前 trial 被判定不值得继续时，才启动新的参数配置

### 8.2 保守继续

在 continuation 哲学上，首版偏向保守继续：

- 只要 trial 仍显示出明确潜力，就继续给下一段 15M
- 只有在“比较确定没戏了”时才停止

### 8.3 总新 trial 上限

- 最多创建 200 个新 trial
- 对同一 trial 的继续训练不计入新的 trial 名额


## 9. 程序与 Agent 的职责分工

### 9.1 程序负责

- 训练启动与恢复
- trial 状态机推进
- checkpoint 完成检测
- metrics 抽取
- leaderboard 更新
- 合法性检查
- 硬规则过滤

### 9.2 Agent 负责

- 对边界 trial 做继续/停止判断
- 为下一个新 trial 生成参数提议
- 根据历史 trial 结果调整搜索方向

### 9.3 为什么不让 Agent 完全自由

完全自由的 agent 方案在长时间夜间运行中存在问题：

- 参数空间漂移
- 低可复现性
- 不容易 debug
- 容易重复探索无效区域

因此首版采用混合模式：

- 程序做硬约束
- agent 在动态边界内做策略决策


## 10. 新 trial 参数生成策略

首版不采用“提前写死的固定枚举搜索空间”，而采用“动态搜索边界”。

程序只负责定义：

- 哪些参数允许被搜索
- 每个参数的合法范围
- 哪些组合显然非法
- 去重与规范化规则
- 是否需要对齐到更规整的结构形状

agent 负责根据历史结果动态塑造“下一阶段值得搜索的局部区域”。

在这个前提下，首版 agent 只能产生 3 类新配置动作：

### 10.1 邻域微调

围绕当前强 trial 小步修改 1 到 2 个参数，例如：

- 调整 `K`
- 微调 `NUM_EXPERTS`
- 微调 `LATENTS_PER_EXPERT`

### 10.2 结构重平衡

保持大体容量近似，但改变结构形状，例如：

- 更多 experts + 更小单 expert
- 更少 experts + 更大单 expert

这类动作对静态 cost 与 exceed 的折中很重要。

### 10.3 探索性跳跃

少量尝试与当前最优区域差异较大的配置，用于防止局部最优。

默认建议比例：

- 60% 邻域微调
- 30% 结构重平衡
- 10% 探索性跳跃

### 10.4 动态边界而非固定表

首版不提前维护一张静态候选值总表，而是让 agent 根据以下信息动态决定当前该搜哪一片区域：

- 当前 incumbent
- 当前 trial 的 continuation 表现
- 最近若干 trial 的结果摘要
- 已确认的潜力区域
- 已确认的低效区域

程序只做最后一道边界检查与规范化落地，避免 agent 无约束漂移。

### 10.5 规整化优先但不预先枚举

尽管不使用固定枚举表，程序在最终落地参数时仍应优先保留更规整的结构形状，尤其是：

- `NUM_EXPERTS`
- `LATENTS_PER_EXPERT`

这样更利于后续部署、分析与去重。


## 11. Continue / Stop 决策机制

### 11.1 程序化硬判断

若出现以下问题，直接停止当前 trial：

- 训练未成功完成到 checkpoint
- 无法解析指标
- 无法定位可恢复 checkpoint

### 11.2 程序化强信号继续

若出现以下强信号，可直接继续：

- 当前 `best_objective_so_far` 已明显优于 incumbent
- 最近一个 15M 窗口里 `best_objective_so_far` 仍有明显下降
- 虽然当前 objective 尚未获胜，但 `total_cost_ratio` 显著更低，存在明显部署潜力

### 11.3 程序化强信号停止

若满足以下组合条件，则直接停止：

- 最近一个 15M 窗口几乎无改善
- 当前 `best_objective_so_far` 仍明显落后 incumbent
- 历史相近结构没有翻盘迹象

### 11.4 边界情况交给 Agent

边界 trial 由 agent 结合历史摘要做判断，输出：

- `continue`
- `stop`
- `spawn_next`

必要时同时附带下一组新 trial 参数建议。


## 12. Agent 输入摘要

每次 agent 决策只看到压缩后的结构化摘要，不直接阅读散乱原始日志。

建议输入字段：

- 当前 trial 参数
- 当前 trial 的 `total_cost_ratio`
- 当前 trial 的 `best_exceed_alpha_0.50_so_far`
- 当前 trial 的 `best_objective_so_far`
- 当前 trial 的 `latest_exceed_alpha_0.50`
- 当前 trial 的 `latest_fvu`
- 当前 trial 最近一个窗口的改善量
- 当前 token 数与 checkpoint 次数
- incumbent 摘要
- 最近若干 trial 的结果摘要
- 已知失败区域摘要
- 已知潜力区域摘要


## 13. 最大价值与风险

### 13.1 价值

该设计解决了用户关心的核心问题：

- 不用提前静态枚举全部配置
- 能根据已跑结果动态生成后续配置
- 差配置不会无意义地一直训练
- 潜力 trial 可以不断恢复继续
- 最终目标直接对齐真实部署目标

### 13.2 风险

首版的主要风险点：

- checkpoint 恢复链路可能不稳定
- `best_exceed` 的提取必须可靠
- 深度优先策略可能让系统在局部区域停留太久
- agent 的提议若约束不够强，可能产生无效参数

对应控制策略：

- 所有状态都结构化持久化
- 程序化硬约束先过滤
- 仅开放有限参数集合，但不预先写死固定候选值表
- 对 agent 输入做摘要压缩而非开放原始日志


## 14. 首版实施顺序

建议实施顺序：

1. 搭建新框架代码骨架
2. 先实现 state / runner / metrics / scheduler
3. 先接“规则 + agent 接口壳”，保证状态机打通
4. 再接真实 agent 决策
5. 最后补 resume、可视化与夜间长跑增强


## 15. 验收标准

当以下条件都满足时，认为首版设计落地成功：

- 新框架完全不依赖旧 `research/AutoResearch/`
- 能独立维护 run 状态目录
- 能创建新 trial
- 能在 15M checkpoint 停下
- 能从 checkpoint 恢复到下一个 15M
- 能正确计算 `best_objective_so_far`
- 能在程序硬规则与 agent 决策之间完成 continue / stop / spawn next
- 能稳定连续运行，支持长周期探索
