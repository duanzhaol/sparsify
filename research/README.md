# SAE 自动科研系统说明

这份文档介绍 `sparsify-ascend` 里当前这套 SAE 自动科研系统的目标、设计和使用方式。

它面向的不是“单次训练”，而是下面这种工作流：

- 你已经有一个稳定可运行的 SAE 训练入口
- 你希望大模型根据历史实验结果，自己决定下一轮要跑什么
- 你希望它不仅能调参数，还能在必要时修改 `sparsify/` 下的代码
- 你希望它能连续跑很多轮，形成有记忆的实验探索，而不是每轮都从零开始

这套系统的目标，就是把这些事情自动化，并且把风险压在可控范围内。

## 1. 为什么要做这件事

当前这项研究的核心目标不是“跑一个 SAE”，而是持续回答下面几个问题：

- 在更小的 `K` 下，能不能保持更好的重建质量
- 不同 SAE 架构之间，哪个方向更值得继续探索
- 某个架构表现差，到底是架构本身差，还是实现性能太差
- 如果一个方向慢得离谱，应该丢掉，还是先优化代码实现

这些问题如果完全手工做，会遇到几个现实困难：

- 实验数量会越来越多，人会记不住
- 每一轮的结论都应该依赖历史实验，但历史太长，直接塞进上下文会爆掉
- 很多轮实验只是为了排除明显错误方向，如果都手工盯着，很浪费时间
- 真正有价值的系统，不只是“自动跑脚本”，而是能根据结果决定下一步

因此这套系统的核心目标是：

- 把训练入口固定下来
- 把结果记录标准化
- 把长期和短期记忆结构化
- 让大模型在一个受限但真实的执行层上持续做实验决策

## 2. 这套系统是如何逐步升级出来的

这套系统不是一次做成的，而是从一个很简单的自动实验器逐步升级出来的。

### 第一阶段：固定实验执行

最早只有一个固定训练脚本：

- [`scripts/autoresearch_test.sh`](/root/sparsify-ascend/scripts/autoresearch_test.sh)

这一步解决的是“训练怎么稳定跑起来”。

### 第二阶段：结果记录与状态管理

后来增加了：

- [`research/controller.py`](/root/sparsify-ascend/research/controller.py)

它负责：

- 初始化历史目录
- 解析训练结果
- 维护 `state.json`
- 维护 `results.tsv`
- 维护 `frontier.json`
- 给实验打出 `promote / keep / archive / discard / crash`

这一步解决的是“实验结果如何被机器稳定读取和比较”。

### 第三阶段：真正的 Agent Loop

再后来加入：

- [`research/agent_loop.py`](/root/sparsify-ascend/research/agent_loop.py)

它负责：

- 读取结构化记忆
- 调用 `codex exec`
- 获取一轮实验动作
- 执行训练
- 调用 controller 记录结果
- 更新记忆
- 再进入下一轮

这一步解决的是“如何让大模型真正基于历史做下一步决策”。

### 第四阶段：加入性能感知和失败处理

当系统开始长时间跑实验后，一个关键问题出现了：

- 慢实验不一定说明架构差，可能只是实现太慢

所以系统又加了：

- watchdog
- throughput baseline
- `perf_regression` 分类
- `current_status.json`

这一步解决的是：

- 不再只看“训练死没死”
- 还要看“训练是不是慢得不值得继续”
- 还要区分“架构问题”和“实现问题”

## 3. 当前系统的整体设计

当前系统可以理解为三层结构。

### 第一层：固定执行层

固定执行层只做“把训练跑起来”。

核心入口：

- [`scripts/autoresearch_test.sh`](/root/sparsify-ascend/scripts/autoresearch_test.sh)

这个脚本负责：

- 组装 `torchrun -m sparsify`
- 读取环境变量作为训练参数
- 执行单轮训练

这层不负责决策，不负责记忆，不负责研究策略。

### 第二层：控制层

控制层负责把每轮实验“变成结构化历史”。

核心入口：

- [`research/controller.py`](/root/sparsify-ascend/research/controller.py)

它负责：

- 初始化历史文件
- 解析 checkpoint 和日志
- 维护 frontier
- 记录实验事实
- 产出决策标签

这层的职责是让系统“知道自己跑过什么、结果怎么样”。

### 第三层：Agent 决策层

决策层负责“下一步跑什么”。

核心入口：

- [`research/agent_loop.py`](/root/sparsify-ascend/research/agent_loop.py)

它负责：

- 读取历史
- 组织 prompt
- 让大模型输出结构化 action
- 允许它在 `sparsify/` 下做代码修改
- 跑 proxy / full
- 更新长期记忆

这层的职责是让系统从“自动执行”变成“自动研究”。

## 4. 记忆是怎么设计的

这套系统最关键的不是训练脚本，而是记忆。

如果没有记忆，它只能不断重复类似实验；如果把所有历史直接塞进上下文，又会很快爆掉。

所以现在的记忆分成两类。

### 长期记忆

长期记忆主要放在：

- [`research/history/state.json`](/root/sparsify-ascend/research/history/state.json)
- [`research/history/frontier.json`](/root/sparsify-ascend/research/history/frontier.json)
- [`research/history/memory.json`](/root/sparsify-ascend/research/history/memory.json)
- [`research/history/results.tsv`](/root/sparsify-ascend/research/history/results.tsv)

它们分别承担不同角色：

- `state.json`
  记录总状态、计数器、当前 frontier、最近结果、agent 进度
- `frontier.json`
  只保留当前最重要的前沿结果，便于快速读取
- `memory.json`
  存压缩后的研究结论、性能问题、失败模式和下一步假设
- `results.tsv`
  存所有实验的事实记录，是最完整的历史来源

### 短期记忆

短期记忆主要来自：

- 最近几轮实验结果
- 最近几轮 round summaries
- 当前代码状态
- 最新运行状态

对应文件包括：

- [`research/history/round_summaries/`](/root/sparsify-ascend/research/history/round_summaries)
- [`research/history/current_status.json`](/root/sparsify-ascend/research/history/current_status.json)
- [`research/history/logs/`](/root/sparsify-ascend/research/history/logs)

这种设计的好处是：

- 长期实验信息不会丢
- 每轮 prompt 不需要把整个历史全塞进去
- 大模型可以用“压缩后的结论 + 最近几轮细节”做下一步决策

## 5. proxy 和 full 是如何设计的

当前系统默认采用两级实验预算。

### proxy

用途：

- 快速筛选候选
- 建立粗粒度 frontier
- 尽快排除明显不值得继续的方向

特点：

- token 少
- 成本低
- 可以高频跑

### full

用途：

- 验证 proxy 的好结果是否站得住
- 把真正值得保留的点写进更可信的 frontier

特点：

- token 多
- 成本高
- 只在值得时才跑

### 当前的升级规则

默认规则是：

- Agent 默认先提 `proxy`
- 运行时默认不允许直接跳 `full`
- 只有当 controller 判定这轮 `proxy` 值得继续时，系统才自动跑 `full`

这样做的目的是：

- 不让每个候选都直接烧掉大量预算
- 先用便宜实验筛选，再用贵实验验证

## 6. 为什么要加入性能 watchdog

如果没有 watchdog，会出现一个很糟糕的问题：

- 某个架构可能不是质量差
- 而是当前实现极慢
- 如果系统只会等固定 30 分钟或 2 小时，夜里就会浪费大量时间

因此现在系统会做几类检查：

- 首个 step 是否迟迟不出现
- `metrics.jsonl` 是否长时间不更新
- 吞吐是否远低于 baseline
- 总运行时间是否超过上限

如果某个实验特别慢，但没有完全挂死，它不会被简单地当成“架构差”，而会被归类成：

- `perf_regression`

这意味着：

- 它更像实现问题
- 下一轮更应该考虑优化代码
- 而不是直接得出“这个架构不值得做”

## 7. 一步一步的操作说明

下面是最推荐的使用流程。

### 第一步：进入仓库

```bash
cd /root/sparsify-ascend
```

### 第二步：初始化历史目录

如果是第一次运行，或者你刚清空了历史：

```bash
python research/controller.py init
```

### 第三步：设置代理

当前 `codex exec` 需要经过你的本地代理：

```bash
export http_proxy=http://127.0.0.1:23234
export https_proxy=http://127.0.0.1:23234
```

### 第四步：先跑一个小预算验证

推荐先用这一条确认系统能正常工作：

```bash
python research/agent_loop.py \
  --rounds 1 \
  --budget-hours 0.2 \
  --proxy-max-tokens 200000 \
  --full-max-tokens 500000 \
  --first-step-timeout-sec 180 \
  --slow-run-grace-sec 120 \
  --min-tokens-per-sec-ratio 0.25 \
  --min-progress-steps 4 \
  --reset-failure-counters
```

这一步的目的是：

- 检查 agent 能否产出 action
- 检查训练能否正常跑完
- 检查 controller 能否正常记录结果
- 检查 watchdog 是否不会误杀正常训练

### 第五步：查看运行状态

运行时最常用的监控方式有三个。

查看当前状态：

```bash
cat research/history/current_status.json
```

查看最近日志：

```bash
ls -lt research/history/logs | head
```

查看最近 metrics：

```bash
find checkpoints/research_agent -name metrics.jsonl | tail -n 1 | xargs -r tail -f
```

### 第六步：开始长时间实验

如果小预算验证没有问题，就可以开始长时间实验。

一个比较保守的夜跑命令是：

```bash
python research/agent_loop.py \
  --rounds 12 \
  --budget-hours 4 \
  --proxy-max-tokens 5000000 \
  --full-max-tokens 50000000
```

如果你已经确认系统很稳定，再放大到更长预算：

```bash
python research/agent_loop.py \
  --rounds 20 \
  --budget-hours 8 \
  --proxy-max-tokens 20000000 \
  --full-max-tokens 200000000
```

## 8. 具体用途和使用方法

这套系统主要有三类用途。

### 用途一：自动扫 low-K frontier

这是目前最直接的用途。

目标是：

- 先固定一个架构，比如 `topk`
- 逐步把 `K` 往下压
- 看在不同预算下，`FVU` 如何变化

这种用法下，系统最常做的是：

- 先试 `K=64`
- 再试 `K=32`
- 必要时回到更高 `K`
- 形成一条逐步下降的 frontier

### 用途二：自动探索不同 SAE 架构

当已有 low-K frontier 之后，Agent 就可以开始比较不同架构：

- `topk`
- `gated`
- `jumprelu`
- `group_topk`

这时系统会利用历史记忆决定：

- 哪些架构值得试
- 哪些架构在当前实现下只是太慢
- 哪些架构是真正质量不行

### 用途三：自动做局部性能优化

这也是这套系统和普通 sweep 最大的区别。

如果某个方向质量可能不错，但运行速度严重异常，系统可以把它识别成：

- `perf_regression`

这时候大模型下一轮更可能做的是：

- 修改 `sparsify/` 里的局部实现
- 减少不必要的开销
- 再重新验证这个架构

也就是说，这套系统不仅是在“调参数”，也是在做“受限代码研究”。

## 9. 平时最常需要改的参数

虽然系统支持很多参数，但你平时最常需要手工改的，其实很少。

### 夜跑层面最常改的

- `--rounds`
- `--budget-hours`
- `--proxy-max-tokens`
- `--full-max-tokens`

### 训练层面最常改的

- `COMPILE_MODEL`
- `HOOKPOINTS`

其他大多数训练参数，通常让 Agent 去选就可以了。

## 10. 结果应该怎么看

如果一轮实验完成后，你最应该先看的文件是：

- [`research/history/state.json`](/root/sparsify-ascend/research/history/state.json)
- [`research/history/frontier.json`](/root/sparsify-ascend/research/history/frontier.json)
- [`research/history/memory.json`](/root/sparsify-ascend/research/history/memory.json)

它们分别回答：

- 当前最优结果是什么
- 哪些 `K` 已经被验证过
- 最近几轮学到了什么
- 有没有出现性能异常或失败模式

如果你要看单轮细节，再去看：

- `research/history/logs/`
- `research/history/results.tsv`
- `checkpoints/research_agent/.../metrics.jsonl`

## 11. 当前系统的边界

这套系统已经能做真实自动科研，但它不是无限制的。

当前边界包括：

- 只允许修改 `sparsify/`
- 不允许修改执行层和历史文件
- 默认必须先走 proxy
- 训练环境目前假设为 2x CUDA
- 当前的 baseline runtime 是按已有正常实验逐步建立的

这意味着它更像一个：

- 受限的、可控的、可追踪的自动科研系统

而不是一个完全自由的通用 agent。

## 12. 推荐的使用习惯

比较稳的使用方式是：

1. 先跑小预算验证
2. 再跑 4 小时左右的稳定性夜测
3. 最后再上 8 小时级别的长实验

每次长跑之后，优先看：

- frontier 有没有变好
- memory 里有没有积累出有价值的新结论
- performance findings 有没有提示你该优化实现

如果这三项都在积累，这套系统就是在正常工作。
