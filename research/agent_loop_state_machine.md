# AutoTest / AutoResearch 状态机分析

本文档基于当前代码实现，对 `research/agent_loop.py` 及其相关执行层进行状态机级别的梳理。

目标：

- 把主流程拆成明确状态和状态转换
- 标出每个状态依赖的上下文
- 说明在不同上下文下，系统可能做出的决策
- 识别现有流程中的异常分支、模糊分支和潜在问题

---

## 1. 总体分层

当前 AutoTest / AutoResearch 流程大致分成 5 层：

1. Agent 决策层
2. Runtime 约束层
3. Sanity / Training 执行层
4. Result / Memory / State 持久化层
5. Session / Crash Recovery 控制层

它不是一个纯 agent-driven 流程，而是：

- Agent 给出动作
- Runtime 根据策略和当前上下文做二次约束
- 执行层决定实验是否真的能运行
- 持久化层把结果折叠进 memory/state/history

因此，最终执行的动作不一定等于 agent 原始想做的动作。

---

## 2. 总状态机

```text
[Loop Start]
  -> [Preflight]
    -> backend unavailable -> [Exit: failure]
    -> ok -> [Round Begin]

[Round Begin]
  -> load state/memory/results/session brief
  -> detect stagnation/crash recovery
  -> resolve session mode
  -> invoke agent

[Agent Invoke]
  -> agent failure after retries -> [Round End: crash(agent_invocation)]
  -> action=stop -> coerce to run
  -> action=full and direct full not allowed -> coerce to proxy
  -> action received -> [Round Execute]

[Round Execute]
  -> if crash recovery and code edit -> coerce to param_only
  -> enter round-local execution loop

[Round-local Execution Loop]
  -> detect touched files / build patch
  -> validate env_overrides
  -> variable isolation warning
  -> incubation policy check
  -> optional behavioral diff test
  -> proxy budget selection
  -> remaining time check
  -> optional sanity
  -> training

[sanity]
  -> pass -> [training]
  -> fail and repairable and attempts left -> [repair action]
  -> fail otherwise -> [Round End: crash(sanity_failed)]

[training]
  -> result=crash and repairable and attempts left -> [repair action]
  -> result=promote on proxy -> [full training]
  -> result=ok/discard/incubate/keep -> [Round End: normal]

[repair action]
  -> send failure context back into same session
  -> constrain experiment target/family/env_overrides
  -> get new code-fix action
  -> loop back to [Round-local Execution Loop]

[full training]
  -> keep/crash/etc -> [Round End: normal or crash]

[Round End]
  -> write round summary
  -> append memory
  -> update agent state
  -> maybe commit
  -> next round or exit on budget / limits
```

---

## 3. 主流程展开

### 3.1 Loop Start / Preflight

上下文：

- CLI 参数
- 是否启用 `auto_commit`
- 当前 git worktree 是否干净
- `codex` CLI 是否可用
- 代理是否可达

可能决策：

- `--reset-failure-counters` 时先清空 crash / no-improve 计数
- backend 不可用时直接退出
- auto-commit 模式下要求起始 worktree 干净

说明：

- 这是一个比较硬的启动门
- 没有进入 round 前，不会进行任何 agent / training 动作

---

### 3.2 Round Begin

每一轮开始时会加载：

- `state.json`
- `memory.json`
- `results.tsv`
- `session_brief.json`

并计算：

- stagnation 状态
- crash recovery 模式
- session 模式

可能分支：

- 正常模式
- mild stagnation
- severe stagnation
- crash streak recovery

其中：

- stagnation 有一部分只影响 prompt guidance
- crash recovery 会直接影响 runtime 行为，例如强制 `param_only`

---

### 3.3 Session 分支

```text
resume-session
  -> session healthy -> resume
  -> session stale/broken/too old/too many rounds -> rebuild
  -> repeated session failures -> degrade to fresh-each-round

fresh-each-round
  -> every round creates a new session
```

上下文：

- `active_session_id`
- `active_session_status`
- `active_session_rounds`
- `active_session_started_at`
- `session_failure_count`

状态解释：

- `active`
- `stale`
- `broken`
- `closed`

潜在风险：

- repair loop 当前也复用同一个 session，这是合理的
- 但 repair 过程中如果 session 本身坏掉，repair 目前没有主 agent invoke 那么完整的 retry/rebuild 包装

---

## 4. Agent 决策层

### 4.1 Agent 原始输出

agent 返回 JSON action，常见类型：

- `param_only`
- `edit_sae_code`
- `edit_perf_code`
- `no_change`

以及 tier：

- `proxy`
- `full`

### 4.2 Runtime 对 Agent 动作的二次约束

可能的 runtime 重写：

- `command=stop` 会被强制改成 `run`
- direct `full` 可能被强制改成 `proxy`
- crash recovery 中的 code edit 会被强制改成 `param_only`

因此需要区分：

- agent 原始意图
- runtime 最终执行动作

最终 round summary 记录的是“最终执行动作”。

---

## 5. Round-local Execution Loop

这是当前最关键的执行循环。

大致逻辑：

```text
while True:
  1. 检查本轮当前代码 diff
  2. 生成 patch
  3. 校验 env_overrides
  4. 变量隔离检查
  5. incubation limit 检查
  6. behavioral diff test
  7. proxy budget 选择
  8. budget 剩余时间检查
  9. sanity
  10. training
  11. crash 可修则进入 repair
  12. 否则退出本轮局部循环
```

这与旧逻辑相比的主要升级是：

- code-fix 失败不再天然跨轮
- 可以在单轮内部做最多 N 次 repair attempt

当前默认：

- `max_repair_attempts = 5`

---

## 6. Param-only 路径

### 6.1 正常路径

```text
param_only
  -> no touched files
  -> no sanity
  -> training
    -> discard/incubate/promote/keep/crash
  -> if promote and proxy -> full
  -> finalize
```

这是当前最稳定、最常见的路径。

尤其是 gated 主线基本都走这一条。

### 6.2 上下文决定的决策

如果上下文表明：

- crash 刚发生过
- 有连续 crash
- 当前 family 已验证稳定

则系统更倾向：

- 继续 `param_only`
- 选择已知 working family
- 避免 code edit

---

## 7. Code-edit 路径

### 7.1 单次修通的理想路径

```text
edit_sae_code
  -> touched files
  -> behavioral diff pass
  -> sanity fail
  -> repair 1
  -> touched files updated
  -> sanity pass
  -> training pass
  -> finalize
```

适用场景：

- `Unknown architecture`
- `AttributeError`
- `NameError`
- 局部 shape bug

### 7.2 多次 repair 的路径

```text
edit_sae_code
  -> sanity fail
  -> repair 1
  -> sanity fail
  -> repair 2
  -> sanity pass
  -> training crash
  -> repair 3
  -> training pass
  -> finalize
```

这类路径现在是被允许的。

---

## 8. Repair Loop

当前 repair loop 的核心设计是：

- repair 发生在同一轮内部
- repair prompt 会把失败 traceback / error summary 喂回同一个 session
- repair action 会被 runtime 强约束

约束内容：

- 不允许改 `family_name`
- 不允许改 `experiment_tier`
- 不允许改 `env_overrides`
- `primary_variable` 被强制为 `code_fix`
- 只允许修 blocker，不允许偷偷换实验目标

### 8.1 Repair 状态机

```text
repairable failure
  -> build repair prompt
  -> resume same session
  -> get repair action
  -> coerce repair action back to original experiment target
  -> re-enter round-local execution loop
```

### 8.2 Repair 触发条件

满足以下任一：

- `SanityCheckError`
- `SyntaxError`
- `NameError`
- `AttributeError`
- `TypeError`
- `ValueError`
- `RuntimeError`
- `AssertionError`
- `KeyError`
- `IndexError`
- `NotImplementedError`
- `termination_reason == sanity_failed`
- 或 training crash 且没有得到有效实验信号，同时存在结构化错误摘要

### 8.3 当前 repair loop 的真实风险

#### 风险 1：repair 空转

存在这种可能：

```text
training crash
  -> repair action 返回了合法 JSON
  -> 但没有真正改任何文件
  -> touched_files = []
  -> 继续下一次 training
  -> 再 crash
  -> 再 repair
```

当前没有硬阻断这种空转，只是最多尝试 5 次。

#### 风险 2：repair 覆盖 round 最终语义

最终 `round_summary` 中记录的 action 是“最后一次 repair 后的 action”，不一定仍然是最初科研问题的自然描述。

因此：

- timeline 是完整的
- round summary 是压缩后的最终态

这对执行正确性无害，但对复盘语义有影响。

---

## 9. Sanity 分支

### 9.1 Sanity 做什么

当前 sanity 的最小语义是：

- 构造 `SparseCoderConfig`
- 实例化 `SparseCoder`
- 跑一次前向
- 对 `out.fvu.backward()`

也就是：

- 架构名是否可构造
- forward 是否可走通
- backward 是否可走通

### 9.2 Sanity 状态转换

```text
needs_sanity and code_edit
  -> pass -> training
  -> fail -> record structured sanity failure
          -> maybe repair
          -> else abort_round(sanity_failed)
```

### 9.3 Sanity 的边界

Sanity 不是 formal training config 的完整代理。

因此合法存在这种状态：

```text
sanity pass
  -> training still crash
```

原因可能包括：

- formal config path
- launcher path
- training-only dependency
- DDP / torchrun path

所以不能把 `sanity pass` 等价理解成“正式训练一定没问题”。

---

## 10. Training 分支

### 10.1 Training 运行时分支

训练执行层中，watchdog 和结果记录层会共同决定状态。

可能出现：

- `completed`
- `first_step_timeout`
- `throughput_too_low`
- `stall_timeout`
- `hard_timeout`

训练结果维度：

- `decision`
- `status`
- `run_health`
- `termination_reason`

### 10.2 一个重要事实

`termination_reason` 和 `decision` 不是一回事。

例如：

```text
termination_reason = completed
decision = crash
```

这表示：

- 从进程角度，它已经退出
- 但从实验语义角度，它没有产生有效结果

这类情况常见于：

- 程序很快报错退出
- watchdog 没有杀它
- controller 通过 log / metrics 把它判成 `crash`

### 10.3 Training crash 的结构化回写

现在 training crash 已经会抽取：

- `error_type`
- `error_summary`
- `traceback_excerpt`
- `log_excerpt`

并写入：

- `recent_training_failures`
- prompt context
- session brief

---

## 11. Full Promotion 分支

### 11.1 路径

```text
proxy decision = promote
  -> budget enough -> full training
  -> budget not enough -> skip full
```

### 11.2 当前设计特点

- full 不会再经过 repair loop
- full 也不会再跑 sanity
- 它直接沿用 proxy 通过后的 action

这意味着：

- proxy pass 只是 full 的前提，不是 full 稳定性的保证
- 如果 full crash，当前会直接作为 full crash 结束该轮

---

## 12. Finalize / Abort 分支

### 12.1 成功或正常结束

```text
_finalize_round
  -> write round summary
  -> append memory
  -> update agent state
  -> save session brief
  -> maybe auto-commit
```

### 12.2 中途失败

```text
_abort_round
  -> build minimal result
  -> still write round summary
  -> append memory
  -> update agent state
```

### 12.3 信息保留特性

当前：

- timeline 最完整
- round summary 是折叠后的最终摘要
- memory 是进一步压缩后的长期工作记忆

这意味着某些 repair 子过程只在 timeline 中可见，不会完全展开到 round summary。

---

## 13. 树状图：按大类决策

```text
Round
├─ Agent invocation failed
│  └─ round crash
├─ Agent returns stop
│  └─ coerce to run
├─ Action blocked by crash recovery
│  └─ code_edit -> param_only
├─ Param-only path
│  ├─ proxy discard/incubate
│  ├─ proxy promote -> full keep/crash
│  └─ proxy/full crash
└─ Code-edit path
   ├─ invalid env_overrides
   │  └─ abort
   ├─ incubation limit exceeded
   │  └─ abort
   ├─ behavioral diff identical
   │  └─ abort
   ├─ sanity pass
   │  └─ training
   ├─ sanity fail
   │  ├─ repairable and attempts left
   │  │  └─ repair loop
   │  └─ not repairable / attempts exhausted
   │     └─ abort
   └─ training crash
      ├─ repairable and attempts left
      │  └─ repair loop
      └─ not repairable / attempts exhausted
         └─ finalize as crash
```

---

## 14. 关键上下文如何影响决策

### 14.1 如果上下文是“连续 crash”

系统倾向于：

- 强制 `param_only`
- 回到 strongest known working family
- 暂时阻止 code edit

### 14.2 如果上下文是“连续 no-improve”

系统倾向于：

- mild：进入 exploitation mode
- severe：要求探索 K 或新架构

### 14.3 如果上下文是“family 还没拿到有效 signal”

系统倾向于：

- 先修 blocker
- 而不是直接下质量结论

### 14.4 如果上下文是“mainline 已稳定”

系统倾向于：

- 在 working family 上继续做相邻 K / small param sweep
- 直到探索收益下降，再回去开新 family

---

## 15. 当前流程中的奇怪问题

下面这些不是纯理论问题，而是当前实现里真实可能出现的异常状态。

### 15.1 Repair 空转

repair action 可能没有改任何代码，但系统仍会继续下一次 repair attempt。

结果：

- 同一根因可能在单轮内重复 5 次
- 浪费预算
- 产生“看起来在修，实际上没动”的假进展

### 15.2 Repair 会覆盖最终 round action 语义

最终 round summary 保存的是 repair 后的 action，不一定仍是最初的研究假设文本。

结果：

- 执行没问题
- 但复盘时会混淆“本轮是在做科研比较”还是“本轮是在修 blocker”

### 15.3 Same root cause 还没有被硬策略利用

虽然 training / sanity 的 root cause 已经被结构化写回，但：

- 没有 `same_root_cause_repeat_count`
- 没有“同根因重复 N 次后禁止继续同类 repair”的硬策略

### 15.4 Full 没有 repair loop

当前：

- proxy 可以 repair
- full crash 不能在同轮 repair

这是一个明确的设计选择，不一定错误，但要知道它存在。

### 15.5 Repair attempt 不影响 crash recovery 计数

当前 crash recovery 是按“轮”算，不按 repair attempt 算。

结果：

- 单轮内部连炸 5 次，也不会立刻触发全局 crash recovery
- 只有最终 round result=crash 才会计入 `consecutive_crashes`

### 15.6 `edit_perf_code` 可能绕过 behavioral diff

`behavioral_diff_test` 只对 `edit_sae_code` 启用。

如果某个本质上改变编码行为的修改被标成 `edit_perf_code`，可能绕过该检测。

---

## 16. 我对当前系统的总体评价

当前系统已经不是“流程混乱”状态，而是：

- 主状态机大体合理
- 主干分支可用
- 但异常分支仍然偏宽松

尤其相比旧版本，当前已明显改善：

- training crash root cause 已进 prompt
- operator guide 已进 prompt
- code-fix 已支持单轮 repair loop

所以现在最需要做的不是重写主流程，而是继续收紧异常路径。

---

## 17. 建议的下一步收紧方向

### 17.1 阻断 repair 空转

建议增加硬规则：

- 如果 repair attempt 后 `touched_files == []`
- 且 root cause 未变化
- 则直接终止 repair loop

### 17.2 引入 same-root-cause 计数

建议新增结构化字段：

- `same_root_cause_repeat_count`
- `last_fix_attempt_effective`

并在 runtime 中作为硬策略使用。

### 17.3 Round summary 增加 repair 摘要

建议在 `round_summary` 中记录：

- repair attempt 次数
- 每次 repair 的 root cause
- 每次是否实际改了代码

这样复盘会更清晰。

---

## 18. 一句话总结

这套 AutoTest / AutoResearch 现在的主状态机是合理的；真正的问题不在主干，而在少数异常分支还不够硬。  
当前最值得继续加强的是：

- repair 空转阻断
- 同根因重复失败的硬约束
- repair 子过程的可观测性

