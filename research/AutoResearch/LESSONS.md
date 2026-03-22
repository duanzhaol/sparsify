# Auto Research 旧系统经验总结

本文档记录了旧 autoresearch 系统（`research/agent_loop.py` + `research/controller.py` 等）
在 ~100 轮实验中积累的经验教训，供新系统设计参考。

---

## 1. Frontier 管理

### 旧设计：Proxy / Full 双 Frontier

- `proxy_frontier`: 短训练（20M tokens）的最佳结果，高方差
- `full_frontier`: 长训练（200M tokens）的最佳结果，低方差
- `frontier` = `full_frontier` 的副本
- `pareto_proxy_frontier` / `pareto_full_frontier`: 各自的 Pareto 前沿

**实际数据差异**（从 state.json 观察）：
- K=128: proxy FVU=0.0663 vs full FVU=0.0663 — 几乎相同
- K=40:  proxy FVU=0.169  vs full FVU=0.091  — 差距巨大（proxy 高估了 86%）
- K=64:  proxy FVU=0.154  vs full FVU=0.081  — 差距巨大

**教训**：
- Proxy 在低 K 值下的 FVU 严重偏高，不能作为可靠信号
- 旧系统的 "promote" 逻辑（proxy 好 → 跑 full 验证）实际上有价值：它避免了
  用 noisy proxy 结果做最终判断
- 但双 frontier 增加了大量代码复杂度（~300 行分支逻辑）

**新系统决策**：
- 统一为单 frontier，所有实验用同一个 token 预算
- 如果未来需要快速筛选，可以在 runner 层加 "early stopping if clearly worse"
  而不是维护两套 frontier

### 决策标签

| 标签 | 旧含义 | 新含义 |
|------|--------|--------|
| `keep` | full 训练改善了 full_frontier | 训练改善了 frontier |
| `promote` | proxy 训练改善了 proxy_frontier | （已移除，统一为 keep） |
| `discard` | 未改善对应 frontier | 未改善 frontier |
| `archive` | 与当前最佳 FVU 差距 < 0.001 | 同上 |
| `crash` | 训练失败 | 同上 |
| `incubate` | 新 family 的原型实验 | （已移除，由 family status 管理） |

---

## 2. Session 管理

### 旧设计

- Codex session 跨轮复用（最多 8 轮或 4 小时）
- 新 session: 完整 prompt（~40K chars）
- Resume session: 精简 update prompt（~5K chars）
- Session 失败 3 次后降级为 fresh-each-round

**踩过的坑**：
- Session resume 偶尔失败（codex 服务端状态丢失），需要重试 + 降级逻辑
- 长 session 后 agent 会 "忘记" 早期指令，出现行为漂移
- Session ID 提取依赖 codex stdout 的 `session id: xxx` 格式，格式变化会导致静默失败

**新系统保留了 session 复用**，但简化了降级逻辑：resume 失败直接切 fresh，
不计数。

---

## 3. Repair Loop

### 设计

- 代码编辑后 sanity check 或训练崩溃 → 在同一 round 内让 agent 修复
- 最多 5 次修复尝试
- 只修复 "可修复" 的错误类型（SyntaxError, TypeError, AttributeError 等）

**踩过的坑**：
- Agent 有时修了一个错但引入另一个错 → 需要 failure signature 去重
- Agent 有时 "修复" 时不改任何文件 → 需要检测 no-code-changes 并终止
- 同一个 root cause 重复出现 2 次以上 → 说明 agent 无法修复，必须终止

**有效的约束**：
- Repair action 锁定 `family_name`, `env_overrides`, `experiment_tier`
  （不允许 agent 借修复之名改实验目标）
- `primary_variable` 强制为 `"code_fix"`
- `needs_sanity` 强制为 True

---

## 4. Policy 层

### Variable Isolation（单变量原则）

每轮只改一个主维度：architecture / optimizer / K / code_edit。
LR 和 optimizer 可以耦合改（换优化器通常需要调 LR）。

**教训**：多变量同时改导致无法归因结果，浪费轮次。

### Incubation 管理

- 新 architecture family 最多 3 轮 proxy 观察期（incubating）
- 同时最多 10 个 incubating family
- 超过配额 → 自动 archive

**教训**：
- 没有 incubation 限制时，agent 会无限开新 family 而不验证已有的
- 3 轮上限对简单 bug 修复太短，但对真正不 work 的架构够用
- Archive 后 agent 有时仍然尝试复活同一个 family → 需要在 prompt 里
  明确 "do not repeat archived families"

### Behavioral Diff Test

新架构的 `encode()` 和 baseline `topk` 比较：如果输出完全相同 → 阻止
（除非是 prototype 阶段）。

**教训**：Agent 经常提交的 "新架构" 实际上没有改变任何计算路径，
只是换了个名字。这个检查非常有效。

### Stagnation Detection

- 3 轮无改善 → 建议 exploitation（系统性参数搜索）
- 5 轮无改善 → 强制 K 探索或新架构
- 2 轮连续 crash → 强制 param_only（禁止代码编辑）

**教训**：
- Crash recovery mode 非常重要，否则 agent 会在坏代码上反复尝试代码编辑
- Stagnation 建议对 agent 有效，但不是强制的 → agent 有时忽略

---

## 5. Prompt 工程

### 信息密度

旧 prompt 中：
- `operator_guide_excerpt` 占 48% — 18.5K chars，完整注入
- `memory_digest` 占 31% — 嵌套 JSON，每个 family ~1500 chars
- `recent_insights` 每条 ~150 chars，格式冗余

**教训**：
- 嵌套 JSON 浪费大量 token 在结构字符（`{`, `"key":`, 缩进）上
- 结构化单行文本（`r45 topk k128 keep fvu=.0489`）信息密度高 3x，
  LLM 理解无障碍
- 但 operator_guide 不能压缩，它是研究方向的灵感来源

### Agent 行为偏差

观察到的典型偏差：
1. **K=128 锚定**：agent 倾向只优化 K=128，需要反复提示探索低 K
2. **EF 膨胀**：agent 喜欢增大 expansion_factor 来 "改善" FVU，
   但这不是公平比较 → 需要强调同 EF 比较
3. **新 family 偏好**：agent 喜欢开新 family 而不是验证已有的
4. **Stop 倾向**：agent 有时想停止搜索 → 必须禁止 `command="stop"`
5. **Cross-tier 混淆**：proxy 和 full 的结果混在一起比较 → 需要
   分 lane 显示

---

## 6. 训练执行

### Watchdog 机制

| 检测 | 超时 | 说明 |
|------|------|------|
| 首步超时 | 180s | 训练脚本启动但没产生第一条 metrics |
| 停滞超时 | 15min | metrics.jsonl 停止更新 |
| 硬超时 | 30min (proxy) / 2h (full) | 绝对墙钟限制 |
| 吞吐异常 | 25% of baseline, 连续 2 次 | 性能退化早停 |

**教训**：
- 首步超时对 import 错误非常有效（代码编辑后最常见的失败模式）
- 吞吐 watchdog 的 baseline 需要按 `architecture + hookpoints` 分别记录
- 硬超时是最后保障，但有时训练确实需要更长时间（尤其是第一次编译）

### 进程管理

- 用 `start_new_session=True` 创建新进程组
- 先 SIGTERM，15 秒后 SIGKILL
- 必须杀进程组（`os.killpg`）而不是单个进程，否则子进程会变孤儿

---

## 7. 状态持久化

### 文件布局

| 文件 | 格式 | 更新频率 | 说明 |
|------|------|---------|------|
| `state.json` | JSON | 每轮 | Agent 计数器 + frontier |
| `frontier.json` | JSON | 每轮 | state.frontier 的副本 |
| `memory.json` | JSON | 每轮 | Agent 记忆（families, insights, failures） |
| `results.tsv` | TSV | 每轮 | Append-only 实验记录 |
| `timeline.jsonl` | JSONL | 每事件 | Append-only 事件流 |
| `round_summaries/` | JSON | 每轮 | 完整 round 快照 |
| `operator_hints.json` | JSON | 手动 | 人工注入的指导 |
| `session_brief.json` | JSON | 每轮 | Session resume 用的精简快照 |

**教训**：
- `compress_history.py`（1621 行）试图压缩历史文件，但和 `state_io.py`
  大量重复 → 新系统直接删除
- `results.tsv` 是最可靠的数据源（append-only），但列定义和实际写入
  不完全一致 → 新系统用 `Result.to_dict()` 统一

### 旧 frontier 迁移

旧 state.json 包含：
```
frontier         = full_frontier 的副本
proxy_frontier   = proxy 最佳结果
full_frontier    = full 最佳结果
pareto_frontier  = pareto_full_frontier 的副本
pareto_proxy_frontier = proxy Pareto 前沿
pareto_full_frontier  = full Pareto 前沿
```

新系统只保留 `frontier`（= 旧 `full_frontier`），因为 full 结果更可信。
`StateManager._load()` 会自动迁移旧格式。

---

## 8. Git 操作

- 每轮自动 commit 到 `research/nightly-YYYYMMDD` 分支
- Commit 包含：state 文件 + 修改的 `sparsify/` 代码
- Agent 只允许编辑 `sparsify/` 下的文件
- Patch 文件记录每轮的代码变更

**教训**：
- 自动 commit 非常有用，可以回溯任何一轮的代码状态
- 但 commit message 需要包含 decision 信息（方便 git log 浏览）

---

## 9. 关键数字（来自 ~100 轮实验）

- 总实验：183 轮
- Keep: 30 (16%)，Promote: 36 (20%)，Incubate: 51 (28%)
- Discard: 37 (20%)，Crash: 26 (14%)，Archive: 3 (2%)
- 最佳 K=128 FVU: 0.0663（lowrank_gated_residual, ef=32）
- 最佳 K=32 FVU: 0.0958（lowrank_gated_residual, ef=32）
- Pareto front 覆盖 K: 32, 40, 48, 56, 64, 72, 80, 96, 128
- 14% crash rate → repair loop + crash recovery mode 是必要的
