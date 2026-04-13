# Qwen3-4B Layer17 Q-Proj Deploy Objective

本目录用于维护本次 AutoResearch 探索的独立状态与文档。

目标：
- 面向 `Qwen3-4B`
- 目标 hookpoint：`layers.[17].self_attn.q_proj`
- 目标 family：`product_key_expert_jumprelu`
- 单目标：最小化端到端部署开销

本次探索采用的端到端部署开销定义：
- `objective = total_cost_ratio + best_exceed_alpha_0_50`
- 其中 `total_cost_ratio` 来自训练启动前的 cost proxy
- `best_exceed_alpha_0_50` 指训练过程中达到的最佳 `exceed_alpha_0.50`

规划中的目录用途：
- `README.md`：本次探索目标、约束、计划
- 后续可放置：
  - 运行计划
  - 启动配置
  - 人工总结
  - 独立的 autoresearch 状态/结果文件路径说明

当前已确认事实：
- 仓库已有 `research/AutoResearch/`，当前 objective 近似为
  `total_cost_ratio + exceed_alpha_0_50`
- 但现有实现使用的是“最后一步”的 `exceed_alpha_0.50`
- 现有 auto-continuation 主要依据最近窗口 `FVU` 下降量，而不是 objective 改善
- 当前主历史目录固定为 `research/history/`，若要隔离本次探索，需要单独定制状态路径
- 新框架将采用“动态搜索边界”，而不是提前写死整张固定搜索空间表

当前新增的 repo 内执行入口：
- `python -m research.deploy_objective_autoresearch.main bootstrap --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective`
- `python -m research.deploy_objective_autoresearch.main status --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective`
- `python -m research.deploy_objective_autoresearch.main metric --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective`
- `python -m research.deploy_objective_autoresearch.main step --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective --resume-latest`
- `python -m research.deploy_objective_autoresearch.main write-decision --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective --action continue_current --rationale '...'`

当前 step 行为：
- 默认自动恢复最近一个未结束 trial
- 每次 step 最多推进一个 15M checkpoint 决策周期
- gray-zone 会先落盘 `current_agent_context.json` / `trials/<trial_id>/agent_context.json`
- 若 `pending_agent_decision.json` 存在，会优先读取外部 agent 决策；否则走保守 heuristic fallback
- `--dry-run-agent` 模式只预览下一步，不会真正启动训练

为 `codex-autoresearch` 适配的补充入口：
- `metric`：默认只输出一个数值，表示当前 incumbent 的 `best_objective`，适合作为外层 autoresearch 的机械指标
- `metric --bootstrap-if-missing`：如果当前还没有 incumbent，会先推进一次 inner step，再输出首个可用 objective，适合作为 baseline 命令
- `write-decision`：把外层 Codex/agent 生成的决策安全写入 `pending_agent_decision.json`，并自动对 `next_params` 做合法化/规整化

推荐的 `$codex-autoresearch` 外层使用方式：
- 外层 skill 负责 background runtime、`research-results.tsv`、`autoresearch-state.json`
- 内层 `research.deploy_objective_autoresearch` 负责真正的 SAE 参数搜索
- 推荐让外层把机械指标设为：
  - `python -m research.deploy_objective_autoresearch.main metric --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective --bootstrap-if-missing`
- 推荐让外层每轮真正执行的动作是：
  1. 读取 `status` / `current_agent_context.json`
  2. 若需要 agent 决策，则调用 `write-decision`
  3. 调用 `step --resume-latest`
  4. 再次调用 `metric` 读取新的 best objective

一个推荐的中文启动描述可以是：

```text
$codex-autoresearch
我想后台持续优化 Qwen3-4B 的 layer17 q_proj SAE 部署目标。请把 research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective 作为 inner search run root，真正的优化指标是 objective = total_cost_ratio + best_exceed_alpha_0.50，方向是越低越好。请使用 python -m research.deploy_objective_autoresearch.main metric --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective --bootstrap-if-missing 作为机械指标读取命令；每轮先看 status 和 current_agent_context.json，如有需要用 write-decision 写决策，再执行 step --resume-latest，然后重新读取 metric。请使用 background 模式。
```
