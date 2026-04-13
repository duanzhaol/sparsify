# Qwen3-4B Layer17 Q-Proj Deploy Objective

本目录用于维护本次 AutoResearch 探索的独立状态与文档。

目标：
- 面向 `Qwen3-4B`
- 目标 hookpoint：`layers.[17].self_attn.q_proj`
- 目标 family：`product_key_expert_jumprelu`
- 单目标：最小化端到端部署开销

当前采用的端到端部署开销定义：
- `objective = total_cost_ratio + latest_exceed_alpha_0.50`
- 其中 `total_cost_ratio` 来自训练启动前的 cost proxy
- `latest_exceed_alpha_0.50` 指当前/latest checkpoint 的 `exceed_alpha_0.50`

目录用途：
- `README.md`：本次探索目标、约束、启动方式
- 运行中的状态文件、trial 记录、leaderboard 由 autoresearch 运行时自动生成

当前 repo 内执行入口：
- `python -m research.deploy_objective_autoresearch.main bootstrap --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective`
- `python -m research.deploy_objective_autoresearch.main status --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective`
- `python -m research.deploy_objective_autoresearch.main metric --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective`
- `python -m research.deploy_objective_autoresearch.main step --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective --resume-latest`
- `python -m research.deploy_objective_autoresearch.main write-decision --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective --action continue_current --rationale '...'`

外层 `codex-autoresearch` 的机械指标建议使用：
- `python -m research.deploy_objective_autoresearch.main metric --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective --bootstrap-if-missing`

推荐的中文启动描述：

```text
$codex-autoresearch
我想后台持续优化 Qwen3-4B 的 layer17 q_proj SAE 部署目标。请把 research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective 作为 inner search run root，真正的优化指标是 objective = total_cost_ratio + latest_exceed_alpha_0.50，方向是越低越好。请使用 python -m research.deploy_objective_autoresearch.main metric --run-root research/autoresearch_runs/qwen3-4b-layer17-qproj-deploy-objective --bootstrap-if-missing 作为机械指标读取命令；每轮先看 status 和 current_agent_context.json，如有需要用 write-decision 写决策，再执行 step --resume-latest，然后重新读取 metric。请使用 background 模式。
```
