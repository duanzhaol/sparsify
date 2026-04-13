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
