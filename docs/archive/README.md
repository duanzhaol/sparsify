# 归档

这里保存的是历史文档，用于回溯背景与设计过程，不作为当前实现依据。

除非你已经对照当前代码重新核对，否则应默认这些文档可能过时。

当前用法和实现细节请以 `docs/` 下的主文档为准。

## 如何使用此归档

- 先阅读 `docs/README.md` 和主文档，确认当前代码主线。
- 只有在需要了解历史背景、旧实验或重构原因时，再查归档文档。
- 如果归档内容与当前代码冲突，优先以代码和主文档为准。

## 子目录

- `ascend/`：历史 Ascend/NPU 性能分析报告
- `refactor/`：仓库简化工作的设计笔记
- `ideas/`：不属于当前代码主线的探索性设计文档
- `legacy/`：为参考保留的旧代码走查和被取代的用户文档

## 索引

### `ascend/`

以 Ascend 为中心的历史性能分析和优化笔记：

- `ascend_profling.md`
- `npu_profiling_analysis.md`
- `sae_training_profiling_report_20260306.md`

如果你想了解历史上的 NPU 瓶颈，或追踪某些融合路径的来历，可以看这一组文档；但它们不代表当前平台建议。

### `refactor/`

关于仓库如何被简化的历史文档：

- `docs_migration_plan.md`
- `hook_mode_design.md`
- `refactoring_design.md`

如果你想知道为什么很多旧分支从主线移除，这一组文档会有帮助。

### `ideas/`

不属于当前代码主线的探索性设计：

- `moe_lowrank_encoder_design.md`
- `tiling_lowrank_two_stage.md`

这部分是研究思路记录，不是实现指南。

### `legacy/`

来自旧仓库阶段的代码走查和用户文档：

- `code_walkthrough.md`
- `feature_analysis.md`
- `fused_decoder_design.md`
- `hook_function_analysis.md`
- `lut_format.md`
- `qwen3_sae_training_guide.md`
- `talking_with_claude.md`
- `trainer_code_walkthrough.md`
- `training_acceleration_guide.md`

这些文档可以作为背景参考，但其中不少内容对应旧功能、旧代码路径或旧平台假设，阅读时请与当前代码交叉核对。
