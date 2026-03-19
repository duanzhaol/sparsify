# 归档

此目录存储对上下文仍有用的历史材料，但不应被视为当前的事实来源。

除非明确针对当前代码库重新验证，否则应假设每个归档的 markdown 文件可能已过时。

使用父目录中的主文档获取当前使用和实现细节。

## 如何使用此归档

- 使用 `docs/README.md` 和活跃文档树获取当前行为。
- 仅在需要历史原理、旧实验或实现考古时使用此归档。
- 如果归档文档与当前代码冲突，优先选择代码和活跃文档。

## 子目录

- `ascend/`：历史 Ascend/NPU 性能分析报告
- `refactor/`：仓库简化工作的设计笔记
- `ideas/`：不属于活跃主线的探索性设计文档
- `legacy/`：为参考保留的旧代码走查和被取代的用户文档

## 索引

### `ascend/`

以 Ascend 为中心的历史性能分析和优化笔记：

- `ascend_profling.md`
- `npu_profiling_analysis.md`
- `sae_training_profiling_report_20260306.md`

如果你想了解旧的 NPU 瓶颈或为什么存在某些融合路径，这些是有用的，但它们不再是主要平台指导。

### `refactor/`

关于仓库如何被简化的历史文档：

- `docs_migration_plan.md`
- `hook_mode_design.md`
- `refactoring_design.md`

当你想了解为什么许多旧分支从当前主线中移除时，这些是有用的。

### `ideas/`

不属于活跃代码路径的探索性设计：

- `moe_lowrank_encoder_design.md`
- `tiling_lowrank_two_stage.md`

将这些视为研究笔记，而非实现指南。

### `legacy/`

来自旧仓库状态的被取代代码走查和用户文档：

- `code_walkthrough.md`
- `feature_analysis.md`
- `fused_decoder_design.md`
- `hook_function_analysis.md`
- `lut_format.md`
- `qwen3_sae_training_guide.md`
- `talking_with_claude.md`
- `trainer_code_walkthrough.md`
- `training_acceleration_guide.md`

这些仍可作为背景上下文有用，但许多包含过时的特性、旧代码路径或与当前仓库不再匹配的平台假设。
