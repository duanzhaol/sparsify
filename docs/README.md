# Sparsify 文档

Sparsify 是 LUTurbo 的稀疏自编码器（SAE）训练与导出层。本仓库现在聚焦于精简的 SAE 训练路径，主要在 NVIDIA/CUDA 上运行，Ascend/NPU 仅保留作为兼容性路径和历史参考。

## 本仓库的功能

- 在 Transformer 模块输入上训练稀疏自编码器
- 保存 SAE 检查点供后续分析和导出
- 计算 LUTurbo 在线补偿逻辑使用的肘部阈值
- 将训练好的 SAE 检查点转换为 LUT 友好的制品

## 本仓库不做什么

- 它不是完整的 LUTurbo 推理运行时
- 它不是通用的可解释性框架
- 它不再将 Ascend 作为主要平台

## 推荐阅读顺序

1. [项目概览](overview.md)
2. [训练快速开始](training/quickstart.md)
3. [配置参考](training/config-reference.md)
4. [Qwen3 指南](training/qwen3-guide.md)
5. [训练流水线](architecture/training-pipeline.md)
6. [核心组件](architecture/core-components.md)
7. [性能说明](architecture/performance.md)
8. [SAE 到 LUT 导出](export/sae-to-lut.md)

## 当前文档策略

- 主要文档仅描述当前代码路径。
- 历史设计笔记和性能分析报告存放于 [archive/](archive/README.md) 下。
- 当文档与代码不一致时，以代码为准，特别是 `sparsify/__main__.py`、`sparsify/config.py` 和 `sparsify/trainer.py`。
