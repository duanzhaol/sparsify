# Sparsify 文档

Sparsify 是 LUTurbo 的 SAE 训练与导出模块，本仓库文档聚焦当前精简后的训练主线：面向 NVIDIA/CUDA，Ascend/NPU 仅保留兼容支持和历史参考。

## 本仓库的功能

- 在 Transformer 模块输入上训练稀疏自编码器
- 保存 SAE 检查点供后续分析和导出
- 计算 LUTurbo 在线补偿逻辑使用的肘部阈值
- 将训练好的 SAE 检查点转换为 LUT 友好的产物

## 本仓库不包含

- 不包含完整的 LUTurbo 推理运行时
- 不定位为通用可解释性框架
- 不再以 Ascend 作为主要平台

## 推荐阅读顺序

1. [项目概览](overview.md)
2. [训练快速开始](training/quickstart.md)
3. [配置参考](training/config-reference.md)
4. [Qwen3 指南](training/qwen3-guide.md)
5. [训练流程](architecture/training-pipeline.md)
6. [核心组件](architecture/core-components.md)
7. [性能说明](architecture/performance.md)
8. [SAE 到 LUT 导出](export/sae-to-lut.md)

## 当前文档策略

- 主文档仅描绘当前代码主线，历史设计笔记和性能分析保留在 [archive/](archive/README.md)。
- 当文档与代码产生差异，以 `sparsify/__main__.py`、`sparsify/config.py` 和 `sparsify/trainer.py` 为准。
