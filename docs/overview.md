# 概览

本仓库用于为 LUTurbo 训练稀疏自编码器，并生成基于 LUT 的推理路径所需的中间制品。

## 在 LUTurbo 中的角色

在更广泛的 LUTurbo 项目中，核心思想是用基于 SAE 基向量的查找表流水线替代昂贵的在线矩阵乘法。Sparsify 负责该工作流的训练端：

- 从 Transformer 收集模块输入激活值
- 在这些激活值上训练稀疏自编码器
- 测量重建行为和阈值统计信息
- 将训练好的检查点导出为面向 LUT 的表

Sparsify 不实现完整的 LUTurbo 推理运行时，它专注于训练和导出阶段。

另一种有用的表述方式是：

- LUTurbo 问："如何将昂贵的在线投影输入转换为稀疏基组合和查找表读取？"
- Sparsify 通过生成 SAE 基、重建统计信息和可导出的检查点制品来回答训练端的问题。

## 主要输入

- 通过 `sparsify/__main__.py` 加载的 Hugging Face Transformer 模型
- 来自 Hugging Face 数据集或 memmap `.bin` 数据集的已分词文本数据
- 选定的钩入点，通常是 Transformer 层或投影模块

## 主要输出

- `checkpoints/` 下的 SAE 检查点
- 可选的最佳检查点快照
- `compute_elbow_thresholds.py` 生成的肘部阈值 JSON 文件
- `convert_sae_to_lut.py` 生成的 LUT 导出制品

这些输出形成一个自然的流水线：

```text
模型 + 数据集
    -> SAE 训练
    -> 检查点树
    -> 肘部阈值统计
    -> LUT 导出
    -> 下游 LUTurbo 推理资产
```

## 当前训练范围

当前代码库有意比历史版本 `sparsify-ascend` 更精简：

- 训练目标为模块输入，而非广泛的钩子模式
- 主要目标是通过 FVU 评估局部重建质量
- 默认优化器路径是 SignSGD 配合无调度训练（schedule-free）
- 仍支持分块 SAE、Hadamard 旋转、超出指标、微调和恢复

当前主线有意不包含的内容：

- 与活跃训练器不再匹配的广泛旧实验分支
- 大量替代损失函数和优化模式
- 将 Ascend 作为主要平台的文档

## 平台优先级

本仓库在 `sparsify/device.py` 中仍包含 CUDA 和 NPU 设备抽象，但实际默认值现在应理解为：

- NVIDIA/CUDA：主要运行时路径
- Ascend/NPU：兼容性路径和历史性能分析目标

历史 Ascend 专用报告保存在 `archive/ascend/` 下。

## 关键文件

- `sparsify/__main__.py`：CLI 入口、模型和数据集加载
- `sparsify/config.py`：活跃配置接口
- `sparsify/trainer.py`：主训练循环和钩子驱动的 SAE 更新
- `compute_elbow_thresholds.py`：阈值统计信息生成
- `convert_sae_to_lut.py`：从 SAE 检查点导出到 LUT 制品
- `LUTurbo-doc/research-log.md`：项目级 LUTurbo 设计上下文

## 端到端工作流

对于大多数用户，本仓库应被理解为一个四阶段工作流。

### 1. 选择目标投影

为 LUTurbo 选择你关心的模块族，通常是投影输入，例如：

- `layers.X.self_attn.q_proj`
- `layers.X.self_attn.o_proj`
- `layers.X.mlp.up_proj`

这些与 `convert_sae_to_lut.py` 中的导出器逻辑很好地对应。

### 2. 训练 SAE 检查点

使用 `python -m sparsify` 来：

- 加载基础模型
- 钩入选定模块
- 每个钩入点或每个钩入点每个种子训练一个 SAE
- 在 `checkpoints/` 下保存检查点

活跃训练器使用模块输入作为 SAE 训练激活值。

### 3. 计算肘部阈值

使用 `compute_elbow_thresholds.py` 收集激活样本并计算每个钩入点的肘部统计信息。这些值随后被 LUTurbo 端的补偿启发式方法和 Sparsify 的超出指标使用。

### 4. 导出 LUT 制品

使用 `convert_sae_to_lut.py` 组合：

- 基础模型权重
- 训练好的 SAE 解码器权重和偏置
- 可选的阈值元数据

到面向 LUT 的输出文件中供下游消费。

## 按目标推荐阅读

如果你的目标是：

- 训练 SAE：阅读 `training/quickstart.md`
- 理解参数：阅读 `training/config-reference.md`
- 使用 Qwen3：阅读 `training/qwen3-guide.md`
- 理解训练器内部：阅读 `architecture/training-pipeline.md`
- 理解制品如何流入 LUTurbo：阅读 `export/sae-to-lut.md`
