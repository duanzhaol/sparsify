# 概览

本仓库用于为 LUTurbo 训练稀疏自编码器，并生成供 LUT 推理使用的中间产物。

## 在 LUTurbo 中的角色

LUTurbo 的核心目标是用基于 SAE 基向量的查表流程替代昂贵的在线矩阵乘法。Sparsify 负责训练侧环节：

- 收集 Transformer 模块输入激活值
- 在这些激活值上训练稀疏自编码器
- 测量重建结果和阈值统计
- 将完成训练的检查点导出为面向 LUT 的产物

Sparsify 不实现完整的 LUTurbo 推理运行时，仅聚焦训练和导出阶段。

## 主要输入

- 通过 `sparsify/__main__.py` 加载的 Hugging Face Transformer 模型
- 来自 Hugging Face 数据集或 memmap `.bin` 数据集的已分词文本
- 选定的 hookpoint，通常对应 Transformer 层或投影模块

## 主要输出

- `checkpoints/` 下的 SAE 检查点
- 可选的最佳检查点快照
- `compute_elbow_thresholds.py` 生成的肘部阈值 JSON 文件
- `convert_sae_to_lut.py` 生成的 LUT 导出产物

这些输出对应一个清晰流程：

```text
模型 + 数据集
    -> SAE 训练
    -> 检查点树
    -> 肘部阈值统计
    -> LUT 导出
    -> 下游 LUTurbo 推理侧使用
```

## 当前训练范围

与历史版本 `sparsify-ascend` 相比，当前主线有意精简为：

- 将训练目标聚焦到模块输入激活，而非复杂的 hook 模式组合
- 以 FVU 评估局部重建质量为核心指标
- 使用 SignSGD + schedule-free 作为默认优化器路径
- 仍保留分块 SAE、Hadamard 旋转、超出指标、微调及恢复能力

当前主线主动剔除以下旧实验内容：

- 与当前训练流程不再匹配的实验分支
- 替代损失函数与优化模式
- 针对 Ascend 的主平台文档

## 平台优先级

`sparsify/device.py` 仍保留 CUDA 和 NPU 设备抽象，但当前默认路径为：

- NVIDIA/CUDA：默认主线
- Ascend/NPU：兼容性路径与历史性能分析的参考

历史 Ascend 专用报告保存在 `archive/ascend/`。

## 关键文件

- `sparsify/__main__.py`：CLI 入口、模型和数据集加载
- `sparsify/config.py`：当前生效的配置定义
- `sparsify/trainer.py`：主训练循环和 hook 驱动的 SAE 更新
- `compute_elbow_thresholds.py`：阈值统计信息生成
- `convert_sae_to_lut.py`：从 SAE 检查点导出到 LUT 产物
- `LUTurbo-doc/research-log.md`：项目级 LUTurbo 设计上下文

## 端到端流程

对大多数用户来说，可以按四个阶段理解本仓库。

### 1. 选择目标投影

先确定要做 LUT 近似的模块，通常是以下投影输入：

- `layers.X.self_attn.q_proj`
- `layers.X.self_attn.o_proj`
- `layers.X.mlp.up_proj`

这些与 `convert_sae_to_lut.py` 的导出逻辑一一对应。

### 2. 训练 SAE 检查点

使用 `python -m sparsify`：

- 加载基础模型
- 在选定模块上注册 hook
- 为每个 hookpoint（以及每个 seed）训练 SAE
- 在 `checkpoints/` 下保存检查点

当前实现使用模块输入作为 SAE 训练激活值。

### 3. 计算肘部阈值

使用 `compute_elbow_thresholds.py` 收集激活样本，并为每个 hookpoint 计算肘部统计值。这些值会用于 LUTurbo 侧补偿策略，也会用于 Sparsify 的 exceed 指标。

### 4. 导出 LUT 产物

使用 `convert_sae_to_lut.py` 组合以下信息：

- 基础模型权重
- 训练好的 SAE 解码器权重和偏置
- 可选阈值元数据

最终生成面向 LUT 的输出文件，交给下游推理侧使用。

## 按目标推荐阅读

如果你的目标是：

- 训练 SAE：看 `training/quickstart.md`
- 理解参数：看 `training/config-reference.md`
- 使用 Qwen3：看 `training/qwen3-guide.md`
- 理解训练器内部：看 `architecture/training-pipeline.md`
- 理解产物如何接入 LUTurbo：看 `export/sae-to-lut.md`
