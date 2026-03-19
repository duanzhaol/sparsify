# SAE 到 LUT 导出

本页描述训练好的 SAE 检查点如何连接到 LUTurbo 的查找表流水线。

## 目标

在目标投影的输入激活上训练 SAE 后，LUTurbo 希望重用 SAE 解码器基向量作为查找表行。`convert_sae_to_lut.py` 执行该转换。

在当前仓库布局中，此导出步骤是 SAE 训练到 LUTurbo 下游推理制品的主要桥梁。

## 导出步骤的输入

`convert_sae_to_lut.py` 组合三部分信息：

- 训练好的基础模型
- 一个或多个 SAE 检查点目录
- `compute_elbow_thresholds.py` 生成的可选阈值文件

该脚本已包含常见投影族的映射：

- `qproj`
- `oproj`
- `upproj`
- `kproj`
- `vproj`
- 融合导出如 `qkv` 和 `gate_up`

融合情况的工作方式：

- 选择一个 SAE 检查点族作为源基
- 连接多个目标模型权重
- 导出一个组合的 LUT 制品

## 导出脚本从 SAE 检查点读取的内容

从每个检查点加载：

- `encoder.weight`
- `encoder.bias`
- `W_dec`
- `b_dec`
- `cfg.json`

这些足以重建 SAE 基并将其与目标模型权重配对。

检查点查找目前支持两种目录风格：

- 嵌套：`best/<checkpoint_name>/<checkpoint_name>/sae.safetensors`
- 扁平：`best/<checkpoint_name>/sae.safetensors`

## 概念导出结果

对于目标线性权重矩阵 `W_target`，导出步骤预计算解码器-基乘积，以便在线推理可以使用：

- 来自 SAE 端的稀疏潜在变量选择
- 查找表组合替代完整的在线矩阵乘法

实际上，这是 Sparsify 训练输出与 LUTurbo 推理时查找路径之间的桥梁。

导出脚本显式计算：

- `precomputed_products = W_dec @ W_target.T`
- `bias_product = b_dec @ W_target.T`

可选分批次以提高内存效率。

## 阈值文件

`compute_elbow_thresholds.py` 生成激活分布上的肘部统计信息。这些文件对下游补偿启发式方法有用，应与它们对应的投影族一起存储。

该脚本：

- 钩入模型模块输入
- 收集激活样本直到令牌预算上限
- 构建绝对值分位数曲线
- 使用 Kneedle 风格启发式方法找到肘部点
- 在 JSON 中存储 `elbow_p` 和 `elbow_value`

## 推荐工作流

1. 为目标钩入点训练 SAE 检查点
2. 为相同的投影族计算肘部阈值 JSON 文件
3. 使用对齐的投影名称和层范围运行 `convert_sae_to_lut.py`
4. 将导出制品送入 LUTurbo 推理端

实用建议：

- 保持 SAE 运行名称和检查点布局与每个投影类型一致
- 生成阈值文件时使用与导出脚本的 `THRESHOLD_FILES` 映射对齐的名称
- 验证训练中使用的钩入点是否与导出器中内置的模块族假设匹配

## 关于格式稳定性

早期文档试图定义固定的 LUT 存储规范。在这个阶段，最好将导出的布局视为当前仓库约定，而非永久冻结的公共标准。

关于旧格式草案的历史说明已移至 `archive/legacy/lut_format.md`。

## 需要记住的当前限制

- 导出约定是仓库本地的，不应被视为冻结的外部标准
- 导出器中的检查点发现假设命名模式如 `*-qproj`、`*-oproj` 和 `*-upproj`
- 导出器是投影族感知的，因此不匹配的检查点命名可能破坏查找，即使原始 SAE 文件存在
