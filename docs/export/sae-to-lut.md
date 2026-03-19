# SAE 到 LUT 导出

本文说明训练好的 SAE 检查点如何进入 LUTurbo 的查表流程。

## 目标

在目标投影输入上完成 SAE 训练后，可以将 SAE 解码器中的基向量转换为查找表行。这个转换由 `convert_sae_to_lut.py` 完成。

在当前仓库中，这一步是从 SAE 训练结果到 LUTurbo 推理产物的关键衔接点。

## 导出步骤的输入

`convert_sae_to_lut.py` 会组合三类输入：

- 训练好的基础模型
- 一个或多个 SAE 检查点目录
- `compute_elbow_thresholds.py` 生成的可选阈值文件

脚本内置了常见投影类型的映射：

- `qproj`
- `oproj`
- `upproj`
- `kproj`
- `vproj`
- 融合导出如 `qkv` 和 `gate_up`

融合导出的处理方式如下：

- 先选一个 SAE 检查点类型作为基底来源
- 连接多个目标模型权重
- 导出一个合并后的 LUT 产物

## 导出脚本从 SAE 检查点读取的内容

每个检查点会读取：

- `encoder.weight`
- `encoder.bias`
- `W_dec`
- `b_dec`
- `cfg.json`

这些信息足以重建 SAE 基向量，并与目标模型权重配对计算。

检查点查找当前支持两种目录结构：

- 嵌套：`best/<checkpoint_name>/<checkpoint_name>/sae.safetensors`
- 扁平：`best/<checkpoint_name>/sae.safetensors`

## 导出结果（概念层面）

对于目标线性权重矩阵 `W_target`，导出步骤会提前计算解码器基向量与目标权重的乘积。这样在线推理时可以使用：

- SAE 端给出的稀疏潜在变量选择
- 查找表组合替代完整的在线矩阵乘法

导出脚本会明确计算：

- `precomputed_products = W_dec @ W_target.T`
- `bias_product = b_dec @ W_target.T`

如果显存或内存紧张，也可以按批计算以降低峰值占用。

## 阈值文件

`compute_elbow_thresholds.py` 用于生成激活分布的肘部统计结果。这些结果会被下游补偿策略使用，建议按对应投影类型分类保存。

该脚本的核心过程是：

- 在模型模块输入处注册 hook
- 收集激活样本直到令牌预算上限
- 构建绝对值分位数曲线
- 使用 Kneedle 风格启发式方法找到肘部点
- 在 JSON 中存储 `elbow_p` 和 `elbow_value`

## 推荐工作流

1. 针对目标 hookpoint 训练 SAE 检查点
2. 为同一投影类型计算肘部阈值 JSON 文件
3. 使用对齐的投影名称和层范围运行 `convert_sae_to_lut.py`
4. 将导出产物接入 LUTurbo 推理端

实用建议：

- 保持 SAE 运行命名和检查点目录结构在同一投影类型下的一致性
- 生成阈值文件时使用与导出脚本的 `THRESHOLD_FILES` 映射对齐的名称
- 确认训练所用 hookpoint 与导出脚本内置的模块映射一致

## 关于格式稳定性

早期文档曾尝试给出固定的 LUT 存储规范。就当前阶段而言，更准确的表述是：导出格式属于仓库内约定，还不是冻结的公共标准。

关于旧格式草案的历史说明已移至 `archive/legacy/lut_format.md`。

## 需要记住的当前限制

- 当前导出约定属于仓库内部约定，不应视为对外冻结标准
- 导出脚本默认按 `*-qproj`、`*-oproj`、`*-upproj` 这类命名模式查找检查点
- 如果检查点命名与投影类型不匹配，即使 SAE 文件存在，也可能被导出脚本遗漏
