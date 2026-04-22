# SAE 到 LUT 导出

本文说明训练好的 SAE 检查点如何进入 LUTurbo 的查表流程。

## 目标

在目标投影输入上完成 SAE 训练后，可以将 SAE 检查点转换为 LUTurbo 运行时直接可加载的查表产物。

在当前仓库中，这一步是从 SAE 训练结果到 LUTurbo 推理产物的关键衔接点。

当前主线导出器包括：

- `scripts/export/export_product_key_expert_jumprelu_lut.py`：面向 `product_key_expert_jumprelu`
- `scripts/export/export_expert_jumprelu_lut.py`：面向 `expert_jumprelu`

旧的 `convert_sae_to_lut.py` 已移除。

## 导出步骤的输入

这些导出脚本都会组合两类核心输入：

- 训练好的基础模型
- 一个包含 `qproj` / `upproj` 运行的检查点根目录

脚本当前导出两类运行时算子：

- `self_attn.qkv_proj`
- `mlp.gate_up_proj`

它会自动发现分散在不同 run 中的层级检查点，并按层挑选最新且配置兼容的来源进行 merge。

## 导出脚本从 SAE 检查点读取的内容

对于 `product_key_expert_jumprelu`，每个单层检查点会读取：

- `left_router.weight` / `left_router.bias`
- `right_router.weight` / `right_router.bias`
- `pair_left_index` / `pair_right_index`
- `expert_encoders`
- `expert_encoder_bias`
- `log_threshold`
- `W_dec` / `b_dec`
- `cfg.json`

对于 `expert_jumprelu`，每个单层检查点会读取：

- `router.weight` / `router.bias`
- `expert_encoders`
- `expert_encoder_bias`
- `log_threshold`
- `W_dec` / `b_dec`
- `cfg.json`

检查点查找支持直接扫描根目录下的 run 树，只要层目录包含 `cfg.json` 与 `sae.safetensors` 即可。

## 导出结果（概念层面）

对于目标线性权重矩阵 `W_target`，导出器会把运行时需要的三个阶段数据一次性写入单个 `.lut.safetensors`：

- Stage A：路由器、专家编码器、阈值、解码器
- Stage B：`precomputed_products = W_dec @ W_target.T`
- Stage B：`bias_product = b_dec @ W_target.T + b_target`
- Stage C：`compensation_weight_t = W_target.T`

如果显存或内存紧张，也可以按批计算 Stage B 以降低峰值占用。

## 输出目录

导出目录至少包含：

- `metadata.json`
- `layers.<idx>.self_attn.qkv_proj.lut.safetensors`
- `layers.<idx>.mlp.gate_up_proj.lut.safetensors`

可选的 `--merge-output-dir` 还会额外生成一个方便排查来源的合并视图：

- `qproj_best/`
- `upproj_best/`
- `merge_manifest.json`

## 推荐工作流

1. 针对目标 hookpoint 训练 SAE 检查点
2. 将分段训练得到的 `qproj` / `upproj` run 放在同一个检查点根目录下
3. 运行 `scripts/export/export_product_key_expert_jumprelu_lut.py`
4. 将导出产物接入 LUTurbo 推理端

实用建议：

- 保持 SAE 运行命名和检查点目录结构在同一投影类型下的一致性
- 对拆分层段使用兼容的架构配置，避免 merge 时出现签名不一致
- 确认训练所用 hookpoint 与导出脚本内置的模块映射一致

## 关于格式稳定性

早期文档曾尝试给出固定的 LUT 存储规范。就当前阶段而言，更准确的表述是：导出格式属于仓库内约定，还不是冻结的公共标准。

关于旧格式草案的历史说明已移至 `archive/legacy/lut_format.md`。

## 需要记住的当前限制

- 当前导出约定属于仓库内部约定，不应视为对外冻结标准
- 当前导出器按架构拆分：`product_key_expert_jumprelu` 与 `expert_jumprelu` 各自使用独立脚本
- 当前脚本默认按 `layers.*.self_attn.q_proj` 与 `layers.*.mlp.up_proj` 自动查找检查点
- 如果检查点配置不兼容，merge 会被拒绝
