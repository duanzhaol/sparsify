# Qwen3 TorchAO W8A8 Teacher 训练设计

## 1. 背景

当前 `product_key_expert_jumprelu` SAE 已经完成了多轮训练后量化（PTQ）与 I/O QAT 相关尝试，现阶段可以得到两个比较稳定的判断：

- SAE 主干本身对 8bit 已经比较稳；
- 仅在 SAE 输入输出侧插入 fake quant，带来的训练收益并不明显。

这意味着下一步更值得回答的问题，已经不再是“SAE 本体要不要继续做更激进的量化训练”，而是：

**如果上游 LLM 本身就是一个真实的 W8A8 backbone，那么它产生活激活的分布是否会改变，进而让 SAE 重建任务变得更简单。**

本设计围绕这个问题展开。

## 2. 目标

本设计的目标是为当前 SAE 在线训练链路增加一种新的 activation source：

- 默认仍支持现有 BF16 / HF backbone；
- 新增支持 `Qwen3 + torchao W8A8` backbone 在线产生活激活；
- SAE 仍然保持当前浮点训练、浮点重构、浮点优化器状态。

第一版的目标不是“做最优雅的抽象”，而是：

**先让真实 W8A8 teacher backbone 能在线接进当前 SAE 训练框架并跑起来。**

## 3. 非目标

第一版明确不做以下事情：

- 不继续扩展 SAE 本体内部的 QAT；
- 不把 `W8A8 teacher` 与 `io_quant_mode=qat_io_int8` 混合到同一个实验里；
- 不先做完整的 BF16 / W8A8 cross-source 评估矩阵；
- 不先做独立 runtime / RPC activation provider；
- 不先设计一个通用的多后端 backbone 插件系统；
- 不在第一版里在线量化 Qwen3 backbone，本设计只消费“已经可加载的 torchao W8A8 teacher”。

## 4. 方法选择

对于 `Qwen3`，第一版不再沿用之前讨论过的 SmoothQuant 官方实现路径，而改用：

- `torchao`
- 具体量化方式：`Int8DynamicActivationInt8WeightConfig`

选择这条路线的原因是：

1. `torchao` 已经与 Hugging Face / PyTorch 模型加载路径集成；
2. 对 `Qwen3` 来说，`torchao` 比官方 SmoothQuant 更现实，更容易贴近现有训练代码；
3. 第一版需要的是“真实 W8A8 teacher 在线产生活激活”，而不是最极致的部署实现。

## 5. 整体思路

第一版只替换上游 teacher activation 的来源，不改 SAE 本体。

训练链路可以概括为：

```text
token batch
 -> Qwen3 BF16 backbone 或 Qwen3 torchao W8A8 backbone
 -> hookpoint activations
 -> SAE training step
 -> loss / optimizer / checkpoint
```

其中：

- `activation_source=hf_bf16` 时，保持现有训练路径；
- `activation_source=w8a8_backbone` 时，使用真实 `Qwen3 + torchao W8A8` 模型前向并收集中间激活；
- 下游 SAE forward / backward / checkpoint 逻辑尽量完全复用当前实现。

## 6. 设计原则

第一版遵循四条原则。

### 6.1 只替换激活来源

这一版要验证的是 teacher activation distribution 的变化，而不是 SAE 内部量化逻辑。因此：

- 上游 teacher 可以变；
- SAE 训练壳尽量不变；
- SAE 架构实现不变。

### 6.2 优先兼容现有 hookpoint 命名

当前训练、阈值文件、checkpoint 命名都围绕现有 hookpoint 工作，例如：

- `layers.0.self_attn.q_proj`
- `layers.13.self_attn.q_proj`

第一版应尽量保住这套命名，以减少训练链路和实验脚本改动。

### 6.3 先追求“能跑起来”

第一版的成功标准不是“设计最优雅”，而是：

- 能加载真实 W8A8 teacher；
- 能抓到目标 hookpoint 激活；
- 能正常训练 SAE。

### 6.4 不和 I/O QAT 混在一起

为了让实验结果更好解释，第一版只改变：

- teacher source

而不同时改变：

- SAE 本地 fake quant 行为

推荐第一版对照组合是：

- `activation_source=hf_bf16, io_quant_mode=off`
- `activation_source=w8a8_backbone, io_quant_mode=off`

## 7. 代码改动边界

第一版只建议改动以下位置：

- `sparsify/config.py`
- `sparsify/__main__.py`
- `sparsify/trainer.py`
- 新增一个最小工具文件，例如 `sparsify/quantized_backbone.py`
- `scripts/autoresearch_test.sh`

第一版明确不改：

- `sparsify/sparse_coder.py`
- 任何具体 SAE 架构实现
- 现有 `sparsify/train_quantization.py` 的主逻辑

## 8. 配置设计

第一版只新增最小配置集。

### 8.1 新增配置项

- `activation_source: str = "hf_bf16"`
  - 允许值：
    - `hf_bf16`
    - `w8a8_backbone`

- `activation_backbone_path: str | None = None`
  - 当 `activation_source=w8a8_backbone` 时必填
  - 表示真实 W8A8 teacher 模型路径

- `activation_threshold_path: str | None = None`
  - 可选
  - 供未来 W8A8 teacher 独立阈值文件使用
  - 第一版可以先不强制启用，但接口应预留

### 8.2 与现有 I/O QAT 配置的关系

这套配置与现有 `io_quant_*` 配置是正交关系：

- `activation_source`：控制 teacher activation 来自哪里
- `io_quant_mode`：控制 SAE 本地是否做 fake quant

第一版只建议使用：

- `activation_source=hf_bf16, io_quant_mode=off`
- `activation_source=w8a8_backbone, io_quant_mode=off`

## 9. 运行时设计

第一版不引入复杂的 activation provider 抽象系统，而是采用一个最小可用方案。

### 9.1 默认路径

当 `activation_source=hf_bf16` 时：

- 沿用当前训练器通过 `named_modules + register_forward_hook` 采集中间激活的路径。

### 9.2 W8A8 路径

当 `activation_source=w8a8_backbone` 时：

- 使用 `torchao` 配置加载 `Qwen3`；
- 仍然优先尝试沿用现有 `named_modules + register_forward_hook` 路径；
- 只有在 hookpoint 不兼容时，才考虑进一步做 fallback 适配。

这一点是第一版的重要约束：

**默认假设 `Qwen3 + torchao W8A8` 仍然是一个可 hook 的 PyTorch/HF 模型。**

## 10. 第一版实现策略

### 10.1 `sparsify/config.py`

增加最小 activation-source 配置及校验。

### 10.2 `sparsify/quantized_backbone.py`

新增一个最小辅助模块，负责：

- 构造 `torchao W8A8` 的 `Qwen3` 模型；
- 在训练启动前检查目标 hookpoints 是否存在；
- 为 trainer 提供一致的模型对象。

### 10.3 `sparsify/__main__.py`

根据 `activation_source` 选择：

- 现有 BF16 teacher 模型；
- `torchao W8A8` teacher 模型。

### 10.4 `sparsify/trainer.py`

第一版尽量少改：

- 保持当前基于模型 hook 的训练主路径；
- 只确保这一路径在 `torchao W8A8 Qwen3` 上也能工作；
- 如果需要，加入少量启动前检查和错误信息。

### 10.5 `scripts/autoresearch_test.sh`

新增两个 env / CLI 参数透传：

- `ACTIVATION_SOURCE`
- `ACTIVATION_BACKBONE_PATH`

## 11. 技术验证要求

在开始正式训练之前，第一版必须先完成一轮最小技术验证。

验证内容包括：

1. `Qwen3 + torchao W8A8` 可以成功加载；
2. 量化后的模型仍然能列出目标 hookpoints；
3. 目标 hookpoints 上的 forward hook 能成功收到 activation tensor；
4. 至少一个 smoke training step 能正常 forward / backward。

## 12. 成功标准

第一版只要求满足以下工程目标：

- `activation_source=w8a8_backbone` 可启动；
- 量化后的 Qwen3 可匹配目标 hookpoints；
- SAE 能在线使用真实 W8A8 teacher activation 训练；
- 训练不会在第一步就失败；
- checkpoint 和基础日志仍然正常。

第一版暂不要求：

- 完整的精度结论；
- W8A8 teacher 专用阈值体系；
- BF16 / W8A8 交叉评估矩阵；
- 与 I/O QAT 的组合实验。

## 13. 后续阶段

如果第一版顺利跑通，后续再按顺序推进：

1. 验证 BF16 与 W8A8 teacher 的 A/B 对照；
2. 为 W8A8 teacher 单独重算 threshold；
3. 再考虑是否需要：
   - W8A8 teacher + I/O QAT
   - 更完整的 deploy-aligned training

## 14. 总结

本设计的核心不是继续量化 SAE 本体，而是：

**让当前 SAE 训练系统能够直接消费真实 `Qwen3 + torchao W8A8` backbone 产生的 activation。**

第一版的关键目标很明确：

- 不追求通用化；
- 不追求一步到位；
- 先用最小改动把真实 W8A8 teacher 接进当前训练架构，并让训练跑起来。
