# Qwen3 Teacher 量化与 SAE I/O Int8 当前实现总结

## 目的

本文档总结截至 `2026-04-10` 已经完成的两条量化相关落地：

- `Qwen3-0.6B + torchao W8A8 teacher` 在线产生活激活；
- `SAE I/O Int8 QAT Phase 1`，即训练时对输入输出做 8bit 感知。

这两条路线的共同目标不是单纯证明“量化能跑”，而是逐步回答下面两个更关键的问题：

1. 上游 teacher 如果换成真实 `W8A8` backbone，SAE 重建任务会不会明显更简单；
2. 如果 SAE 在训练时就显式感知部署侧 `int8` 输入输出，最终指标会不会更贴近真实上线形态。

## 当前已完成的实现

### 1. Qwen3 + torchao W8A8 teacher

当前仓库已经支持通过配置切换 teacher 激活来源。

相关文件：

- `sparsify/config.py`
- `sparsify/__main__.py`
- `sparsify/trainer.py`
- `sparsify/quantized_backbone.py`
- `tests/test_quantized_backbone.py`
- `tests/test_train_quantization.py`
- `scripts/autoresearch_test.sh`

新增能力：

- `activation_source=hf_bf16`
  - 保持原始 BF16 teacher 路线；
- `activation_source=w8a8_backbone`
  - 使用真实 `Qwen3 + torchao W8A8` backbone 在线前向并采集中间激活；
- `activation_backbone_path`
  - 指定本地基础模型目录；
- 训练侧 hookpoint 匹配仍沿用现有逻辑，不需要额外改 SAE 架构。

当前 teacher 侧量化方法为：

- `torchao` 的 `Int8DynamicActivationInt8WeightConfig`
- 本质上是 `dynamic W8A8`
- activation 为对称动态量化；
- weight 为 int8；
- 接入方式是 **加载时量化**，不是预下载独立量化 checkpoint。

### 2. SAE I/O Int8 QAT Phase 1

相关文件：

- `sparsify/train_quantization.py`
- `sparsify/config.py`
- `sparsify/trainer.py`
- `scripts/autoresearch_test.sh`
- `scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_io_int8.sh`

当前 Phase 1 的做法是：

```text
acts_fp -> fake_quant_in -> SAE -> recon_fp -> fake_quant_out -> loss
```

核心特点：

- 只让 SAE 显式感知部署时的 `int8` 输入输出；
- SAE 权重、梯度、优化器状态仍然保持浮点；
- 不引入真实 int8 kernel；
- 当前 fake quant 方法为：
  - `8bit`
  - `symmetric`
  - `per_token`
  - `absmax`
  - `QDQ + STE`

训练中已经可以记录：

- `fvu_fp_teacher`
- `fvu_deploy`
- `quant_floor`
- `exceed_alpha_xx_fp_teacher`
- `exceed_alpha_xx_deploy`
- `input_clip_rate`
- `output_clip_rate`
- `input_scale_mean`
- `output_scale_mean`

## 新增脚本

为了做对照实验，当前额外补充了两个轻量脚本：

- `scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_w8a8_teacher_smoke_q_0_13.sh`
- `scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_bf16_teacher_smoke_q_0_13.sh`

它们都只覆盖：

- `product_key_expert_jumprelu`
- `q_proj`
- `layers.[0-13]`
- smoke 级训练预算

这样可以把变量尽量压到最少，只比较：

- BF16 teacher
- W8A8 teacher
- 后续 I/O Int8 QAT

## 关键实现结论

### 1. W8A8 teacher 已经可以稳定接入现有训练框架

当前实测说明：

- `Qwen3-0.6B` 可以通过 `torchao` 成功加载成 `W8A8 teacher`；
- 现有 hookpoint 机制在量化 backbone 上仍然可用；
- SAE 主训练代码不需要为 teacher 切换做大规模重写。

也就是说，**在线从真实 W8A8 backbone 采激活** 这条路径已经打通。

### 2. 单纯把 teacher 从 BF16 换成 W8A8，任务难度没有明显变化

从 smoke 对照实验看：

- `BF16 teacher` 与 `W8A8 teacher` 的 `FVU` / `exceed_alpha_0.50` 非常接近；
- 差异存在，但整体仍处于很小的范围；
- 当前没有证据说明“teacher 换成 W8A8”会显著简化 SAE 重建任务。

这意味着：

- `W8A8 teacher` 更像是一个 **部署一致性增强变量**；
- 而不是当前主要的精度改进来源。

### 3. I/O Int8 QAT 当前是更值得继续推进的主线

从已经跑出的 `IO int8` 本地结果看：

- `fvu_fp_teacher` 与普通训练主 `FVU` 仍然接近；
- `fvu_deploy` 与 `fvu_fp_teacher` 也非常接近；
- `quant_floor` 很低，当前观察值约为 `0.0028`；
- `exceed_alpha_0.50_fp_teacher` 与 `exceed_alpha_0.50_deploy` 也基本贴合。

这说明当前 Phase 1 至少支持下面这个判断：

**在 `q_proj layers.[0-13]` 这个 slice 上，8bit I/O 感知训练没有明显破坏 SAE 重建质量。**

## 已跑实验摘要

以下结果主要用于记录阶段性趋势，不强调严格统计显著性；因为不同实验的 token 数并不完全一致。

### 1. BF16 teacher smoke

实验目录：

- `checkpoints/product_key_expert_jumprelu_qproj_bf16_teacher_smoke/product_key_expert_jumprelu_q_bf16_teacher_smoke_dp2_bs1_ga8_ef1_k32_20260410_193051`

末尾指标（`layers.13.self_attn.q_proj`）：

- `FVU`: `0.3410`
- `exceed_alpha_0.50`: `0.4089`

### 2. W8A8 teacher smoke

实验目录：

- `checkpoints/product_key_expert_jumprelu_qproj_w8a8_teacher_smoke/product_key_expert_jumprelu_q_w8a8_teacher_smoke_dp2_bs1_ga8_ef1_k32_20260410_190420`

末尾指标（`layers.13.self_attn.q_proj`）：

- `FVU`: `0.3449`
- `exceed_alpha_0.50`: `0.4131`

解读：

- 相对 BF16 teacher，数值略高，但没有出现量级上的退化；
- 现阶段更合理的结论是“接近”，而不是“显著更优”或“显著更差”。

### 3. I/O Int8 QAT

实验目录：

- `checkpoints/product_key_expert_jumprelu_qproj_io_int8/product_key_expert_jumprelu_q_io_int8_dp2_bs1_ga8_ef1_k32_20260410_193651`

最近尾部指标（`layers.13.self_attn.q_proj`）：

- `FVU`: `0.3230`
- `fvu_fp_teacher`: `0.3240`
- `fvu_deploy`: `0.3257`
- `exceed_alpha_0.50_fp_teacher`: `0.4001`
- `exceed_alpha_0.50_deploy`: `0.4005`
- `quant_floor`: `0.00279`

解读：

- 当前没有观察到 I/O Int8 QAT 导致明显劣化；
- 部署口径指标和 teacher 口径指标很接近；
- 这是一个偏正面的信号，说明 8bit-aware 训练值得继续深入。

## 当前方法与 SmoothQuant 的关系

当前落地的方法并不是 SmoothQuant。

### 当前 teacher 侧方法

- `torchao dynamic W8A8`
- 重点是运行时动态量化 activation
- 优点是接入简单、与当前训练框架兼容性好

### 当前 SAE I/O 方法

- `per-token symmetric absmax fake quant`
- 重点是让 SAE 在训练时感知部署态的输入输出约束

### 与 SmoothQuant 的关键区别

SmoothQuant 的本质是：

- 先做离线 calibration；
- 通过通道级平滑，把 activation 的 outlier 压到 weight 一侧；
- 再做更稳定的 W8A8。

而当前方法更偏：

- teacher 侧：直接用 `torchao` 动态 W8A8 跑起来；
- SAE 侧：直接用简单 fake quant 先验证训练目标是否需要靠近部署目标。

因此，SmoothQuant 更适合被视为下一阶段的 **teacher 侧增强方案**，而不是当前第一阶段必须依赖的基础设施。

## 当前阶段结论

截至当前，可以给出三个比较稳的判断：

1. `Qwen3-0.6B + torchao W8A8 teacher` 已经成功接入 SAE 在线训练。
2. 单纯把 teacher 从 BF16 换成 W8A8，并不会显著改变当前 SAE 重建任务难度。
3. `SAE I/O Int8 QAT Phase 1` 目前表现稳定，且更接近真正部署问题，因此是更值得继续推进的主线。

## 建议的下一步

当前最合理的后续顺序是：

1. 继续把 `I/O Int8 QAT` 做成更正式的 q_proj 实验；
2. 在 teacher 侧单独尝试更强的离线量化方案；
3. 优先评估 `LLM Compressor + SmoothQuant + Qwen3-0.6B` 是否能作为新的 `W8A8 teacher` 来源；
4. 先做 teacher A/B，再决定是否需要把 SmoothQuant teacher 与 SAE I/O QAT 组合。

也就是说，下一步的重点不再是“量化能不能跑”，而是：

**哪一种量化 teacher / 哪一种量化训练目标，能真正让 SAE 在部署口径下变得更好。**
