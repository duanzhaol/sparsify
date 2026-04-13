# SAE Full W8A8 QAT 训练设计

## 1. 背景

当前量化探索已经得到几个比较稳定的阶段性结论：

- `expert-only / full-encoder / decoder` 的 PTQ 结果整体都很稳；
- `BF16 teacher -> torchao W8A8 teacher` 的替换，没有明显降低 SAE 重建难度；
- `BF16 teacher -> SmoothQuant teacher` 的替换，训练曲线依然与 BF16 teacher 非常接近；
- `I/O int8 QAT` 已经证明：让 SAE 感知部署态输入输出约束本身是可行的。

这意味着下一步最值得推进的问题，不再是“teacher 要不要量化”，而是：

**在保持 BF16 teacher 监督稳定的前提下，SAE 自身能否直接做更完整的 W8A8 QAT，并最终逼近 CPU 高性能部署目标。**

## 2. 目标

本设计的目标是实现一版第一阶段的 **full-SAE W8A8 QAT**：

- teacher 仍然使用 `BF16` 激活；
- `product_key_expert_jumprelu` SAE 的主要线性/乘法主干都显式进行 `W8A8 fake quant`；
- 训练阶段直接观察 `FVU / exceed_alpha_0.50 / deploy FVU` 是否仍然稳定；
- 为后续真实 int8 SAE 导出和 CPU 高性能推理打基础。

第一版的成功标准不是“直接交付可部署 int8 SAE”，而是：

- 训练稳定；
- 单层和小规模多层实验可运行；
- 指标退化可接受；
- 量化边界定义清晰，便于后续导出真实 int8 版本。

## 3. 非目标

第一版明确不做以下事情：

- 不修改上游 teacher，teacher 固定使用 `BF16`；
- 不在第一版里实现真实 CPU int8 kernel；
- 不在第一版里导出最终可部署的 int8 SAE 文件格式；
- 不在第一版里覆盖所有 SAE 架构，只先支持 `product_key_expert_jumprelu`；
- 不强行把所有非线性、离散选择、索引逻辑都改成 int8。

## 4. 设计原则

### 4.1 优先量化“热点主干”，不强求“全图全 int8”

CPU 上真正决定吞吐的，通常是：

- `Linear / matmul`
- 主要激活输入输出边界

而不是：

- `topk`
- 路由离散索引
- `JumpReLU` 阈值逻辑
- gather/scatter 的索引本身

因此第一版追求的是：

- **主干 W8A8**

而不是：

- **全图所有变量全 int8**

### 4.2 保持 BF16 teacher 稳定监督

teacher 量化没有显示出明确收益，因此这一版继续使用：

- `activation_source=hf_bf16`

这样可以把问题收敛到：

- SAE 自身量化是否稳定；
- SAE 是否更接近部署态。

### 4.3 量化训练优先最小侵入

第一版优先复用：

- `sparsify/train_quantization.py`
- `sparsify/config.py`
- `sparsify/trainer.py`

尽量把量化 helper 和 fake quant 逻辑收敛在训练外壳与少量架构分支中，不大面积重写所有 SAE 实现。

### 4.4 先支持单一主架构

第一版只支持：

- `product_key_expert_jumprelu`

这是当前最关键、最常用、也最值得验证 CPU 部署价值的架构。

## 5. 量化边界

### 5.1 需要进入 W8A8 QAT 的部分

针对 `product_key_expert_jumprelu`，第一版量化以下主干：

- 输入激活 `x`
- `left_router`
- `right_router`
- `expert_encoders`
- `W_dec`
- 输出激活边界

### 5.2 保持浮点的部分

第一版继续保持浮点逻辑：

- `b_dec`
- `expert_encoder_bias`
- `log_threshold -> softplus(threshold)`
- `sigmoid` / `JumpReLU` 门控
- `selected_probs`
- `topk` / 路由离散选择
- latent index 相关操作
- `auxk` 逻辑
- loss 统计与监控逻辑

### 5.3 这样划分的原因

这套边界兼顾了两件事：

1. 把 CPU 未来真正可能受益的热点主干纳入量化训练；
2. 避免把离散选择/门控逻辑一并 int8 化，导致训练与调试难度大幅上升。

## 6. 量化规则

### 6.1 激活量化

第一版激活统一采用：

- `8 bit`
- `symmetric`
- `per-token`
- `absmax`
- `QDQ + STE`

这一点与当前已有的 `I/O int8 QAT` 保持一致。

### 6.2 权重量化

第一版权重统一采用：

- `8 bit`
- `symmetric`
- `per-output-channel`
- `absmax`
- `QDQ + STE`

原因是：

- 比 `per-tensor` 更稳；
- 更接近 CPU int8 linear 常见实现；
- 更适合 `router / expert encoder / decoder` 这类按输出行组织的权重。

## 7. 前向链路

第一版 full-SAE W8A8 QAT 的训练链路定义如下：

```text
x_fp
 -> fake_quant_act(x_fp)
 -> router linear W8A8 QAT
 -> expert_logits_fp
 -> topk / route select (fp)
 -> expert encoder W8A8 QAT
 -> pre_acts_fp
 -> JumpReLU gate (fp)
 -> top_acts_fp
 -> fake_quant_act(top_acts_fp)
 -> decoder W8A8 QAT
 -> recon_fp
 -> fake_quant_act(recon_fp)
 -> loss / metrics
```

要点：

- `router` 的线性层走 W8A8 fake quant；
- `expert encoder` 的张量乘主干走 W8A8 fake quant；
- `decoder` 的主重构路径走 W8A8 fake quant；
- `topk / route select / gate` 仍然在浮点中进行；
- 部署态输出由 `recon_fp -> fake_quant(recon_fp)` 模拟。

## 8. Loss 设计

第一版不建议只对 deploy target 训练，而是继续保留 BF16 teacher 监督。

定义：

- `target_fp = x_fp`
- `target_deploy = fake_quant(x_fp)`
- `recon_fp = SAE_full_qat(x_fp)`
- `recon_deploy = fake_quant(recon_fp)`

主损失建议为：

```text
L = FVU(recon_fp, target_fp) + lambda_deploy * FVU(recon_deploy, target_deploy)
```

推荐初始值：

- `lambda_deploy = 0.25`

这样做的目的：

- `FVU(recon_fp, target_fp)` 保住浮点 teacher 语义；
- `FVU(recon_deploy, target_deploy)` 让结果贴近最终部署态输出。

## 9. 配置设计

第一版建议扩展现有配置体系，而不是直接新起一套大配置。

建议新增：

- `io_quant_mode=qat_full_w8a8`

保留并继续使用：

- `io_quant_bits=8`
- `io_quant_granularity=per_token`
- `io_quant_clip_mode=absmax`
- `io_loss_mode=dual_target`
- `io_loss_deploy_weight=0.25`

这样做的好处：

- 训练入口改动小；
- 现有日志、脚本、监控框架可以最大程度复用；
- 后续如果 full QAT 稳定，再重构成更通用的 `sae_quant_mode` 也不迟。

## 10. 代码改动边界

### 10.1 `sparsify/train_quantization.py`

新增 full-QAT helper，建议包括：

- `fake_quantize_weight_per_output_channel(...)`
- `fake_quantized_linear(...)`
- `fake_quantized_expert_einsum(...)`
- `fake_quantized_decoder_path(...)`

这些 helper 负责把 QDQ + scale 逻辑从架构代码中抽离出来。

### 10.2 `sparsify/config.py`

扩展配置校验：

- 允许 `io_quant_mode=qat_full_w8a8`

### 10.3 `sparsify/sparse_coder.py`

第一版只在 `ProductKeyExpertJumpReLUSparseCoder` 中接入 full-QAT 路径，重点覆盖：

- `_expert_logits_from_flat(...)`
- `_expert_candidate_acts(...)`
- `_decode_sparse(...)`

必要时增加少量内部 helper，用于：

- 判断当前是否启用 full QAT；
- 走普通路径还是 fake-quant 路径。

### 10.4 `sparsify/trainer.py`

尽量少改，主要处理：

- 输入 fake quant 的入口；
- 输出 fake quant 与 deploy 口径 loss；
- 新增 full-QAT 相关监控指标。

## 11. 监控指标

第一版建议至少记录：

- `FVU`
- `fvu_fp_teacher`
- `fvu_deploy`
- `quant_floor`
- `exceed_alpha_0.50`
- `exceed_alpha_0.50_fp_teacher`
- `exceed_alpha_0.50_deploy`
- `encoder_input_scale_mean`
- `router_weight_scale_mean`
- `expert_weight_scale_mean`
- `decoder_act_scale_mean`
- `decoder_weight_scale_mean`
- `encoder_input_clip_rate`
- `decoder_output_clip_rate`

其中最关键的核心判断指标仍然是：

- `FVU`
- `fvu_fp_teacher`
- `fvu_deploy`
- `exceed_alpha_0.50`

## 12. 测试与验证策略

第一版至少需要以下验证：

### 12.1 配置测试

- `qat_full_w8a8` 能通过配置校验；
- 非法组合能正确报错。

### 12.2 Helper 单测

- activation fake quant shape / dtype 正常；
- weight fake quant 的 scale 维度正确；
- fake quant linear / expert einsum / decoder path 输出 shape 正常。

### 12.3 架构单测

- `product_key_expert_jumprelu` 的 tiny shape 前向能跑；
- backward 能跑；
- 不会在首步产生 NaN。

### 12.4 训练 smoke

先跑：

- `layers.[13].self_attn.q_proj`

确认：

- 能启动；
- 不在首步崩溃；
- 指标曲线有意义；
- 相比 BF16 baseline 没有灾难性退化。

## 13. 成功标准

第一版不要求比 BF16 更优，只要求：

- 训练稳定；
- 单层 smoke 可运行；
- `FVU` 没有明显灾难性退化；
- `deploy 指标` 可解释；
- 代码结构足够清晰，后续能扩展到真实 int8 导出。

## 14. 推荐实验顺序

建议按下面顺序推进：

1. `BF16 teacher + qat_full_w8a8 + layers.[13].self_attn.q_proj`
2. 对比：
   - BF16 baseline
   - 现有 `qat_io_int8`
   - 新的 `qat_full_w8a8`
3. 如果单层稳定，再扩到：
   - `layers.[0-13].self_attn.q_proj`
4. 如果 full QAT 结果稳定，再讨论：
   - 真实 int8 SAE 导出
   - CPU 推理 kernel 对接

## 15. 总结

这版设计的核心结论是：

- 不再继续把主要希望押在 teacher 量化上；
- 回到 `BF16 teacher`，把量化重点放回 SAE 自身；
- 第一版优先实现 **主干全 W8A8 QAT，离散选择逻辑继续浮点**；
- 这更符合未来 CPU 高性能部署的真实热点，也更适合作为真实 int8 SAE 的前置训练阶段。
