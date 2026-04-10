# Qwen3 指南

本指南说明如何在当前代码主线下，将 Sparsify 用于 Qwen3 系列模型。

## LUTurbo 场景推荐的 hookpoint

当前实现默认在模块输入上训练 SAE。对于 LUTurbo 相关实验，应优先关注后续要用 LUT 近似的线性投影输入。

常见选择：

- `layers.X.self_attn.o_proj`
- `layers.X.self_attn.q_proj`
- `layers.X.mlp.up_proj`

其中 `q_proj` 与 `up_proj` 可直接对应 `scripts/export/export_product_key_expert_jumprelu_lut.py` 的自动发现逻辑；导出脚本会进一步融合为 `qkv` 与 `gate_up` 两类运行时目标。

## 建议的起始配置

对于小型 Qwen3 模型（如 `Qwen/Qwen3-0.6B`），可从以下配置起步：

- `sae.expansion_factor = 8`
- `sae.k = 128`（适用于隐藏层大小约 1024）
- `batch_size = 1`
- `grad_acc_steps = 8`
- `ctx_len = 2048`

建议先用这组参数跑通流程，确认稳定后再逐步放大。

## 示例：训练 `o_proj` 输入 SAE

```bash
python -m sparsify Qwen/Qwen3-0.6B HuggingFaceFW/fineweb \
  --data_args "name=sample-10BT" \
  --text_column text \
  --hookpoints "layers.[7,14].self_attn.o_proj" \
  --batch_size 1 \
  --grad_acc_steps 8 \
  --ctx_len 2048 \
  --sae.expansion_factor 8 \
  --sae.k 128 \
  --run_name qwen3-oproj
```

## 示例：训练 `up_proj` 输入 SAE

```bash
python -m sparsify Qwen/Qwen3-0.6B HuggingFaceFW/fineweb \
  --data_args "name=sample-10BT" \
  --text_column text \
  --hookpoints "layers.[7,14].mlp.up_proj" \
  --batch_size 1 \
  --grad_acc_steps 8 \
  --ctx_len 2048 \
  --sae.expansion_factor 8 \
  --sae.k 128 \
  --run_name qwen3-upproj
```

## 层选择建议

对于探索性运行：

- 从少量层开始
- 优先选择早期、中期和后期深度的代表性层
- 在扩展到所有层之前确认重建质量

## 导出导向建议

如果最终目标是 LUT 转换：

- 保持 hookpoint 与导出脚本识别的 `q_proj` / `up_proj` 训练目标一致
- 对拆分训练的层段保持相同架构配置，便于后续 merge
- 运行名称尽量包含投影类型，方便导出脚本自动检索

## 本指南不再涵盖的内容

历史版本里提到的替代 hook 模式、低秩编码器、端到端 CE/KL 目标和部分实验分支，不属于当前推荐路径，已移至 `docs/archive/`。
