# Qwen3 指南

本指南聚焦于将当前 Sparsify 代码库与 Qwen3 系列模型一起使用。

## LUTurbo 工作推荐的钩入点

当前训练器在模块输入上学习 SAE。对于面向 LUTurbo 的实验，最相关的钩入点通常是那些稍后将用 LUT 近似的线性投影的输入。

常见选择：

- `layers.X.self_attn.o_proj`
- `layers.X.self_attn.q_proj`
- `layers.X.mlp.up_proj`

这些与 `convert_sae_to_lut.py` 中的导出脚本自然对应，该脚本已包含 `qproj`、`oproj`、`upproj` 的映射，以及融合目标如 `qkv` 和 `gate_up`。

## 建议的起始配置

对于小型 Qwen3 模型如 `Qwen/Qwen3-0.6B`，一个实用的起点是：

- `sae.expansion_factor = 8`
- `sae.k = 128`（适用于隐藏层大小约 1024）
- `batch_size = 1`
- `grad_acc_steps = 8`
- `ctx_len = 2048`

仅在基础流水线稳定后才向上调整这些参数。

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

- 保持钩入点与 `convert_sae_to_lut.py` 识别的投影名称对齐
- 尽可能为每个投影族保存肘部阈值
- 保持运行名称与投影类型一致，以简化后续导出脚本

## 本指南不再涵盖的内容

Sparsify 的历史版本记录了替代钩子模式、低秩编码器、端到端 CE/KL 目标以及几个实验分支。这些不是当前推荐路径的一部分，已移至 `docs/archive/`。
