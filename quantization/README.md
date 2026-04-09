# Quantization Experiments

This folder contains isolated research scripts for SAE quantization work.

## Current scope

`eval_w8a8_expert_only.py` evaluates `product_key_expert_jumprelu` checkpoints by simulating W8A8 only on the `expert_encoders` matmul path.

What stays in floating point:
- `left_router`
- `right_router`
- `expert_encoder_bias`
- `log_threshold` / JumpReLU thresholding
- decoder path

What is quantized:
- weights: symmetric int8, per-row/per-latent
- activations: symmetric int8, per-token dynamic

This is a metric-impact study, not a real int8 kernel integration.

## Example

```bash
python quantization/eval_w8a8_expert_only.py \
  --checkpoint-root checkpoints/product_key_expert_jumprelu_qproj/product_key_expert_jumprelu_q_dp2_bs1_ga8_ef1_k32_20260407_001803/best \
  --hookpoints 'layers.[0-13].self_attn.q_proj' \
  --elbow-threshold-path thresholds/Qwen3-0.6B/thresholds_q.json \
  --num-samples 128 \
  --batch-size 1
```

## Outputs

Each run writes:
- `summary.json`
- `summary.csv`
- `config.json`

under `quantization/results/<timestamp>/` unless `--output-dir` is provided.
