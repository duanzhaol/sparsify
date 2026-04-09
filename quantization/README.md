# Quantization Experiments

This folder contains isolated research scripts for SAE quantization work.

## Current scope

`eval_w8a8_expert_only.py` evaluates `product_key_expert_jumprelu` checkpoints by simulating W8A8 only on the `expert_encoders` matmul path.

Expert-only mode:
- quantized: `expert_encoders`
- floating point: `left_router`, `right_router`, `expert_encoder_bias`, `log_threshold` / JumpReLU thresholding, decoder path

`eval_w8a8_full_encoder.py` extends the simulated quantized scope to the full encoder path.

Full-encoder mode:
- quantized: `left_router`, `right_router`, `expert_encoders`
- floating point: `expert_encoder_bias`, `log_threshold` / JumpReLU thresholding, decoder path

`eval_w8a8_full_encoder_w8_decoder.py` extends the full-encoder run with decoder weight-only quantization.

Full-encoder + decoder-W8 mode:
- quantized: `left_router`, `right_router`, `expert_encoders`, `W_dec`
- floating point: `top_acts`, `expert_encoder_bias`, `b_dec`, `log_threshold` / JumpReLU thresholding

`eval_w8a8_full_encoder_w8a8_decoder.py` extends the full-encoder run with decoder W8A8 simulation.

Full-encoder + decoder-W8A8 mode:
- quantized: `left_router`, `right_router`, `expert_encoders`, `W_dec`, sparse `top_acts`
- floating point: `expert_encoder_bias`, `b_dec`, `log_threshold` / JumpReLU thresholding

Quantization arithmetic:
- weights: symmetric int8, per-row/per-latent
- activations: symmetric int8, per-token dynamic

This is a metric-impact study, not a real int8 kernel integration.

## Example

Expert-only:

```bash
python quantization/eval_w8a8_expert_only.py \
  --checkpoint-root checkpoints/product_key_expert_jumprelu_qproj/product_key_expert_jumprelu_q_dp2_bs1_ga8_ef1_k32_20260407_001803/best \
  --hookpoints 'layers.[0-13].self_attn.q_proj' \
  --elbow-threshold-path thresholds/Qwen3-0.6B/thresholds_q.json \
  --num-samples 128 \
  --batch-size 1
```

Full encoder:

```bash
python quantization/eval_w8a8_full_encoder.py \
  --checkpoint-root checkpoints/product_key_expert_jumprelu_qproj/product_key_expert_jumprelu_q_dp2_bs1_ga8_ef1_k32_20260407_001803/best \
  --hookpoints 'layers.[0-13].self_attn.q_proj' \
  --elbow-threshold-path thresholds/Qwen3-0.6B/thresholds_q.json \
  --num-samples 128 \
  --batch-size 1
```

Full encoder + decoder W8:

```bash
python quantization/eval_w8a8_full_encoder_w8_decoder.py \
  --checkpoint-root checkpoints/product_key_expert_jumprelu_qproj/product_key_expert_jumprelu_q_dp2_bs1_ga8_ef1_k32_20260407_001803/best \
  --hookpoints 'layers.[0-13].self_attn.q_proj' \
  --elbow-threshold-path thresholds/Qwen3-0.6B/thresholds_q.json \
  --num-samples 128 \
  --batch-size 1
```

Full encoder + decoder W8A8:

```bash
python quantization/eval_w8a8_full_encoder_w8a8_decoder.py \
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
