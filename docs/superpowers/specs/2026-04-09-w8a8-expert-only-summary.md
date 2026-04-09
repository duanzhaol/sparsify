# W8A8 Expert-Only Quantization Summary

## Purpose

This note records the first larger-scale validation result for the isolated W8A8 post-training quantization study on `product_key_expert_jumprelu` SAE checkpoints.

The goal of this experiment was to answer a narrow question first: if we quantize only the `expert_encoders` matmul path to W8A8 and leave routing, bias, thresholding, and decoding in floating point, do the reconstruction metrics degrade meaningfully?

## Evaluated setup

- Script: `quantization/eval_w8a8_expert_only.py`
- Model: `/root/models/Qwen3-0.6B`
- Dataset: `/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048`
- Checkpoint root: `checkpoints/product_key_expert_jumprelu_qproj/product_key_expert_jumprelu_q_dp2_bs1_ga8_ef1_k32_20260406_221636/best`
- Hookpoints: `layers.[0-13].self_attn.q_proj`
- Threshold file: `thresholds/Qwen3-0.6B/thresholds_q.json`
- Sample count: `1024`
- Batch size: `1`
- Device: `cuda:0`
- Result artifact directory: `quantization/results/20260409_155713`

## Quantization scope

This study quantizes only the expert matmul path:

- Quantized:
  - `expert_encoders`
  - weights: symmetric int8, per-row static scale
  - activations: symmetric int8, per-token dynamic scale
- Kept in floating point:
  - `left_router`
  - `right_router`
  - `expert_encoder_bias`
  - JumpReLU threshold logic
  - decoder path

This is a metric-impact simulation, not a deployment kernel benchmark.

## Command

```bash
python quantization/eval_w8a8_expert_only.py \
  --checkpoint-root checkpoints/product_key_expert_jumprelu_qproj/product_key_expert_jumprelu_q_dp2_bs1_ga8_ef1_k32_20260406_221636/best \
  --hookpoints 'layers.[0-13].self_attn.q_proj' \
  --elbow-threshold-path thresholds/Qwen3-0.6B/thresholds_q.json \
  --num-samples 1024 \
  --batch-size 1
```

## Aggregate result

- Number of evaluated hookpoints: `14`
- Mean `FVU` delta: `+0.00004316`
- Mean `exceed_alpha_0.50` delta: `+0.00004222`
- Worst `FVU` hookpoint: `layers.13.self_attn.q_proj`
- Worst `exceed_alpha_0.50` hookpoint: `layers.13.self_attn.q_proj`

These deltas are extremely small. For this checkpoint family and this metric suite, expert-only W8A8 PTQ looks very promising.

## Per-hookpoint result

```text
hookpoint                                                 fvu_base   fvu_w8a8      delta exceed_alpha_0.50_base exceed_alpha_0.50_w8a8      delta
-------------------------------------------------------------------------------------------------------------------------------------------------
layers.0.self_attn.q_proj                                 0.041858   0.041955   0.000097           0.042893           0.042973   0.000079
layers.1.self_attn.q_proj                                 0.114717   0.114704  -0.000012           0.117915           0.117919   0.000004
layers.2.self_attn.q_proj                                 0.135097   0.135167   0.000069           0.155376           0.155424   0.000048
layers.3.self_attn.q_proj                                 0.188581   0.188619   0.000037           0.205522           0.205592   0.000070
layers.4.self_attn.q_proj                                 0.194873   0.194899   0.000025           0.240904           0.240919   0.000014
layers.5.self_attn.q_proj                                 0.187294   0.187348   0.000054           0.216245           0.216305   0.000059
layers.6.self_attn.q_proj                                 0.240321   0.240351   0.000030           0.296241           0.296274   0.000033
layers.7.self_attn.q_proj                                 0.255928   0.255935   0.000006           0.324274           0.324295   0.000021
layers.8.self_attn.q_proj                                 0.307073   0.307097   0.000023           0.361075           0.361110   0.000035
layers.9.self_attn.q_proj                                 0.299410   0.299422   0.000012           0.362374           0.362398   0.000024
layers.10.self_attn.q_proj                                0.300086   0.300121   0.000035           0.386498           0.386526   0.000028
layers.11.self_attn.q_proj                                0.297008   0.297082   0.000074           0.379995           0.380054   0.000059
layers.12.self_attn.q_proj                                0.338732   0.338767   0.000036           0.393126           0.393154   0.000029
layers.13.self_attn.q_proj                                0.324919   0.325036   0.000117           0.401269           0.401356   0.000087
```

## Interpretation

The observed regression is negligible across all 14 evaluated `q_proj` layers:

- `FVU` drift stays around the `1e-5` to `1e-4` level.
- `exceed_alpha_0.50` drift also stays around the `1e-5` to `1e-4` level.
- No layer shows a qualitatively concerning jump.
- One layer (`layers.1.self_attn.q_proj`) even shows a tiny negative `FVU` delta, which is consistent with quantization noise and should not be interpreted as a systematic improvement.

The main conclusion is that the current expert-only PTQ formulation preserves reconstruction quality well enough to justify further work in this direction.

## What this result does and does not prove

This result supports:

- continuing expert-only PTQ exploration,
- expanding evaluation to more checkpoints and `up_proj`,
- delaying training-time quantization work until PTQ ceilings are clearer.

This result does not yet prove:

- that full encoder W8A8 will remain this stable,
- that end-to-end inference latency improves,
- that a real deployment int8 kernel will exactly match the simulation,
- that routing or threshold-path quantization is safe.

## Recommended next steps

1. Run the same evaluation on `layers.[14-27].mlp.up_proj`.
2. Repeat on additional checkpoints to check whether the conclusion generalizes beyond one run.
3. If the metric trend remains stable, consider widening the quantized scope from expert-only matmul to more of the encoder path.
4. After metric validation, measure actual inference speed and memory changes with a real kernel path rather than simulation alone.
