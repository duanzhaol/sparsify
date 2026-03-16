---
name: luturbo_experiment
description: LUTurbo 实验开发指南。在用户要求实现新实验、运行评估、或在 experiments/ 目录下工作时自动应用。提供 LUT 权重加载、数据集使用、指标计算等关键上下文。
keywords:
  - experiment
  - experiments
  - LUTurbo
  - LUT
  - SAE
  - exceed
  - CG
  - evaluation
  - eval
---

# LUTurbo Experiment Development Guide

You are implementing experiments for the LUTurbo project, which replaces LLM matrix
multiplications with lookup table operations using Sparse Autoencoders (SAE).

## Experiment Structure

```
experiments/
  <experiment_name>/
    __init__.py
    *.py                   # core logic modules
    run_eval.sh            # shell entry point (per model)
    results/
      <ModelName>/         # separate results by model
        *.json             # per-layer results
        summary.csv
        *.png              # plots
```

- Place experiments in `experiments/`, NOT `scripts/`.
- Make modules importable: `python -m experiments.<name>.<module>`.
- Separate results by model name (e.g. `results/Qwen3-0.6B/`, `results/Qwen3-4B/`).

## Key Paths

| Resource | Path |
|----------|------|
| Models | `/root/models/<ModelName>/` |
| **LUT weights** | `/root/models/<ModelName>/lut/` |
| LUT metadata | `/root/models/<ModelName>/lut/metadata.json` |
| Elbow thresholds | `/root/sparsify-ascend/thresholds/<ModelName>/` |
| Tokenized dataset | `/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048/` (80 shards) |
| LUTurbo docs | `/root/sparsify-ascend/LUTurbo-doc/` |
| Reference eval | `/root/sparsify-ascend/scripts/eval_exceed.py` |

## Model Configurations

| Model | Layers | hidden_size | N (latents) | K (active) | LUT ops |
|-------|--------|------------|-------------|------------|---------|
| Qwen3-0.6B | 28 | 1024 | 8192 | 128 | mlp, qkv, o_proj |
| Qwen3-4B | 36 | 2560 | 20480 | 256 | mlp, qkv (no o_proj) |
| Qwen3-8B | 36 | 4096 | ? | ? | check lut/ dir |

- **Read K from `metadata.json`**, never hardcode. Field: `sae_config.k_active`.
- All Qwen3 models share the same tokenizer; one tokenized dataset works for all.
- Layer sampling: pick 6-8 layers spread across the model (e.g. 0, 5, 10, 15, 20, 27 for 28-layer model).

## Critical Rules (Lessons from Past Mistakes)

### 1. Use LUT Weights, NOT Raw Checkpoints

LUT files are the **deployed** SAE weights. They differ significantly from raw training
checkpoints (max diff ~0.5). Always load from `/root/models/<Model>/lut/`.

Each `.lut.safetensors` contains: `encoder_weight`, `encoder_bias`, `decoder_weight`,
`decoder_bias`, `precomputed_products`, `bias_product`.

### 2. Hookpoint Convention — SAE Trains on INPUT

| LUT layer name | Hookpoint (captures input) |
|---------------|---------------------------|
| `layers.{i}.mlp.gate_up_proj` | `model.layers.{i}.mlp.up_proj` |
| `layers.{i}.self_attn.qkv_proj` | `model.layers.{i}.self_attn.q_proj` |
| `layers.{i}.self_attn.o_proj` | `model.layers.{i}.self_attn.o_proj` |

Use `register_forward_hook`, capture `inputs[0]`, not output.

### 3. Exceed Ratio — Absolute Error with Elbow Threshold

```python
# CORRECT:
exceed = (abs_error > tau * elbow_threshold).sum() / total_elements

# WRONG (gives ~70% instead of expected 10-30%):
exceed = (abs_error / abs_x > tau).sum() / total_elements
```

Threshold files:
- `thresholds_up.json` → MLP, key `layer_{i}/mlp_up_proj`
- `thresholds_q.json` → QKV, key `layer_{i}/self_attn_q_proj`
- `thresholds_o.json` → O proj, key `layer_{i}/self_attn_o_proj`

Sweep τ from 0.1 to 1.0 at 0.1 intervals.

### 4. Numerical Precision — bf16 Encode, fp32 Solve

```python
# Encode in SAE dtype for correct TopK selection
x_sae = x_orig.to(sae.dtype)  # bf16
top_acts, top_indices, _ = sae.encode(x_sae)

# ALL downstream computation in float32
x = x_orig.float()
W_dec_f32 = sae.W_dec.float()
b_dec_f32 = sae.b_dec.float()
top_acts_f32 = top_acts.float()
```

Note: `torch.linalg.svdvals` / `torch.linalg.lstsq` do not support bf16.

### 5. Dataset — Load ALL Shards

```python
from datasets import Dataset, concatenate_datasets
arrow_files = sorted(path.glob("data-*.arrow"))
shards = [Dataset.from_file(str(f)) for f in arrow_files]  # all 80 shards
ds = concatenate_datasets(shards)
```

Loading only the first shard biases the sample.

### 6. Metrics — Global Accumulation, Not Batch Averaging

```python
# WRONG: mean of per-batch metrics
fvu = mean([batch_sse / batch_var for batch in batches])

# CORRECT: accumulate raw sums globally
global_fvu = total_sse / total_variance
```

Track: SSE, sum(x), sum(x²), total_elements, total_samples per method.

### 7. Always Include an Exact Baseline

Use `torch.linalg.lstsq` as an upper bound to separate "method hasn't converged"
from "theoretical ceiling is low".

## Shell Script Template

```bash
#!/bin/bash
set -euo pipefail

MODEL=/root/models/<ModelName>
LUT_DIR=/root/models/<ModelName>/lut
THRESHOLD_DIR=/root/sparsify-ascend/thresholds/<ModelName>
DATASET=/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048
RESULT_DIR=experiments/<name>/results/<ModelName>

NUM_SAMPLES="${NUM_SAMPLES:-4096}"
LAYERS="${LAYERS:-"0 5 10 15 20 27"}"

mkdir -p "${RESULT_DIR}"

for LAYER in ${LAYERS}; do
    # MLP
    python -m experiments.<name>.<module> \
        --lut_dir "${LUT_DIR}" \
        --lut_layer "layers.${LAYER}.mlp.gate_up_proj" \
        --model "${MODEL}" \
        --hookpoint "model.layers.${LAYER}.mlp.up_proj" \
        --elbow_threshold_file "${THRESHOLD_DIR}/thresholds_up.json" \
        --elbow_key "layer_${LAYER}/mlp_up_proj" \
        --output "${RESULT_DIR}/layer${LAYER}_mlp.json"

    # QKV
    python -m experiments.<name>.<module> \
        --lut_dir "${LUT_DIR}" \
        --lut_layer "layers.${LAYER}.self_attn.qkv_proj" \
        --model "${MODEL}" \
        --hookpoint "model.layers.${LAYER}.self_attn.q_proj" \
        --elbow_threshold_file "${THRESHOLD_DIR}/thresholds_q.json" \
        --elbow_key "layer_${LAYER}/self_attn_q_proj" \
        --output "${RESULT_DIR}/layer${LAYER}_qkv.json"
done

# Summarize
python -m experiments.<name>.summarize \
    --results_dir "${RESULT_DIR}" \
    --output "${RESULT_DIR}/summary.csv"
```

## Pre-Implementation Checklist

Before writing experiment code, verify:

- [ ] Using LUT weights (not raw SAE checkpoint)?
- [ ] K value read from metadata.json (not hardcoded)?
- [ ] Encoding in SAE dtype, solving in float32?
- [ ] Loading all dataset shards (not just first)?
- [ ] Absolute error with elbow threshold (not relative error)?
- [ ] Global metric accumulation (not batch averaging)?
- [ ] Exact baseline included for validation?
- [ ] Results separated by model name?
- [ ] Hookpoints capture INPUT (not output)?
