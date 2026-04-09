# W8A8 Expert-Only Quantization Design

**Goal:** Add an isolated quantization research area that can evaluate the metric impact of simulating W8A8 only on the `expert_encoders` matmul path for `product_key_expert_jumprelu` SAEs.

**Scope:**
- Create a standalone `quantization/` folder.
- Add a simple evaluation script that supports single or multiple hookpoints using the same range-pattern syntax as training scripts.
- Compare baseline vs W8A8-simulated `FVU` and `exceed_alpha_0.50` on a fixed sample subset from the existing tokenized dataset.
- Keep router, bias, and JumpReLU threshold logic in floating point; quantize only the expert matmul path.

**Non-goals:**
- No real int8 kernel integration yet.
- No training-time quantization.
- No modification of the main training path.

**Architecture:**
- `quantization/quant_utils.py` holds symmetric int8 quantization helpers and W8A8 matmul simulation.
- `quantization/eval_utils.py` holds hookpoint expansion, checkpoint discovery, elbow-threshold matching, and metrics aggregation helpers.
- `quantization/eval_w8a8_expert_only.py` loads model, dataset, and SAEs; captures activations; evaluates baseline and W8A8-simulated outputs; writes JSON/CSV summaries.
- Tests focus on the quantization math and helper behavior rather than end-to-end model execution.

**Quantization strategy:**
- Weight quantization: symmetric int8, per-row/per-latent static scales over `expert_encoders[..., d_in]`.
- Activation quantization: symmetric int8, per-token dynamic scales over `flat_x[d_in]`.
- Accumulation: int32 simulation, then dequantize to float for downstream bias/gating/top-k.

**Outputs:**
- Terminal summary table.
- `summary.json`, `summary.csv`, and `config.json` under `quantization/results/<timestamp>/`.

**Current validation status:**
- A 14-layer `q_proj` evaluation on `layers.[0-13].self_attn.q_proj` with 1024 samples has been recorded in `docs/superpowers/specs/2026-04-09-w8a8-expert-only-summary.md`.
- That run shows negligible metric drift for expert-only W8A8 PTQ, which supports continuing with this direction before considering training-time quantization.
