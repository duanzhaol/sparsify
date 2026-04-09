# W8A8 Full-Encoder Quantization Design

**Goal:** Extend the isolated quantization research area so it can evaluate the metric impact of simulating W8A8 on the full encoder path of `product_key_expert_jumprelu` SAEs.

**Scope:**
- Reuse the standalone `quantization/` folder and result format from the expert-only study.
- Add a new evaluation script for baseline vs full-encoder W8A8 simulation.
- Quantize `left_router`, `right_router`, and `expert_encoders`.
- Keep `expert_encoder_bias`, `log_threshold`, JumpReLU thresholding, and the decoder path in floating point.
- Continue reporting `FVU` and `exceed_alpha_0.50` so results stay directly comparable with the expert-only run.

**Non-goals:**
- No real int8 kernel integration yet.
- No training-time quantization.
- No router-consistency or latency benchmarking in this first full-encoder pass.
- No modification of the main training path.

**Architecture:**
- Extend `quantization/quant_utils.py` with reusable symmetric-int8 linear helpers for router layers.
- Keep `quantization/eval_utils.py` as the shared home for hookpoint matching, checkpoint resolution, threshold loading, and summary aggregation.
- Add `quantization/eval_w8a8_full_encoder.py` that mirrors the expert-only CLI but quantizes both router logits and expert-local matmul.
- Keep tests focused on helper behavior and script-adjacent logic rather than a full end-to-end model run.

**Quantization strategy:**
- Router weights: symmetric int8, per-output-row static scales.
- Router activations: symmetric int8, per-token dynamic scales.
- Expert weights: symmetric int8, per-row/per-latent static scales.
- Expert activations: symmetric int8, per-token dynamic scales.
- Accumulation: int32 simulation, then dequantize to float before top-k routing, bias addition, JumpReLU gating, and decoding.

**Outputs:**
- Terminal summary table.
- `summary.json`, `summary.csv`, and `config.json` under `quantization/results/<timestamp>/`.

**Current rationale:**
- The prior expert-only W8A8 run on `layers.[0-13].self_attn.q_proj` showed negligible drift.
- The next highest-value experiment is to quantify whether router quantization remains similarly stable while keeping the rest of the pipeline unchanged.
