# W8A8 Decoder Quantization Design

**Goal:** Extend the quantization research area with two decoder-side experiments for `product_key_expert_jumprelu` SAEs: a conservative decoder-weight-only path and a more aggressive decoder W8A8 path.

**Scope:**
- Reuse the existing full-encoder W8A8 evaluation as the baseline encoder path.
- Add one script that evaluates `left_router + right_router + expert_encoders + W_dec(weight-only)`.
- Add one script that evaluates `left_router + right_router + expert_encoders + W_dec + top_acts` with decoder-side W8A8 simulation.
- Keep `expert_encoder_bias`, `b_dec`, `log_threshold`, JumpReLU thresholding, and control-flow ops in floating point.
- Continue reporting `FVU` and `exceed_alpha_0.50`.

**Non-goals:**
- No real int8 kernel integration yet.
- No training-time quantization.
- No threshold/bias quantization in this round.

**Architecture:**
- Extend `quantization/quant_utils.py` with reusable sparse decoder simulation helpers.
- Keep two independent CLI scripts so decoder weight-only and decoder W8A8 runs stay easy to compare.
- Reuse the existing full-encoder W8A8 flow to produce sparse `top_acts` and `top_indices`, then swap only the decoder simulation step.

**Quantization strategy:**
- Decoder weights (`W_dec`): symmetric int8, per-latent-row static scales.
- Decoder weight-only mode: `top_acts` remain floating point.
- Decoder W8A8 mode: `top_acts` are quantized symmetrically per sparse token vector.
- Decoder reconstruction is simulated by gathering selected decoder rows from `W_dec`, applying quantized arithmetic, and summing over the active sparse support.

**Outputs:**
- Terminal summary table.
- `summary.json`, `summary.csv`, and `config.json` under `quantization/results/<timestamp>/`.

**Rationale:**
- After full-encoder W8A8 proved acceptable, the next highest-value expansion is `W_dec`, because it is the other large matrix in the SAE.
- Splitting decoder weight-only and decoder W8A8 lets us isolate whether decoder weights are easy to quantize before introducing activation quantization on the sparse code itself.
