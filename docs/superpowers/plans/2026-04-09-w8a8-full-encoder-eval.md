# W8A8 Full-Encoder Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone quantization evaluation script that measures baseline vs full-encoder W8A8-simulated `FVU` and `exceed_alpha_0.50` for `product_key_expert_jumprelu` SAE checkpoints.

**Architecture:** The implementation stays inside `quantization/` and mirrors the expert-only tooling so runs remain easy to compare. Shared quantization helpers cover both router linear layers and expert-local matmuls, while the new CLI script expands the simulated quantized scope to `left_router + right_router + expert_encoders`.

**Tech Stack:** Python, PyTorch, HuggingFace Transformers/Datasets, pytest, JSON/CSV output.

---

### Task 1: Add failing tests for full-encoder quantization helpers

**Files:**
- Modify: `tests/test_quantization.py`
- Test: `tests/test_quantization.py`

- [ ] Add tests for simulated W8A8 linear layers used by router quantization.
- [ ] Add tests for any new summary/config helper behavior introduced by the full-encoder script.
- [ ] Run `pytest tests/test_quantization.py -q` and confirm the new tests fail for the expected missing behavior.

### Task 2: Implement reusable helper support for router quantization

**Files:**
- Modify: `quantization/quant_utils.py`
- Test: `tests/test_quantization.py`

- [ ] Implement a reusable W8A8 linear simulation helper that accepts batched activation vectors plus standard linear-layer weights.
- [ ] Reuse the existing symmetric quantization conventions so router and expert paths share the same arithmetic assumptions.
- [ ] Run `pytest tests/test_quantization.py -q` and confirm helper tests pass.

### Task 3: Implement the full-encoder CLI evaluation script

**Files:**
- Create: `quantization/eval_w8a8_full_encoder.py`
- Modify: `quantization/README.md`
- Test: `tests/test_quantization.py`

- [ ] Implement the CLI that loads model/dataset/checkpoints, captures activations, evaluates baseline vs full-encoder W8A8 simulation, and writes result files.
- [ ] Keep the CLI argument surface close to `quantization/eval_w8a8_expert_only.py` so usage stays familiar.
- [ ] Document the new script and clarify the difference between expert-only and full-encoder evaluation modes in `quantization/README.md`.

### Task 4: Verify end-to-end quality

**Files:**
- Test: `tests/test_quantization.py`
- Test: `quantization/eval_w8a8_full_encoder.py`

- [ ] Run `pytest tests/test_quantization.py -q`.
- [ ] Run `python -m py_compile quantization/quant_utils.py quantization/eval_utils.py quantization/eval_w8a8_expert_only.py quantization/eval_w8a8_full_encoder.py`.
- [ ] If the environment allows a small smoke run, execute the full-encoder CLI on one layer with a tiny sample count.
