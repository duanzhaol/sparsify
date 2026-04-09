# W8A8 Decoder Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two standalone quantization evaluation scripts that extend full-encoder W8A8 evaluation with decoder weight-only W8 and decoder W8A8 simulation for `product_key_expert_jumprelu` SAE checkpoints.

**Architecture:** Shared sparse decoder quantization helpers live in `quantization/quant_utils.py`. Two separate CLI entrypoints keep the conservative decoder-weight-only run and the more aggressive decoder-W8A8 run independently reproducible while reusing the same encoder-side W8A8 path.

**Tech Stack:** Python, PyTorch, HuggingFace Transformers/Datasets, pytest, JSON/CSV output.

---

### Task 1: Add failing tests for decoder quantization helpers

**Files:**
- Modify: `tests/test_quantization.py`
- Test: `tests/test_quantization.py`

- [ ] Add a test for sparse decoder weight-only simulation against a float reference decode.
- [ ] Add a test for sparse decoder W8A8 simulation against a float reference decode.
- [ ] Run `pytest tests/test_quantization.py -q` and confirm the new tests fail for the expected missing symbols.

### Task 2: Implement shared decoder quantization helpers

**Files:**
- Modify: `quantization/quant_utils.py`
- Test: `tests/test_quantization.py`

- [ ] Implement a reusable helper for decoder W8 weight-only sparse reconstruction.
- [ ] Implement a reusable helper for decoder W8A8 sparse reconstruction.
- [ ] Run `pytest tests/test_quantization.py -q` and confirm helper tests pass.

### Task 3: Implement the two decoder evaluation CLIs

**Files:**
- Create: `quantization/eval_w8a8_full_encoder_w8_decoder.py`
- Create: `quantization/eval_w8a8_full_encoder_w8a8_decoder.py`
- Modify: `quantization/README.md`

- [ ] Implement the decoder-weight-only CLI by reusing the full-encoder W8A8 path and swapping in W8 sparse decode.
- [ ] Implement the decoder-W8A8 CLI by reusing the same encoder path and applying W8A8 sparse decode.
- [ ] Update `quantization/README.md` with usage examples and scope descriptions for both new scripts.

### Task 4: Verify end-to-end quality

**Files:**
- Test: `tests/test_quantization.py`
- Test: `quantization/eval_w8a8_full_encoder_w8_decoder.py`
- Test: `quantization/eval_w8a8_full_encoder_w8a8_decoder.py`

- [ ] Run `pytest tests/test_quantization.py -q`.
- [ ] Run `python -m py_compile quantization/quant_utils.py quantization/eval_w8a8_full_encoder.py quantization/eval_w8a8_full_encoder_w8_decoder.py quantization/eval_w8a8_full_encoder_w8a8_decoder.py`.
- [ ] If the environment allows a small smoke run, execute both new CLIs on one layer with a tiny sample count.
