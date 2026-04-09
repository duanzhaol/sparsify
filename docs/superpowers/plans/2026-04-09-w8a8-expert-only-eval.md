# W8A8 Expert-Only Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone quantization evaluation script that measures baseline vs W8A8-simulated `FVU` and `exceed_alpha_0.50` for `product_key_expert_jumprelu` SAE checkpoints.

**Architecture:** The implementation lives in a new `quantization/` folder. Quant math and evaluation helpers are separated so tests can validate the core logic without loading a full model. The CLI script reuses existing hookpoint expansion and checkpoint conventions from training.

**Tech Stack:** Python, PyTorch, HuggingFace Transformers/Datasets, pytest, JSON/CSV output.

---

### Task 1: Add failing tests for quantization helpers and evaluation helpers

**Files:**
- Create: `tests/test_quantization.py`
- Create: `quantization/__init__.py`

- [ ] Add tests for per-row weight quantization, per-token activation quantization, W8A8 simulation, hookpoint resolution, elbow-threshold mapping, and summary aggregation.
- [ ] Run `pytest tests/test_quantization.py -v` and confirm it fails because the helper modules do not exist yet.

### Task 2: Implement reusable quantization and evaluation helpers

**Files:**
- Create: `quantization/quant_utils.py`
- Create: `quantization/eval_utils.py`
- Modify: `quantization/__init__.py`
- Test: `tests/test_quantization.py`

- [ ] Implement symmetric int8 quantization helpers and W8A8 matmul simulation.
- [ ] Implement hookpoint expansion/matching, checkpoint discovery, elbow threshold mapping, and summary aggregation helpers.
- [ ] Run `pytest tests/test_quantization.py -v` and confirm helper tests pass.

### Task 3: Implement the CLI evaluation script

**Files:**
- Create: `quantization/eval_w8a8_expert_only.py`
- Create: `quantization/README.md`
- Test: `tests/test_quantization.py`

- [ ] Implement the CLI that loads model/dataset/checkpoints, captures activations, compares baseline vs W8A8 simulation, and writes result files.
- [ ] Document usage and current limitations in `quantization/README.md`.
- [ ] Run focused tests plus a light syntax check on the new module.

### Task 4: Verify end-to-end quality

**Files:**
- Test: `tests/test_quantization.py`
- Test: `quantization/eval_w8a8_expert_only.py`

- [ ] Run `pytest tests/test_quantization.py -v`.
- [ ] Run `python -m py_compile quantization/quant_utils.py quantization/eval_utils.py quantization/eval_w8a8_expert_only.py`.
- [ ] If the environment allows a small smoke run, execute the CLI on one layer with a tiny sample count.
