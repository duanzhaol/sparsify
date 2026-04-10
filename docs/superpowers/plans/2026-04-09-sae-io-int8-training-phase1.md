# SAE Int8 I/O QAT Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal I/O quantization-aware training path so SAE training explicitly simulates `int8 input + int8 output` while keeping model parameters and optimizer state in floating point.

**Architecture:** Introduce a small pure-PyTorch helper module for per-token activation fake quantization and deployment-side metric computation, then wire it into `Trainer` behind new `TrainConfig` flags. Keep SAE architecture implementations unchanged in Phase 1; `Trainer` owns the I/O quantization, dual-target loss, and new logging keys.

**Tech Stack:** Python, PyTorch, simple-parsing dataclasses, existing `sparsify` trainer/checkpoint/logging stack, pytest

---

## File Structure

**Create**

- `sparsify/train_quantization.py` — Phase 1 fake-quant helpers, clip-rate helpers, deploy-side metric helpers, and a small result container
- `tests/test_train_quantization.py` — unit tests for fake quantization, deploy metrics, and config validation
- `scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_io_int8.sh` — convenience launch script for Phase 1 I/O-QAT runs

**Modify**

- `sparsify/config.py` — add I/O quantization config fields and validation
- `sparsify/trainer.py` — wire fake quant input/output path, dual-target loss, extra metrics, and checkpoint-selection semantics
- `scripts/autoresearch_test.sh` — plumb new env vars into `torchrun -m sparsify`

**Verification**

- `pytest tests/test_train_quantization.py -q`
- `python -m py_compile sparsify/train_quantization.py sparsify/config.py sparsify/trainer.py`
- `bash -n scripts/autoresearch_test.sh scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_io_int8.sh`

---

### Task 1: Add pure fake-quant and deploy-metric helpers

**Files:**
- Create: `sparsify/train_quantization.py`
- Test: `tests/test_train_quantization.py`

- [ ] **Step 1: Write the failing helper tests**

```python
import pytest
import torch

from sparsify.train_quantization import (
    IOQuantMetrics,
    compute_exceed_ratio,
    compute_fvu_scalar,
    fake_quantize_activation_per_token,
    summarize_io_quant_batch,
)


def test_fake_quantize_activation_per_token_preserves_shape_and_reports_clip_rate():
    x = torch.tensor([[1.0, -2.0, 0.5], [0.0, 4.0, -4.0]], dtype=torch.float32)
    qdq, scales, clip_rate = fake_quantize_activation_per_token(x, num_bits=8)

    assert qdq.shape == x.shape
    assert scales.shape == (2, 1)
    assert clip_rate.ndim == 0
    assert 0.0 <= float(clip_rate) <= 1.0
    assert torch.allclose(qdq, x, atol=0.05)


def test_compute_fvu_scalar_matches_manual_reference():
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    recon = torch.tensor([[1.0, 1.0], [2.0, 4.0]], dtype=torch.float32)

    fvu = compute_fvu_scalar(target, recon)

    total_variance = (target - target.mean(0)).pow(2).sum()
    expected = ((target - recon).pow(2).sum() / total_variance).item()
    assert fvu.item() == pytest.approx(expected)


def test_summarize_io_quant_batch_returns_dual_target_metrics():
    target_fp = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    recon_fp = torch.tensor([[1.0, 1.5], [2.5, 4.0]], dtype=torch.float32)

    metrics = summarize_io_quant_batch(
        target_fp=target_fp,
        recon_fp=recon_fp,
        num_bits=8,
        alpha=0.5,
        elbow_value=0.5,
        deploy_weight=0.25,
    )

    assert isinstance(metrics, IOQuantMetrics)
    assert metrics.recon_deploy.shape == target_fp.shape
    assert metrics.target_deploy.shape == target_fp.shape
    assert metrics.fvu_fp_teacher.ndim == 0
    assert metrics.fvu_deploy.ndim == 0
    assert metrics.quant_floor.ndim == 0
    assert metrics.main_loss.ndim == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_train_quantization.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'sparsify.train_quantization'`

- [ ] **Step 3: Write the minimal helper module**

```python
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class IOQuantMetrics:
    recon_deploy: Tensor
    target_deploy: Tensor
    fvu_fp_teacher: Tensor
    fvu_deploy: Tensor
    quant_floor: Tensor
    exceed_fp_teacher: Tensor | None
    exceed_deploy: Tensor | None
    input_clip_rate: Tensor
    output_clip_rate: Tensor
    input_scale_mean: Tensor
    output_scale_mean: Tensor
    main_loss: Tensor


def fake_quantize_activation_per_token(x: Tensor, num_bits: int = 8) -> tuple[Tensor, Tensor, Tensor]:
    qmax = float((1 << (num_bits - 1)) - 1)
    absmax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
    normalized = x / scale
    clipped = normalized.clamp(-qmax, qmax)
    rounded = clipped.round()
    qdq = rounded * scale
    clip_rate = (normalized.abs() > qmax).float().mean()
    return qdq, scale, clip_rate


def compute_fvu_scalar(target: Tensor, recon: Tensor) -> Tensor:
    total_variance = (target - target.mean(0)).pow(2).sum().clamp_min(1e-12)
    return (target - recon).pow(2).sum() / total_variance


def compute_exceed_ratio(target: Tensor, recon: Tensor, threshold: float) -> Tensor:
    return (torch.abs(target - recon) > threshold).float().mean()


def summarize_io_quant_batch(
    target_fp: Tensor,
    recon_fp: Tensor,
    *,
    num_bits: int,
    alpha: float | None,
    elbow_value: float | None,
    deploy_weight: float,
) -> IOQuantMetrics:
    target_deploy, in_scales, in_clip = fake_quantize_activation_per_token(target_fp, num_bits=num_bits)
    recon_deploy, out_scales, out_clip = fake_quantize_activation_per_token(recon_fp, num_bits=num_bits)
    fvu_fp_teacher = compute_fvu_scalar(target_fp, recon_deploy)
    fvu_deploy = compute_fvu_scalar(target_deploy, recon_deploy)
    quant_floor = compute_fvu_scalar(target_fp, target_deploy)
    exceed_threshold = None if alpha is None or elbow_value is None else alpha * elbow_value
    exceed_fp = None if exceed_threshold is None else compute_exceed_ratio(target_fp, recon_deploy, exceed_threshold)
    exceed_deploy = None if exceed_threshold is None else compute_exceed_ratio(target_deploy, recon_deploy, exceed_threshold)
    main_loss = fvu_fp_teacher + deploy_weight * fvu_deploy
    return IOQuantMetrics(
        recon_deploy=recon_deploy,
        target_deploy=target_deploy,
        fvu_fp_teacher=fvu_fp_teacher,
        fvu_deploy=fvu_deploy,
        quant_floor=quant_floor,
        exceed_fp_teacher=exceed_fp,
        exceed_deploy=exceed_deploy,
        input_clip_rate=in_clip,
        output_clip_rate=out_clip,
        input_scale_mean=in_scales.mean(),
        output_scale_mean=out_scales.mean(),
        main_loss=main_loss,
    )
```

- [ ] **Step 4: Run the helper tests and make sure they pass**

Run: `pytest tests/test_train_quantization.py -q`

Expected: PASS for the helper tests added in this task

- [ ] **Step 5: Commit the helper module**

```bash
git add sparsify/train_quantization.py tests/test_train_quantization.py
git commit -m "Add SAE I/O fake quant helpers"
```

---

### Task 2: Add config flags and validation for I/O-QAT

**Files:**
- Modify: `sparsify/config.py`
- Modify: `tests/test_train_quantization.py`

- [ ] **Step 1: Add failing config tests**

```python
from sparsify.config import SparseCoderConfig, TrainConfig


def test_train_config_accepts_io_quant_defaults():
    cfg = TrainConfig(
        sae=SparseCoderConfig(),
        io_quant_mode="qat_io_int8",
        io_quant_bits=8,
        io_quant_granularity="per_token",
        io_quant_clip_mode="absmax",
        io_loss_mode="dual_target",
        io_loss_deploy_weight=0.25,
    )

    assert cfg.io_quant_mode == "qat_io_int8"
    assert cfg.io_quant_bits == 8


def test_train_config_rejects_invalid_io_quant_bits():
    with pytest.raises(ValueError, match="io_quant_bits"):
        TrainConfig(sae=SparseCoderConfig(), io_quant_bits=7)


def test_train_config_rejects_invalid_io_loss_mode():
    with pytest.raises(ValueError, match="io_loss_mode"):
        TrainConfig(sae=SparseCoderConfig(), io_loss_mode="bad_mode")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_train_quantization.py -q`

Expected: FAIL with `TypeError` about unexpected `TrainConfig` keyword arguments

- [ ] **Step 3: Extend `TrainConfig` with Phase 1 flags**

```python
io_quant_mode: str = "off"
"""Training-time I/O quantization mode: 'off' | 'qat_io_int8'."""

io_quant_bits: int = 8
"""Bit width for Phase 1 I/O fake quantization."""

io_quant_granularity: str = "per_token"
"""Activation quantization granularity. Phase 1 only supports 'per_token'."""

io_quant_clip_mode: str = "absmax"
"""Activation clipping mode. Phase 1 only supports 'absmax'."""

io_loss_mode: str = "dual_target"
"""Loss mode for I/O quantization: 'fp_teacher' | 'dual_target' | 'deploy_target'."""

io_loss_deploy_weight: float = 0.25
"""Weight applied to deploy-target FVU inside Phase 1 main loss."""
```

```python
valid_io_quant_modes = {"off", "qat_io_int8"}
if self.io_quant_mode not in valid_io_quant_modes:
    raise ValueError(f"io_quant_mode must be one of {sorted(valid_io_quant_modes)}")

if self.io_quant_bits != 8:
    raise ValueError(f"io_quant_bits must be 8 in Phase 1, got {self.io_quant_bits}")

if self.io_quant_granularity != "per_token":
    raise ValueError("Phase 1 only supports io_quant_granularity='per_token'")

if self.io_quant_clip_mode != "absmax":
    raise ValueError("Phase 1 only supports io_quant_clip_mode='absmax'")

valid_loss_modes = {"fp_teacher", "dual_target", "deploy_target"}
if self.io_loss_mode not in valid_loss_modes:
    raise ValueError(f"io_loss_mode must be one of {sorted(valid_loss_modes)}")

if self.io_loss_deploy_weight < 0:
    raise ValueError("io_loss_deploy_weight must be non-negative")
```

- [ ] **Step 4: Run config tests and make sure they pass**

Run: `pytest tests/test_train_quantization.py -q`

Expected: PASS for both helper tests and new config validation tests

- [ ] **Step 5: Commit the config changes**

```bash
git add sparsify/config.py tests/test_train_quantization.py
git commit -m "Add SAE I/O quantization config flags"
```

---

### Task 3: Wire I/O fake quantization into `Trainer`

**Files:**
- Modify: `sparsify/trainer.py`
- Modify: `tests/test_train_quantization.py`
- Verify: `sparsify/checkpoint.py` behavior remains compatible through existing `best_loss` usage

- [ ] **Step 1: Add a failing metric-shape regression test for the trainer-side summary helper**

```python
from sparsify.train_quantization import summarize_io_quant_batch


def test_summarize_io_quant_batch_respects_loss_modes():
    target_fp = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    recon_fp = torch.tensor([[1.0, 1.5], [2.5, 4.0]], dtype=torch.float32)

    dual = summarize_io_quant_batch(
        target_fp=target_fp,
        recon_fp=recon_fp,
        num_bits=8,
        alpha=None,
        elbow_value=None,
        deploy_weight=0.25,
    )
    assert dual.main_loss.item() == pytest.approx(
        dual.fvu_fp_teacher.item() + 0.25 * dual.fvu_deploy.item()
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_train_quantization.py -q`

Expected: FAIL because `summarize_io_quant_batch` does not yet support trainer-driven loss mode semantics cleanly enough for integration

- [ ] **Step 3: Update helpers and `Trainer` to use Phase 1 objective and metrics**

Add trainer import and mode selection:

```python
from .train_quantization import summarize_io_quant_batch
```

Inside the hook body, preserve the floating-point target and build quantized input:

```python
acts_fp = acts
if self.cfg.io_quant_mode == "qat_io_int8":
    acts_in, _, _ = fake_quantize_activation_per_token(
        acts_fp, num_bits=self.cfg.io_quant_bits
    )
else:
    acts_in = acts_fp

out = wrapped(
    x=acts_in,
    dead_mask=...,
)
```

Immediately after `out` is computed, build deploy-side metrics:

```python
io_metrics = None
if self.cfg.io_quant_mode == "qat_io_int8":
    io_metrics = summarize_io_quant_batch(
        target_fp=acts_fp,
        recon_fp=out.sae_out,
        num_bits=self.cfg.io_quant_bits,
        alpha=max(self.cfg.exceed_alphas) if self.cfg.exceed_alphas else None,
        elbow_value=self.elbow_thresholds.get(name),
        deploy_weight=self.cfg.io_loss_deploy_weight,
    )
```

Select the main objective:

```python
if io_metrics is None:
    main_loss = out.fvu
    main_metric = out.fvu
else:
    if self.cfg.io_loss_mode == "fp_teacher":
        main_loss = io_metrics.fvu_fp_teacher
    elif self.cfg.io_loss_mode == "deploy_target":
        main_loss = io_metrics.fvu_deploy
    else:
        main_loss = io_metrics.main_loss
    main_metric = main_loss
```

Use `main_loss` for backprop and checkpoint selection, but log both old and new metrics:

```python
loss = main_loss + self.cfg.auxk_alpha * out.auxk_loss
avg_losses[sae_key] = float(main_metric.detach())

if io_metrics is not None:
    avg_monitoring[sae_key]["fvu_fp_teacher"] += float(io_metrics.fvu_fp_teacher.detach() / denom)
    avg_monitoring[sae_key]["fvu_deploy"] += float(io_metrics.fvu_deploy.detach() / denom)
    avg_monitoring[sae_key]["quant_floor"] += float(io_metrics.quant_floor.detach() / denom)
    avg_monitoring[sae_key]["input_clip_rate"] += float(io_metrics.input_clip_rate.detach() / denom)
    avg_monitoring[sae_key]["output_clip_rate"] += float(io_metrics.output_clip_rate.detach() / denom)
    avg_monitoring[sae_key]["input_scale_mean"] += float(io_metrics.input_scale_mean.detach() / denom)
    avg_monitoring[sae_key]["output_scale_mean"] += float(io_metrics.output_scale_mean.detach() / denom)
```

Keep exceed metrics split by target space when quantization is enabled:

```python
if io_metrics is not None and io_metrics.exceed_fp_teacher is not None:
    avg_monitoring[sae_key][f"exceed_alpha_{alpha:.2f}_fp_teacher"] += ...
    avg_monitoring[sae_key][f"exceed_alpha_{alpha:.2f}_deploy"] += ...
```

- [ ] **Step 4: Run targeted verification for trainer-adjacent helpers**

Run:

```bash
pytest tests/test_train_quantization.py -q
python -m py_compile sparsify/train_quantization.py sparsify/config.py sparsify/trainer.py
```

Expected:

- `tests/test_train_quantization.py` passes
- `py_compile` exits `0`

- [ ] **Step 5: Commit the trainer integration**

```bash
git add sparsify/trainer.py sparsify/train_quantization.py tests/test_train_quantization.py
git commit -m "Wire SAE trainer to int8 I/O fake quantization"
```

---

### Task 4: Plumb CLI/env flags and add a runnable training script

**Files:**
- Modify: `scripts/autoresearch_test.sh`
- Create: `scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_io_int8.sh`

- [ ] **Step 1: Add a failing shell-level smoke expectation**

Document the new flags that must appear in the launch command:

```bash
--io_quant_mode qat_io_int8
--io_quant_bits 8
--io_quant_granularity per_token
--io_quant_clip_mode absmax
--io_loss_mode dual_target
--io_loss_deploy_weight 0.25
```

- [ ] **Step 2: Run syntax-only shell validation before edits**

Run: `bash -n scripts/autoresearch_test.sh`

Expected: PASS before modification, giving a clean baseline

- [ ] **Step 3: Add env plumbing and a Phase 1 launch script**

Add defaults near the top of `scripts/autoresearch_test.sh`:

```bash
IO_QUANT_MODE="${IO_QUANT_MODE:-off}"
IO_QUANT_BITS="${IO_QUANT_BITS:-8}"
IO_QUANT_GRANULARITY="${IO_QUANT_GRANULARITY:-per_token}"
IO_QUANT_CLIP_MODE="${IO_QUANT_CLIP_MODE:-absmax}"
IO_LOSS_MODE="${IO_LOSS_MODE:-dual_target}"
IO_LOSS_DEPLOY_WEIGHT="${IO_LOSS_DEPLOY_WEIGHT:-0.25}"
```

Add them to the `cmd` array:

```bash
  --io_quant_mode "${IO_QUANT_MODE}"
  --io_quant_bits "${IO_QUANT_BITS}"
  --io_quant_granularity "${IO_QUANT_GRANULARITY}"
  --io_quant_clip_mode "${IO_QUANT_CLIP_MODE}"
  --io_loss_mode "${IO_LOSS_MODE}"
  --io_loss_deploy_weight "${IO_LOSS_DEPLOY_WEIGHT}"
```

Create `scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_io_int8.sh`:

```bash
WANDB_PROJECT=qwen3-0.6B-product_key_expert_jumprelu-qproj-io-int8 \
    SAVE_DIR=checkpoints/product_key_expert_jumprelu_qproj_io_int8 \
    RUN_NAME=product_key_expert_jumprelu_q_io_int8 \
    MAX_TOKENS=100000000 \
    ARCHITECTURE=product_key_expert_jumprelu \
    K=32 \
    EXPANSION_FACTOR=1 \
    NUM_EXPERTS=512 \
    ACTIVE_EXPERTS=2 \
    LATENTS_PER_EXPERT=56 \
    OPTIMIZER=adam \
    LR=8e-4 \
    HOOKPOINTS='layers.[0-13].self_attn.q_proj' \
    BATCH_SIZE=1 \
    GRAD_ACC_STEPS=8 \
    MICRO_ACC_STEPS=1 \
    AUXK_ALPHA=0.03125 \
    DEAD_FEATURE_THRESHOLD=10000000 \
    USE_HADAMARD=0 \
    COMPILE_MODEL=1 \
    IO_QUANT_MODE=qat_io_int8 \
    IO_QUANT_BITS=8 \
    IO_QUANT_GRANULARITY=per_token \
    IO_QUANT_CLIP_MODE=absmax \
    IO_LOSS_MODE=dual_target \
    IO_LOSS_DEPLOY_WEIGHT=0.25 \
    bash scripts/autoresearch_test.sh
```

- [ ] **Step 4: Run shell validation**

Run:

```bash
bash -n scripts/autoresearch_test.sh
bash -n scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_io_int8.sh
```

Expected: both commands exit `0`

- [ ] **Step 5: Commit script plumbing**

```bash
git add scripts/autoresearch_test.sh scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_io_int8.sh
git commit -m "Add Phase 1 SAE int8 I/O training launch script"
```

---

### Task 5: Run final Phase 1 verification and document the dry-run commands

**Files:**
- Modify: `docs/superpowers/plans/2026-04-09-sae-io-int8-training-phase1.md` (check off completed steps during execution)
- Verify: working tree and CLI help output

- [ ] **Step 1: Run the focused unit test suite**

Run: `pytest tests/test_train_quantization.py -q`

Expected: PASS

- [ ] **Step 2: Run syntax and import verification**

Run:

```bash
python -m py_compile sparsify/train_quantization.py sparsify/config.py sparsify/trainer.py
bash -n scripts/autoresearch_test.sh
bash -n scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_io_int8.sh
```

Expected: all commands exit `0`

- [ ] **Step 3: Verify CLI surfaces the new flags**

Run:

```bash
python -m sparsify --help | rg "io_quant|io_loss"
```

Expected:

- help output includes `--io_quant_mode`
- help output includes `--io_quant_bits`
- help output includes `--io_loss_mode`
- help output includes `--io_loss_deploy_weight`

- [ ] **Step 4: Verify git state before handoff**

Run:

```bash
git status --short
git log --oneline -n 5
```

Expected:

- only intended files are modified or the worktree is clean after commits
- recent history includes the four task commits from this plan

- [ ] **Step 5: Optional smoke command for first real run**

```bash
IO_QUANT_MODE=qat_io_int8 \
IO_QUANT_BITS=8 \
IO_QUANT_GRANULARITY=per_token \
IO_QUANT_CLIP_MODE=absmax \
IO_LOSS_MODE=dual_target \
IO_LOSS_DEPLOY_WEIGHT=0.25 \
MAX_TOKENS=200000 \
SAVE_EVERY=10 \
RUN_NAME=smoke_io_int8 \
bash scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_io_int8.sh
```

Expected:

- training starts successfully
- metrics logs include `fvu_fp_teacher`, `fvu_deploy`, and `quant_floor`

---

## Self-Review

### Spec coverage

- `sparsify/config.py` task covers new Phase 1 knobs from the spec
- `sparsify/train_quantization.py` task covers fake quant helper reuse and metric split
- `sparsify/trainer.py` task covers the new I/O-QAT data path, loss, and logging
- shell plumbing task covers repo-local execution entry points

### Placeholder scan

- No `TODO`, `TBD`, or “similar to above” placeholders remain
- Each code-edit step includes a concrete code block and a concrete command

### Type consistency

- Config names are consistent across plan tasks:
  - `io_quant_mode`
  - `io_quant_bits`
  - `io_quant_granularity`
  - `io_quant_clip_mode`
  - `io_loss_mode`
  - `io_loss_deploy_weight`
- Helper names are reused consistently:
  - `fake_quantize_activation_per_token`
  - `summarize_io_quant_batch`
  - `IOQuantMetrics`

