# Qwen3 TorchAO W8A8 Activation Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal path that lets SAE training consume online activations from a real `Qwen3 + torchao W8A8` backbone while keeping SAE itself in floating-point training mode.

**Architecture:** Keep the existing SAE training loop intact as much as possible. Add a small torchao-backed model loader plus minimal config and script plumbing so `Trainer` can continue using the current hook-based activation capture path on either the default BF16 model or a quantized `Qwen3` backbone.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, torchao, simple-parsing dataclasses, pytest, existing `sparsify` trainer/scripts

---

## File Structure

**Create**

- `sparsify/quantized_backbone.py` — minimal torchao-backed `Qwen3` model loading helpers plus hookpoint validation helpers
- `tests/test_quantized_backbone.py` — unit tests for torchao config construction, error handling, and hookpoint validation

**Modify**

- `sparsify/config.py` — add activation-source config fields and validation
- `sparsify/__main__.py` — choose BF16 or W8A8 teacher model at load time
- `scripts/autoresearch_test.sh` — plumb activation-source env vars into the training CLI

**Verification**

- `pytest tests/test_train_quantization.py tests/test_quantized_backbone.py -q`
- `python -m py_compile sparsify/config.py sparsify/__main__.py sparsify/quantized_backbone.py`
- `bash -n scripts/autoresearch_test.sh`

---

### Task 1: Add activation-source config fields and validation

**Files:**
- Modify: `sparsify/config.py`
- Modify: `tests/test_train_quantization.py`

- [ ] **Step 1: Write the failing config tests**

```python
def test_train_config_accepts_w8a8_activation_source():
    cfg = TrainConfig(
        sae=SparseCoderConfig(),
        activation_source="w8a8_backbone",
        activation_backbone_path="/tmp/qwen3-w8a8",
    )

    assert cfg.activation_source == "w8a8_backbone"
    assert cfg.activation_backbone_path == "/tmp/qwen3-w8a8"


def test_train_config_rejects_unknown_activation_source():
    with pytest.raises(ValueError, match="activation_source"):
        TrainConfig(sae=SparseCoderConfig(), activation_source="bad_source")


def test_train_config_requires_backbone_path_for_w8a8_source():
    with pytest.raises(ValueError, match="activation_backbone_path"):
        TrainConfig(sae=SparseCoderConfig(), activation_source="w8a8_backbone")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_train_quantization.py -q`

Expected: FAIL with `TypeError` for unknown `TrainConfig` keyword arguments

- [ ] **Step 3: Add the minimal config fields and validation**

```python
activation_source: str = "hf_bf16"
"""Activation source: 'hf_bf16' | 'w8a8_backbone'."""

activation_backbone_path: str | None = None
"""Optional model path for a quantized activation backbone."""

activation_threshold_path: str | None = None
"""Optional threshold file matched to the activation source."""
```

```python
valid_activation_sources = {"hf_bf16", "w8a8_backbone"}
if self.activation_source not in valid_activation_sources:
    raise ValueError(
        f"activation_source must be one of {sorted(valid_activation_sources)}"
    )

if self.activation_source == "w8a8_backbone" and not self.activation_backbone_path:
    raise ValueError(
        "activation_backbone_path must be set when activation_source='w8a8_backbone'"
    )

if self.activation_threshold_path and not Path(self.activation_threshold_path).exists():
    raise ValueError(
        f"Activation threshold file not found: {self.activation_threshold_path}"
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_train_quantization.py -q`

Expected: PASS with the new config tests included

- [ ] **Step 5: Commit**

```bash
git add sparsify/config.py tests/test_train_quantization.py
git commit -m "Add activation source config for W8A8 teacher"
```

---

### Task 2: Add a minimal torchao-backed quantized backbone helper

**Files:**
- Create: `sparsify/quantized_backbone.py`
- Create: `tests/test_quantized_backbone.py`

- [ ] **Step 1: Write the failing helper tests**

```python
import pytest

from sparsify.quantized_backbone import (
    build_torchao_int8_quantization_config,
    validate_requested_hookpoints,
)


def test_build_torchao_int8_quantization_config_has_expected_repr():
    cfg = build_torchao_int8_quantization_config()
    assert cfg is not None


def test_validate_requested_hookpoints_rejects_missing_entries():
    available = ["layers.0.self_attn.q_proj", "layers.1.self_attn.q_proj"]
    with pytest.raises(ValueError, match="Missing hookpoints"):
        validate_requested_hookpoints(
            ["layers.0.self_attn.q_proj", "layers.9.self_attn.q_proj"],
            available,
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_quantized_backbone.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'sparsify.quantized_backbone'`

- [ ] **Step 3: Write the minimal helper implementation**

```python
from __future__ import annotations

from fnmatch import fnmatchcase

from transformers import AutoModel, TorchAoConfig


def build_torchao_int8_quantization_config() -> TorchAoConfig:
    return TorchAoConfig("int8_dynamic_activation_int8_weight")


def resolve_available_hookpoints(model) -> list[str]:
    return [name for name, _ in model.base_model.named_modules()]


def validate_requested_hookpoints(requested: list[str], available: list[str]) -> None:
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"Missing hookpoints in quantized backbone: {missing}")
```

Add the loader helper in the same file:

```python
def load_torchao_w8a8_model(
    model_name_or_path: str,
    *,
    device_map,
    revision: str | None,
    torch_dtype,
    token: str | None,
):
    quantization_config = build_torchao_int8_quantization_config()
    return AutoModel.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        revision=revision,
        torch_dtype=torch_dtype,
        token=token,
        quantization_config=quantization_config,
    )
```

- [ ] **Step 4: Add one more test for hookpoint pattern coverage**

```python
def test_validate_requested_hookpoints_accepts_exact_matches():
    available = ["layers.0.self_attn.q_proj", "layers.1.self_attn.q_proj"]
    validate_requested_hookpoints(["layers.0.self_attn.q_proj"], available)
```

- [ ] **Step 5: Run the helper tests and verify they pass**

Run: `pytest tests/test_quantized_backbone.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sparsify/quantized_backbone.py tests/test_quantized_backbone.py
git commit -m "Add torchao W8A8 backbone helper"
```

---

### Task 3: Wire W8A8 backbone loading into the training entrypoint

**Files:**
- Modify: `sparsify/__main__.py`
- Modify: `tests/test_quantized_backbone.py`

- [ ] **Step 1: Add a failing loader-selection test**

Create a small selection helper and test it:

```python
from sparsify.quantized_backbone import select_activation_model_path


def test_select_activation_model_path_prefers_quantized_backbone():
    path = select_activation_model_path(
        activation_source="w8a8_backbone",
        default_model="Qwen/Qwen3-0.6B",
        activation_backbone_path="/tmp/qwen3-w8a8",
    )
    assert path == "/tmp/qwen3-w8a8"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_quantized_backbone.py -q`

Expected: FAIL because the helper does not exist yet

- [ ] **Step 3: Add the minimal helper and update `load_artifacts()`**

Add to `sparsify/quantized_backbone.py`:

```python
def select_activation_model_path(
    *,
    activation_source: str,
    default_model: str,
    activation_backbone_path: str | None,
) -> str:
    if activation_source == "w8a8_backbone":
        assert activation_backbone_path is not None
        return activation_backbone_path
    return default_model
```

Update `sparsify/__main__.py`:

```python
from .quantized_backbone import (
    load_torchao_w8a8_model,
    resolve_available_hookpoints,
    select_activation_model_path,
    validate_requested_hookpoints,
)
```

```python
model_path = select_activation_model_path(
    activation_source=args.activation_source,
    default_model=args.model,
    activation_backbone_path=args.activation_backbone_path,
)

if args.activation_source == "w8a8_backbone":
    model = load_torchao_w8a8_model(
        model_path,
        device_map={"": get_device_string(rank)},
        revision=args.revision,
        torch_dtype=dtype,
        token=args.hf_token,
    )
else:
    model = AutoModel.from_pretrained(
        model_path,
        device_map={"": get_device_string(rank)},
        revision=args.revision,
        torch_dtype=dtype,
        token=args.hf_token,
    )
```

After model load, reuse existing hookpoint expansion and add a validation pass:

```python
if args.hookpoints:
    available = resolve_available_hookpoints(model)
    # after expansion in Trainer init, the names must exist in the loaded model
    validate_requested_hookpoints(args.hookpoints, available)
```

- [ ] **Step 4: Run focused tests and compile check**

Run:

```bash
pytest tests/test_quantized_backbone.py -q
python -m py_compile sparsify/__main__.py sparsify/quantized_backbone.py
```

Expected: both commands exit `0`

- [ ] **Step 5: Commit**

```bash
git add sparsify/__main__.py sparsify/quantized_backbone.py tests/test_quantized_backbone.py
git commit -m "Load Qwen3 torchao W8A8 teacher models"
```

---

### Task 4: Plumb the new activation-source flags through training scripts

**Files:**
- Modify: `scripts/autoresearch_test.sh`

- [ ] **Step 1: Add a failing shell expectation**

Document the new flags that must appear in the launch command:

```bash
--activation_source "${ACTIVATION_SOURCE}"
--activation_backbone_path "${ACTIVATION_BACKBONE_PATH}"
```

- [ ] **Step 2: Run shell syntax check before edits**

Run: `bash -n scripts/autoresearch_test.sh`

Expected: PASS

- [ ] **Step 3: Add the minimal env plumbing**

Near the top of the script:

```bash
ACTIVATION_SOURCE="${ACTIVATION_SOURCE:-hf_bf16}"
ACTIVATION_BACKBONE_PATH="${ACTIVATION_BACKBONE_PATH:-}"
```

Inside the `cmd` array:

```bash
  --activation_source "${ACTIVATION_SOURCE}"
```

Append the optional path only when present:

```bash
if [[ -n "${ACTIVATION_BACKBONE_PATH}" ]]; then
  cmd+=(--activation_backbone_path "${ACTIVATION_BACKBONE_PATH}")
fi
```

- [ ] **Step 4: Run shell validation**

Run: `bash -n scripts/autoresearch_test.sh`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/autoresearch_test.sh
git commit -m "Add W8A8 activation source script plumbing"
```

---

### Task 5: Run final verification for the minimal W8A8-teacher path

**Files:**
- Verify only

- [ ] **Step 1: Run unit tests**

Run:

```bash
pytest tests/test_train_quantization.py tests/test_quantized_backbone.py -q
```

Expected: PASS

- [ ] **Step 2: Run compile checks**

Run:

```bash
python -m py_compile sparsify/config.py sparsify/__main__.py sparsify/quantized_backbone.py
```

Expected: PASS

- [ ] **Step 3: Run shell syntax verification**

Run:

```bash
bash -n scripts/autoresearch_test.sh
```

Expected: PASS

- [ ] **Step 4: Record a first smoke command**

```bash
ACTIVATION_SOURCE=w8a8_backbone \
ACTIVATION_BACKBONE_PATH=/path/to/qwen3-w8a8 \
COMPILE_MODEL=0 \
PRINT_COST_BREAKDOWN=0 \
MAX_TOKENS=200000 \
RUN_NAME=smoke_qwen3_w8a8_teacher \
bash scripts/autoresearch_test.sh
```

Expected:

- model loading reaches the torchao W8A8 branch
- target hookpoints resolve successfully
- training starts without failing during activation capture

---

## Self-Review

### Spec coverage

- Config support is covered in Task 1
- torchao backbone helper is covered in Task 2
- entrypoint loading logic is covered in Task 3
- script plumbing is covered in Task 4
- final verification is covered in Task 5

### Placeholder scan

- No `TODO` / `TBD` placeholders remain
- Every code-edit task includes concrete code blocks and exact commands

### Type consistency

- Config field names are used consistently:
  - `activation_source`
  - `activation_backbone_path`
  - `activation_threshold_path`
- Helper names are reused consistently:
  - `build_torchao_int8_quantization_config`
  - `load_torchao_w8a8_model`
  - `resolve_available_hookpoints`
  - `validate_requested_hookpoints`
  - `select_activation_model_path`
