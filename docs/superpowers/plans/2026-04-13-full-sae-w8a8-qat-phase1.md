# Full SAE W8A8 QAT Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first runnable `qat_full_w8a8` training mode for `product_key_expert_jumprelu` using a BF16 teacher while quantizing SAE encoder and decoder matmul paths with fake W8A8 during training.

**Architecture:** Reuse the existing training-shell quantization framework in `sparsify/train_quantization.py` and `sparsify/trainer.py`, then add a narrow architecture-specific full-QAT path inside `ProductKeyExpertJumpReLUSparseCoder`. Keep routing, TopK, JumpReLU thresholds, and index logic in floating point; quantize only activations and weights on the main linear/einsum/decode paths.

**Tech Stack:** PyTorch, existing `sparsify` trainer/config/sparse coder stack, `pytest`

---

## File Structure

- Modify: `sparsify/config.py`
  - Add config validation for `io_quant_mode=qat_full_w8a8`.
- Modify: `sparsify/train_quantization.py`
  - Add reusable weight fake-quant and fake-quant linear/einsum/decode helpers.
- Modify: `sparsify/sparse_coder.py`
  - Add architecture-local full-QAT path for `ProductKeyExpertJumpReLUSparseCoder`.
- Modify: `sparsify/trainer.py`
  - Route full-QAT metrics through existing logging and deploy-target loss logic.
- Modify: `tests/test_train_quantization.py`
  - Add config and helper regression tests for the new mode.
- Modify: `tests/test_sparse_coder.py`
  - Add tiny-shape forward/backward tests for `product_key_expert_jumprelu` full QAT.
- Create: `scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_full_w8a8_qat_q_13.sh`
  - Add a single-layer smoke/formal entrypoint for the new mode.

### Task 1: Add Quantization Helpers And Config Gate

**Files:**
- Modify: `sparsify/config.py`
- Modify: `sparsify/train_quantization.py`
- Modify: `tests/test_train_quantization.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_train_config_accepts_full_w8a8_qat_mode():
    cfg = TrainConfig(
        sae=SparseCoderConfig(),
        io_quant_mode="qat_full_w8a8",
        io_quant_bits=8,
    )
    assert cfg.io_quant_mode == "qat_full_w8a8"


def test_fake_quantize_weight_per_output_channel_returns_expected_shapes():
    weight = torch.tensor([[1.0, -2.0], [0.5, -0.5]], dtype=torch.float32)
    qdq, scales = fake_quantize_weight_per_output_channel(weight, num_bits=8)
    assert qdq.shape == weight.shape
    assert scales.shape == (2, 1)


def test_fake_quantized_linear_matches_dense_shape():
    x = torch.randn(3, 4)
    weight = torch.randn(5, 4)
    bias = torch.randn(5)
    out, stats = fake_quantized_linear(x, weight, bias, num_bits=8)
    assert out.shape == (3, 5)
    assert "weight_scale_mean" in stats
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_train_quantization.py -q
```

Expected:

- FAIL because `qat_full_w8a8` is not yet accepted and helper functions do not exist.

- [ ] **Step 3: Write the minimal implementation**

```python
# sparsify/config.py
valid_io_quant_modes = {"off", "qat_io_int8", "qat_full_w8a8"}


# sparsify/train_quantization.py
def fake_quantize_weight_per_output_channel(weight: Tensor, num_bits: int = 8):
    ...


def fake_quantized_linear(x: Tensor, weight: Tensor, bias: Tensor | None, num_bits: int = 8):
    ...


def fake_quantized_expert_einsum(x: Tensor, weight: Tensor, bias: Tensor, num_bits: int = 8):
    ...


def fake_quantized_decoder_path(top_acts: Tensor, top_indices: Tensor, W_dec: Tensor, num_bits: int = 8):
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_train_quantization.py -q
```

Expected:

- PASS with new config and helper tests green.

- [ ] **Step 5: Commit**

```bash
git add sparsify/config.py sparsify/train_quantization.py tests/test_train_quantization.py
git commit -m "Add full W8A8 QAT quantization helpers"
```

### Task 2: Add Product-Key Expert Full-QAT Forward Path

**Files:**
- Modify: `sparsify/sparse_coder.py`
- Modify: `tests/test_sparse_coder.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_product_key_expert_jumprelu_full_qat_forward_runs():
    cfg = SparseCoderConfig(
        architecture="product_key_expert_jumprelu",
        k=4,
        num_experts=4,
        active_experts=2,
        latents_per_expert=4,
    )
    model = SparseCoder(8, cfg, device="cpu", dtype=torch.float32)
    model.set_quantization_mode("qat_full_w8a8", num_bits=8)
    x = torch.randn(2, 8)
    out = model(x)
    assert out.sae_out.shape == x.shape
    assert torch.isfinite(out.fvu)


def test_product_key_expert_jumprelu_full_qat_backward_runs():
    cfg = SparseCoderConfig(
        architecture="product_key_expert_jumprelu",
        k=4,
        num_experts=4,
        active_experts=2,
        latents_per_expert=4,
    )
    model = SparseCoder(8, cfg, device="cpu", dtype=torch.float32)
    model.set_quantization_mode("qat_full_w8a8", num_bits=8)
    x = torch.randn(2, 8)
    out = model(x)
    out.fvu.backward()
    assert model.left_router.weight.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_sparse_coder.py -q
```

Expected:

- FAIL because `set_quantization_mode` or the full-QAT path does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
# sparsify/sparse_coder.py
class SparseCoder(...):
    def set_quantization_mode(self, mode: str, *, num_bits: int = 8) -> None:
        self.quantization_mode = mode
        self.quantization_num_bits = num_bits


class ProductKeyExpertJumpReLUSparseCoder(...):
    def _full_qat_enabled(self) -> bool:
        return getattr(self, "quantization_mode", "off") == "qat_full_w8a8"

    def _expert_logits_from_flat(self, flat_x: Tensor) -> Tensor:
        if self._full_qat_enabled():
            left_logits, _ = fake_quantized_linear(...)
            right_logits, _ = fake_quantized_linear(...)
            ...
        ...

    def _expert_candidate_acts(...):
        if self._full_qat_enabled():
            pre_acts, _ = fake_quantized_expert_einsum(...)
            ...

    def _decode_sparse(...):
        if self._full_qat_enabled():
            return fake_quantized_decoder_path(...)
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_sparse_coder.py -q
```

Expected:

- PASS with tiny-shape forward/backward stable on CPU.

- [ ] **Step 5: Commit**

```bash
git add sparsify/sparse_coder.py tests/test_sparse_coder.py
git commit -m "Add full W8A8 QAT path for product-key SAE"
```

### Task 3: Trainer Wiring, Metrics, And Smoke Script

**Files:**
- Modify: `sparsify/trainer.py`
- Modify: `tests/test_train_quantization.py`
- Create: `scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_full_w8a8_qat_q_13.sh`

- [ ] **Step 1: Write the failing tests**

```python
def test_select_main_loss_supports_full_w8a8_metrics_path():
    metrics = summarize_io_quant_batch(
        target_fp=torch.randn(4, 8),
        recon_fp=torch.randn(4, 8),
        num_bits=8,
        alpha=None,
        elbow_value=None,
        deploy_weight=0.25,
    )
    loss = select_main_loss(metrics, metrics.fvu_fp_teacher, "dual_target")
    assert torch.isfinite(loss)
```

Add a smoke-level trainer regression that asserts:

```python
cfg = TrainConfig(
    sae=SparseCoderConfig(architecture="product_key_expert_jumprelu", ...),
    io_quant_mode="qat_full_w8a8",
    hookpoints=["layers.13.self_attn.q_proj"],
)
assert cfg.io_quant_mode == "qat_full_w8a8"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_train_quantization.py -q
```

Expected:

- FAIL if trainer/config wiring does not yet recognize the new mode end-to-end.

- [ ] **Step 3: Write the minimal implementation**

```python
# sparsify/trainer.py
if self.cfg.io_quant_mode in {"qat_io_int8", "qat_full_w8a8"}:
    acts_in, _, _ = fake_quantize_activation_per_token(...)
else:
    acts_in = acts_fp

raw.set_quantization_mode(self.cfg.io_quant_mode, num_bits=self.cfg.io_quant_bits)

# log extra monitoring emitted by full-QAT path when available
for metric_name, metric_value in monitoring_metrics.items():
    avg_monitoring[sae_key][metric_name] += metric_value / denom
```

Script:

```bash
ACTIVATION_SOURCE=hf_bf16 \
IO_QUANT_MODE=qat_full_w8a8 \
IO_QUANT_BITS=8 \
IO_LOSS_MODE=dual_target \
IO_LOSS_DEPLOY_WEIGHT=0.25 \
HOOKPOINTS='layers.[13].self_attn.q_proj' \
bash scripts/autoresearch_test.sh
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_train_quantization.py tests/test_sparse_coder.py -q
bash -n scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_full_w8a8_qat_q_13.sh
```

Expected:

- PASS for Python tests;
- shell syntax check passes for the new script.

- [ ] **Step 5: Commit**

```bash
git add sparsify/trainer.py tests/test_train_quantization.py scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_full_w8a8_qat_q_13.sh
git commit -m "Wire full W8A8 QAT into trainer"
```

### Task 4: End-To-End Verification

**Files:**
- Modify: none
- Test: existing files above

- [ ] **Step 1: Run the targeted unit/integration suite**

Run:

```bash
python -m pytest \
  tests/test_train_quantization.py \
  tests/test_sparse_coder.py \
  tests/test_main.py \
  -q
```

Expected:

- PASS with no failures.

- [ ] **Step 2: Run bytecode validation**

Run:

```bash
python -m py_compile \
  sparsify/config.py \
  sparsify/train_quantization.py \
  sparsify/trainer.py \
  sparsify/sparse_coder.py
```

Expected:

- PASS with no syntax errors.

- [ ] **Step 3: Run the new script syntax check**

Run:

```bash
bash -n scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_full_w8a8_qat_q_13.sh
```

Expected:

- PASS.

- [ ] **Step 4: Commit the final integrated state**

```bash
git add \
  sparsify/config.py \
  sparsify/train_quantization.py \
  sparsify/trainer.py \
  sparsify/sparse_coder.py \
  tests/test_train_quantization.py \
  tests/test_sparse_coder.py \
  scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_full_w8a8_qat_q_13.sh \
  docs/superpowers/plans/2026-04-13-full-sae-w8a8-qat-phase1.md
git commit -m "Implement full SAE W8A8 QAT phase 1"
```

## Self-Review

- **Spec coverage:** helper/config, architecture-local full-QAT path, trainer wiring, smoke script, and verification are all covered.
- **Placeholder scan:** no `TODO` or implicit “appropriate handling” placeholders remain; each task has explicit files, commands, and expected failures/passes.
- **Type consistency:** the plan consistently uses `qat_full_w8a8`, `set_quantization_mode(...)`, and the helper names introduced in Task 1 across later tasks.
