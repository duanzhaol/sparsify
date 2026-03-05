"""Verify FusedDecoder custom autograd Function on Ascend NPU."""

import warnings

import pytest
import torch
import torch.nn.functional as F

from sparsify.fused_decoder import fused_decode
from sparsify.utils import eager_decode


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _naive_decode(top_indices, top_acts, W_dec):
    """Reference implementation using basic ops for gradient verification.

    Same signature as eager_decode: W_dec shape is [d_in, M] from call site.
    """
    # W_dec: [d_in, M], W_dec.mT: [M, d_in]
    W_T = W_dec.mT  # [M, d_in]
    # Gather selected rows: [N, k, d_in]
    selected = W_T[top_indices]
    # Weighted sum: [N, k, 1] * [N, k, d_in] → sum over k → [N, d_in]
    return (top_acts.unsqueeze(-1) * selected).sum(dim=1)


# ===================================================================
# Original basic tests
# ===================================================================

def test_forward_shapes():
    """Test that fused_decode produces correct output shapes."""
    N, M, d_in, k = 16, 256, 64, 8
    # W_dec arrives from call site as self.W_dec.mT: [d_in, M]
    W_dec = torch.randn(d_in, M, device="npu")
    indices = torch.randint(0, M, (N, k), device="npu")
    acts = torch.randn(N, k, device="npu")

    out = fused_decode(indices, acts, W_dec)
    assert out.shape == (N, d_in)


def test_forward_matches_naive():
    """Test that fused_decode forward matches a naive reference."""
    N, M, d_in, k = 16, 128, 64, 8
    W_dec = torch.randn(d_in, M, device="npu")
    indices = torch.randint(0, M, (N, k), device="npu")
    acts = torch.randn(N, k, device="npu")

    out_fused = fused_decode(indices, acts, W_dec)
    out_naive = _naive_decode(indices, acts, W_dec)

    torch.testing.assert_close(out_fused, out_naive, atol=1e-5, rtol=1e-5)


def test_gradient_vs_naive():
    """Compare fused_decode gradients against naive reference on NPU."""
    N, M, d_in, k = 32, 128, 64, 8
    indices = torch.randint(0, M, (N, k), device="npu")

    # Naive reference
    W_naive = torch.randn(d_in, M, device="npu", requires_grad=True)
    acts_naive = torch.randn(N, k, device="npu", requires_grad=True)
    out_naive = _naive_decode(indices, acts_naive, W_naive)
    out_naive.sum().backward()
    gW_naive = W_naive.grad.clone()
    ga_naive = acts_naive.grad.clone()

    # Fused path
    W_fused = W_naive.detach().clone().requires_grad_(True)
    acts_fused = acts_naive.detach().clone().requires_grad_(True)
    out_fused = fused_decode(indices, acts_fused, W_fused)
    out_fused.sum().backward()

    torch.testing.assert_close(out_fused, out_naive, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(W_fused.grad, gW_naive, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(acts_fused.grad, ga_naive, atol=1e-5, rtol=1e-5)


def test_gradient_vs_cpu():
    """Compare NPU fused_decode gradients against CPU reference."""
    N, M, d_in, k = 16, 64, 32, 4
    indices_data = torch.randint(0, M, (N, k))
    W_data = torch.randn(d_in, M)
    acts_data = torch.randn(N, k)

    # CPU reference (fused_decode works on CPU too)
    W_cpu = W_data.clone().requires_grad_(True)
    acts_cpu = acts_data.clone().requires_grad_(True)
    out_cpu = fused_decode(indices_data, acts_cpu, W_cpu)
    out_cpu.sum().backward()

    # NPU
    W_npu = W_data.to("npu").requires_grad_(True)
    acts_npu = acts_data.to("npu").requires_grad_(True)
    out_npu = fused_decode(indices_data.to("npu"), acts_npu, W_npu)
    out_npu.sum().backward()

    torch.testing.assert_close(out_npu.cpu(), out_cpu, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(W_npu.grad.cpu(), W_cpu.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(acts_npu.grad.cpu(), acts_cpu.grad, atol=1e-5, rtol=1e-5)


def test_no_cpu_fallback_warnings():
    """Verify that fused_decode does NOT trigger CPU fallback warnings."""
    N, M, d_in, k = 16, 128, 64, 8
    W = torch.randn(d_in, M, device="npu", requires_grad=True)
    indices = torch.randint(0, M, (N, k), device="npu")
    acts = torch.randn(N, k, device="npu", requires_grad=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fused_decode(indices, acts, W)
        out.sum().backward()

    fallback_warnings = [
        w for w in caught
        if "fall back" in str(w.message).lower() or "cpu" in str(w.message).lower()
    ]
    assert len(fallback_warnings) == 0, (
        f"CPU fallback detected: {[str(w.message) for w in fallback_warnings]}"
    )


def test_large_batch():
    """Stress test with a large batch."""
    N, M, d_in, k = 4096, 4096, 128, 32
    W = torch.randn(d_in, M, device="npu", requires_grad=True)
    indices = torch.randint(0, M, (N, k), device="npu")
    acts = torch.randn(N, k, device="npu", requires_grad=True)

    out = fused_decode(indices, acts, W)
    out.sum().backward()

    assert W.grad.shape == (d_in, M)
    assert acts.grad.shape == (N, k)


# ===================================================================
# 1. CPU fallback profiler-based detection
# ===================================================================

def test_no_cpu_fallback_profiler():
    """Verify no aten::_embedding_bag_backward CPU execution via profiler."""
    from torch.profiler import ProfilerActivity, profile

    N, M, d_in, k = 32, 128, 64, 8
    W = torch.randn(d_in, M, device="npu", requires_grad=True)
    indices = torch.randint(0, M, (N, k), device="npu")
    acts = torch.randn(N, k, device="npu", requires_grad=True)

    # Warm up
    out = fused_decode(indices, acts, W)
    out.sum().backward()
    W.grad, acts.grad = None, None

    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        out = fused_decode(indices, acts, W)
        out.sum().backward()
        torch.npu.synchronize()

    event_names = [evt.key for evt in prof.key_averages()]
    bad_events = [n for n in event_names if "embedding_bag_backward" in n.lower()]
    assert len(bad_events) == 0, (
        f"CPU fallback detected via profiler: {bad_events}"
    )


# ===================================================================
# 2. Duplicate indices gradient tests
# ===================================================================

def test_duplicate_indices_all_same():
    """All k indices per row point to the same latent -- extreme duplication."""
    N, M, d_in, k = 16, 128, 64, 8
    single_idx = torch.randint(0, M, (N, 1), device="npu")
    indices = single_idx.expand(N, k).contiguous()

    # Naive reference
    W_naive = torch.randn(d_in, M, device="npu", requires_grad=True)
    acts_naive = torch.randn(N, k, device="npu", requires_grad=True)
    out_naive = _naive_decode(indices, acts_naive, W_naive)
    out_naive.sum().backward()

    # Fused
    W_fused = W_naive.detach().clone().requires_grad_(True)
    acts_fused = acts_naive.detach().clone().requires_grad_(True)
    out_fused = fused_decode(indices, acts_fused, W_fused)
    out_fused.sum().backward()

    torch.testing.assert_close(out_fused, out_naive, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(W_fused.grad, W_naive.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(acts_fused.grad, acts_naive.grad, atol=1e-5, rtol=1e-5)


def test_duplicate_indices_high_collision():
    """90% of indices are duplicates -- NPU vs CPU cross-validation."""
    N, M, d_in, k = 32, 128, 64, 16
    indices = torch.randint(0, M, (N, k))
    # Force 90% columns to repeat the same index per row
    dup_cols = int(k * 0.9)  # 14 out of 16
    single_idx = torch.randint(0, M, (N, 1))
    indices[:, :dup_cols] = single_idx.expand(N, dup_cols)

    W_data = torch.randn(d_in, M)
    acts_data = torch.randn(N, k)

    # CPU reference
    W_cpu = W_data.clone().requires_grad_(True)
    acts_cpu = acts_data.clone().requires_grad_(True)
    out_cpu = fused_decode(indices, acts_cpu, W_cpu)
    out_cpu.sum().backward()

    # NPU
    W_npu = W_data.to("npu").requires_grad_(True)
    acts_npu = acts_data.to("npu").requires_grad_(True)
    out_npu = fused_decode(indices.to("npu"), acts_npu, W_npu)
    out_npu.sum().backward()

    torch.testing.assert_close(out_npu.cpu(), out_cpu, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(W_npu.grad.cpu(), W_cpu.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(acts_npu.grad.cpu(), acts_cpu.grad, atol=1e-5, rtol=1e-5)


# ===================================================================
# 3. k boundary tests
# ===================================================================

@pytest.mark.parametrize("k", [1, 2, 128])
def test_k_boundary(k):
    """Forward and gradient correctness at boundary k values."""
    N, d_in = 32, 64
    M = max(256, k * 2)

    indices = torch.randint(0, M, (N, k), device="npu")

    # Naive reference
    W_naive = torch.randn(d_in, M, device="npu", requires_grad=True)
    acts_naive = torch.randn(N, k, device="npu", requires_grad=True)
    out_naive = _naive_decode(indices, acts_naive, W_naive)
    out_naive.sum().backward()

    # Fused
    W_fused = W_naive.detach().clone().requires_grad_(True)
    acts_fused = acts_naive.detach().clone().requires_grad_(True)
    out_fused = fused_decode(indices, acts_fused, W_fused)
    out_fused.sum().backward()

    torch.testing.assert_close(out_fused, out_naive, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(W_fused.grad, W_naive.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(acts_fused.grad, acts_naive.grad, atol=1e-5, rtol=1e-5)


# ===================================================================
# 4. dtype combination tests
# ===================================================================

_DTYPE_TOLERANCES = {
    torch.float32: (1e-5, 1e-5),
    torch.bfloat16: (2e-2, 2e-2),
    torch.float16: (5e-3, 5e-3),
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtype_forward_and_gradient(dtype):
    """Forward and gradient correctness for each dtype."""
    N, M, d_in, k = 16, 128, 64, 8
    atol, rtol = _DTYPE_TOLERANCES[dtype]

    indices = torch.randint(0, M, (N, k), device="npu")

    # Check if this dtype is supported on NPU for embedding_bag
    try:
        test_w = torch.randn(2, 2, device="npu", dtype=dtype)
        F.embedding_bag(torch.tensor([[0]], device="npu"), test_w, mode="sum")
    except RuntimeError:
        pytest.skip(f"{dtype} not supported for embedding_bag on this NPU")

    # Naive reference
    W_naive = torch.randn(d_in, M, device="npu", dtype=dtype, requires_grad=True)
    acts_naive = torch.randn(N, k, device="npu", dtype=dtype, requires_grad=True)
    out_naive = _naive_decode(indices, acts_naive, W_naive)
    out_naive.sum().backward()

    # Fused
    W_fused = W_naive.detach().clone().requires_grad_(True)
    acts_fused = acts_naive.detach().clone().requires_grad_(True)
    out_fused = fused_decode(indices, acts_fused, W_fused)
    out_fused.sum().backward()

    torch.testing.assert_close(out_fused, out_naive, atol=atol, rtol=rtol)
    torch.testing.assert_close(W_fused.grad, W_naive.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(acts_fused.grad, acts_naive.grad, atol=atol, rtol=rtol)


# ===================================================================
# 5. Non-contiguous input tests
# ===================================================================

def test_non_contiguous_W_dec():
    """W_dec created via transpose is non-contiguous; must match contiguous."""
    N, M, d_in, k = 16, 128, 64, 8
    indices = torch.randint(0, M, (N, k), device="npu")
    acts = torch.randn(N, k, device="npu")

    # Non-contiguous W_dec: start from [M, d_in], transpose to [d_in, M]
    W_base = torch.randn(M, d_in, device="npu")
    W_dec_nc = W_base.mT  # [d_in, M], non-contiguous
    assert not W_dec_nc.is_contiguous()

    W_dec_c = W_dec_nc.contiguous()

    out_nc = fused_decode(indices, acts, W_dec_nc)
    out_c = fused_decode(indices, acts, W_dec_c)

    torch.testing.assert_close(out_nc, out_c, atol=1e-5, rtol=1e-5)


def test_non_contiguous_acts():
    """top_acts via stride-2 slicing is non-contiguous; gradients must match."""
    N, M, d_in, k = 16, 128, 64, 8
    indices = torch.randint(0, M, (N, k), device="npu")

    W = torch.randn(d_in, M, device="npu", requires_grad=True)

    # Create non-contiguous acts via stride-2 slicing
    acts_big = torch.randn(N, k * 2, device="npu", requires_grad=True)
    acts_nc = acts_big[:, ::2]  # every other column
    assert not acts_nc.is_contiguous()
    assert acts_nc.shape == (N, k)

    out = fused_decode(indices, acts_nc, W)
    out.sum().backward()

    # Compare against contiguous copy
    W2 = W.detach().clone().requires_grad_(True)
    acts_c = acts_nc.detach().clone().contiguous().requires_grad_(True)
    out2 = fused_decode(indices, acts_c, W2)
    out2.sum().backward()

    torch.testing.assert_close(out, out2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(W.grad, W2.grad, atol=1e-5, rtol=1e-5)


# ===================================================================
# 6. Large-scale stress + peak memory
# ===================================================================

@pytest.mark.parametrize(
    "N,M,d_in,k",
    [
        (4096, 4096, 128, 32),
        (8192, 4096, 256, 32),
        (4096, 8192, 128, 64),
        (4096, 4096, 256, 128),
    ],
    ids=["baseline", "large_N_d", "large_M_k", "max_k"],
)
def test_large_scale_stress(N, M, d_in, k):
    """Stress test: verify no error, correct shapes, and measure peak memory."""
    has_mem_api = hasattr(torch.npu, "reset_peak_memory_stats")

    if has_mem_api:
        torch.npu.reset_peak_memory_stats()
        mem_before = torch.npu.max_memory_allocated()

    W = torch.randn(d_in, M, device="npu", requires_grad=True)
    indices = torch.randint(0, M, (N, k), device="npu")
    acts = torch.randn(N, k, device="npu", requires_grad=True)

    out = fused_decode(indices, acts, W)
    out.sum().backward()
    torch.npu.synchronize()

    assert out.shape == (N, d_in)
    assert W.grad.shape == (d_in, M)
    assert acts.grad.shape == (N, k)

    if has_mem_api:
        mem_peak = torch.npu.max_memory_allocated()
        mem_used_mb = (mem_peak - mem_before) / (1024 ** 2)

        # Naive approach would allocate N*k*d_in*4 bytes intermediate
        naive_mem_mb = N * k * d_in * 4 / (1024 ** 2)

        print(f"\n  [{N=}, {M=}, {d_in=}, {k=}] "
              f"peak={mem_used_mb:.1f}MB, naive_estimate={naive_mem_mb:.1f}MB")

        # For large cases, assert for-k loop uses significantly less memory
        if naive_mem_mb > 100:
            assert mem_used_mb < naive_mem_mb * 0.5, (
                f"Memory {mem_used_mb:.1f}MB exceeds 50% of naive estimate "
                f"{naive_mem_mb:.1f}MB"
            )


# ===================================================================
# 7. End-to-end SAE training regression
# ===================================================================

def test_end_to_end_sae_training_step():
    """Mini SAE training loop to verify FusedDecoder integration."""
    from sparsify.config import SparseCoderConfig
    from sparsify.fused_decoder import fused_decode as fd
    from sparsify.sparse_coder import SparseCoder
    from sparsify.utils import decoder_impl

    # Confirm fused_decode is active
    assert decoder_impl is fd, "decoder_impl should be fused_decode on NPU"

    d_in = 128
    cfg = SparseCoderConfig(expansion_factor=4, k=8)
    sae = SparseCoder(d_in, cfg, device="npu", dtype=torch.float32)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    losses = []
    for step in range(5):
        x = torch.randn(32, d_in, device="npu")
        if step == 0:
            sae.b_dec.data = x.mean(0)

        out = sae(x)
        loss = out.fvu + 1 / 32 * out.auxk_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    # Loss should be finite
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), (
        f"Non-finite loss: {losses}"
    )
    # Loss should not explode
    assert losses[-1] < losses[0] * 2, (
        f"Loss exploded: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
