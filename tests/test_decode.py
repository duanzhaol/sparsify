import pytest
import torch

from sparsify.device import get_device_type, is_accelerator_available
from sparsify.fused_decoder import fused_decode
from sparsify.utils import decoder_impl, eager_decode


def _naive_decode(top_indices, top_acts, W_dec):
    """Reference implementation: W_dec shape is [d_in, M] from call site."""
    W_T = W_dec.mT  # [M, d_in]
    selected = W_T[top_indices]  # [N, k, d_in]
    return (top_acts.unsqueeze(-1) * selected).sum(dim=1)


@pytest.mark.skipif(not is_accelerator_available(), reason="CUDA or NPU required")
@pytest.mark.parametrize("d_in", [48, 64])
def test_decode(d_in: int):
    batch = 2
    d_sae = 128
    k = 10

    dev = get_device_type()
    latents = torch.rand(batch, d_sae, device=dev)
    W_dec = torch.randn(d_sae, d_in, device=dev)

    top_vals, top_idx = latents.topk(k)
    eager_res = eager_decode(top_idx, top_vals, W_dec.mT)
    fused_res = fused_decode(top_idx, top_vals, W_dec.mT)

    torch.testing.assert_close(eager_res, fused_res, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not is_accelerator_available(), reason="CUDA or NPU required")
def test_fused_decode_gradient():
    """Verify fused_decode gradients match naive reference."""
    N, M, d_in, k = 32, 128, 64, 8
    dev = get_device_type()
    indices = torch.randint(0, M, (N, k), device=dev)

    # Naive reference
    W_naive = torch.randn(d_in, M, device=dev, requires_grad=True)
    acts_naive = torch.randn(N, k, device=dev, requires_grad=True)
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


def test_fused_decode_fallback_backward_avoids_full_nkd_expand(monkeypatch):
    """Fallback backward should not materialize [N, k, d_in] grad tensors."""
    import sparsify.fused_decoder as fused_decoder

    N, M, d_in, k = 4, 32, 8, 5
    indices = torch.randint(0, M, (N, k))
    W_dec = torch.randn(d_in, M, requires_grad=True)
    acts = torch.randn(N, k, requires_grad=True)

    monkeypatch.setattr(fused_decoder, "_MATMUL_THRESHOLD", 0)
    original_expand = torch.Tensor.expand

    def guarded_expand(self, *sizes):
        normalized = sizes[0] if len(sizes) == 1 and isinstance(sizes[0], tuple) else sizes
        if tuple(self.shape) == (N, 1, d_in) and tuple(normalized) == (N, k, d_in):
            raise AssertionError("fallback decoder backward expanded full [N, k, d_in]")
        return original_expand(self, *sizes)

    monkeypatch.setattr(torch.Tensor, "expand", guarded_expand)

    out = fused_decode(indices, acts, W_dec)
    out.square().sum().backward()

    assert W_dec.grad is not None
    assert acts.grad is not None


def test_fused_decode_fallback_gradient_matches_naive(monkeypatch):
    """Forced fallback backward should match the naive dense reference."""
    import sparsify.fused_decoder as fused_decoder

    N, M, d_in, k = 6, 40, 10, 7
    indices = torch.randint(0, M, (N, k))
    W_naive = torch.randn(d_in, M, requires_grad=True)
    acts_naive = torch.randn(N, k, requires_grad=True)

    out_naive = _naive_decode(indices, acts_naive, W_naive)
    out_naive.square().sum().backward()

    monkeypatch.setattr(fused_decoder, "_MATMUL_THRESHOLD", 0)
    W_fused = W_naive.detach().clone().requires_grad_(True)
    acts_fused = acts_naive.detach().clone().requires_grad_(True)
    out_fused = fused_decode(indices, acts_fused, W_fused)
    out_fused.square().sum().backward()

    torch.testing.assert_close(out_fused, out_naive, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(W_fused.grad, W_naive.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(acts_fused.grad, acts_naive.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_decode_bf16_autocast():
    """Verify fused_decode works under bf16 autocast (mixed dtype backward)."""
    N, M, d_in, k = 32, 256, 64, 8
    indices = torch.randint(0, M, (N, k), device="cuda")

    # float32 parameters (as in real SAE training)
    W_dec = torch.randn(d_in, M, device="cuda", requires_grad=True)
    acts = torch.randn(N, k, device="cuda", requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = fused_decode(indices, acts, W_dec)
        out.sum().backward()

    assert out.shape == (N, d_in)
    assert W_dec.grad is not None
    assert acts.grad is not None
    assert torch.isfinite(W_dec.grad).all()
    assert torch.isfinite(acts.grad).all()


@pytest.mark.skipif(not is_accelerator_available(), reason="CUDA or NPU required")
def test_fused_decode_is_default():
    """Verify that fused_decode is the default decoder_impl on CUDA/NPU."""
    assert decoder_impl is fused_decode, (
        f"Expected fused_decode as default decoder_impl, got {decoder_impl}"
    )
