"""Verify SAE eager decoder on Ascend NPU."""

import torch

from sparsify.utils import eager_decode


def test_eager_decode_forward():
    batch, d_sae, d_in, k = 16, 128, 64, 8
    W_dec = torch.randn(d_sae, d_in, device="npu")
    latents = torch.rand(batch, d_sae, device="npu")
    top_vals, top_idx = latents.topk(k)

    result = eager_decode(top_idx, top_vals, W_dec.mT)
    assert result.shape == (batch, d_in)


def test_eager_decode_vs_manual():
    """Compare eager_decode against a manual loop reference."""
    batch, d_sae, d_in, k = 4, 64, 32, 4
    W_dec = torch.randn(d_sae, d_in, device="npu")
    latents = torch.rand(batch, d_sae, device="npu")
    top_vals, top_idx = latents.topk(k)

    result = eager_decode(top_idx, top_vals, W_dec.mT)

    expected = torch.zeros(batch, d_in, device="npu")
    for b in range(batch):
        for j in range(k):
            expected[b] += top_vals[b, j] * W_dec[top_idx[b, j]]
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


def test_eager_decode_backward():
    W_dec = torch.randn(64, 32, device="npu", requires_grad=True)
    top_vals = torch.randn(8, 4, device="npu", requires_grad=True)
    top_idx = torch.randint(0, 64, (8, 4), device="npu")

    result = eager_decode(top_idx, top_vals, W_dec.mT)
    result.sum().backward()

    assert W_dec.grad is not None and W_dec.grad.abs().sum() > 0
    assert top_vals.grad is not None and top_vals.grad.abs().sum() > 0


def test_eager_decode_vs_cpu():
    d_sae, d_in, k = 64, 32, 4
    W = torch.randn(d_sae, d_in)
    idx = torch.randint(0, d_sae, (8, k))
    vals = torch.randn(8, k)

    r_cpu = eager_decode(idx, vals, W.mT)
    r_npu = eager_decode(idx.to("npu"), vals.to("npu"), W.to("npu").mT)
    torch.testing.assert_close(r_npu.cpu(), r_cpu, atol=1e-5, rtol=1e-5)


def test_decoder_impl_is_fused_on_npu():
    """On NPU, the module-level decoder_impl should be fused_decode."""
    from sparsify.fused_decoder import fused_decode
    from sparsify.utils import decoder_impl

    assert decoder_impl is fused_decode
