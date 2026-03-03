"""Verify FusedEncoder custom autograd Function on Ascend NPU."""

import torch
import torch.nn.functional as F

from sparsify.fused_encoder import fused_encoder


def test_topk_forward_shapes():
    N, D, M, k = 16, 64, 256, 8
    x = torch.randn(N, D, device="npu")
    W = torch.randn(M, D, device="npu")
    b = torch.randn(M, device="npu")
    values, indices, preacts = fused_encoder(x, W, b, k, "topk")

    assert values.shape == (N, k)
    assert indices.shape == (N, k)
    assert preacts.shape == (N, M)


def test_groupmax_forward_shapes():
    N, D, M, k = 16, 64, 256, 8
    x = torch.randn(N, D, device="npu")
    W = torch.randn(M, D, device="npu")
    b = torch.randn(M, device="npu")
    values, indices, preacts = fused_encoder(x, W, b, k, "groupmax")

    assert values.shape == (N, k)
    assert indices.shape == (N, k)


def test_gradient_vs_naive():
    """Compare fused gradients against a naive step-by-step implementation."""
    N, D, M, k = 32, 64, 256, 8
    x = torch.randn(N, D, requires_grad=True, device="npu")
    W = torch.randn(M, D, requires_grad=True, device="npu")
    b = torch.randn(M, requires_grad=True, device="npu")

    # Naive implementation
    preacts = F.relu(F.linear(x, W, b))
    vals_naive, _ = preacts.topk(k, sorted=False)
    vals_naive.sum().backward()
    gx_naive = x.grad.clone()
    gW_naive = W.grad.clone()
    gb_naive = b.grad.clone()
    x.grad, W.grad, b.grad = None, None, None

    # Fused implementation
    vals, _, _ = fused_encoder(x, W, b, k, "topk")
    vals.sum().backward()

    torch.testing.assert_close(vals, vals_naive)
    torch.testing.assert_close(x.grad, gx_naive, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(W.grad, gW_naive, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b.grad, gb_naive, atol=1e-5, rtol=1e-5)


def test_gradient_vs_cpu():
    """Compare NPU gradients against CPU gradients."""
    N, D, M, k = 16, 64, 128, 4
    x_data = torch.randn(N, D)
    W_data = torch.randn(M, D)
    b_data = torch.randn(M)

    # CPU
    x_cpu = x_data.clone().requires_grad_(True)
    W_cpu = W_data.clone().requires_grad_(True)
    b_cpu = b_data.clone().requires_grad_(True)
    v_cpu, _, _ = fused_encoder(x_cpu, W_cpu, b_cpu, k, "topk")
    v_cpu.sum().backward()

    # NPU
    x_npu = x_data.to("npu").requires_grad_(True)
    W_npu = W_data.to("npu").requires_grad_(True)
    b_npu = b_data.to("npu").requires_grad_(True)
    v_npu, _, _ = fused_encoder(x_npu, W_npu, b_npu, k, "topk")
    v_npu.sum().backward()

    torch.testing.assert_close(x_npu.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(W_npu.grad.cpu(), W_cpu.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b_npu.grad.cpu(), b_cpu.grad, atol=1e-5, rtol=1e-5)


def test_large_batch():
    """Stress test with a large batch."""
    N, D, M, k = 4096, 128, 4096, 32
    x = torch.randn(N, D, device="npu", requires_grad=True)
    W = torch.randn(M, D, device="npu", requires_grad=True)
    b = torch.randn(M, device="npu", requires_grad=True)
    vals, _, _ = fused_encoder(x, W, b, k, "topk")
    vals.sum().backward()

    assert x.grad.shape == (N, D)
    assert W.grad.shape == (M, D)
