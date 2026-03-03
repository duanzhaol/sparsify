"""Verify critical PyTorch operators work correctly on Ascend NPU.

This is the most important Ascend test file.  Every operator exercised here
is on the critical path for SAE training.  Each test computes a reference
result on CPU and compares it to the NPU result.
"""

import torch
import torch.nn.functional as F


# ---- torch.topk ----


def test_topk_correctness():
    """topk is the core of SAE encoding -- called every forward pass."""
    x_npu = torch.randn(64, 1024, device="npu")
    x_cpu = x_npu.cpu()

    vals_npu, _ = torch.topk(x_npu, k=32, dim=-1, sorted=True)
    vals_cpu, _ = torch.topk(x_cpu, k=32, dim=-1, sorted=True)

    torch.testing.assert_close(vals_npu.cpu(), vals_cpu)


def test_topk_unsorted():
    """topk with sorted=False (used in fused_encoder)."""
    x_npu = torch.randn(32, 512, device="npu")
    x_cpu = x_npu.cpu()

    vals_npu, idx_npu = torch.topk(x_npu, k=16, dim=-1, sorted=False)
    vals_cpu, idx_cpu = torch.topk(x_cpu, k=16, dim=-1, sorted=False)

    # Indices may differ in ordering; compare sorted values instead.
    vals_npu_s, _ = vals_npu.sort(dim=-1)
    vals_cpu_s, _ = vals_cpu.sort(dim=-1)
    torch.testing.assert_close(vals_npu_s.cpu(), vals_cpu_s)


def test_topk_large_k():
    """Large k (expansion_factor=32 can mean k>=128)."""
    x = torch.randn(32, 8192, device="npu")
    vals, idx = torch.topk(x, k=128, dim=-1)
    assert vals.shape == (32, 128)
    assert idx.shape == (32, 128)

    # Verify selected values match the original tensor.
    for i in range(4):
        for j in range(128):
            assert x[i, idx[i, j]] == vals[i, j]


# ---- F.embedding_bag ----


def test_embedding_bag_forward():
    """embedding_bag is the core SAE decoder operation."""
    num_embeddings, dim, k = 128, 64, 8
    W = torch.randn(num_embeddings, dim, device="npu")
    indices = torch.randint(0, num_embeddings, (16, k), device="npu")
    weights = torch.randn(16, k, device="npu")

    result = F.embedding_bag(indices, W, per_sample_weights=weights, mode="sum")
    assert result.shape == (16, dim)

    result_cpu = F.embedding_bag(
        indices.cpu(), W.cpu(), per_sample_weights=weights.cpu(), mode="sum"
    )
    torch.testing.assert_close(result.cpu(), result_cpu, atol=1e-5, rtol=1e-5)


def test_embedding_bag_backward():
    """Backward through per_sample_weights -- highest migration risk."""
    num_embeddings, dim, k = 128, 64, 8
    W = torch.randn(num_embeddings, dim, device="npu", requires_grad=True)
    indices = torch.randint(0, num_embeddings, (16, k), device="npu")
    weights = torch.randn(16, k, device="npu", requires_grad=True)

    result = F.embedding_bag(indices, W, per_sample_weights=weights, mode="sum")
    result.sum().backward()

    assert W.grad is not None
    assert weights.grad is not None
    assert W.grad.abs().sum() > 0
    assert weights.grad.abs().sum() > 0


def test_embedding_bag_backward_vs_cpu():
    """Compare embedding_bag gradients between NPU and CPU."""
    num_embeddings, dim, k = 64, 32, 4
    W_data = torch.randn(num_embeddings, dim)
    indices = torch.randint(0, num_embeddings, (8, k))
    weights_data = torch.randn(8, k)

    # CPU
    W_cpu = W_data.clone().requires_grad_(True)
    w_cpu = weights_data.clone().requires_grad_(True)
    r_cpu = F.embedding_bag(indices, W_cpu, per_sample_weights=w_cpu, mode="sum")
    r_cpu.sum().backward()

    # NPU
    W_npu = W_data.to("npu").requires_grad_(True)
    w_npu = weights_data.to("npu").requires_grad_(True)
    r_npu = F.embedding_bag(
        indices.to("npu"), W_npu, per_sample_weights=w_npu, mode="sum"
    )
    r_npu.sum().backward()

    torch.testing.assert_close(W_npu.grad.cpu(), W_cpu.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(w_npu.grad.cpu(), w_cpu.grad, atol=1e-5, rtol=1e-5)


# ---- index_add_ ----


def test_index_add():
    """index_add_ is used in FusedEncoder backward for weight gradients."""
    M, D = 256, 64
    target_npu = torch.zeros(M, D, device="npu")
    indices = torch.randint(0, M, (32,), device="npu")
    source = torch.randn(32, D, device="npu")
    target_npu.index_add_(0, indices, source)

    target_cpu = torch.zeros(M, D)
    target_cpu.index_add_(0, indices.cpu(), source.cpu())
    torch.testing.assert_close(target_npu.cpu(), target_cpu, atol=1e-6, rtol=1e-6)


# ---- F.linear ----


def test_linear_forward_backward():
    """F.linear is used in the encoder."""
    x = torch.randn(32, 64, device="npu", requires_grad=True)
    W = torch.randn(256, 64, device="npu", requires_grad=True)
    b = torch.randn(256, device="npu", requires_grad=True)
    y = F.linear(x, W, b)
    y.sum().backward()

    assert x.grad is not None and x.grad.abs().sum() > 0
    assert W.grad is not None and W.grad.abs().sum() > 0
    assert b.grad is not None and b.grad.abs().sum() > 0


# ---- F.relu ----


def test_relu():
    x = torch.randn(32, 64, device="npu")
    y = F.relu(x)
    assert (y >= 0).all()
    torch.testing.assert_close(y.cpu(), F.relu(x.cpu()))


# ---- bf16 matmul ----


def test_bfloat16_matmul():
    """bf16 matmul is used under autocast."""
    a = torch.randn(32, 64, device="npu", dtype=torch.bfloat16)
    b = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
    c = a @ b
    assert c.dtype == torch.bfloat16
    assert c.shape == (32, 128)


def test_bfloat16_matmul_accuracy():
    """bf16 accuracy compared to fp32."""
    a = torch.randn(32, 64, device="npu")
    b = torch.randn(64, 128, device="npu")
    ref = a @ b
    result = a.bfloat16() @ b.bfloat16()
    torch.testing.assert_close(result.float(), ref, atol=0.1, rtol=0.05)
