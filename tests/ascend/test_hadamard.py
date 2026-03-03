"""Verify Hadamard transform on Ascend NPU (pure PyTorch fallback path)."""

import torch

from sparsify.hadamard import (
    HadamardRotation,
    block_hadamard_transform,
    fast_hadamard_transform,
)


def test_fast_hadamard_uses_fallback_on_npu():
    """On NPU x.is_cuda is False, so fast_hadamard_transform should use the
    pure PyTorch fallback and still produce correct results."""
    x = torch.randn(8, 128, device="npu")
    assert not x.is_cuda
    y = fast_hadamard_transform(x, block_size=64)
    assert y.shape == x.shape
    assert y.device.type == "npu"


def test_block_hadamard_self_inverse():
    """H(H(x)) == x  (Hadamard is self-inverse)."""
    x = torch.randn(8, 128, device="npu")
    y = block_hadamard_transform(x, block_size=64)
    x_recovered = block_hadamard_transform(y, block_size=64)
    torch.testing.assert_close(x, x_recovered, atol=1e-4, rtol=1e-4)


def test_hadamard_norm_preservation():
    """Hadamard transform preserves L2 norm."""
    x = torch.randn(8, 128, device="npu")
    y = block_hadamard_transform(x, block_size=64)
    torch.testing.assert_close(
        x.norm(dim=-1), y.norm(dim=-1), atol=1e-4, rtol=1e-4
    )


def test_hadamard_rotation_roundtrip():
    """rotate then unrotate should recover the original."""
    rot = HadamardRotation(d_in=128, block_size=64, use_permutation=True, device="npu")
    x = torch.randn(8, 128, device="npu")
    y = rot.rotate(x)
    x_back = rot.unrotate(y)
    torch.testing.assert_close(x, x_back, atol=1e-4, rtol=1e-4)


def test_hadamard_npu_vs_cpu():
    """NPU and CPU should produce identical results."""
    x = torch.randn(8, 128)
    y_cpu = block_hadamard_transform(x, block_size=64)
    y_npu = block_hadamard_transform(x.to("npu"), block_size=64)
    torch.testing.assert_close(y_npu.cpu(), y_cpu, atol=1e-5, rtol=1e-5)
