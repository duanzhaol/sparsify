"""Verify device_autocast decorator and bf16 mixed precision on Ascend NPU."""

import torch

from sparsify.device import device_autocast


def test_autocast_enables_bf16():
    @device_autocast
    def fn(x):
        return x @ x.T

    x = torch.randn(8, 8, device="npu", dtype=torch.float32)
    result = fn(x)
    assert result.dtype == torch.bfloat16


def test_autocast_preserves_metadata():
    @device_autocast
    def my_func():
        """my doc"""
        pass

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "my doc"


def test_autocast_gradient_flow():
    @device_autocast
    def fn(x, W):
        return (x @ W).sum()

    x = torch.randn(8, 64, device="npu", requires_grad=True)
    W = torch.randn(64, 32, device="npu", requires_grad=True)
    loss = fn(x, W)
    loss.backward()
    assert x.grad is not None
    assert W.grad is not None


def test_autocast_context_manager_directly():
    """torch.autocast('npu', ...) must be supported by torch_npu."""
    with torch.autocast("npu", dtype=torch.bfloat16):
        x = torch.randn(8, 64, device="npu")
        y = x @ x.T
    assert y.dtype == torch.bfloat16
