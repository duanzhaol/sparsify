"""Verify OutlierClipper on Ascend NPU."""

import pytest
import torch

from sparsify.outlier_clip import OutlierClipper


def test_outlier_clipper_basic():
    clipper = OutlierClipper(d_in=64, k=3.0, device="npu")
    x = torch.randn(32, 64, device="npu")
    clipper.update_stats(x)
    x_inlier, residual, mask = clipper.clip(x)

    assert x_inlier.shape == x.shape
    assert residual.shape == x.shape
    assert mask.shape == x.shape
    # inlier + residual should reconstruct the original
    torch.testing.assert_close(x_inlier + residual, x)


def test_outlier_clipper_ema_convergence():
    clipper = OutlierClipper(d_in=64, k=3.0, device="npu")
    for _ in range(50):
        x = torch.randn(32, 64, device="npu")
        clipper.update_stats(x)
    # After 50 steps of standard normal input, E[x^2] should converge to ~1.0
    assert clipper.ema_sq.mean().item() == pytest.approx(1.0, abs=0.2)
