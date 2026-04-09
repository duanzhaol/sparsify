from __future__ import annotations

import torch
import pytest

from sparsify.train_quantization import (
    IOQuantMetrics,
    compute_exceed_ratio,
    compute_fvu_scalar,
    fake_quantize_activation_per_token,
    summarize_io_quant_batch,
)


def test_fake_quantize_activation_per_token_preserves_shape_and_clip_rate():
    tensor = torch.tensor([[1.0, -2.0, 0.2], [0.0, 3.5, -3.5]], dtype=torch.float32)
    qdq, scales, clip_rate = fake_quantize_activation_per_token(tensor, num_bits=8)

    assert qdq.shape == tensor.shape
    assert scales.shape == (2, 1)
    assert 0.0 <= float(clip_rate) <= 1.0
    assert torch.allclose(qdq, tensor, atol=0.1)
    expected_scales = torch.tensor([[2.0 / 127.0], [3.5 / 127.0]])
    torch.testing.assert_close(scales, expected_scales, atol=1e-4, rtol=0.0)


def test_compute_fvu_scalar_matches_reference():
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    recon = torch.tensor([[1.0, 1.0], [2.0, 4.0]], dtype=torch.float32)

    fvu = compute_fvu_scalar(target, recon)
    total_variance = (target - target.mean(0)).pow(2).sum()
    expected = ((target - recon).pow(2).sum() / total_variance).item()

    assert fvu.item() == pytest.approx(expected)


def test_compute_fvu_scalar_shape_mismatch():
    target = torch.randn(2, 4)
    recon = torch.randn(3, 4)
    with pytest.raises(ValueError, match="target and recon must share shape"):
        compute_fvu_scalar(target, recon)


def test_compute_exceed_ratio_shape_mismatch():
    target = torch.randn(2, 4)
    recon = torch.randn(2, 5)
    with pytest.raises(ValueError, match="target and recon must share shape"):
        compute_exceed_ratio(target, recon, threshold=0.1)


def test_compute_fvu_scalar_vector_matches_batch():
    vector = torch.tensor([1.0, 2.0, 3.0])
    recon = torch.tensor([0.5, 1.5, 2.5])
    fvu_vec = compute_fvu_scalar(vector, recon)
    fvu_batch = compute_fvu_scalar(vector.unsqueeze(0), recon.unsqueeze(0))
    assert float(fvu_vec) == pytest.approx(float(fvu_batch))


def test_fake_quantize_activation_rejects_invalid_bits():
    with pytest.raises(ValueError, match="num_bits must be"):
        fake_quantize_activation_per_token(torch.zeros(1, 4), num_bits=1)


def test_fake_quant_has_ste_gradients():
    tensor = torch.tensor([[0.2, -0.7], [1.0, -1.5]], dtype=torch.float32, requires_grad=True)
    qdq, _, _ = fake_quantize_activation_per_token(tensor, num_bits=8)
    qdq.sum().backward()
    assert torch.allclose(tensor.grad, torch.ones_like(tensor))


def test_fake_quant_clip_boundary_saturation():
    tensor = torch.tensor([[1.0, -127.0], [63.5, -1.0]], dtype=torch.float32)
    _, _, clip_rate = fake_quantize_activation_per_token(tensor, num_bits=8)
    assert float(clip_rate) == pytest.approx(0.5)


def test_compute_fvu_scalar_handles_ndims():
    target = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32
    )
    recon = target - 0.5
    flat_target = target.flatten(0, -2)
    flat_recon = recon.flatten(0, -2)
    mean = flat_target.mean(dim=0, keepdim=True)
    variance = (flat_target - mean).pow(2).sum()
    mse = (flat_target - flat_recon).pow(2).sum()
    expected = (mse / variance).item()
    fvu = compute_fvu_scalar(target, recon)
    assert float(fvu) == pytest.approx(expected)


def test_compute_exceed_ratio_bounds_and_threshold():
    target = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    recon = torch.tensor([[0.0, 0.5], [2.1, 3.2]], dtype=torch.float32)

    ratio = compute_exceed_ratio(target, recon, threshold=0.1)
    assert 0.0 <= float(ratio) <= 1.0

    high_threshold = compute_exceed_ratio(target, recon, threshold=10.0)
    assert high_threshold == pytest.approx(0.0)


def test_summarize_io_quant_batch_outputs_metrics():
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    recon = torch.tensor([[1.0, 1.5], [2.5, 4.0]], dtype=torch.float32)

    metrics = summarize_io_quant_batch(
        target_fp=target,
        recon_fp=recon,
        num_bits=8,
        alpha=0.5,
        elbow_value=0.5,
        deploy_weight=0.25,
    )

    assert isinstance(metrics, IOQuantMetrics)
    assert metrics.recon_deploy.shape == target.shape
    assert metrics.target_deploy.shape == target.shape
    assert metrics.fvu_fp_teacher.ndim == 0
    assert metrics.fvu_deploy.ndim == 0
    assert metrics.quant_floor.ndim == 0
    assert metrics.main_loss.ndim == 0
    assert metrics.exceed_fp_teacher is not None
    assert metrics.exceed_deploy is not None
    assert 0.0 <= float(metrics.input_clip_rate) <= 1.0
    assert 0.0 <= float(metrics.output_clip_rate) <= 1.0
    expected_input_mean = (2.0 / 127.0 + 4.0 / 127.0) / 2.0
    expected_output_mean = (1.5 / 127.0 + 4.0 / 127.0) / 2.0
    assert float(metrics.input_scale_mean) == pytest.approx(expected_input_mean)
    assert float(metrics.output_scale_mean) == pytest.approx(expected_output_mean)


def test_summarize_io_quant_batch_handles_missing_alpha_elbow():
    target = torch.randn(3, 4)
    recon = torch.randn(3, 4)

    metrics = summarize_io_quant_batch(
        target_fp=target,
        recon_fp=recon,
        num_bits=8,
        alpha=None,
        elbow_value=None,
        deploy_weight=1.0,
    )

    assert metrics.exceed_fp_teacher is None
    assert metrics.exceed_deploy is None
