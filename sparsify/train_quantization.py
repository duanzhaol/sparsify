from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class IOQuantMetrics:
    recon_deploy: Tensor
    target_deploy: Tensor
    fvu_fp_teacher: Tensor
    fvu_deploy: Tensor
    quant_floor: Tensor
    exceed_fp_teacher: Tensor | None
    exceed_deploy: Tensor | None
    input_clip_rate: Tensor
    output_clip_rate: Tensor
    input_scale_mean: Tensor
    output_scale_mean: Tensor
    main_loss: Tensor


def _symmetric_qmax(num_bits: int) -> float:
    if num_bits < 2:
        raise ValueError("num_bits must be >= 2 for symmetric quantization")
    return float((1 << (num_bits - 1)) - 1)


def fake_quantize_activation_per_token(
    x: Tensor,
    num_bits: int = 8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-token symmetric fake quantization returning qdq, scales, and clip-rate."""

    qmax = _symmetric_qmax(num_bits)
    x_fp = x.to(torch.float32)
    absmax = x_fp.abs().amax(dim=-1, keepdim=True)
    scales = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
    normalized = x_fp / scales
    clipped = normalized.clamp(-qmax, qmax)
    rounded = clipped.round()
    qdq_fp = rounded * scales
    clip_rate = (normalized.abs() > qmax).float().mean()
    qdq = x_fp + (qdq_fp - x_fp).detach()
    return qdq, scales, clip_rate


def _ensure_matching_shape(target: Tensor, recon: Tensor) -> None:
    if target.shape != recon.shape:
        raise ValueError(
            f"target and recon must share shape, got {target.shape} vs {recon.shape}"
        )


def _flatten_except_last(tensor: Tensor) -> Tensor:
    if tensor.ndim == 0:
        raise ValueError("tensor must have at least one dimension")
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor.reshape(-1, tensor.shape[-1])


def compute_fvu_scalar(target: Tensor, recon: Tensor) -> Tensor:
    _ensure_matching_shape(target, recon)
    target_fp = target.to(torch.float32)
    recon_fp = recon.to(torch.float32)

    if target_fp.ndim == 1:
        mean = target_fp.mean()
        variance = (target_fp - mean).pow(2).sum().clamp_min(1e-12)
        mse = (target_fp - recon_fp).pow(2).sum()
        return mse / variance

    target_flat = _flatten_except_last(target_fp)
    recon_flat = _flatten_except_last(recon_fp)
    mean = target_flat.mean(dim=0, keepdim=True)
    variance = (target_flat - mean).pow(2).sum().clamp_min(1e-12)
    mse = (target_flat - recon_flat).pow(2).sum()
    return mse / variance


def compute_exceed_ratio(target: Tensor, recon: Tensor, threshold: float) -> Tensor:
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    _ensure_matching_shape(target, recon)
    target_fp = target.to(torch.float32)
    recon_fp = recon.to(torch.float32)
    exceed_mask = (target_fp - recon_fp).abs() > threshold
    return exceed_mask.float().mean()


def summarize_io_quant_batch(
    target_fp: Tensor,
    recon_fp: Tensor,
    *,
    num_bits: int,
    alpha: float | None,
    elbow_value: float | None,
    deploy_weight: float,
) -> IOQuantMetrics:
    target_fp = target_fp.to(torch.float32)
    recon_fp = recon_fp.to(torch.float32)

    target_deploy, input_scales, input_clip = fake_quantize_activation_per_token(
        target_fp, num_bits=num_bits
    )
    recon_deploy, output_scales, output_clip = fake_quantize_activation_per_token(
        recon_fp, num_bits=num_bits
    )

    fvu_fp_teacher = compute_fvu_scalar(target_fp, recon_deploy)
    fvu_deploy = compute_fvu_scalar(target_deploy, recon_deploy)
    quant_floor = compute_fvu_scalar(target_fp, target_deploy)

    exceed_fp_teacher: Tensor | None = None
    exceed_deploy: Tensor | None = None
    if alpha is not None and elbow_value is not None:
        threshold = alpha * elbow_value
        exceed_fp_teacher = compute_exceed_ratio(target_fp, recon_deploy, threshold)
        exceed_deploy = compute_exceed_ratio(target_deploy, recon_deploy, threshold)

    main_loss = fvu_fp_teacher + deploy_weight * fvu_deploy

    return IOQuantMetrics(
        recon_deploy=recon_deploy,
        target_deploy=target_deploy,
        fvu_fp_teacher=fvu_fp_teacher,
        fvu_deploy=fvu_deploy,
        quant_floor=quant_floor,
        exceed_fp_teacher=exceed_fp_teacher,
        exceed_deploy=exceed_deploy,
        input_clip_rate=input_clip,
        output_clip_rate=output_clip,
        input_scale_mean=input_scales.mean(),
        output_scale_mean=output_scales.mean(),
        main_loss=main_loss,
    )
