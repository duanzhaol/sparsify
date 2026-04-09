from __future__ import annotations

import torch
from torch import Tensor


_QMAX = 127.0


def _safe_symmetric_scales(max_abs: Tensor) -> Tensor:
    max_abs = max_abs.to(torch.float32)
    return torch.where(max_abs > 0, max_abs / _QMAX, torch.ones_like(max_abs))


def quantize_weight_per_row_symmetric(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a weight tensor symmetrically per trailing row.

    The last dimension is treated as the inner-product axis. All leading
    dimensions are quantized independently.
    """
    weight_fp = weight.to(torch.float32)
    scales = _safe_symmetric_scales(weight_fp.abs().amax(dim=-1, keepdim=True))
    q_weight = torch.round(weight_fp / scales).clamp(-_QMAX, _QMAX).to(torch.int8)
    return q_weight, scales


def quantize_activation_per_token_symmetric(acts: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize activations symmetrically per token/vector."""
    acts_fp = acts.to(torch.float32)
    scales = _safe_symmetric_scales(acts_fp.abs().amax(dim=-1, keepdim=True))
    q_acts = torch.round(acts_fp / scales).clamp(-_QMAX, _QMAX).to(torch.int8)
    return q_acts, scales


def simulate_w8a8_matmul_prequantized(
    acts: Tensor,
    q_weight: Tensor,
    weight_scales: Tensor,
) -> Tensor:
    """Simulate int8 x int8 -> int32 accumulation and dequantization.

    Args:
        acts: `[B, D]` activation matrix.
        q_weight: `[..., D]` int8 weights. If the leading dimension does not
            match `B`, the weights are broadcast across the batch.
        weight_scales: Matching scales with trailing singleton dim.
    """
    q_acts, act_scales = quantize_activation_per_token_symmetric(acts)
    q_acts_i32 = q_acts.to(torch.int32)

    batch_size = q_acts.shape[0]
    if q_weight.ndim == 2 or q_weight.shape[0] != batch_size:
        q_weight = q_weight.unsqueeze(0).expand(batch_size, *q_weight.shape)
        weight_scales = weight_scales.unsqueeze(0).expand(
            batch_size, *weight_scales.shape
        )

    broadcast_dims = [1] * (q_weight.ndim - 2)
    q_acts_view = q_acts_i32.view(batch_size, *broadcast_dims, q_acts.shape[-1])
    accum = (q_acts_view * q_weight.to(torch.int32)).sum(dim=-1)
    act_scales_view = act_scales.view(batch_size, *broadcast_dims)
    dequant = (
        accum.to(torch.float32)
        * act_scales_view
        * weight_scales.squeeze(-1)
    )
    return dequant


def simulate_w8a8_matmul(acts: Tensor, weight: Tensor) -> Tensor:
    """Convenience wrapper that quantizes weights on the fly before simulating."""
    q_weight, weight_scales = quantize_weight_per_row_symmetric(weight)
    return simulate_w8a8_matmul_prequantized(acts, q_weight, weight_scales)


def simulate_w8a8_linear_prequantized(
    acts: Tensor,
    q_weight: Tensor,
    weight_scales: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Simulate a standard linear layer with symmetric W8A8 quantization."""
    q_acts, act_scales = quantize_activation_per_token_symmetric(acts)
    # PyTorch does not provide int32 addmm kernels on CUDA, so keep the
    # accumulation as explicit elementwise multiply + reduce. Router matrices
    # are small enough that this remains practical for our metric study.
    accum = (
        q_acts.to(torch.int32).unsqueeze(1)
        * q_weight.to(torch.int32).unsqueeze(0)
    ).sum(dim=-1)
    dequant = accum.to(torch.float32) * act_scales * weight_scales.squeeze(-1).unsqueeze(0)
    if bias is not None:
        dequant = dequant + bias.to(torch.float32)
    return dequant


def simulate_w8a8_linear(
    acts: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Quantize a standard linear layer on the fly and simulate W8A8 inference."""
    q_weight, weight_scales = quantize_weight_per_row_symmetric(weight)
    return simulate_w8a8_linear_prequantized(acts, q_weight, weight_scales, bias=bias)
