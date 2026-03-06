from typing import NamedTuple

import torch
import torch.nn.functional as F


class EncoderOutput(NamedTuple):
    top_acts: torch.Tensor
    """Activations of the top-k latents."""

    top_indices: torch.Tensor
    """Indices of the top-k features."""

    pre_acts: torch.Tensor
    """Activations before the top-k selection."""


_MATMUL_THRESHOLD = 256 * 1024 * 1024  # 256 MB


class FusedEncoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, k: int):
        """
        input:  (N, D)
        weight: (M, D)
        bias:   (M,)
        k:      int (number of top elements to select along dim=1)
        """
        preacts = F.relu(F.linear(input, weight, bias))

        # Get top-k values and indices for each row
        values, indices = torch.topk(preacts, k, dim=-1, sorted=False)

        # Save tensors needed for the backward pass
        ctx.save_for_backward(input, weight, bias, indices)
        ctx.k = k
        return values, indices, preacts

    @staticmethod
    def backward(ctx, grad_values, grad_indices, grad_preacts):
        input, weight, bias, indices = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        k = ctx.k
        N, d_in = input.shape
        M = weight.shape[0]

        # --- Grad w.r.t. input ---
        if ctx.needs_input_grad[0]:
            grad_input = F.embedding_bag(
                indices,
                weight,
                mode="sum",
                per_sample_weights=grad_values.type_as(weight),
            )

        # --- Grad w.r.t. weight (and bias via shared S matrix) ---
        use_matmul = M * N * input.element_size() <= _MATMUL_THRESHOLD
        S = None

        if ctx.needs_input_grad[1]:
            if use_matmul:
                S = torch.zeros(M, N, dtype=input.dtype, device=input.device)
                S.scatter_add_(0, indices.t().long(), grad_values.t().to(input.dtype))
                grad_weight = (S @ input).type_as(weight)
            else:
                grad_weight = torch.zeros_like(weight)
                flat_idx = indices.reshape(-1)
                gv_flat = grad_values.reshape(-1, 1)
                input_exp = input.unsqueeze(1).expand(N, k, d_in).reshape(-1, d_in)
                contrib = (gv_flat * input_exp).type_as(weight)
                grad_weight.index_add_(0, flat_idx, contrib)

        # --- Grad w.r.t. bias ---
        if bias is not None and ctx.needs_input_grad[2]:
            if S is not None:
                grad_bias = S.sum(1).type_as(bias)
            else:
                grad_bias = torch.zeros_like(bias)
                grad_bias.index_add_(
                    0, indices.flatten(), grad_values.flatten().type_as(bias)
                )

        # The k parameter is an int, so return None for its gradient.
        return grad_input, grad_weight, grad_bias, None


def fused_encoder(
    input,
    weight,
    bias,
    k: int,
) -> EncoderOutput:
    """
    Convenience wrapper that performs an nn.Linear followed by top-k with
    a backward pass optimized using scatter-matmul.
    """
    return EncoderOutput(
        *FusedEncoder.apply(input, weight, bias, k)  # type: ignore
    )
