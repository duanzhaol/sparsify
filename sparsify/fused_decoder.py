"""Fused decoder with custom backward pass for NPU compatibility.

The standard F.embedding_bag backward (aten::_embedding_bag_backward) is not
natively supported on Ascend NPU and falls back to CPU. This module provides
FusedDecoder, a custom autograd Function that replaces the backward pass with
index_add_ and gather operations that ARE natively supported on NPU.

The approach mirrors FusedEncoder (fused_encoder.py), which uses the same
strategy for the encoder side.
"""

import torch
import torch.nn.functional as F


class FusedDecoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, top_indices, top_acts, W_T):
        """
        top_indices: (N, k)         indices of top-k latents
        top_acts:    (N, k)         activation values of top-k latents
        W_T:         (M, d_in)      decoder weight (num_latents rows, d_in cols)

        Returns:     (N, d_in)      reconstructed output
        """
        out = F.embedding_bag(
            top_indices, W_T, per_sample_weights=top_acts, mode="sum"
        )
        ctx.save_for_backward(top_indices, top_acts, W_T)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        top_indices, top_acts, W_T = ctx.saved_tensors
        grad_acts = grad_W_T = None
        k = top_indices.shape[-1]

        # --- Grad w.r.t. top_acts ---
        # grad_acts[n, i] = dot(grad_output[n], W_T[top_indices[n, i]])
        if ctx.needs_input_grad[1]:
            grad_acts = torch.empty_like(top_acts)
            for i in range(k):
                selected = W_T[top_indices[:, i]]  # (N, d_in)
                grad_acts[:, i] = (grad_output * selected).sum(-1)

        # --- Grad w.r.t. W_T ---
        # grad_W_T[top_indices[n, i]] += top_acts[n, i] * grad_output[n]
        if ctx.needs_input_grad[2]:
            grad_W_T = torch.zeros_like(W_T)
            for i in range(k):
                contrib = top_acts[:, i].unsqueeze(-1) * grad_output  # (N, d_in)
                grad_W_T.index_add_(
                    0, top_indices[:, i], contrib.type_as(W_T)
                )

        return None, grad_acts, grad_W_T


def fused_decode(
    top_indices: torch.Tensor, top_acts: torch.Tensor, W_dec: torch.Tensor
) -> torch.Tensor:
    """Drop-in replacement for eager_decode with NPU-native backward.

    Same signature as eager_decode / triton_decode:
        fused_decode(top_indices, top_acts, W_dec) -> (N, d_in)

    W_dec arrives as self.W_dec.mT from the call site (shape [d_in, M]).
    We transpose it to [M, d_in] for embedding_bag, matching eager_decode's
    internal convention. The .mT is outside the custom Function so autograd
    handles the transpose gradient automatically.
    """
    return FusedDecoder.apply(top_indices, top_acts, W_dec.mT)  # type: ignore
