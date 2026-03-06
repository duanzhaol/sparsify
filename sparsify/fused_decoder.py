"""Fused decoder with custom backward pass for NPU compatibility.

The standard F.embedding_bag backward (aten::_embedding_bag_backward) is not
natively supported on Ascend NPU and falls back to CPU. This module provides
FusedDecoder, a custom autograd Function that replaces the backward pass with
NPU-native operations.

Backward strategy (when M*N fits in memory):
  - grad_acts:  full matmul grad_output @ W_T.T then scalar gather (AI_CORE)
  - grad_W_T:   scatter scalars into (M, N) coefficient matrix, then dense
                matmul S @ grad_output (AI_CORE)
  Falls back to gather+bmm / vectorized index_add_ when M*N exceeds threshold.
"""

import torch
import torch.nn.functional as F


_MATMUL_THRESHOLD = 256 * 1024 * 1024  # 256 MB


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
        N, k = top_indices.shape
        d_in = W_T.shape[-1]
        M = W_T.shape[0]
        use_matmul = M * N * grad_output.element_size() <= _MATMUL_THRESHOLD

        # --- Grad w.r.t. top_acts ---
        if ctx.needs_input_grad[1]:
            if use_matmul:
                full_scores = grad_output @ W_T.t()
                grad_acts = full_scores.gather(1, top_indices.long())
            else:
                selected = W_T[top_indices]
                grad_acts = torch.bmm(
                    grad_output.unsqueeze(1),
                    selected.transpose(1, 2)
                ).squeeze(1)

        # --- Grad w.r.t. W_T ---
        if ctx.needs_input_grad[2]:
            if use_matmul:
                S = torch.zeros(M, N, dtype=grad_output.dtype, device=grad_output.device)
                S.scatter_add_(0, top_indices.t().long(), top_acts.t().to(grad_output.dtype))
                grad_W_T = (S @ grad_output).type_as(W_T)
            else:
                grad_W_T = torch.zeros_like(W_T)
                flat_idx = top_indices.reshape(-1)
                acts_flat = top_acts.reshape(-1, 1)
                grad_exp = grad_output.unsqueeze(1).expand(N, k, d_in).reshape(-1, d_in)
                contrib = (acts_flat * grad_exp).type_as(W_T)
                grad_W_T.index_add_(0, flat_idx, contrib)

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
