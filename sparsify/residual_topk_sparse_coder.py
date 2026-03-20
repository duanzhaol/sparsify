"""Residual two-branch TopK sparse coder."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput
from .sparse_coder import SparseCoder
from .utils import decoder_impl


class ResidualTopKSparseCoder(SparseCoder):
    """Run a second top-k encoder pass on the first branch residual.

    The first branch behaves like a standard TopK encoder. Its partial
    reconstruction is subtracted from the centered input and a second encoder
    proposes residual-focused features. The final support is selected globally
    from the union of both candidate sets.
    """

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device, dtype, decoder=decoder)
        self.residual_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.residual_encoder.bias.data.zero_()

    def _encode_branch(self, x: Tensor, encoder: nn.Linear, k: int) -> tuple[Tensor, Tensor, Tensor]:
        pre_acts = F.relu(F.linear(x, encoder.weight, encoder.bias))
        top_acts, top_indices = pre_acts.topk(k, dim=-1, sorted=False)
        return pre_acts, top_acts, top_indices

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode from the centered input, then refine with a residual branch."""
        x_centered = x - self.b_dec

        # Let both branches over-propose candidates, then keep only the final
        # global top-k. This removes the fixed half-and-half split, which can
        # otherwise block the better branch from using more of the support.
        branch_k = min(self.cfg.k, self.num_latents)

        first_pre, first_acts, first_indices = self._encode_branch(
            x_centered, self.encoder, branch_k
        )

        partial_recon = decoder_impl(
            first_indices, first_acts.to(self.dtype), self.W_dec.mT
        )
        residual_input = x_centered - partial_recon

        second_pre, second_acts, second_indices = self._encode_branch(
            residual_input, self.residual_encoder, branch_k
        )

        merged_acts = torch.cat((first_acts, second_acts), dim=-1)
        merged_indices = torch.cat((first_indices, second_indices), dim=-1)
        top_acts, top_pos = merged_acts.topk(self.cfg.k, dim=-1, sorted=False)
        top_indices = merged_indices.gather(-1, top_pos)

        pre_acts = torch.maximum(first_pre, second_pre)
        return EncoderOutput(top_acts, top_indices, pre_acts)
