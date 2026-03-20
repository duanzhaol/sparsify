"""Refined-coefficient TopK sparse coder."""

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput
from .sparse_coder import SparseCoder


class RefinedTopKSparseCoder(SparseCoder):
    """TopK SAE with ridge-refined coefficients on the selected support.

    The linear encoder still chooses the sparse support, but the final
    coefficients are recomputed against the decoder atoms on that support. This
    targets coefficient error directly, which recent support-selection variants
    have not improved.
    """

    ridge_lambda: float = 1e-3

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

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode with baseline support, then refit acts on that support."""
        x = x - self.b_dec
        pre_acts = F.relu(F.linear(x, self.encoder.weight, self.encoder.bias))
        top_acts, top_indices = pre_acts.topk(self.cfg.k, dim=-1, sorted=False)

        # Solve a small ridge system on the selected decoder atoms to reduce
        # coefficient error while preserving the linear top-k support.
        solve_dtype = torch.float32
        selected_decoder = self.W_dec[top_indices].to(solve_dtype)
        x_solve = x.to(solve_dtype)
        gram = torch.matmul(selected_decoder, selected_decoder.transpose(-1, -2))
        eye = torch.eye(self.cfg.k, device=x.device, dtype=solve_dtype).expand_as(gram)
        rhs = torch.matmul(selected_decoder, x_solve.unsqueeze(-1))
        refined_acts = torch.linalg.solve(
            gram + self.ridge_lambda * eye,
            rhs,
        ).squeeze(-1).to(top_acts.dtype)
        refined_acts = refined_acts + (top_acts - top_acts.detach())

        return EncoderOutput(refined_acts, top_indices, pre_acts)
