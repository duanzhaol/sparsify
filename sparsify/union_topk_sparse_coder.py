"""Union TopK sparse coder."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput
from .sparse_coder import SparseCoder


class UnionTopKSparseCoder(SparseCoder):
    """TopK SAE with separate linear and nonlinear candidate proposal branches.

    The linear branch preserves the proven baseline support. A nonlinear branch
    proposes an alternate set of candidates from the same input, and the final
    support is selected from the union of both proposal sets rather than by
    directly perturbing the linear logits.
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
        hidden_dim = min(self.num_latents, d_in * 2)
        self.nonlinear_hidden = nn.Linear(d_in, hidden_dim, device=device, dtype=dtype)
        self.nonlinear_hidden.bias.data.zero_()
        self.nonlinear_gate = nn.Linear(d_in, hidden_dim, device=device, dtype=dtype)
        self.nonlinear_gate.bias.data.zero_()
        self.nonlinear_out = nn.Linear(
            hidden_dim, self.num_latents, device=device, dtype=dtype
        )
        self.nonlinear_out.bias.data.zero_()

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode by taking a global top-k from linear and nonlinear proposals."""
        x = x - self.b_dec

        linear_pre = F.relu(F.linear(x, self.encoder.weight, self.encoder.bias))
        hidden = F.silu(self.nonlinear_hidden(x)) * torch.sigmoid(
            self.nonlinear_gate(x)
        )
        nonlinear_pre = F.relu(self.nonlinear_out(hidden))

        linear_acts, linear_indices = linear_pre.topk(self.cfg.k, dim=-1, sorted=False)
        nonlinear_acts, nonlinear_indices = nonlinear_pre.topk(
            self.cfg.k, dim=-1, sorted=False
        )

        merged_acts = torch.cat((linear_acts, nonlinear_acts), dim=-1)
        merged_indices = torch.cat((linear_indices, nonlinear_indices), dim=-1)
        top_acts, top_pos = merged_acts.topk(self.cfg.k, dim=-1, sorted=False)
        top_indices = merged_indices.gather(-1, top_pos)

        pre_acts = torch.maximum(linear_pre, nonlinear_pre)
        return EncoderOutput(top_acts, top_indices, pre_acts)
