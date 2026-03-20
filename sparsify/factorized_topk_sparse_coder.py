"""Factorized nonlinear TopK sparse coder."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput
from .sparse_coder import SparseCoder


class FactorizedTopKSparseCoder(SparseCoder):
    """TopK SAE with a low-rank nonlinear encoder.

    A widened gated hidden layer lets the encoder compose input dimensions
    before the final latent projection, increasing routing expressivity without
    changing the fixed-K decode path or requiring extra launcher parameters.
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
        hidden_dim = min(self.num_latents, d_in * 4)
        self.encoder.requires_grad_(False)
        self.encoder_hidden = nn.Linear(d_in, hidden_dim, device=device, dtype=dtype)
        self.encoder_hidden.bias.data.zero_()
        self.encoder_gate = nn.Linear(d_in, hidden_dim, device=device, dtype=dtype)
        self.encoder_gate.bias.data.zero_()
        self.encoder_out = nn.Linear(
            hidden_dim, self.num_latents, device=device, dtype=dtype
        )
        self.encoder_out.bias.data.zero_()

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """Exclude the unused base encoder from optimization."""
        return [{
            "params": (
                list(self.encoder_hidden.parameters())
                + list(self.encoder_gate.parameters())
                + list(self.encoder_out.parameters())
                + [self.W_dec, self.b_dec]
            ),
            "lr": base_lr,
        }]

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode with a widened gated hidden mixing stage before latent scoring."""
        x = x - self.b_dec
        hidden = F.silu(self.encoder_hidden(x)) * torch.sigmoid(self.encoder_gate(x))
        pre_acts = F.relu(self.encoder_out(hidden))
        top_acts, top_indices = pre_acts.topk(self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, pre_acts)
