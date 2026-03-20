"""Hybrid TopK sparse coder with linear and nonlinear encoder paths."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput
from .sparse_coder import SparseCoder


class HybridTopKSparseCoder(SparseCoder):
    """TopK SAE with a baseline linear path plus a gated nonlinear residual path.

    The base linear encoder preserves the proven TopK behavior. A gated MLP path
    adds a residual correction to the latent logits so the family can express
    nonlinear routing without discarding the strong linear anchor.
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
        self.encoder_hidden = nn.Linear(d_in, hidden_dim, device=device, dtype=dtype)
        self.encoder_hidden.bias.data.zero_()
        self.encoder_gate = nn.Linear(d_in, hidden_dim, device=device, dtype=dtype)
        self.encoder_gate.bias.data.zero_()
        self.encoder_residual = nn.Linear(
            hidden_dim, self.num_latents, device=device, dtype=dtype
        )
        self.encoder_residual.bias.data.zero_()
        self.encoder_modulation = nn.Linear(
            hidden_dim, self.num_latents, device=device, dtype=dtype
        )
        self.encoder_modulation.weight.data.zero_()
        self.encoder_modulation.bias.data.zero_()

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode with linear logits plus a gated nonlinear residual."""
        x = x - self.b_dec
        linear_logits = F.linear(x, self.encoder.weight, self.encoder.bias)
        hidden = F.silu(self.encoder_hidden(x)) * torch.sigmoid(self.encoder_gate(x))
        residual_logits = self.encoder_residual(hidden)
        modulation = torch.tanh(self.encoder_modulation(hidden))
        combined_logits = linear_logits * (1.0 + modulation) + residual_logits
        pre_acts = F.relu(combined_logits)
        top_acts, top_indices = pre_acts.topk(self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, pre_acts)
