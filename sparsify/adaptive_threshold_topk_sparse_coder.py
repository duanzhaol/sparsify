"""Adaptive-threshold TopK sparse coder."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput
from .sparse_coder import SparseCoder


class AdaptiveThresholdTopKSparseCoder(SparseCoder):
    """TopK SAE with a baseline linear path and learned input-dependent suppression.

    The linear encoder remains the anchor. A small nonlinear branch predicts a
    nonnegative per-latent threshold that only suppresses linear logits, so the
    family starts exactly at the plain TopK behavior and can learn cleaner support
    without needing an unconstrained additive residual path.
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
        self.threshold_hidden = nn.Linear(d_in, hidden_dim, device=device, dtype=dtype)
        self.threshold_hidden.bias.data.zero_()
        self.threshold_gate = nn.Linear(d_in, hidden_dim, device=device, dtype=dtype)
        self.threshold_gate.bias.data.zero_()
        self.threshold_out = nn.Linear(
            hidden_dim, self.num_latents, device=device, dtype=dtype
        )
        self.threshold_out.weight.data.zero_()
        self.threshold_out.bias.data.zero_()

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode with linear logits minus a learned nonnegative threshold."""
        x = x - self.b_dec
        linear_logits = F.linear(x, self.encoder.weight, self.encoder.bias)
        hidden = F.silu(self.threshold_hidden(x)) * torch.sigmoid(
            self.threshold_gate(x)
        )
        suppression = F.relu(self.threshold_out(hidden))
        pre_acts = F.relu(linear_logits - suppression)
        top_acts, top_indices = pre_acts.topk(self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, pre_acts)
