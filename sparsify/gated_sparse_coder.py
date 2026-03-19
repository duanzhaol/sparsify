"""Gated Sparse Coder — independent gate and magnitude branches."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput
from .sparse_coder import ForwardOutput, SparseCoder


class GatedSparseCoder(SparseCoder):
    """Gated SAE with independent gate (selection) and magnitude (coefficient) branches.

    gate:      g = TopK(σ(W_gate · x + b_gate))   — continuous selection scores
    magnitude: m = ReLU(W_mag · x + b_mag)         — coefficient values
    output:    z = g ⊙ m

    The gate and magnitude branches are fully independent linear projections.
    Decoder is shared (same as base SparseCoder).
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
        # Skip SparseCoder.__init__ to avoid creating the unused self.encoder
        nn.Module.__init__(self)
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        # Gate branch: sigmoid → topk for selection
        self.W_gate = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.W_gate.bias.data.zero_()

        # Magnitude branch: ReLU for coefficients
        self.W_mag = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.W_mag.bias.data.zero_()

        # Decoder
        if decoder:
            self.W_dec = nn.Parameter(self.W_mag.weight.data.clone())
            if cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode using independent gate and magnitude branches."""
        x = x - self.b_dec

        # Gate: sigmoid scores, then topk for sparsity
        gate_scores = torch.sigmoid(self.W_gate(x))
        top_gate_vals, top_indices = gate_scores.topk(self.cfg.k, dim=-1, sorted=False)

        # Magnitude: ReLU activations at selected positions
        mag_pre = F.relu(self.W_mag(x))
        top_mag = mag_pre.gather(-1, top_indices)

        # Final activations = gate * magnitude
        top_acts = top_gate_vals * top_mag

        # pre_acts uses magnitude branch (for auxk dead feature detection)
        return EncoderOutput(top_acts, top_indices, mag_pre)

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """Separate param groups for gate and magnitude (same LR for now)."""
        return [{"params": self.parameters(), "lr": base_lr}]
