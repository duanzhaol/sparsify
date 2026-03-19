"""JumpReLU Sparse Coder (fixedK) — per-feature learnable thresholds with STE."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput
from .sparse_coder import SparseCoder


class JumpReLUSparseCoder(SparseCoder):
    """JumpReLU SAE with per-feature learnable thresholds.

    z_i = ReLU(pre_i) * H(pre_i - θ_i)

    where H is the Heaviside step function, approximated by sigmoid for backward
    (straight-through estimator). This is the fixedK variant: output is still top-K
    for fixed tensor shapes, compatible with the standard decode() interface.

    Native variable-K mode (where the number of active features varies per sample)
    would require changes to decode/ForwardOutput and is left for future work.
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
        self.threshold = nn.Parameter(
            torch.full(
                (self.num_latents,),
                cfg.jumprelu_init_threshold,
                device=device,
                dtype=dtype,
            )
        )

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode with learnable per-feature thresholds and STE backward."""
        x = x - self.b_dec
        pre_acts = F.linear(x, self.encoder.weight, self.encoder.bias)

        # JumpReLU: relu(z) * H(z - θ)
        relu_acts = F.relu(pre_acts)

        # Hard mask (forward path)
        mask_hard = (pre_acts > self.threshold).float()

        if self.training:
            # STE: forward uses hard mask, backward flows through sigmoid approx
            mask_soft = torch.sigmoid(
                (pre_acts - self.threshold) / self.cfg.jumprelu_bandwidth
            )
            mask = mask_hard + (mask_soft - mask_soft.detach())
        else:
            mask = mask_hard

        acts = relu_acts * mask

        # Output top-K for fixed tensor shape (fixedK mode)
        top_acts, top_indices = acts.topk(self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, pre_acts)
