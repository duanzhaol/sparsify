"""Differentiable routed Group TopK sparse coder."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput
from .sparse_coder import SparseCoder


class RoutedGroupTopKSparseCoder(SparseCoder):
    """Group-routed top-k SAE with a straight-through coarse router.

    The encoder first learns coarse group selection, then performs exact latent top-k
    within the routed groups. Forward routing is hard and capacity-limited, while the
    backward pass uses a normalized sigmoid mask so the router receives gradients.
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

        if cfg.num_groups <= 0:
            raise ValueError(f"num_groups must be > 0, got {cfg.num_groups}")
        if self.num_latents % cfg.num_groups != 0:
            raise ValueError(
                f"num_latents ({self.num_latents}) must be divisible by "
                f"num_groups ({cfg.num_groups})"
            )

        self.group_size = self.num_latents // cfg.num_groups

        selectable = cfg.active_groups * self.group_size
        if cfg.k > selectable:
            raise ValueError(
                f"k ({cfg.k}) > active_groups * group_size "
                f"({cfg.active_groups} * {self.group_size} = {selectable}). "
                f"This would select -inf values from masked groups."
            )

        self.group_router = nn.Linear(
            d_in, cfg.num_groups, device=device, dtype=dtype
        )
        self.group_router.bias.data.zero_()

    def _group_mask(self, group_logits: Tensor) -> Tensor:
        _, top_groups = group_logits.topk(self.cfg.active_groups, dim=-1, sorted=False)

        hard_mask = torch.zeros_like(group_logits)
        hard_mask.scatter_(1, top_groups, 1.0)

        if not self.training:
            return hard_mask

        # Keep the soft mask mass near active_groups so the router learns a
        # capacity-constrained allocation rather than opening every group.
        soft_mask = torch.sigmoid(group_logits)
        soft_mass = soft_mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        soft_mask = (soft_mask * (self.cfg.active_groups / soft_mass)).clamp_(0.0, 1.0)
        return hard_mask + (soft_mask - soft_mask.detach())

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode with differentiable group routing and exact latent top-k."""
        x_centered = x - self.b_dec
        pre_acts = F.relu(F.linear(x_centered, self.encoder.weight, self.encoder.bias))

        group_logits = self.group_router(x_centered)
        group_mask = self._group_mask(group_logits)
        latent_mask = group_mask.repeat_interleave(self.group_size, dim=1)

        routed_pre_acts = pre_acts * latent_mask
        top_acts, top_indices = routed_pre_acts.topk(self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, routed_pre_acts)
