"""Group TopK Sparse Coder (v0/hard routing) — router selects groups, global top-K within."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput
from .sparse_coder import SparseCoder


class GroupTopKSparseCoder(SparseCoder):
    """Group TopK SAE with independent group router.

    1) Router R ∈ R^{G×h} scores groups, select top-g groups
    2) Global top-K within the union of selected groups' latents

    v0 (hard routing): The router's topk group selection is not differentiable.
    The encoder gets gradients through selected groups' pre_acts, but the router
    weights do NOT receive gradients from FVU loss. v0 primarily validates the
    training framework, logging, and artifact pipeline. Its training effectiveness
    should NOT be used as a formal judgment of structured SAE feasibility.

    v1 (future): Introduce differentiable routing (Gumbel-Softmax, STE on group
    selection, or auxiliary load-balance loss) for effective router training.
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

        # Independent group router (small: G × d_in)
        self.group_router = nn.Linear(
            d_in, cfg.num_groups, device=device, dtype=dtype
        )
        self.group_router.bias.data.zero_()

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode with group routing and global top-K within selected groups."""
        x_centered = x - self.b_dec

        # Full pre-activations (needed for auxk and correctness in v0)
        pre_acts = F.relu(
            F.linear(x_centered, self.encoder.weight, self.encoder.bias)
        )

        # 1) Router selects top-g groups (hard, not differentiable)
        group_scores = self.group_router(x_centered)  # [batch, G]
        _, top_groups = group_scores.topk(
            self.cfg.active_groups, dim=-1, sorted=False
        )  # [batch, g]

        # 2) Build latent-level mask from selected groups
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)  # [batch, G]
        group_mask.scatter_(1, top_groups, True)
        latent_mask = group_mask.repeat_interleave(
            self.group_size, dim=1
        )  # [batch, num_latents]

        # 3) Global top-K within selected groups' union
        masked_pre_acts = pre_acts.masked_fill(~latent_mask, -float("inf"))
        top_acts, top_indices = masked_pre_acts.topk(
            self.cfg.k, dim=-1, sorted=False
        )

        return EncoderOutput(top_acts, top_indices, pre_acts)
