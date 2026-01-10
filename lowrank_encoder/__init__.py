"""Low-rank encoder module for SAE distillation.

This module provides components for training low-rank sparse autoencoders
through knowledge distillation from full-rank teachers.
"""

from .lowrank_encoder import (
    LowRankFusedEncoder,
    LowRankSparseCoder,
    lowrank_fused_encoder,
    from_pretrained_lowrank,
    compute_distillation_loss,
    DistillationLoss,
)

__all__ = [
    "LowRankFusedEncoder",
    "LowRankSparseCoder",
    "lowrank_fused_encoder",
    "from_pretrained_lowrank",
    "compute_distillation_loss",
    "DistillationLoss",
]
