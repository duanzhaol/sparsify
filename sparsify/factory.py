"""Factory for creating SAE instances."""

import torch

from .sparse_coder import SparseCoder
from .switch_sae import SwitchSAE
from .config import TrainConfig


def create_sae(
    d_in: int,
    cfg: TrainConfig,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    **kwargs,
) -> SparseCoder | SwitchSAE:
    """
    Create SAE or Switch SAE based on configuration.

    Args:
        d_in: Input dimension
        cfg: Training configuration
        device: Device to create the model on
        dtype: Data type for model parameters
        **kwargs: Additional arguments passed to the constructor

    Returns:
        SparseCoder or SwitchSAE instance
    """
    if cfg.use_switch_sae:
        return SwitchSAE(
            d_in=d_in,
            cfg=cfg.sae,
            num_experts=cfg.num_experts,
            load_balance_alpha=cfg.load_balance_alpha,
            device=device,
            dtype=dtype,
            **kwargs,
        )
    else:
        return SparseCoder(
            d_in=d_in,
            cfg=cfg.sae,
            device=device,
            dtype=dtype,
            **kwargs,
        )
