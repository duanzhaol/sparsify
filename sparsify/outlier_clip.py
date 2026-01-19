"""Online outlier clipping for SAE training.

This module provides the OutlierClipper class which separates outlier dimensions
from SAE training, allowing the SAE to focus on the main distribution of activations.
"""

import torch
from torch import Tensor


class OutlierClipper:
    """Online outlier clipper using EMA statistics for dynamic thresholds.

    For each dimension, computes threshold_d = k * RMS_d where RMS_d is estimated
    via exponential moving average of squared values. Values exceeding the threshold
    are separated into a residual tensor.
    """

    def __init__(
        self,
        d_in: int,
        k: float = 3.0,
        ema_alpha: float = 0.01,
        warmup_steps: int = 100,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the outlier clipper.

        Args:
            d_in: Input dimension
            k: Threshold multiplier (threshold = k * RMS)
            ema_alpha: EMA decay factor for RMS estimation
            warmup_steps: Number of warmup steps with larger alpha for faster convergence
            device: Device to store statistics on
            dtype: Data type for statistics
        """
        self.d_in = d_in
        self.k = k
        self.ema_alpha = ema_alpha
        self.warmup_steps = warmup_steps

        # EMA statistics: E[|x|^2] per dimension
        self.ema_sq = torch.zeros(d_in, device=device, dtype=dtype)
        self.step_count = 0
        self.initialized = False

    def update_stats(self, x: Tensor) -> None:
        """Update EMA statistics with a batch of activations.

        Args:
            x: Input tensor [batch, d_in]
        """
        x_sq = (x.float().abs() ** 2).mean(dim=0)  # [d_in]

        if not self.initialized:
            self.ema_sq = x_sq.to(self.ema_sq.device, self.ema_sq.dtype)
            self.initialized = True
        else:
            alpha = self.ema_alpha
            # Warmup period uses larger alpha for faster convergence
            if self.step_count < self.warmup_steps:
                alpha = max(self.ema_alpha, 1.0 / (self.step_count + 1))
            self.ema_sq = (1 - alpha) * self.ema_sq + alpha * x_sq.to(self.ema_sq)

        self.step_count += 1

    @property
    def threshold(self) -> Tensor:
        """Compute current threshold: k * RMS per dimension."""
        return self.k * torch.sqrt(self.ema_sq + 1e-8)

    @property
    def rms(self) -> Tensor:
        """Current RMS estimate per dimension."""
        return torch.sqrt(self.ema_sq + 1e-8)

    def clip(
        self,
        x: Tensor,
        update_stats: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Clip outlier values from input.

        Args:
            x: Input tensor [batch, d_in]
            update_stats: Whether to update EMA statistics (True for training, False for inference)

        Returns:
            x_inlier: Main distribution with outlier dims zeroed [batch, d_in]
            residual: Outlier residual with non-outlier dims zeroed [batch, d_in]
            mask: Outlier mask (1 = outlier, 0 = normal) [batch, d_in]
        """
        if update_stats:
            self.update_stats(x)

        threshold = self.threshold.to(x.device, x.dtype)
        mask = (x.abs() > threshold).float()

        x_inlier = x * (1 - mask)
        residual = x * mask

        return x_inlier, residual, mask

    def to(self, device: str | torch.device) -> "OutlierClipper":
        """Move statistics to specified device."""
        self.ema_sq = self.ema_sq.to(device)
        return self

    def state_dict(self) -> dict:
        """Serialize clipper state for checkpointing."""
        return {
            "d_in": self.d_in,
            "k": self.k,
            "ema_alpha": self.ema_alpha,
            "warmup_steps": self.warmup_steps,
            "ema_sq": self.ema_sq.cpu(),
            "step_count": self.step_count,
            "initialized": self.initialized,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict,
        device: str | torch.device = "cpu"
    ) -> "OutlierClipper":
        """Deserialize clipper state from checkpoint."""
        instance = cls(
            d_in=state["d_in"],
            k=state["k"],
            ema_alpha=state["ema_alpha"],
            warmup_steps=state.get("warmup_steps", 100),
            device=device,
        )
        instance.ema_sq = state["ema_sq"].to(device)
        instance.step_count = state["step_count"]
        instance.initialized = state["initialized"]
        return instance
