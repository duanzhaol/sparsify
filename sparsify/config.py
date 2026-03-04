from dataclasses import dataclass
from pathlib import Path

from simple_parsing import Serializable, list_field


@dataclass
class SparseCoderConfig(Serializable):
    """Configuration for a sparse coder (SAE) architecture."""

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the sparse coder dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""


# Support different naming conventions for the same configuration
SaeConfig = SparseCoderConfig


@dataclass
class TrainConfig(Serializable):
    sae: SparseCoderConfig

    batch_size: int = 32
    """Batch size measured in sequences."""

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for training."""

    max_tokens: int | None = None
    """Maximum number of tokens to train on. Training stops when this limit is reached."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    auxk_alpha: float = 0.0
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    # Exceed evaluation metrics
    exceed_alphas: list[float] = list_field(0.05, 0.10, 0.25, 0.50)
    """List of alpha coefficients for exceed metrics (error > alpha * elbow_value)."""

    elbow_threshold_path: str | None = None
    """Path to JSON file with pre-computed elbow thresholds per layer/operation."""

    # Hookpoint selection
    hookpoints: list[str] = list_field()
    """List of hookpoints to train sparse coders on."""

    init_seeds: list[int] = list_field(0)
    """List of random seeds to use for initialization. If more than one, train a sparse
    coder for each seed."""

    layers: list[int] = list_field()
    """List of layer indices to train sparse coders on."""

    layer_stride: int = 1
    """Stride between layers to train sparse coders on."""

    # Tiling
    num_tiles: int = 1
    """Number of tiles to split input activations. Each tile trains a separate SAE.
    d_in must be divisible by num_tiles. Set to 1 (default) for standard training."""

    global_topk: bool = False
    """Use global top-k selection across all tiles instead of per-tile top-k.
    Only effective when num_tiles > 1."""

    input_mixing: bool = False
    """Apply learnable T×T mixing matrix on tile dimension before encoding.
    Only effective when num_tiles > 1."""

    # Hadamard rotation
    use_hadamard: bool = False
    """Apply block-diagonal Hadamard rotation to activations before SAE."""

    hadamard_block_size: int = 128
    """Block size for Hadamard transform (must be power of 2)."""

    hadamard_seed: int = 0
    """Random seed for Hadamard permutation."""

    hadamard_use_perm: bool = True
    """Whether to use random permutation before Hadamard transform."""

    # Saving & logging
    save_every: int = 1000
    """Save sparse coders every `save_every` steps."""

    save_best: bool = False
    """Save the best checkpoint found for each hookpoint."""

    save_dir: str = "checkpoints"

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_project: str | None = None
    """WandB project name. If None, uses WANDB_PROJECT env var or defaults to 'sparsify'."""
    wandb_log_frequency: int = 1

    # Lifecycle
    finetune: str | None = None
    """Finetune the sparse coders from a pretrained checkpoint."""

    def __post_init__(self):
        """Validate the configuration."""
        if self.layers and self.layer_stride != 1:
            raise ValueError("Cannot specify both `layers` and `layer_stride`.")

        if not self.init_seeds:
            raise ValueError("Must specify at least one random seed.")

        if self.exceed_alphas and not all(alpha > 0 for alpha in self.exceed_alphas):
            raise ValueError("All exceed_alphas must be positive.")

        if self.elbow_threshold_path and not Path(self.elbow_threshold_path).exists():
            raise ValueError(f"Elbow threshold file not found: {self.elbow_threshold_path}")

        if self.use_hadamard:
            bs = self.hadamard_block_size
            if bs <= 0 or (bs & (bs - 1)) != 0:
                raise ValueError(
                    f"hadamard_block_size must be a positive power of 2, got {bs}"
                )
