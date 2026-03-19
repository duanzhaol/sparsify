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

    # Architecture selection
    architecture: str = "topk"
    """Encoding architecture: 'topk' | 'gated' | 'jumprelu' | 'group_topk'."""

    # JumpReLU parameters (only used when architecture='jumprelu')
    jumprelu_init_threshold: float = 0.001
    """Initial per-feature threshold for JumpReLU activation."""

    jumprelu_bandwidth: float = 0.001
    """Bandwidth for STE approximation in JumpReLU backward."""

    # Group TopK parameters (only used when architecture='group_topk')
    num_groups: int = 0
    """Number of groups for Group TopK. Must divide num_latents evenly."""

    active_groups: int = 0
    """Number of groups to select per input in Group TopK."""


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

    # Compilation
    compile_model: bool = False
    """Compile transformer layers with torch.compile to fuse small kernels.
    Reduces kernel launch overhead from elementwise/layernorm/dtype ops."""

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

    # Optimizer selection
    optimizer: str = "signum"
    """Optimizer: 'signum' | 'adam'."""

    # Matryoshka multi-K training
    matryoshka_ks: list[int] = list_field()
    """List of K values for Matryoshka multi-K loss. Empty = disabled."""

    matryoshka_weights: list[float] = list_field()
    """Weights for each Matryoshka K. Must match len(matryoshka_ks). Empty = uniform."""

    # Orthogonality regularization
    ortho_lambda: float = 0.0
    """Weight for orthogonality regularization on active decoder columns."""

    # Residual SAE training
    residual_from: str | None = None
    """Level 1 SAE checkpoint path. When set, trains Level 2 SAE on residual."""

    # Local metrics saving
    save_metrics_jsonl: bool = True
    """Save per-step metrics to JSONL file alongside W&B."""

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

        if self.compile_model:
            from .device import get_device_type
            if get_device_type() != "cuda":
                self.compile_model = False

        if self.use_hadamard:
            bs = self.hadamard_block_size
            if bs <= 0 or (bs & (bs - 1)) != 0:
                raise ValueError(
                    f"hadamard_block_size must be a positive power of 2, got {bs}"
                )

        # Architecture validation
        valid_archs = ("topk", "gated", "jumprelu", "group_topk")
        if self.sae.architecture not in valid_archs:
            raise ValueError(
                f"Unknown architecture: {self.sae.architecture!r}. "
                f"Must be one of {valid_archs}"
            )

        if self.sae.architecture == "group_topk":
            if self.sae.num_groups <= 0:
                raise ValueError("num_groups must be > 0 for group_topk architecture")
            if self.sae.active_groups <= 0:
                raise ValueError("active_groups must be > 0 for group_topk architecture")
            if self.sae.active_groups > self.sae.num_groups:
                raise ValueError("active_groups must be <= num_groups")
            # Ensure enough selectable latents for k
            num_latents = self.sae.num_latents or 1  # actual value depends on d_in
            if self.sae.num_latents > 0:
                group_size = self.sae.num_latents // self.sae.num_groups
                selectable = self.sae.active_groups * group_size
                if self.sae.k > selectable:
                    raise ValueError(
                        f"k ({self.sae.k}) > active_groups * group_size "
                        f"({self.sae.active_groups} * {group_size} = {selectable}). "
                        f"This would select -inf values from masked groups."
                    )

        # Optimizer validation
        valid_opts = ("signum", "adam")
        if self.optimizer not in valid_opts:
            raise ValueError(
                f"Unknown optimizer: {self.optimizer!r}. Must be one of {valid_opts}"
            )

        # JumpReLU validation
        if self.sae.architecture == "jumprelu":
            if self.sae.jumprelu_bandwidth <= 0:
                raise ValueError("jumprelu_bandwidth must be > 0")

        # Matryoshka validation
        if self.matryoshka_ks:
            for mk in self.matryoshka_ks:
                if mk <= 0 or mk > self.sae.k:
                    raise ValueError(
                        f"Each matryoshka K must be in (0, sae.k={self.sae.k}], got {mk}"
                    )
        if self.matryoshka_ks and self.matryoshka_weights:
            if len(self.matryoshka_ks) != len(self.matryoshka_weights):
                raise ValueError(
                    "matryoshka_ks and matryoshka_weights must have same length"
                )
