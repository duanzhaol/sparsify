from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from simple_parsing import Serializable, list_field


@dataclass
class SparseCoderConfig(Serializable):
    """
    Configuration for training a sparse coder on a language model.
    """

    activation: Literal["groupmax", "topk"] = "topk"
    """Activation function to use."""

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the sparse coder dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""

    multi_topk: bool = False
    """Use Multi-TopK loss."""

    skip_connection: bool = False
    """Include a linear skip connection."""

    encoder_rank: int = 0
    """Low-rank encoder rank. 0 means full-rank (default)."""


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

    loss_fn: Literal["ce", "fvu", "kl"] = "fvu"
    """Loss function to use for training the sparse coders.

    - `ce`: Cross-entropy loss of the final model logits.
    - `fvu`: Fraction of variance explained.
    - `kl`: KL divergence of the final model logits w.r.t. the original logits.
    """

    optimizer: Literal["adam", "muon", "signum"] = "signum"
    """Optimizer to use."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    lr_warmup_steps: int = 1000
    """Number of steps over which to warm up the learning rate. Only used if
    `optimizer` is `adam`."""

    k_decay_steps: int = 0
    """Number of steps over which to decay the number of active latents. Starts at
    input width * 10 and decays to k. Experimental feature."""

    auxk_alpha: float = 0.0
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    exclude_tokens: list[int] = list_field()
    """List of tokens to ignore during sparse coders training."""

    exceed_alphas: list[float] = list_field(0.05, 0.10, 0.25, 0.50)
    """List of alpha coefficients for exceed metrics (error > alpha * elbow_value)."""

    elbow_threshold_path: str | None = None
    """Path to JSON file with pre-computed elbow thresholds per layer/operation."""

    hookpoints: list[str] = list_field()
    """List of hookpoints to train sparse coders on."""

    hook_mode: Literal["output", "input", "transcode"] = "output"
    """Activation hook mode:
    - output: autoencoder on module outputs (default)
    - input: autoencoder on module inputs
    - transcode: predict module outputs from inputs
    """

    init_seeds: list[int] = list_field(0)
    """List of random seeds to use for initialization. If more than one, train a sparse
    coder for each seed."""

    layers: list[int] = list_field()
    """List of layer indices to train sparse coders on."""

    layer_stride: int = 1
    """Stride between layers to train sparse coders on."""

    distribute_modules: bool = False
    """Store one copy of each sparse coder, instead of copying them across devices."""

    num_tiles: int = 1
    """Number of tiles to split input activations. Each tile trains a separate SAE.
    d_in must be divisible by num_tiles. Set to 1 (default) for standard training."""

    global_topk: bool = False
    """Use global top-k selection across all tiles instead of per-tile top-k.
    When enabled, all tiles compete for the same k activation budget, allowing
    more important tiles to use more capacity. Only effective when num_tiles > 1."""

    input_mixing: bool = False
    """Apply learnable T×T mixing matrix on tile dimension before encoding.
    This allows the model to learn a better coordinate system for tiling.
    Only effective when num_tiles > 1."""

    save_every: int = 1000
    """Save sparse coders every `save_every` steps."""

    save_best: bool = False
    """Save the best checkpoint found for each hookpoint."""

    finetune: str | None = None
    """Finetune the sparse coders from a pretrained checkpoint."""

    distill_from: str | None = None
    """Path to teacher SAE checkpoint for distillation training."""

    distill_lambda_decode: float = 0.5
    """Weight for decode distillation loss."""

    distill_lambda_acts: float = 0.1
    """Weight for top-k activation distillation loss."""

    freeze_decoder: bool = True
    """Freeze decoder during distillation (use teacher's decoder)."""

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_project: str | None = None
    """WandB project name. If None, uses WANDB_PROJECT env var or defaults to 'sparsify'."""
    wandb_log_frequency: int = 1

    save_dir: str = "checkpoints"

    def __post_init__(self):
        """Validate the configuration."""
        if self.layers and self.layer_stride != 1:
            raise ValueError("Cannot specify both `layers` and `layer_stride`.")

        if self.distribute_modules and self.loss_fn in ("ce", "kl"):
            raise ValueError(
                "Distributing modules across ranks is not compatible with the "
                "cross-entropy or KL divergence losses."
            )

        if not self.init_seeds:
            raise ValueError("Must specify at least one random seed.")

        # Validate exceed configuration
        if self.exceed_alphas and not all(alpha > 0 for alpha in self.exceed_alphas):
            raise ValueError("All exceed_alphas must be positive.")

        if self.elbow_threshold_path and not Path(self.elbow_threshold_path).exists():
            raise ValueError(f"Elbow threshold file not found: {self.elbow_threshold_path}")

        # Validate distillation configuration
        if self.distill_from and self.sae.encoder_rank <= 0:
            raise ValueError(
                "distill_from requires encoder_rank > 0 in sae config."
            )

        # Validate tiling configuration
        if self.num_tiles > 1 and self.hook_mode == "transcode":
            raise ValueError("Tiled training does not support transcode mode.")

        if self.num_tiles > 1 and self.distill_from:
            raise ValueError(
                "Tiled training (num_tiles > 1) does not support distillation mode. "
                "Use num_tiles=1 for distillation training."
            )

