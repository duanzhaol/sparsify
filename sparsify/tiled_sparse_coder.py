"""Tiled Sparse Coder - splits input activations into tiles for independent SAE training."""

import copy
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from .config import SparseCoderConfig
from .sparse_coder import ForwardOutput, SparseCoder


class TiledSparseCoder(nn.Module):
    """
    Splits input activations along hidden_dim into T tiles,
    each trained by an independent SAE.

    Input [N, D] -> chunk -> T x [N, D/T] -> T SAEs -> concat -> [N, D]

    Note: Total active features = cfg.k (distributed as k // num_tiles per tile).
    """

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        num_tiles: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_in % num_tiles == 0, (
            f"d_in ({d_in}) must be divisible by num_tiles ({num_tiles})"
        )
        assert cfg.k % num_tiles == 0, (
            f"k ({cfg.k}) must be divisible by num_tiles ({num_tiles})"
        )

        self.cfg = cfg  # Original config (with original k)
        self.d_in = d_in
        self.num_tiles = num_tiles
        self.tile_size = d_in // num_tiles
        self.k_per_tile = cfg.k // num_tiles

        # Create per-tile config with adjusted k
        tile_cfg = copy.deepcopy(cfg)
        tile_cfg.k = self.k_per_tile

        self.saes = nn.ModuleList([
            SparseCoder(self.tile_size, tile_cfg, device, dtype)
            for _ in range(num_tiles)
        ])

    @property
    def num_latents(self) -> int:
        """Total number of latents across all tiles."""
        first_sae: SparseCoder = self.saes[0]  # type: ignore
        return first_sae.num_latents * self.num_tiles

    @property
    def device(self):
        first_sae: SparseCoder = self.saes[0]  # type: ignore
        return first_sae.device

    @property
    def dtype(self):
        first_sae: SparseCoder = self.saes[0]  # type: ignore
        return first_sae.dtype

    @property
    def W_dec(self):
        """Return first SAE's W_dec for interface compatibility (e.g., requires_grad check)."""
        first_sae: SparseCoder = self.saes[0]  # type: ignore
        return first_sae.W_dec

    @property
    def b_dec(self) -> Tensor:
        """Concatenated decoder bias from all tiles (read-only view)."""
        return torch.cat([sae.b_dec for sae in self.saes])  # type: ignore

    def set_b_dec_data(self, value: Tensor):
        """Set b_dec data distributed across tiles."""
        tiles = value.chunk(self.num_tiles)
        for sae, tile in zip(self.saes, tiles):
            sae.b_dec.data = tile.clone()  # clone() to avoid shared storage (safetensors issue)

    def freeze_decoder(self):
        """Freeze decoder weights and biases for all tiles."""
        for sae in self.saes:
            sae: SparseCoder  # type: ignore
            sae.W_dec.requires_grad_(False)
            sae.b_dec.requires_grad_(False)

    def set_k(self, k: int):
        """Set k value, propagating to all tiles as k // num_tiles."""
        assert k % self.num_tiles == 0, (
            f"k ({k}) must be divisible by num_tiles ({self.num_tiles})"
        )
        self.cfg.k = k
        self.k_per_tile = k // self.num_tiles
        for sae in self.saes:
            sae: SparseCoder  # type: ignore
            sae.cfg.k = self.k_per_tile

    @property
    def encoder(self):
        """Return None - tiling doesn't support transcode mode's bias initialization."""
        return None

    @torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_bf16_supported(),
    )
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        # Split input into tiles
        x_tiles = x.chunk(self.num_tiles, dim=-1)
        y_tiles = y.chunk(self.num_tiles, dim=-1) if y is not None else [None] * self.num_tiles

        # Split dead_mask if provided
        if dead_mask is not None:
            dead_masks = dead_mask.chunk(self.num_tiles)
        else:
            dead_masks = [None] * self.num_tiles

        # Forward each tile through its SAE
        outputs = [
            sae(x_tile, y_tile, dead_mask=dm)
            for sae, x_tile, y_tile, dm in zip(self.saes, x_tiles, y_tiles, dead_masks)
        ]

        # Merge outputs
        fvu = torch.stack([o.fvu for o in outputs]).mean()
        auxk_loss = torch.stack([o.auxk_loss for o in outputs]).mean()
        multi_topk_fvu = torch.stack([o.multi_topk_fvu for o in outputs]).mean()

        return ForwardOutput(
            sae_out=torch.cat([o.sae_out for o in outputs], dim=-1),
            latent_acts=torch.cat([o.latent_acts for o in outputs], dim=-1),
            latent_indices=self._merge_indices(outputs),
            fvu=fvu,
            auxk_loss=auxk_loss,
            multi_topk_fvu=multi_topk_fvu,
        )

    def _merge_indices(self, outputs: list[ForwardOutput]) -> Tensor:
        """Merge latent indices with offset for each tile."""
        first_sae: SparseCoder = self.saes[0]  # type: ignore
        num_latents_per_tile = first_sae.num_latents
        merged = []
        for i, o in enumerate(outputs):
            merged.append(o.latent_indices + i * num_latents_per_tile)
        return torch.cat(merged, dim=-1)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Normalize decoder weights for all tiles."""
        for sae in self.saes:
            sae: SparseCoder  # type: ignore
            sae.set_decoder_norm_to_unit_norm()

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """Remove gradient parallel to decoder directions for all tiles."""
        for sae in self.saes:
            sae: SparseCoder  # type: ignore
            sae.remove_gradient_parallel_to_decoder_directions()

    def save_to_disk(self, path: Path | str):
        """Save tiled SAE to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config with tiling info
        # Note: cfg.k is the global k (total active features across all tiles)
        # Each tile's cfg.json will have k = k_per_tile
        with open(path / "cfg.json", "w") as f:
            json.dump({
                **self.cfg.to_dict(),
                "d_in": self.d_in,
                "num_tiles": self.num_tiles,
                "k_per_tile": self.k_per_tile,  # For clarity to external tooling
            }, f)

        # Save each tile
        for i, sae in enumerate(self.saes):
            sae: SparseCoder  # type: ignore
            sae.save_to_disk(path / f"tile_{i}")

    @classmethod
    def load_from_disk(
        cls,
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "TiledSparseCoder":
        """Load tiled SAE from disk."""
        path = Path(path)

        with open(path / "cfg.json") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            num_tiles = cfg_dict.pop("num_tiles")
            cfg = SparseCoderConfig.from_dict(cfg_dict, drop_extra_fields=True)

        instance = cls(d_in, cfg, num_tiles, device=device)

        # Load each tile
        for i in range(num_tiles):
            tile_path = path / f"tile_{i}"
            instance.saes[i] = SparseCoder.load_from_disk(
                tile_path, device=device, decoder=decoder
            )

        return instance
