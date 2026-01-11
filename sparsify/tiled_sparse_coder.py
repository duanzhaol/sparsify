"""Tiled Sparse Coder - splits input activations into tiles for independent SAE training."""

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import SparseCoderConfig
from .sparse_coder import ForwardOutput, SparseCoder
from .utils import decoder_impl


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
        global_topk: bool = False,
        input_mixing: bool = False,
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
        self.global_topk = global_topk
        self.input_mixing = input_mixing

        # Create per-tile config with adjusted k
        tile_cfg = copy.deepcopy(cfg)
        tile_cfg.k = self.k_per_tile

        self.saes = nn.ModuleList([
            SparseCoder(self.tile_size, tile_cfg, device, dtype)
            for _ in range(num_tiles)
        ])

        # Input mixing matrix (T×T) for cross-tile information flow
        if input_mixing:
            self.mixing = nn.Parameter(
                torch.eye(num_tiles, device=device, dtype=dtype or torch.float32)
            )

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
        # Store original input for FVU calculation with input_mixing
        original_x = x
        original_y = y

        # Apply input mixing if enabled
        if self.input_mixing:
            x = self._apply_mixing(x)
            if y is not None:
                y = self._apply_mixing(y)

        # Route to appropriate forward implementation
        if self.global_topk:
            out = self._forward_global_topk(x, y, dead_mask=dead_mask)
        else:
            out = self._forward_per_tile(x, y, dead_mask=dead_mask)

        # Apply inverse mixing and recalculate FVU in original space
        if self.input_mixing:
            sae_out = self._apply_mixing_inverse(out.sae_out)

            # Recalculate FVU in original space
            target = original_y if original_y is not None else original_x
            e = target - sae_out
            total_variance = (target - target.mean(0)).pow(2).sum()
            fvu = e.pow(2).sum() / total_variance

            out = ForwardOutput(
                sae_out=sae_out,
                latent_acts=out.latent_acts,
                latent_indices=out.latent_indices,
                fvu=fvu,
                auxk_loss=out.auxk_loss,
                multi_topk_fvu=out.multi_topk_fvu,
            )

        return out

    def _apply_mixing(self, x: Tensor) -> Tensor:
        """Apply T×T mixing matrix on tile dimension.

        The mixing operates on contiguous tile blocks to match chunk() behavior:
        - tile_i = x[:, i*tile_size : (i+1)*tile_size]
        - mixed_tile_i = sum_j(mixing[i,j] * tile_j)
        """
        # x: [N, D] -> [N, num_tiles, tile_size] (contiguous blocks per tile)
        x_reshaped = x.view(x.shape[0], self.num_tiles, self.tile_size)
        # Mix on tile dimension: mixed[n,i,d] = sum_j x[n,j,d] * mixing[i,j]
        x_mixed = torch.einsum('njd,ij->nid', x_reshaped, self.mixing)
        return x_mixed.reshape(x.shape[0], -1)

    def _apply_mixing_inverse(self, x: Tensor) -> Tensor:
        """Apply inverse of T×T mixing matrix on tile dimension.

        Uses true matrix inverse (not transpose) since mixing is not constrained
        to be orthogonal.
        """
        x_reshaped = x.view(x.shape[0], self.num_tiles, self.tile_size)
        # Compute true inverse: unmixed[n,j,d] = sum_i x[n,i,d] * inv_mixing[j,i]
        mixing_inv = torch.linalg.inv(self.mixing)
        x_unmixed = torch.einsum('nid,ji->njd', x_reshaped, mixing_inv)
        return x_unmixed.reshape(x.shape[0], -1)

    def _forward_per_tile(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        """Original per-tile forward implementation with independent top-k per tile."""
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

    def _forward_global_topk(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        """Global top-k forward: all tiles compete for the same k activation budget.

        Optimized to avoid per-tile loops in decode stage.
        """
        x_tiles = x.chunk(self.num_tiles, dim=-1)

        # Step 1: Compute pre-activations for all tiles
        all_pre_acts = []
        for sae, x_tile in zip(self.saes, x_tiles):
            sae: SparseCoder  # type: ignore
            # Center the input (same as in SparseCoder.encode)
            centered = x_tile - sae.b_dec
            # Linear transform + ReLU
            pre_acts = F.relu(F.linear(centered, sae.encoder.weight, sae.encoder.bias))
            all_pre_acts.append(pre_acts)

        global_pre_acts = torch.cat(all_pre_acts, dim=-1)  # [N, M_total]

        # Step 2: Global top-k selection
        k = self.cfg.k
        top_acts, top_indices = torch.topk(global_pre_acts, k, dim=-1, sorted=False)

        # Step 3: Decode using block-diagonal W_dec (single call, no loop)
        # Build block-diagonal decoder: W_dec_global[i*m:(i+1)*m, i*d:(i+1)*d] = W_dec[i]
        W_dec_global = torch.block_diag(*[sae.W_dec for sae in self.saes])  # [M_total, D]

        # Single decoder call
        sae_out = decoder_impl(top_indices, top_acts.to(self.dtype), W_dec_global.mT)

        # Add concatenated bias
        b_dec_global = torch.cat([sae.b_dec for sae in self.saes])  # [D]
        sae_out = sae_out + b_dec_global

        # Step 4: Compute loss
        target = y if y is not None else x
        e = target - sae_out
        total_variance = (target - target.mean(0)).pow(2).sum()
        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        # AuxK loss is disabled for global_topk mode (dead_mask handling is complex)
        # TODO: Implement global dead feature tracking if needed
        auxk_loss = sae_out.new_tensor(0.0)
        multi_topk_fvu = sae_out.new_tensor(0.0)

        return ForwardOutput(
            sae_out=sae_out,
            latent_acts=top_acts,
            latent_indices=top_indices,
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
                "global_topk": self.global_topk,
                "input_mixing": self.input_mixing,
            }, f)

        # Save each tile
        for i, sae in enumerate(self.saes):
            sae: SparseCoder  # type: ignore
            sae.save_to_disk(path / f"tile_{i}")

        # Save mixing matrix if enabled
        if self.input_mixing:
            torch.save(self.mixing, path / "mixing.pt")

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
            global_topk = cfg_dict.pop("global_topk", False)
            input_mixing = cfg_dict.pop("input_mixing", False)
            cfg = SparseCoderConfig.from_dict(cfg_dict, drop_extra_fields=True)

        instance = cls(
            d_in, cfg, num_tiles, device=device,
            global_topk=global_topk, input_mixing=input_mixing
        )

        # Load each tile
        for i in range(num_tiles):
            tile_path = path / f"tile_{i}"
            instance.saes[i] = SparseCoder.load_from_disk(
                tile_path, device=device, decoder=decoder
            )

        # Load mixing matrix if it exists
        mixing_path = path / "mixing.pt"
        if mixing_path.exists() and input_mixing:
            instance.mixing.data = torch.load(mixing_path, map_location=device)

        return instance
