"""Tests for TiledSparseCoder."""

import tempfile

import pytest
import torch

from sparsify.config import SparseCoderConfig
from sparsify.tiled_sparse_coder import TiledSparseCoder


# Skip tests that require CUDA if not available
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for forward pass"
)


@pytest.fixture
def cfg():
    return SparseCoderConfig(expansion_factor=4, k=4)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestTiledSparseCoder:
    def test_init(self, cfg):
        """Test initialization with valid parameters."""
        d_in = 64
        num_tiles = 4
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device="cpu")

        assert tiled.d_in == d_in
        assert tiled.num_tiles == num_tiles
        assert tiled.tile_size == d_in // num_tiles
        assert len(tiled.saes) == num_tiles
        assert tiled.num_latents == (d_in // num_tiles) * cfg.expansion_factor * num_tiles

    def test_init_invalid_tiles(self, cfg):
        """Test that non-divisible d_in raises error."""
        with pytest.raises(AssertionError):
            TiledSparseCoder(65, cfg, 4, device="cpu")  # 65 not divisible by 4

    def test_init_invalid_k(self):
        """Test that k not divisible by num_tiles raises error."""
        cfg = SparseCoderConfig(expansion_factor=4, k=5)  # k=5 not divisible by 4
        with pytest.raises(AssertionError):
            TiledSparseCoder(64, cfg, 4, device="cpu")

    @requires_cuda
    def test_forward(self, cfg, device):
        """Test forward pass produces correct output shapes."""
        d_in = 64
        num_tiles = 4
        batch_size = 8
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device=device)

        x = torch.randn(batch_size, d_in, device=device)
        out = tiled(x)

        # Check output shapes
        # Total active features = cfg.k (distributed as k // num_tiles per tile)
        assert out.sae_out.shape == (batch_size, d_in)
        assert out.latent_acts.shape == (batch_size, cfg.k)
        assert out.latent_indices.shape == (batch_size, cfg.k)

        # Check indices are in valid range
        assert out.latent_indices.min() >= 0
        assert out.latent_indices.max() < tiled.num_latents

    @requires_cuda
    def test_forward_with_y(self, cfg, device):
        """Test forward pass with explicit target y."""
        d_in = 64
        num_tiles = 4
        batch_size = 8
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device=device)

        x = torch.randn(batch_size, d_in, device=device)
        y = torch.randn(batch_size, d_in, device=device)
        out = tiled(x, y)

        assert out.sae_out.shape == (batch_size, d_in)

    @requires_cuda
    def test_forward_with_dead_mask(self, cfg, device):
        """Test forward pass with dead feature mask."""
        d_in = 64
        num_tiles = 4
        batch_size = 8
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device=device)

        x = torch.randn(batch_size, d_in, device=device)
        dead_mask = torch.zeros(tiled.num_latents, dtype=torch.bool, device=device)
        dead_mask[:10] = True  # Mark first 10 features as dead

        out = tiled(x, dead_mask=dead_mask)
        assert out.sae_out.shape == (batch_size, d_in)

    def test_set_b_dec_data(self, cfg):
        """Test setting b_dec data distributed across tiles."""
        d_in = 64
        num_tiles = 4
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device="cpu")

        new_b_dec = torch.randn(d_in)
        tiled.set_b_dec_data(new_b_dec)

        # Check each tile got the correct portion
        tiles = new_b_dec.chunk(num_tiles)
        for sae, expected in zip(tiled.saes, tiles):
            assert torch.allclose(sae.b_dec.data, expected)

    def test_save_load(self, cfg):
        """Test saving and loading TiledSparseCoder."""
        d_in = 64
        num_tiles = 4
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device="cpu")

        # Modify weights to ensure they're different from default
        for sae in tiled.saes:
            sae.encoder.weight.data.fill_(0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            tiled.save_to_disk(tmpdir)

            # Load
            loaded = TiledSparseCoder.load_from_disk(tmpdir, device="cpu")

            assert loaded.d_in == d_in
            assert loaded.num_tiles == num_tiles
            assert len(loaded.saes) == num_tiles

            # Check weights match
            for orig_sae, loaded_sae in zip(tiled.saes, loaded.saes):
                assert torch.allclose(
                    orig_sae.encoder.weight.data,
                    loaded_sae.encoder.weight.data
                )

    def test_decoder_norm(self, cfg):
        """Test set_decoder_norm_to_unit_norm normalizes all tiles."""
        d_in = 64
        num_tiles = 4
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device="cpu")

        tiled.set_decoder_norm_to_unit_norm()

        for sae in tiled.saes:
            norms = torch.norm(sae.W_dec.data, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_set_k(self, cfg):
        """Test set_k propagates to all tiles."""
        d_in = 64
        num_tiles = 4
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device="cpu")

        # Initial k per tile
        assert tiled.cfg.k == 4
        assert tiled.k_per_tile == 1
        for sae in tiled.saes:
            assert sae.cfg.k == 1

        # Update k (must be divisible by num_tiles)
        new_k = 8
        tiled.set_k(new_k)

        assert tiled.cfg.k == new_k
        assert tiled.k_per_tile == new_k // num_tiles
        for sae in tiled.saes:
            assert sae.cfg.k == new_k // num_tiles

    def test_set_k_invalid(self, cfg):
        """Test set_k with non-divisible k raises error."""
        d_in = 64
        num_tiles = 4
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device="cpu")

        with pytest.raises(AssertionError):
            tiled.set_k(5)  # 5 not divisible by 4

    @requires_cuda
    def test_gradient_flow(self, cfg, device):
        """Test that gradients flow through all tiles."""
        d_in = 64
        num_tiles = 4
        batch_size = 8
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device=device)

        x = torch.randn(batch_size, d_in, device=device)
        out = tiled(x)

        # Backprop
        out.fvu.backward()

        # Check all tiles received gradients
        for i, sae in enumerate(tiled.saes):
            assert sae.encoder.weight.grad is not None, f"Tile {i} encoder has no grad"
            assert sae.encoder.weight.grad.abs().sum() > 0, f"Tile {i} encoder grad is zero"

    @requires_cuda
    def test_indices_offset(self, cfg, device):
        """Test that latent indices are properly offset per tile."""
        d_in = 64
        num_tiles = 4
        batch_size = 8
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device=device)

        x = torch.randn(batch_size, d_in, device=device)
        out = tiled(x)

        # Each tile should have indices in its own range
        num_latents_per_tile = tiled.saes[0].num_latents  # type: ignore
        indices_per_tile = out.latent_indices.chunk(num_tiles, dim=-1)

        for i, tile_indices in enumerate(indices_per_tile):
            min_idx = i * num_latents_per_tile
            max_idx = (i + 1) * num_latents_per_tile
            assert tile_indices.min() >= min_idx, f"Tile {i} has index below range"
            assert tile_indices.max() < max_idx, f"Tile {i} has index above range"

    def test_save_includes_k_per_tile(self, cfg):
        """Test that saved config includes k_per_tile for clarity."""
        import json

        d_in = 64
        num_tiles = 4
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            tiled.save_to_disk(tmpdir)

            # Check top-level config
            with open(f"{tmpdir}/cfg.json") as f:
                saved_cfg = json.load(f)

            assert "k_per_tile" in saved_cfg, "k_per_tile should be in saved config"
            assert saved_cfg["k_per_tile"] == cfg.k // num_tiles
            assert saved_cfg["k"] == cfg.k  # Global k preserved
            assert saved_cfg["num_tiles"] == num_tiles


class TestLoadCheckpointValidation:
    """Test checkpoint loading validation for tiled/non-tiled mismatch."""

    @pytest.fixture
    def cfg(self):
        return SparseCoderConfig(expansion_factor=4, k=4)

    def test_load_tiled_checkpoint_with_regular_sae_fails(self, cfg):
        """Test that loading tiled checkpoint into regular SAE fails."""
        from sparsify.sparse_coder import SparseCoder
        from sparsify.trainer import load_sae_checkpoint

        d_in = 64
        num_tiles = 4

        # Create and save tiled SAE
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            tiled.save_to_disk(tmpdir)

            # Try to load into regular SAE - should fail
            regular_sae = SparseCoder(d_in, cfg, device="cpu")
            with pytest.raises(TypeError, match="TiledSparseCoder"):
                load_sae_checkpoint(regular_sae, tmpdir, device="cpu")

    def test_load_regular_checkpoint_with_tiled_sae_fails(self, cfg):
        """Test that loading regular checkpoint into tiled SAE fails."""
        from sparsify.sparse_coder import SparseCoder
        from sparsify.trainer import load_sae_checkpoint

        d_in = 64
        num_tiles = 4

        # Create and save regular SAE
        regular_sae = SparseCoder(d_in, cfg, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            regular_sae.save_to_disk(tmpdir)

            # Try to load into tiled SAE - should fail
            tiled = TiledSparseCoder(d_in, cfg, num_tiles, device="cpu")
            with pytest.raises(TypeError, match="num_tiles=1"):
                load_sae_checkpoint(tiled, tmpdir, device="cpu")

    def test_load_tiled_checkpoint_with_mismatched_tiles_fails(self, cfg):
        """Test that loading tiled checkpoint with different num_tiles fails."""
        from sparsify.trainer import load_sae_checkpoint

        d_in = 64

        # Create tiled SAE with 4 tiles
        tiled_4 = TiledSparseCoder(d_in, cfg, num_tiles=4, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            tiled_4.save_to_disk(tmpdir)

            # Try to load into tiled SAE with 2 tiles - should fail
            cfg_2tiles = SparseCoderConfig(expansion_factor=4, k=2)  # k divisible by 2
            tiled_2 = TiledSparseCoder(d_in, cfg_2tiles, num_tiles=2, device="cpu")
            with pytest.raises(ValueError, match="num_tiles must match"):
                load_sae_checkpoint(tiled_2, tmpdir, device="cpu")


class TestConfigValidation:
    """Test configuration validation for tiled training."""

    def test_tiled_with_distillation_fails(self):
        """Test that num_tiles > 1 with distill_from raises error."""
        from sparsify.config import TrainConfig

        with pytest.raises(ValueError, match="does not support distillation"):
            TrainConfig(
                sae=SparseCoderConfig(expansion_factor=4, k=4, encoder_rank=64),
                num_tiles=4,
                distill_from="/some/path",
            )

    def test_tiled_with_transcode_fails(self):
        """Test that num_tiles > 1 with transcode mode raises error."""
        from sparsify.config import TrainConfig

        with pytest.raises(ValueError, match="does not support transcode"):
            TrainConfig(
                sae=SparseCoderConfig(expansion_factor=4, k=4),
                num_tiles=4,
                hook_mode="transcode",
            )


class TestGlobalTopK:
    """Test global top-k functionality."""

    @pytest.fixture
    def cfg(self):
        return SparseCoderConfig(expansion_factor=4, k=8)

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_init_global_topk(self, cfg):
        """Test initialization with global_topk enabled."""
        d_in = 64
        num_tiles = 4
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device="cpu", global_topk=True)

        assert tiled.global_topk is True
        assert tiled.input_mixing is False

    @requires_cuda
    def test_forward_global_topk(self, cfg, device):
        """Test forward pass with global top-k."""
        d_in = 64
        num_tiles = 4
        batch_size = 8
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device=device, global_topk=True)

        x = torch.randn(batch_size, d_in, device=device)
        out = tiled(x)

        # Check output shapes - should be same as per-tile mode
        assert out.sae_out.shape == (batch_size, d_in)
        assert out.latent_acts.shape == (batch_size, cfg.k)
        assert out.latent_indices.shape == (batch_size, cfg.k)

        # Indices should span all tiles (not just tile-local)
        assert out.latent_indices.min() >= 0
        assert out.latent_indices.max() < tiled.num_latents

    @requires_cuda
    def test_global_topk_indices_cross_tiles(self, cfg, device):
        """Test that global top-k can select from multiple tiles."""
        d_in = 64
        num_tiles = 4
        batch_size = 16
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device=device, global_topk=True)

        x = torch.randn(batch_size, d_in, device=device)
        out = tiled(x)

        # Check that indices come from multiple tiles (statistical check)
        num_latents_per_tile = tiled.saes[0].num_latents
        tile_counts = torch.zeros(num_tiles, device=device)
        for i in range(num_tiles):
            tile_start = i * num_latents_per_tile
            tile_end = (i + 1) * num_latents_per_tile
            mask = (out.latent_indices >= tile_start) & (out.latent_indices < tile_end)
            tile_counts[i] = mask.sum()

        # At least some indices should come from different tiles
        # (not all from one tile)
        assert (tile_counts > 0).sum() > 1, "Global top-k should select from multiple tiles"

    @requires_cuda
    def test_global_topk_gradient_flow(self, cfg, device):
        """Test gradient flow with global top-k."""
        d_in = 64
        num_tiles = 4
        batch_size = 8
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device=device, global_topk=True)

        x = torch.randn(batch_size, d_in, device=device, requires_grad=True)
        out = tiled(x)
        out.fvu.backward()

        assert x.grad is not None
        for sae in tiled.saes:
            assert sae.encoder.weight.grad is not None

    def test_save_load_global_topk(self, cfg):
        """Test save/load preserves global_topk setting."""
        d_in = 64
        num_tiles = 4
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device="cpu", global_topk=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tiled.save_to_disk(tmpdir)
            loaded = TiledSparseCoder.load_from_disk(tmpdir, device="cpu")

            assert loaded.global_topk is True


class TestInputMixing:
    """Test input mixing functionality."""

    @pytest.fixture
    def cfg(self):
        return SparseCoderConfig(expansion_factor=4, k=8)

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_init_input_mixing(self, cfg):
        """Test initialization with input_mixing enabled."""
        d_in = 64
        num_tiles = 4
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device="cpu", input_mixing=True)

        assert tiled.input_mixing is True
        assert hasattr(tiled, 'mixing')
        assert tiled.mixing.shape == (num_tiles, num_tiles)
        # Should be initialized as identity
        assert torch.allclose(tiled.mixing, torch.eye(num_tiles))

    @requires_cuda
    def test_forward_input_mixing(self, cfg, device):
        """Test forward pass with input mixing."""
        d_in = 64
        num_tiles = 4
        batch_size = 8
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device=device, input_mixing=True)

        x = torch.randn(batch_size, d_in, device=device)
        out = tiled(x)

        assert out.sae_out.shape == (batch_size, d_in)
        assert out.latent_acts.shape == (batch_size, cfg.k)

    @requires_cuda
    def test_input_mixing_gradient(self, cfg, device):
        """Test that mixing matrix receives gradients."""
        d_in = 64
        num_tiles = 4
        batch_size = 8
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device=device, input_mixing=True)

        x = torch.randn(batch_size, d_in, device=device)
        out = tiled(x)
        out.fvu.backward()

        assert tiled.mixing.grad is not None
        assert tiled.mixing.grad.abs().sum() > 0

    def test_save_load_input_mixing(self, cfg):
        """Test save/load preserves input_mixing setting and weights."""
        d_in = 64
        num_tiles = 4
        tiled = TiledSparseCoder(d_in, cfg, num_tiles, device="cpu", input_mixing=True)

        # Modify mixing matrix
        tiled.mixing.data = torch.randn(num_tiles, num_tiles)
        original_mixing = tiled.mixing.data.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            tiled.save_to_disk(tmpdir)
            loaded = TiledSparseCoder.load_from_disk(tmpdir, device="cpu")

            assert loaded.input_mixing is True
            assert hasattr(loaded, 'mixing')
            assert torch.allclose(loaded.mixing, original_mixing)

    @requires_cuda
    def test_combined_global_topk_and_mixing(self, cfg, device):
        """Test both global_topk and input_mixing enabled together."""
        d_in = 64
        num_tiles = 4
        batch_size = 8
        tiled = TiledSparseCoder(
            d_in, cfg, num_tiles, device=device,
            global_topk=True, input_mixing=True
        )

        x = torch.randn(batch_size, d_in, device=device, requires_grad=True)
        out = tiled(x)

        assert out.sae_out.shape == (batch_size, d_in)
        assert out.latent_acts.shape == (batch_size, cfg.k)

        # Test gradient flow
        out.fvu.backward()
        assert x.grad is not None
        assert tiled.mixing.grad is not None
        for sae in tiled.saes:
            assert sae.encoder.weight.grad is not None
