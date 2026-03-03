"""Verify TiledSparseCoder on Ascend NPU."""

import torch

from sparsify.config import SparseCoderConfig
from sparsify.tiled_sparse_coder import TiledSparseCoder


def test_tiled_forward():
    cfg = SparseCoderConfig(expansion_factor=4, k=4)
    tiled = TiledSparseCoder(64, cfg, num_tiles=4, device="npu")
    x = torch.randn(8, 64, device="npu")
    out = tiled(x)

    assert out.sae_out.shape == (8, 64)
    assert out.latent_acts.shape == (8, 4)
    assert out.latent_indices.shape == (8, 4)


def test_tiled_backward():
    cfg = SparseCoderConfig(expansion_factor=4, k=4)
    tiled = TiledSparseCoder(64, cfg, num_tiles=4, device="npu")
    x = torch.randn(8, 64, device="npu")
    out = tiled(x)
    out.fvu.backward()

    for i, sae in enumerate(tiled.saes):
        assert sae.encoder.weight.grad is not None, f"Tile {i} encoder has no grad"
        assert sae.encoder.weight.grad.abs().sum() > 0, f"Tile {i} grad is zero"


def test_global_topk():
    cfg = SparseCoderConfig(expansion_factor=4, k=8)
    tiled = TiledSparseCoder(64, cfg, num_tiles=4, device="npu", global_topk=True)
    x = torch.randn(16, 64, device="npu")
    out = tiled(x)

    assert out.sae_out.shape == (16, 64)
    assert out.latent_acts.shape == (16, 8)
    assert out.latent_indices.min() >= 0
    assert out.latent_indices.max() < tiled.num_latents


def test_input_mixing():
    cfg = SparseCoderConfig(expansion_factor=4, k=8)
    tiled = TiledSparseCoder(64, cfg, num_tiles=4, device="npu", input_mixing=True)
    x = torch.randn(8, 64, device="npu")
    out = tiled(x)
    out.fvu.backward()

    assert tiled.mixing.grad is not None
    assert tiled.mixing.grad.abs().sum() > 0
