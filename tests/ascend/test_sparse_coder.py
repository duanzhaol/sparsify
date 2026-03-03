"""End-to-end SparseCoder tests on Ascend NPU."""

import tempfile

import torch

from sparsify.config import SparseCoderConfig
from sparsify.sparse_coder import SparseCoder


def test_forward_output_shapes():
    cfg = SparseCoderConfig(expansion_factor=4, k=4)
    sae = SparseCoder(64, cfg, device="npu")
    x = torch.randn(16, 64, device="npu")
    out = sae(x)

    assert out.sae_out.shape == (16, 64)
    assert out.latent_acts.shape == (16, 4)
    assert out.latent_indices.shape == (16, 4)


def test_fvu_reasonable_range():
    cfg = SparseCoderConfig(expansion_factor=4, k=4)
    sae = SparseCoder(64, cfg, device="npu")
    x = torch.randn(16, 64, device="npu")
    out = sae(x)

    assert 0 <= out.fvu.item() <= 10


def test_backward_gradient_flow():
    cfg = SparseCoderConfig(expansion_factor=4, k=4)
    sae = SparseCoder(64, cfg, device="npu")
    x = torch.randn(16, 64, device="npu")
    out = sae(x)
    out.fvu.backward()

    assert sae.encoder.weight.grad is not None
    assert sae.encoder.weight.grad.abs().sum() > 0
    assert sae.W_dec.grad is not None


def test_auxk_loss():
    cfg = SparseCoderConfig(expansion_factor=4, k=4)
    sae = SparseCoder(64, cfg, device="npu")
    x = torch.randn(16, 64, device="npu")

    dead_mask = torch.zeros(sae.num_latents, dtype=torch.bool, device="npu")
    dead_mask[:32] = True
    out = sae(x, dead_mask=dead_mask)

    assert out.auxk_loss.item() > 0


def test_decoder_norm():
    cfg = SparseCoderConfig(expansion_factor=4, k=4, normalize_decoder=True)
    sae = SparseCoder(64, cfg, device="npu")
    sae.set_decoder_norm_to_unit_norm()
    norms = sae.W_dec.data.norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_npu_vs_cpu_numerical():
    """Same weights on NPU and CPU should produce close outputs.

    We compare the encode → decode path directly (bypassing device_autocast)
    so that both sides run in fp32 and any difference is purely due to the
    NPU compute kernels, not mixed-precision casting.
    """
    cfg = SparseCoderConfig(expansion_factor=4, k=4)
    sae_npu = SparseCoder(64, cfg, device="npu", dtype=torch.float32)

    # Copy weights to CPU model
    sae_cpu = SparseCoder(64, cfg, device="cpu", dtype=torch.float32)
    sae_cpu.load_state_dict(
        {k: v.cpu() for k, v in sae_npu.state_dict().items()}
    )

    x = torch.randn(8, 64)

    # Use encode + decode directly to avoid device_autocast bf16 differences
    with torch.no_grad():
        acts_cpu, idx_cpu, _ = sae_cpu.encode(x)
        out_cpu = sae_cpu.decode(acts_cpu, idx_cpu)

        acts_npu, idx_npu, _ = sae_npu.encode(x.to("npu"))
        out_npu = sae_npu.decode(acts_npu, idx_npu)

    torch.testing.assert_close(
        out_npu.cpu(), out_cpu, atol=1e-4, rtol=1e-4
    )


def test_save_load_cross_device():
    """SAE trained on NPU should be loadable on CPU."""
    cfg = SparseCoderConfig(expansion_factor=4, k=4)
    sae = SparseCoder(64, cfg, device="npu")

    with tempfile.TemporaryDirectory() as tmpdir:
        sae.save_to_disk(tmpdir)
        loaded = SparseCoder.load_from_disk(tmpdir, device="cpu")

    assert loaded.W_dec.device.type == "cpu"
