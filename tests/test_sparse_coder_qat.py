from __future__ import annotations

import torch

from sparsify.config import SparseCoderConfig
from sparsify.sparse_coder import ProductKeyExpertJumpReLUSparseCoder, SparseCoder


def _tiny_product_key_sae() -> ProductKeyExpertJumpReLUSparseCoder:
    cfg = SparseCoderConfig(
        architecture="product_key_expert_jumprelu",
        k=4,
        num_experts=4,
        active_experts=2,
        latents_per_expert=4,
        normalize_decoder=False,
        jumprelu_bandwidth=0.2,
    )
    return SparseCoder(d_in=8, cfg=cfg, device="cpu", dtype=torch.float32)


def test_sparse_coder_set_quantization_mode_defaults_off():
    cfg = SparseCoderConfig(architecture="topk", k=2, expansion_factor=2)
    coder = SparseCoder(d_in=8, cfg=cfg, device="cpu", dtype=torch.float32)

    assert coder._quantization_mode == "off"
    assert coder._quantization_num_bits == 8

    coder.set_quantization_mode("qat_io_int8", num_bits=8)
    assert coder._quantization_mode == "qat_io_int8"
    assert coder._quantization_num_bits == 8

    coder.set_quantization_mode("qat_full_w8a8", num_bits=8)
    assert coder._quantization_mode == "qat_full_w8a8"
    assert coder._quantization_num_bits == 8


def test_product_key_expert_jumprelu_qat_full_w8a8_forward_backward_smoke_cpu():
    torch.manual_seed(0)
    coder = _tiny_product_key_sae()
    coder.set_quantization_mode("qat_full_w8a8", num_bits=8)

    x = torch.randn(6, 8, dtype=torch.float32)
    y = torch.randn(6, 8, dtype=torch.float32)

    out = coder(x=x, y=y)
    loss = out.fvu + out.auxk_loss + out.sae_out.pow(2).mean()
    loss.backward()

    assert coder.left_router.weight.grad is not None
    assert coder.right_router.weight.grad is not None
    assert coder.expert_encoders.grad is not None
    assert coder.W_dec is not None
    assert coder.W_dec.grad is not None

    assert coder.expert_encoder_bias.grad is not None
    assert coder.b_dec.grad is not None
    assert coder.log_threshold.grad is not None

    monitoring = coder.pop_monitoring_metrics()
    assert "qat_router_act_scale_mean" in monitoring
    assert "qat_router_weight_scale_mean" in monitoring
    assert "qat_encoder_act_scale_mean" in monitoring
    assert "qat_encoder_weight_scale_mean" in monitoring
    assert "qat_decoder_act_scale_mean" in monitoring
    assert "qat_decoder_weight_scale_mean" in monitoring


def test_product_key_expert_jumprelu_decode_uses_full_qat_decoder_path():
    torch.manual_seed(1)
    coder = _tiny_product_key_sae()
    coder.set_quantization_mode("qat_full_w8a8", num_bits=8)

    x = torch.randn(4, 8, dtype=torch.float32)
    encoded = coder.encode(x)
    _ = coder.pop_monitoring_metrics()

    recon = coder.decode(encoded.top_acts, encoded.top_indices)

    assert recon.shape == x.shape
    monitoring = coder.pop_monitoring_metrics()
    assert "qat_decoder_act_scale_mean" in monitoring
    assert "qat_decoder_weight_scale_mean" in monitoring
