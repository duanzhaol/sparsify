from __future__ import annotations

from enum import Enum

import torch
import pytest

from sparsify.quantized_backbone import (
    activation_source_requires_backbone_path,
    activation_source_uses_torchao_loader,
    build_torchao_int8_quantization_config,
    load_smoothquant_w8a8_model,
    load_torchao_w8a8_model,
    materialize_compressed_linears,
    resolve_available_hookpoints,
    select_activation_model_path,
    validate_requested_hookpoints,
)


def test_build_torchao_int8_quantization_config_uses_expected_quant_type():
    cfg = build_torchao_int8_quantization_config()
    assert "Int8DynamicActivationInt8WeightConfig" in str(cfg.quant_type)


def test_validate_requested_hookpoints_rejects_missing_entries():
    available = ["layers.0.self_attn.q_proj", "layers.1.self_attn.q_proj"]
    with pytest.raises(ValueError, match="Missing hookpoints"):
        validate_requested_hookpoints(
            ["layers.0.self_attn.q_proj", "layers.9.self_attn.q_proj"],
            available,
        )


def test_validate_requested_hookpoints_accepts_exact_matches():
    available = ["layers.0.self_attn.q_proj", "layers.1.self_attn.q_proj"]
    validate_requested_hookpoints(["layers.0.self_attn.q_proj"], available)


def test_select_activation_model_path_prefers_quantized_backbone():
    path = select_activation_model_path(
        activation_source="w8a8_backbone",
        default_model="Qwen/Qwen3-0.6B",
        activation_backbone_path="/tmp/qwen3-w8a8",
    )
    assert path == "/tmp/qwen3-w8a8"


def test_select_activation_model_path_uses_default_for_bf16():
    path = select_activation_model_path(
        activation_source="hf_bf16",
        default_model="Qwen/Qwen3-0.6B",
        activation_backbone_path="/tmp/qwen3-w8a8",
    )
    assert path == "Qwen/Qwen3-0.6B"


def test_select_activation_model_path_prefers_smoothquant_backbone():
    path = select_activation_model_path(
        activation_source="smoothquant_w8a8_backbone",
        default_model="Qwen/Qwen3-0.6B",
        activation_backbone_path="/tmp/qwen3-smoothquant-w8a8",
    )
    assert path == "/tmp/qwen3-smoothquant-w8a8"


def test_activation_source_requires_backbone_path_for_quantized_teachers():
    assert activation_source_requires_backbone_path("w8a8_backbone") is True
    assert activation_source_requires_backbone_path("smoothquant_w8a8_backbone") is True
    assert activation_source_requires_backbone_path("hf_bf16") is False


def test_activation_source_uses_torchao_loader_only_for_torchao_path():
    assert activation_source_uses_torchao_loader("w8a8_backbone") is True
    assert activation_source_uses_torchao_loader("smoothquant_w8a8_backbone") is False
    assert activation_source_uses_torchao_loader("hf_bf16") is False


def test_load_torchao_w8a8_model_passes_quantization_config():
    captured: dict[str, object] = {}

    class FakeLoader:
        @staticmethod
        def from_pretrained(model_name_or_path: str, **kwargs):
            captured["model_name_or_path"] = model_name_or_path
            captured["kwargs"] = kwargs
            return "fake-model"

    model = load_torchao_w8a8_model(
        "Qwen/Qwen3-0.6B",
        device_map={"": "cpu"},
        revision=None,
        torch_dtype="auto",
        token=None,
        model_loader=FakeLoader,
    )

    assert model == "fake-model"
    assert captured["model_name_or_path"] == "Qwen/Qwen3-0.6B"
    assert "Int8DynamicActivationInt8WeightConfig" in str(
        captured["kwargs"]["quantization_config"].quant_type
    )


def test_materialize_compressed_linears_eagerly_decompresses_weights():
    class FakeQuantizationStatus(Enum):
        COMPRESSED = "compressed"
        FROZEN = "frozen"

    class FakeCompressor:
        def decompress_module(self, module):
            return torch.full((2, 3), 1.5, dtype=torch.float32)

    class FakeCompressedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quantization_status = FakeQuantizationStatus.COMPRESSED
            self.compressor = FakeCompressor()
            self.register_parameter(
                "weight",
                torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.int8), requires_grad=False),
            )

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = FakeCompressedLinear()

    model = FakeModel()
    materialized = materialize_compressed_linears(
        model,
        compressed_linear_cls=FakeCompressedLinear,
        quantization_status_enum=FakeQuantizationStatus,
    )

    assert materialized == 1
    assert model.proj.quantization_status == FakeQuantizationStatus.FROZEN
    assert model.proj.weight.dtype == torch.float32
    assert torch.allclose(model.proj.weight, torch.full((2, 3), 1.5))


def test_load_smoothquant_w8a8_model_materializes_after_loading():
    captured: dict[str, object] = {}

    class FakeLoader:
        @staticmethod
        def from_pretrained(model_name_or_path: str, **kwargs):
            captured["model_name_or_path"] = model_name_or_path
            captured["kwargs"] = kwargs
            return "fake-smoothquant-model"

    def fake_materialize(model):
        captured["materialized_model"] = model
        return 7

    model = load_smoothquant_w8a8_model(
        "Qwen/Qwen3-0.6B-SQ",
        device_map={"": "cpu"},
        revision=None,
        torch_dtype="auto",
        token=None,
        model_loader=FakeLoader,
        materialize_fn=fake_materialize,
    )

    assert model == "fake-smoothquant-model"
    assert captured["model_name_or_path"] == "Qwen/Qwen3-0.6B-SQ"
    assert captured["materialized_model"] == "fake-smoothquant-model"


def test_resolve_available_hookpoints_prefers_base_model():
    class FakeBaseModel:
        def named_modules(self):
            return [
                ("", object()),
                ("layers.0.self_attn.q_proj", object()),
                ("layers.1.self_attn.q_proj", object()),
            ]

    class FakeModel:
        base_model = FakeBaseModel()

    names = resolve_available_hookpoints(FakeModel())
    assert "layers.0.self_attn.q_proj" in names
    assert "layers.1.self_attn.q_proj" in names
