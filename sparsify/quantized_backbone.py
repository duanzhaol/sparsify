from __future__ import annotations

import logging

import torch
from torchao.quantization import Int8DynamicActivationInt8WeightConfig
from torch.nn import Parameter
from transformers import AutoModel, TorchAoConfig

logger = logging.getLogger(__name__)


def activation_source_requires_backbone_path(activation_source: str) -> bool:
    return activation_source in {"w8a8_backbone", "smoothquant_w8a8_backbone"}


def activation_source_uses_torchao_loader(activation_source: str) -> bool:
    return activation_source == "w8a8_backbone"


def build_torchao_int8_quantization_config() -> TorchAoConfig:
    return TorchAoConfig(Int8DynamicActivationInt8WeightConfig())


def select_activation_model_path(
    *,
    activation_source: str,
    default_model: str,
    activation_backbone_path: str | None,
) -> str:
    if activation_source_requires_backbone_path(activation_source):
        if activation_backbone_path is None:
            raise ValueError(
                "activation_backbone_path must be provided for external activation sources"
            )
        return activation_backbone_path
    return default_model


def load_torchao_w8a8_model(
    model_name_or_path: str,
    *,
    device_map,
    revision: str | None,
    torch_dtype,
    token: str | None,
    model_loader=AutoModel,
):
    quantization_config = build_torchao_int8_quantization_config()
    return model_loader.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        revision=revision,
        torch_dtype=torch_dtype,
        token=token,
        quantization_config=quantization_config,
    )


def materialize_compressed_linears(
    model,
    *,
    compressed_linear_cls=None,
    quantization_status_enum=None,
) -> int:
    if compressed_linear_cls is None or quantization_status_enum is None:
        from compressed_tensors.linear.compressed_linear import CompressedLinear
        from compressed_tensors.quantization import QuantizationStatus

        compressed_linear_cls = CompressedLinear
        quantization_status_enum = QuantizationStatus

    materialized = 0
    for _, module in model.named_modules():
        if not isinstance(module, compressed_linear_cls):
            continue
        if module.quantization_status != quantization_status_enum.COMPRESSED:
            continue

        weight_data = module.compressor.decompress_module(module)
        module.register_parameter(
            "weight",
            Parameter(weight_data.detach().clone(), requires_grad=False),
        )
        module.quantization_status = quantization_status_enum.FROZEN
        materialized += 1

    return materialized


def load_smoothquant_w8a8_model(
    model_name_or_path: str,
    *,
    device_map,
    revision: str | None,
    torch_dtype,
    token: str | None,
    model_loader=AutoModel,
    materialize_fn=materialize_compressed_linears,
):
    model = model_loader.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        revision=revision,
        torch_dtype=torch_dtype,
        token=token,
    )
    materialized = materialize_fn(model)
    logger.info(
        "Eagerly materialized %d CompressedLinear modules for SmoothQuant teacher",
        materialized,
    )
    return model


def resolve_available_hookpoints(model) -> list[str]:
    module_root = getattr(model, "base_model", model)
    return [name for name, _ in module_root.named_modules() if name]


def validate_requested_hookpoints(requested: list[str], available: list[str]) -> None:
    missing = sorted(name for name in requested if name not in set(available))
    if missing:
        raise ValueError(f"Missing hookpoints in quantized backbone: {missing}")
