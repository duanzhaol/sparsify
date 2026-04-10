from __future__ import annotations

from torchao.quantization import Int8DynamicActivationInt8WeightConfig
from transformers import AutoModel, TorchAoConfig


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


def resolve_available_hookpoints(model) -> list[str]:
    module_root = getattr(model, "base_model", model)
    return [name for name, _ in module_root.named_modules() if name]


def validate_requested_hookpoints(requested: list[str], available: list[str]) -> None:
    missing = sorted(name for name in requested if name not in set(available))
    if missing:
        raise ValueError(f"Missing hookpoints in quantized backbone: {missing}")
