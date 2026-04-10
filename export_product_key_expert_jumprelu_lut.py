#!/usr/bin/env python3
"""
Export product_key_expert_jumprelu SAE checkpoints into standalone LUT bundles.

Each output `.lut.safetensors` file is a runtime-complete artifact containing:
- Stage A routing + encoder tensors
- Stage B precomputed lookup tables
- Stage C compensation weights

The exporter is intentionally separate from `convert_sae_to_lut.py` because the
product-key JumpReLU checkpoints use a different tensor layout and a richer
runtime bundle schema.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


PROJECTION_SPECS = {
    "qkv": {
        "operator_name": "self_attn.qkv_proj",
        "checkpoint_suffix": "self_attn.q_proj",
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "output_name": "layers.{layer_idx}.self_attn.qkv_proj",
    },
    "gate_up": {
        "operator_name": "mlp.gate_up_proj",
        "checkpoint_suffix": "mlp.up_proj",
        "target_modules": ["gate_proj", "up_proj"],
        "output_name": "layers.{layer_idx}.mlp.gate_up_proj",
    },
}

ATTENTION_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
LAYER_PATTERN = re.compile(r"layers\.(\d+)\.")


def parse_layer_range(range_str: str) -> list[int]:
    numbers: list[int] = []
    for part in range_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-"))
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))
    return sorted(set(numbers))


def detect_layer_indices(best_dir: Path) -> list[int]:
    layers = set()
    for child in best_dir.iterdir():
        if not child.is_dir():
            continue
        match = LAYER_PATTERN.search(child.name)
        if match:
            layers.add(int(match.group(1)))
    return sorted(layers)


def resolve_layer_indices(
    qproj_best_dir: Path,
    upproj_best_dir: Path,
    *,
    layers: list[int] | None,
) -> list[int]:
    if layers is not None:
        return sorted(set(layers))

    qproj_layers = set(detect_layer_indices(qproj_best_dir))
    upproj_layers = set(detect_layer_indices(upproj_best_dir))
    shared_layers = sorted(qproj_layers & upproj_layers)
    if not shared_layers:
        raise ValueError(
            "No shared layers found between "
            f"{qproj_best_dir} and {upproj_best_dir}."
        )
    return shared_layers


def _checkpoint_dir(best_dir: Path, layer_idx: int, checkpoint_suffix: str) -> Path:
    return best_dir / f"layers.{layer_idx}.{checkpoint_suffix}"


def _load_tensor_map(path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def load_product_key_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    with open(checkpoint_dir / "cfg.json", "r") as f:
        config = json.load(f)

    if config.get("architecture") != "product_key_expert_jumprelu":
        raise ValueError(
            f"{checkpoint_dir} is not a product_key_expert_jumprelu checkpoint"
        )

    tensors = _load_tensor_map(checkpoint_dir / "sae.safetensors")
    required_keys = {
        "left_router.weight",
        "left_router.bias",
        "right_router.weight",
        "right_router.bias",
        "pair_left_index",
        "pair_right_index",
        "expert_encoders",
        "expert_encoder_bias",
        "log_threshold",
        "W_dec",
        "b_dec",
    }
    missing = sorted(required_keys - set(tensors))
    if missing:
        raise ValueError(f"Missing checkpoint tensors in {checkpoint_dir}: {missing}")

    return {
        "router_left_weight": tensors["left_router.weight"],
        "router_left_bias": tensors["left_router.bias"],
        "router_right_weight": tensors["right_router.weight"],
        "router_right_bias": tensors["right_router.bias"],
        "pair_left_index": tensors["pair_left_index"].to(torch.int32),
        "pair_right_index": tensors["pair_right_index"].to(torch.int32),
        "expert_encoder_weight": tensors["expert_encoders"],
        "expert_encoder_bias": tensors["expert_encoder_bias"],
        "expert_threshold": F.softplus(tensors["log_threshold"]),
        "decoder_weight": tensors["W_dec"],
        "decoder_bias": tensors["b_dec"],
        "config": config,
    }


def _module_path(layer_idx: int, module_name: str) -> str:
    prefix = "self_attn" if module_name in ATTENTION_MODULES else "mlp"
    return f"model.layers.{layer_idx}.{prefix}.{module_name}"


def get_fused_weight_and_bias(
    model: Any,
    layer_idx: int,
    target_modules: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = []
    biases = []
    for module_name in target_modules:
        module = model.get_submodule(_module_path(layer_idx, module_name))
        weight = module.weight.detach().cpu()
        weights.append(weight)

        module_bias = getattr(module, "bias", None)
        if module_bias is None:
            bias = torch.zeros(weight.shape[0], dtype=weight.dtype)
        else:
            bias = module_bias.detach().cpu()
        biases.append(bias)

    return torch.cat(weights, dim=0), torch.cat(biases, dim=0)


def compute_precomputed_products(
    decoder_weight: torch.Tensor,
    target_weight: torch.Tensor,
    *,
    device: str,
    batch_size: int | None,
) -> torch.Tensor:
    decoder_weight = decoder_weight.float().to(device)
    target_weight = target_weight.float().to(device)

    if batch_size is None:
        return (decoder_weight @ target_weight.T).cpu()

    num_latents = decoder_weight.shape[0]
    output_dim = target_weight.shape[0]
    output = torch.empty(num_latents, output_dim, dtype=torch.float32, device=device)
    for start in range(0, num_latents, batch_size):
        end = min(start + batch_size, num_latents)
        output[start:end] = decoder_weight[start:end] @ target_weight.T
    return output.cpu()


def compute_bias_product(
    decoder_bias: torch.Tensor,
    target_weight: torch.Tensor,
    target_bias: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    decoder_bias = decoder_bias.float().to(device)
    target_weight = target_weight.float().to(device)
    target_bias = target_bias.float().to(device)
    return (decoder_bias @ target_weight.T + target_bias).cpu()


def _convert_bundle_dtype(
    tensors: dict[str, torch.Tensor],
    target_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if tensor.is_floating_point():
            converted[key] = tensor.to(target_dtype)
        else:
            converted[key] = tensor
    return converted


def _architecture_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint["config"]
    return {
        "k": config["k"],
        "d_in": config["d_in"],
        "num_experts": config["num_experts"],
        "active_experts": config["active_experts"],
        "latents_per_expert": config["latents_per_expert"],
        "jumprelu_init_threshold": config["jumprelu_init_threshold"],
        "jumprelu_bandwidth": config["jumprelu_bandwidth"],
        "left_keys": int(checkpoint["router_left_weight"].shape[0]),
        "right_keys": int(checkpoint["router_right_weight"].shape[0]),
    }


def export_projection_layer(
    *,
    model: Any,
    checkpoint_dir: Path,
    output_dir: Path,
    layer_idx: int,
    operator_name: str,
    output_name: str,
    target_modules: Sequence[str],
    target_dtype: torch.dtype,
    device: str,
    batch_size: int | None,
) -> dict[str, Any]:
    checkpoint = load_product_key_checkpoint(checkpoint_dir)
    target_weight, target_bias = get_fused_weight_and_bias(
        model,
        layer_idx,
        target_modules,
    )

    d_in = checkpoint["decoder_weight"].shape[1]
    if d_in != target_weight.shape[1]:
        raise ValueError(
            f"Input dimension mismatch for {checkpoint_dir}: "
            f"decoder d_in={d_in}, target in_features={target_weight.shape[1]}"
        )

    precomputed_products = compute_precomputed_products(
        checkpoint["decoder_weight"],
        target_weight,
        device=device,
        batch_size=batch_size,
    )
    bias_product = compute_bias_product(
        checkpoint["decoder_bias"],
        target_weight,
        target_bias,
        device=device,
    )
    compensation_weight_t = target_weight.T.contiguous().cpu()

    bundle = {
        "router_left_weight": checkpoint["router_left_weight"],
        "router_left_bias": checkpoint["router_left_bias"],
        "router_right_weight": checkpoint["router_right_weight"],
        "router_right_bias": checkpoint["router_right_bias"],
        "pair_left_index": checkpoint["pair_left_index"],
        "pair_right_index": checkpoint["pair_right_index"],
        "expert_encoder_weight": checkpoint["expert_encoder_weight"],
        "expert_encoder_bias": checkpoint["expert_encoder_bias"],
        "expert_threshold": checkpoint["expert_threshold"],
        "decoder_weight": checkpoint["decoder_weight"],
        "decoder_bias": checkpoint["decoder_bias"],
        "precomputed_products": precomputed_products,
        "bias_product": bias_product,
        "compensation_weight_t": compensation_weight_t,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{output_name}.lut.safetensors"
    save_file(
        _convert_bundle_dtype(bundle, target_dtype),
        str(output_dir / file_name),
    )

    return {
        "file": file_name,
        "input_dim": int(d_in),
        "output_dim": int(target_weight.shape[0]),
        "operator": operator_name,
        "encoder_architecture": "product_key_expert_jumprelu",
        "architecture_config": _architecture_config(checkpoint),
    }


def build_metadata(
    *,
    model_path: str,
    model_config: dict[str, Any],
    layer_info: dict[str, Any],
    compensation_ratio: float,
    dtype_name: str,
) -> dict[str, Any]:
    return {
        "version": "2.0",
        "architecture": "product_key_expert_jumprelu",
        "runtime_target": "gpu_decode_only",
        "operators": ["self_attn.qkv_proj", "mlp.gate_up_proj"],
        "dtype": dtype_name,
        "compensation": {
            "mode": "ratio",
            "ratio": compensation_ratio,
        },
        "model": {
            "path": model_path,
            "model_type": model_config.get("model_type"),
            "num_layers": model_config.get("num_hidden_layers"),
            "hidden_size": model_config.get("hidden_size"),
            "num_attention_heads": model_config.get("num_attention_heads"),
            "intermediate_size": model_config.get("intermediate_size"),
        },
        "layers": layer_info,
        "creation_info": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "script": "export_product_key_expert_jumprelu_lut.py",
        },
    }


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def _load_model(model_path: str) -> Any:
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export product_key_expert_jumprelu checkpoints to standalone LUT bundles.",
    )
    parser.add_argument("model_path", type=str, help="Path to the base model.")
    parser.add_argument(
        "--qproj-best-dir",
        type=str,
        required=True,
        help="Path to the q_proj checkpoint `best/` directory.",
    )
    parser.add_argument(
        "--upproj-best-dir",
        type=str,
        required=True,
        help="Path to the up_proj checkpoint `best/` directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write metadata.json and *.lut.safetensors bundles.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        help="Layer range, for example `14-27` or `14,16,20`.",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        default=["qkv", "gate_up"],
        choices=sorted(PROJECTION_SPECS),
        help="Projection groups to export.",
    )
    parser.add_argument(
        "--compensation-ratio",
        type=float,
        required=True,
        help="Default compensation ratio to store in metadata.json.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Floating-point dtype for exported tensors.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for offline matmul computation, for example `cpu` or `cuda:0`.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Latent batch size for offline matmul. Use 0 to disable batching.",
    )
    args = parser.parse_args()

    qproj_best_dir = Path(args.qproj_best_dir)
    upproj_best_dir = Path(args.upproj_best_dir)
    output_dir = Path(args.output_dir)
    layer_indices = resolve_layer_indices(
        qproj_best_dir,
        upproj_best_dir,
        layers=parse_layer_range(args.layers) if args.layers else None,
    )
    target_dtype = _dtype_from_name(args.dtype)
    batch_size = None if args.batch_size == 0 else args.batch_size

    print(f"Loading model from {args.model_path}...")
    model = _load_model(args.model_path)

    projection_best_dirs = {
        "qkv": qproj_best_dir,
        "gate_up": upproj_best_dir,
    }
    layer_info: dict[str, Any] = {}

    for layer_idx in layer_indices:
        for operator_key in args.operators:
            spec = PROJECTION_SPECS[operator_key]
            checkpoint_dir = _checkpoint_dir(
                projection_best_dirs[operator_key],
                layer_idx,
                spec["checkpoint_suffix"],
            )
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

            output_name = spec["output_name"].format(layer_idx=layer_idx)
            print(f"Exporting {output_name} from {checkpoint_dir}...")
            layer_info[output_name] = export_projection_layer(
                model=model,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                layer_idx=layer_idx,
                operator_name=spec["operator_name"],
                output_name=output_name,
                target_modules=spec["target_modules"],
                target_dtype=target_dtype,
                device=args.device,
                batch_size=batch_size,
            )

    metadata = build_metadata(
        model_path=args.model_path,
        model_config=model.config.to_dict(),
        layer_info=layer_info,
        compensation_ratio=args.compensation_ratio,
        dtype_name=args.dtype,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {len(layer_info)} LUT bundles to {output_dir}")
    print(f"Wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
