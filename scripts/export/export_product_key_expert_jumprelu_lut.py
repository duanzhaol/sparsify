#!/usr/bin/env python3
"""
Export product_key_expert_jumprelu SAE checkpoints into standalone LUT bundles.

Each output `.lut.safetensors` file is a runtime-complete artifact containing:
- Stage A routing + encoder tensors
- Stage B precomputed lookup tables
- Stage C compensation weights

The exporter lives under `scripts/export/` because the product-key JumpReLU
checkpoints use a different tensor layout and a richer runtime bundle schema
than the older generic LUT exporters.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
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
RUN_TIMESTAMP_PATTERN = re.compile(r"_(\d{8}_\d{6})$")
CONFIG_MATCH_KEYS = (
    "architecture",
    "d_in",
    "k",
    "num_experts",
    "active_experts",
    "latents_per_expert",
    "jumprelu_init_threshold",
    "jumprelu_bandwidth",
)


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


def _run_name_for_layer_dir(layer_dir: Path) -> str:
    if layer_dir.parent.name == "best" and layer_dir.parent.parent.name:
        return layer_dir.parent.parent.name
    return str(layer_dir.parent)


def _priority_for_layer_dir(layer_dir: Path) -> tuple[str, str]:
    run_name = _run_name_for_layer_dir(layer_dir)
    match = RUN_TIMESTAMP_PATTERN.search(run_name)
    timestamp = match.group(1) if match else ""
    return (timestamp, run_name, str(layer_dir))


def _load_checkpoint_cfg(layer_dir: Path) -> dict[str, Any]:
    with open(layer_dir / "cfg.json", "r") as f:
        return json.load(f)


def _config_signature(layer_dir: Path) -> tuple[Any, ...]:
    config = _load_checkpoint_cfg(layer_dir)
    return tuple(config.get(key) for key in CONFIG_MATCH_KEYS)


def _validate_projection_sources(
    operator_key: str,
    sources: dict[int, Path],
    *,
    layers: Sequence[int] | None = None,
) -> None:
    if layers is None:
        selected_sources = dict(sources)
    else:
        selected_sources = {layer_idx: sources[layer_idx] for layer_idx in layers}

    if not selected_sources:
        return

    baseline_layer = min(selected_sources)
    baseline_signature = _config_signature(selected_sources[baseline_layer])
    for layer_idx, layer_dir in selected_sources.items():
        signature = _config_signature(layer_dir)
        if signature != baseline_signature:
            raise ValueError(
                f"Incompatible configs detected while merging {operator_key} layer "
                f"{layer_idx} from {layer_dir}."
            )


def discover_projection_layer_sources(
    checkpoint_root: Path,
    operator_key: str,
) -> dict[int, Path]:
    spec = PROJECTION_SPECS[operator_key]
    layer_pattern = f"layers.*.{spec['checkpoint_suffix']}"
    chosen: dict[int, Path] = {}

    for layer_dir in checkpoint_root.rglob(layer_pattern):
        if not layer_dir.is_dir():
            continue
        if not (layer_dir / "cfg.json").exists():
            continue
        if not (layer_dir / "sae.safetensors").exists():
            continue

        match = LAYER_PATTERN.search(layer_dir.name)
        if match is None:
            continue
        layer_idx = int(match.group(1))

        current = chosen.get(layer_idx)
        if current is None or _priority_for_layer_dir(layer_dir) >= _priority_for_layer_dir(current):
            chosen[layer_idx] = layer_dir

    return dict(sorted(chosen.items()))


def resolve_layer_indices(
    qproj_sources: dict[int, Path],
    upproj_sources: dict[int, Path],
    *,
    layers: list[int] | None,
    operators: Sequence[str] | None = None,
) -> list[int]:
    selected_operators = set(operators or ("qkv", "gate_up"))
    available_sets = []
    if "qkv" in selected_operators:
        available_sets.append(set(qproj_sources))
    if "gate_up" in selected_operators:
        available_sets.append(set(upproj_sources))

    if not available_sets:
        raise ValueError("At least one operator must be selected.")

    available_layers = set.intersection(*available_sets)
    if layers is not None:
        requested_layers = sorted(set(layers))
        missing = [layer for layer in requested_layers if layer not in available_layers]
        if missing:
            raise ValueError(f"Requested layers missing from discovered sources: {missing}")
        return requested_layers

    resolved = sorted(available_layers)
    if not resolved:
        raise ValueError("No shared layers found for the selected operators.")
    return resolved


def _ensure_clean_symlink_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.exists():
        shutil.rmtree(path)


def materialize_merge_view(
    *,
    q_sources: dict[int, Path],
    up_sources: dict[int, Path],
    output_dir: Path,
    layers: Sequence[int],
    operators: Sequence[str] | None = None,
) -> dict[str, Path]:
    selected_operators = set(operators or ("qkv", "gate_up"))
    output_dir.mkdir(parents=True, exist_ok=True)
    qproj_best_dir = output_dir / "qproj_best"
    upproj_best_dir = output_dir / "upproj_best"
    qproj_best_dir.mkdir(parents=True, exist_ok=True)
    upproj_best_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "layers": list(layers),
        "qkv": {},
        "gate_up": {},
    }

    if "qkv" in selected_operators:
        for layer_idx in layers:
            source = q_sources[layer_idx]
            dest = qproj_best_dir / source.name
            _ensure_clean_symlink_path(dest)
            dest.symlink_to(source.resolve(), target_is_directory=True)
            manifest["qkv"][str(layer_idx)] = str(source)

    if "gate_up" in selected_operators:
        for layer_idx in layers:
            source = up_sources[layer_idx]
            dest = upproj_best_dir / source.name
            _ensure_clean_symlink_path(dest)
            dest.symlink_to(source.resolve(), target_is_directory=True)
            manifest["gate_up"][str(layer_idx)] = str(source)

    with open(output_dir / "merge_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "qproj_best_dir": qproj_best_dir,
        "upproj_best_dir": upproj_best_dir,
    }


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
            "script": "scripts/export/export_product_key_expert_jumprelu_lut.py",
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
        description=(
            "Export product_key_expert_jumprelu checkpoints to standalone LUT "
            "bundles from a single checkpoint root, with optional merge view output."
        ),
    )
    parser.add_argument("model_path", type=str, help="Path to the base model.")
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help=(
            "Checkpoint root containing product_key_expert_jumprelu qproj/upproj "
            "runs. The exporter auto-discovers and merges per-layer best dirs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write metadata.json and *.lut.safetensors bundles.",
    )
    parser.add_argument(
        "--merge-output-dir",
        type=str,
        help=(
            "Optional directory to materialize a merged qproj/upproj best-view "
            "using symlinks plus a merge_manifest.json."
        ),
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

    checkpoint_root = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    qproj_sources = discover_projection_layer_sources(checkpoint_root, "qkv")
    upproj_sources = discover_projection_layer_sources(checkpoint_root, "gate_up")
    layer_indices = resolve_layer_indices(
        qproj_sources,
        upproj_sources,
        layers=parse_layer_range(args.layers) if args.layers else None,
        operators=args.operators,
    )
    if "qkv" in args.operators:
        _validate_projection_sources("qkv", qproj_sources, layers=layer_indices)
    if "gate_up" in args.operators:
        _validate_projection_sources("gate_up", upproj_sources, layers=layer_indices)
    target_dtype = _dtype_from_name(args.dtype)
    batch_size = None if args.batch_size == 0 else args.batch_size

    print(f"Loading model from {args.model_path}...")
    model = _load_model(args.model_path)

    if args.merge_output_dir:
        materialize_merge_view(
            q_sources=qproj_sources,
            up_sources=upproj_sources,
            output_dir=Path(args.merge_output_dir),
            layers=layer_indices,
            operators=args.operators,
        )

    projection_sources = {
        "qkv": qproj_sources,
        "gate_up": upproj_sources,
    }
    layer_info: dict[str, Any] = {}

    for layer_idx in layer_indices:
        for operator_key in args.operators:
            spec = PROJECTION_SPECS[operator_key]
            checkpoint_dir = projection_sources[operator_key][layer_idx]

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
