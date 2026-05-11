#!/usr/bin/env python3
"""Export dense JumpReLU lm_head SAE checkpoints into LUT bundles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


REQUIRED_TENSORS = {
    "encoder.weight",
    "encoder.bias",
    "log_threshold",
    "W_dec",
    "b_dec",
}


def _load_tensor_map(path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def _is_checkpoint_dir(path: Path) -> bool:
    return (path / "cfg.json").exists() and (path / "sae.safetensors").exists()


def resolve_lm_head_checkpoint(path: Path | str) -> Path:
    root = Path(path)
    if _is_checkpoint_dir(root):
        return root

    preferred = [
        root / "best" / "lm_head",
        root / "lm_head",
    ]
    for candidate in preferred:
        if _is_checkpoint_dir(candidate):
            return candidate

    candidates = [
        candidate
        for candidate in root.rglob("lm_head")
        if candidate.is_dir() and _is_checkpoint_dir(candidate)
    ]
    if not candidates:
        raise FileNotFoundError(f"No lm_head checkpoint found under {root}")

    def priority(candidate: Path) -> tuple[int, str]:
        is_best = int(candidate.parent.name == "best")
        return (is_best, str(candidate))

    return max(candidates, key=priority)


def load_dense_jumprelu_checkpoint(checkpoint_dir: Path | str) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "cfg.json", "r") as f:
        config = json.load(f)

    if config.get("architecture") != "jumprelu":
        raise ValueError(f"{checkpoint_dir} is not a dense jumprelu checkpoint")

    tensors = _load_tensor_map(checkpoint_dir / "sae.safetensors")
    missing = sorted(REQUIRED_TENSORS - set(tensors))
    if missing:
        raise ValueError(f"Missing checkpoint tensors in {checkpoint_dir}: {missing}")

    return {
        "encoder_weight": tensors["encoder.weight"],
        "encoder_bias": tensors["encoder.bias"],
        "threshold": F.softplus(tensors["log_threshold"]),
        "decoder_weight": tensors["W_dec"],
        "decoder_bias": tensors["b_dec"],
        "config": config,
    }


def get_lm_head_weight_and_bias(model: Any) -> tuple[torch.Tensor, torch.Tensor]:
    lm_head = model.get_submodule("lm_head")
    weight = lm_head.weight.detach().cpu()
    bias = getattr(lm_head, "bias", None)
    if bias is None:
        bias_tensor = torch.zeros(weight.shape[0], dtype=weight.dtype)
    else:
        bias_tensor = bias.detach().cpu()
    return weight, bias_tensor


def compute_precomputed_products(
    decoder_weight: torch.Tensor,
    target_weight: torch.Tensor,
    *,
    device: str,
    batch_size: int | None,
) -> torch.Tensor:
    target_weight = target_weight.float().to(device)
    decoder_weight = decoder_weight.float()

    if batch_size is None:
        return (decoder_weight.to(device) @ target_weight.T).cpu()

    num_latents = decoder_weight.shape[0]
    output_dim = target_weight.shape[0]
    output = torch.empty(num_latents, output_dim, dtype=torch.float32, device="cpu")
    for start in range(0, num_latents, batch_size):
        end = min(start + batch_size, num_latents)
        chunk = decoder_weight[start:end].to(device)
        output[start:end] = (chunk @ target_weight.T).cpu()
    return output


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
        "architecture": config["architecture"],
        "k": config["k"],
        "d_in": config["d_in"],
        "expansion_factor": config["expansion_factor"],
        "num_latents": int(checkpoint["decoder_weight"].shape[0]),
        "jumprelu_init_threshold": config["jumprelu_init_threshold"],
        "jumprelu_bandwidth": config["jumprelu_bandwidth"],
        "encoder_out_dim": int(checkpoint["encoder_weight"].shape[0]),
    }


def export_lm_head(
    *,
    model: Any,
    checkpoint_dir: Path | str,
    output_dir: Path | str,
    target_dtype: torch.dtype,
    device: str,
    batch_size: int | None,
) -> dict[str, Any]:
    checkpoint = load_dense_jumprelu_checkpoint(checkpoint_dir)
    target_weight, target_bias = get_lm_head_weight_and_bias(model)

    d_in = checkpoint["decoder_weight"].shape[1]
    if d_in != target_weight.shape[1]:
        raise ValueError(
            f"Input dimension mismatch for {checkpoint_dir}: "
            f"decoder d_in={d_in}, lm_head in_features={target_weight.shape[1]}"
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
        "encoder_weight": checkpoint["encoder_weight"],
        "encoder_bias": checkpoint["encoder_bias"],
        "threshold": checkpoint["threshold"],
        "decoder_weight": checkpoint["decoder_weight"],
        "decoder_bias": checkpoint["decoder_bias"],
        "precomputed_products": precomputed_products,
        "bias_product": bias_product,
        "compensation_weight_t": compensation_weight_t,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        _convert_bundle_dtype(bundle, target_dtype),
        str(output_dir / "lm_head.lut.safetensors"),
    )

    return {
        "file": "lm_head.lut.safetensors",
        "input_dim": int(d_in),
        "output_dim": int(target_weight.shape[0]),
        "operator": "lm_head",
        "encoder_architecture": "jumprelu",
        "architecture_config": _architecture_config(checkpoint),
    }


def build_metadata(
    *,
    model_path: str,
    model_config: dict[str, Any],
    layer_info: dict[str, Any],
    dtype_name: str,
) -> dict[str, Any]:
    return {
        "version": "1.0",
        "architecture": "jumprelu",
        "runtime_target": "gpu_decode_only",
        "operators": ["lm_head"],
        "dtype": dtype_name,
        "model": {
            "path": model_path,
            "model_type": model_config.get("model_type"),
            "hidden_size": model_config.get("hidden_size"),
            "vocab_size": model_config.get("vocab_size"),
        },
        "layers": layer_info,
        "creation_info": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/export/export_dense_jumprelu_lm_head_lut.py",
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
        description="Export dense jumprelu lm_head SAE checkpoint to a LUT bundle.",
    )
    parser.add_argument("model_path", type=str, help="Path to the base model.")
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Run directory, best/lm_head directory, or lm_head checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write metadata.json and lm_head.lut.safetensors.",
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
        help="Device used for offline matmul computation, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Latent batch size for offline matmul. Use 0 to disable batching.",
    )
    args = parser.parse_args()

    checkpoint_dir = resolve_lm_head_checkpoint(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    target_dtype = _dtype_from_name(args.dtype)
    batch_size = None if args.batch_size == 0 else args.batch_size

    print(f"Resolved lm_head checkpoint: {checkpoint_dir}")
    print(f"Loading model from {args.model_path}...")
    model = _load_model(args.model_path)

    print("Exporting lm_head LUT...")
    layer_info = {
        "lm_head": export_lm_head(
            model=model,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            target_dtype=target_dtype,
            device=args.device,
            batch_size=batch_size,
        )
    }

    metadata = build_metadata(
        model_path=args.model_path,
        model_config=model.config.to_dict(),
        layer_info=layer_info,
        dtype_name=args.dtype,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote lm_head LUT bundle to {output_dir / 'lm_head.lut.safetensors'}")
    print(f"Wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
