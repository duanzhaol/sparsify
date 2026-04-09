from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset as HfDataset
from natsort import natsorted
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import AutoModel

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantization.eval_utils import (
    build_summary,
    compute_reconstruction_metrics,
    load_elbow_thresholds_for_hookpoints,
    resolve_checkpoint_paths,
    resolve_matching_hookpoints,
)
from quantization.quant_utils import (
    quantize_weight_per_row_symmetric,
    simulate_w8a8_linear_prequantized,
    simulate_w8a8_matmul_prequantized,
)
from sparsify.device import get_device_type, is_bf16_supported
from sparsify.sparse_coder import (
    ProductKeyExpertJumpReLUSparseCoder,
    SparseCoder,
    _finalize_routed_expert_acts,
)
from sparsify.utils import get_layer_list, get_max_layer_index, partial_forward_to_layer


DEFAULT_MODEL_PATH = "/root/models/Qwen3-0.6B"
DEFAULT_DATASET_PATH = "/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048"
DEFAULT_RESULTS_ROOT = Path("quantization/results")
DEFAULT_ALPHA = 0.50


@dataclass
class QuantizedLinearState:
    q_weight: Tensor
    weight_scales: Tensor
    bias: Tensor | None


@dataclass
class QuantizedExpertState:
    q_weight: Tensor
    weight_scales: Tensor


@dataclass
class QuantizedDecoderState:
    q_weight: Tensor
    weight_scales: Tensor


@dataclass
class QuantizedFullEncoderState:
    left_router: QuantizedLinearState
    right_router: QuantizedLinearState
    expert_encoders: QuantizedExpertState


@dataclass
class RunningMetric:
    weighted_sum: float = 0.0
    total_vectors: int = 0

    def update(self, value: float, num_vectors: int) -> None:
        self.weighted_sum += float(value) * int(num_vectors)
        self.total_vectors += int(num_vectors)

    @property
    def mean(self) -> float:
        if self.total_vectors == 0:
            return 0.0
        return self.weighted_sum / self.total_vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs full-encoder W8A8 simulation for product_key_expert_jumprelu checkpoints.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--hookpoints", nargs="+", required=True)
    parser.add_argument("--elbow-threshold-path")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--output-dir")
    parser.add_argument("--save-per-layer-json", action="store_true")
    return parser.parse_args()


def resolve_runtime_device(device_arg: str) -> tuple[torch.device, str]:
    if device_arg != "auto":
        device = torch.device(device_arg)
        return device, str(device)

    device_type = get_device_type()
    if device_type == "cpu":
        return torch.device("cpu"), "cpu"
    return torch.device(f"{device_type}:0"), f"{device_type}:0"


def infer_threshold_path(model_path: str, hookpoints: list[str]) -> str | None:
    if not hookpoints:
        return None

    model_name = Path(model_path).name
    threshold_root = Path("thresholds") / model_name
    if all("self_attn.q_proj" in hookpoint for hookpoint in hookpoints):
        candidate = threshold_root / "thresholds_q.json"
    elif all("mlp.up_proj" in hookpoint for hookpoint in hookpoints):
        candidate = threshold_root / "thresholds_up.json"
    else:
        return None

    if candidate.exists():
        return str(candidate)
    return None


def load_model(model_path: str, device_str: str) -> nn.Module:
    dtype = torch.bfloat16 if is_bf16_supported() else torch.float32
    if device_str == "cpu":
        model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
        return model.to(torch.device(device_str))

    return AutoModel.from_pretrained(
        model_path,
        device_map={"": device_str},
        torch_dtype=dtype,
    )


def load_dataset(dataset_path: str, num_samples: int) -> HfDataset:
    dataset = HfDataset.load_from_disk(dataset_path, keep_in_memory=False)
    dataset = dataset.with_format("torch")
    limit = min(len(dataset), num_samples)
    return dataset.select(range(limit))


def prepare_quantized_linear_state(linear: nn.Linear) -> QuantizedLinearState:
    q_weight, weight_scales = quantize_weight_per_row_symmetric(linear.weight.detach())
    bias = None if linear.bias is None else linear.bias.detach()
    return QuantizedLinearState(
        q_weight=q_weight,
        weight_scales=weight_scales,
        bias=bias,
    )


def prepare_quantized_expert_state(
    sae: ProductKeyExpertJumpReLUSparseCoder,
) -> QuantizedExpertState:
    q_weight, weight_scales = quantize_weight_per_row_symmetric(
        sae.expert_encoders.detach()
    )
    return QuantizedExpertState(q_weight=q_weight, weight_scales=weight_scales)


def prepare_quantized_decoder_state(
    sae: ProductKeyExpertJumpReLUSparseCoder,
) -> QuantizedDecoderState:
    assert sae.W_dec is not None, "Decoder weight was not initialized."
    q_weight, weight_scales = quantize_weight_per_row_symmetric(sae.W_dec.detach())
    return QuantizedDecoderState(q_weight=q_weight, weight_scales=weight_scales)


@torch.no_grad()
def encode_w8a8_full_encoder(
    sae: ProductKeyExpertJumpReLUSparseCoder,
    quant_state: QuantizedFullEncoderState,
    x: Tensor,
) -> tuple[Tensor, Tensor]:
    x_centered = x - sae.b_dec
    original_shape = x_centered.shape[:-1]
    flat_x = x_centered.reshape(-1, sae.d_in)

    left_logits = simulate_w8a8_linear_prequantized(
        flat_x,
        quant_state.left_router.q_weight,
        quant_state.left_router.weight_scales,
        bias=quant_state.left_router.bias,
    )
    right_logits = simulate_w8a8_linear_prequantized(
        flat_x,
        quant_state.right_router.q_weight,
        quant_state.right_router.weight_scales,
        bias=quant_state.right_router.bias,
    )
    expert_logits = (
        left_logits[:, sae.pair_left_index] + right_logits[:, sae.pair_right_index]
    )
    selected_expert_idx, selected_probs = sae._select_expert_route(expert_logits)

    selected_weight = quant_state.expert_encoders.q_weight[selected_expert_idx.long()]
    selected_scales = quant_state.expert_encoders.weight_scales[
        selected_expert_idx.long()
    ]
    selected_bias = sae.expert_encoder_bias[selected_expert_idx.long()]
    selected_threshold = sae.threshold[selected_expert_idx.long()]

    pre_acts = (
        simulate_w8a8_matmul_prequantized(flat_x, selected_weight, selected_scales)
        + selected_bias
    )
    positive = F.relu(pre_acts)
    gate = torch.sigmoid(
        (positive - selected_threshold) / sae.cfg.jumprelu_bandwidth
    )
    candidate_acts = positive * gate * selected_probs.unsqueeze(-1)

    top_acts, top_indices, _ = _finalize_routed_expert_acts(
        candidate_acts,
        selected_expert_idx,
        sae.cfg.k,
        sae.latents_per_expert,
        sae.num_latents,
        return_full=False,
    )

    target_shape = (*original_shape, sae.cfg.k)
    top_acts = top_acts.reshape(target_shape)
    top_indices = top_indices.reshape(target_shape)
    return top_acts, top_indices


@torch.no_grad()
def forward_w8a8_full_encoder(
    sae: ProductKeyExpertJumpReLUSparseCoder,
    quant_state: QuantizedFullEncoderState,
    x: Tensor,
) -> Tensor:
    top_acts, top_indices = encode_w8a8_full_encoder(sae, quant_state, x)
    sparse_out = sae._decode_sparse(top_acts, top_indices)
    return sparse_out + sae.b_dec


@torch.no_grad()
def capture_hook_activations(
    model: nn.Module,
    input_ids: Tensor,
    hookpoints: list[str],
) -> dict[str, Tensor]:
    module_map = {
        model.base_model.get_submodule(name): name  # type: ignore[attr-defined]
        for name in hookpoints
    }
    captured: dict[str, Tensor] = {}

    def hook(module: nn.Module, inputs: tuple[Any, ...], _outputs: Any) -> None:
        tensor = inputs[0] if isinstance(inputs, tuple) else inputs
        captured[module_map[module]] = tensor.flatten(0, 1).detach()

    handles = [module.register_forward_hook(hook) for module in module_map]
    try:
        layers_name, _layers = get_layer_list(model)
        max_layer = get_max_layer_index(hookpoints, layers_name)
        if max_layer is not None:
            partial_forward_to_layer(model, input_ids, max_layer)
        else:
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    return captured


def ensure_product_key_sae(sae: SparseCoder, hookpoint: str) -> ProductKeyExpertJumpReLUSparseCoder:
    if not isinstance(sae, ProductKeyExpertJumpReLUSparseCoder):
        raise TypeError(
            f"Hookpoint '{hookpoint}' uses unsupported SAE type '{type(sae).__name__}'. "
            "This script currently supports only ProductKeyExpertJumpReLUSparseCoder."
        )
    return sae


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def print_table(records: list[dict[str, Any]], alpha: float) -> None:
    if not records:
        print("No matching hookpoints were evaluated.")
        return

    exceed_key = f"exceed_alpha_{alpha:.2f}".replace(".", ".")
    header = (
        f"{'hookpoint':55} {'fvu_base':>10} {'fvu_w8a8':>10} {'delta':>10} "
        f"{(exceed_key + '_base'):>18} {(exceed_key + '_w8a8'):>18} {'delta':>10}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        print(
            f"{record['hookpoint']:55} "
            f"{record['fvu_base']:10.6f} {record['fvu_w8a8']:10.6f} {record['fvu_delta']:10.6f} "
            f"{record[f'{exceed_key}_base']:18.6f} {record[f'{exceed_key}_w8a8']:18.6f} {record[f'{exceed_key}_delta']:10.6f}"
        )


def main() -> None:
    args = parse_args()
    device, device_str = resolve_runtime_device(args.device)
    model = load_model(args.model, device_str)
    dataset = load_dataset(args.dataset, args.num_samples)

    available_modules = [
        name for name, _module in model.base_model.named_modules()  # type: ignore[attr-defined]
    ]
    hookpoints = resolve_matching_hookpoints(args.hookpoints, available_modules)
    if not hookpoints:
        raise ValueError(f"No hookpoints matched patterns: {args.hookpoints}")

    threshold_path = args.elbow_threshold_path or infer_threshold_path(args.model, hookpoints)
    if threshold_path is None:
        raise ValueError(
            "Could not infer elbow threshold path; please pass --elbow-threshold-path explicitly."
        )

    threshold_map = load_elbow_thresholds_for_hookpoints(threshold_path, hookpoints)
    missing_thresholds = [hook for hook in hookpoints if hook not in threshold_map]
    if missing_thresholds:
        raise ValueError(f"Missing elbow thresholds for hookpoints: {missing_thresholds}")

    checkpoint_paths = resolve_checkpoint_paths(args.checkpoint_root, hookpoints)
    saes: dict[str, ProductKeyExpertJumpReLUSparseCoder] = {}
    quant_states: dict[str, QuantizedFullEncoderState] = {}
    for hookpoint in hookpoints:
        sae = SparseCoder.load_any(checkpoint_paths[hookpoint], device=device_str).eval()
        sae = ensure_product_key_sae(sae, hookpoint)
        sae.requires_grad_(False)
        saes[hookpoint] = sae
        quant_states[hookpoint] = QuantizedFullEncoderState(
            left_router=prepare_quantized_linear_state(sae.left_router),
            right_router=prepare_quantized_linear_state(sae.right_router),
            expert_encoders=prepare_quantized_expert_state(sae),
        )

    results: dict[str, dict[str, RunningMetric]] = defaultdict(
        lambda: defaultdict(RunningMetric)
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    exceed_key = f"exceed_alpha_{args.alpha:.2f}"

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        activations = capture_hook_activations(model, input_ids, hookpoints)
        for hookpoint in hookpoints:
            acts = activations[hookpoint]
            num_vectors = acts.shape[0]
            elbow_value = threshold_map[hookpoint]
            sae = saes[hookpoint]

            base_recon = sae(acts).sae_out
            quant_recon = forward_w8a8_full_encoder(
                sae,
                quant_states[hookpoint],
                acts,
            )

            base_metrics = compute_reconstruction_metrics(
                acts,
                base_recon,
                elbow_value,
                alpha=args.alpha,
            )
            quant_metrics = compute_reconstruction_metrics(
                acts,
                quant_recon,
                elbow_value,
                alpha=args.alpha,
            )

            results[hookpoint]["fvu_base"].update(base_metrics["fvu"], num_vectors)
            results[hookpoint]["fvu_w8a8"].update(quant_metrics["fvu"], num_vectors)
            results[hookpoint][f"{exceed_key}_base"].update(
                base_metrics[exceed_key], num_vectors
            )
            results[hookpoint][f"{exceed_key}_w8a8"].update(
                quant_metrics[exceed_key], num_vectors
            )

    records: list[dict[str, Any]] = []
    for hookpoint in natsorted(hookpoints):
        fvu_base = results[hookpoint]["fvu_base"].mean
        fvu_w8a8 = results[hookpoint]["fvu_w8a8"].mean
        exceed_base = results[hookpoint][f"{exceed_key}_base"].mean
        exceed_w8a8 = results[hookpoint][f"{exceed_key}_w8a8"].mean
        records.append(
            {
                "hookpoint": hookpoint,
                "fvu_base": fvu_base,
                "fvu_w8a8": fvu_w8a8,
                "fvu_delta": fvu_w8a8 - fvu_base,
                f"{exceed_key}_base": exceed_base,
                f"{exceed_key}_w8a8": exceed_w8a8,
                f"{exceed_key}_delta": exceed_w8a8 - exceed_base,
            }
        )

    summary = build_summary(
        [
            {
                **record,
                "exceed_alpha_0.50_base": record[f"{exceed_key}_base"],
                "exceed_alpha_0.50_w8a8": record[f"{exceed_key}_w8a8"],
                "exceed_alpha_0.50_delta": record[f"{exceed_key}_delta"],
            }
            for record in records
        ]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_csv(output_dir / "summary.csv", records)
    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "dataset": args.dataset,
                "checkpoint_root": args.checkpoint_root,
                "hookpoint_patterns": args.hookpoints,
                "hookpoints": hookpoints,
                "elbow_threshold_path": threshold_path,
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "device": device_str,
                "alpha": args.alpha,
                "quantization_scope": "full_encoder",
                "output_dir": str(output_dir),
            },
            f,
            indent=2,
        )

    if args.save_per_layer_json:
        for record in records:
            safe_name = record["hookpoint"].replace("/", "__").replace(".", "_")
            with open(output_dir / f"{safe_name}.json", "w") as f:
                json.dump(record, f, indent=2)

    print_table(records, args.alpha)
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
