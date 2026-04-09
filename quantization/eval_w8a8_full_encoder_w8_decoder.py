from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from natsort import natsorted
from torch import Tensor
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantization.eval_utils import (
    build_summary,
    compute_reconstruction_metrics,
    load_elbow_thresholds_for_hookpoints,
    resolve_checkpoint_paths,
    resolve_matching_hookpoints,
)
from quantization.eval_w8a8_full_encoder import (
    DEFAULT_ALPHA,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RESULTS_ROOT,
    QuantizedDecoderState,
    QuantizedFullEncoderState,
    RunningMetric,
    capture_hook_activations,
    encode_w8a8_full_encoder,
    ensure_product_key_sae,
    infer_threshold_path,
    load_dataset,
    load_model,
    prepare_quantized_decoder_state,
    prepare_quantized_expert_state,
    prepare_quantized_linear_state,
    print_table,
    resolve_runtime_device,
    write_csv,
)
from quantization.quant_utils import simulate_w8_sparse_decode_prequantized
from sparsify.sparse_coder import ProductKeyExpertJumpReLUSparseCoder, SparseCoder


@dataclass
class QuantizedFullStackState:
    encoder: QuantizedFullEncoderState
    decoder: QuantizedDecoderState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs full-encoder W8A8 plus decoder-W8 simulation for product_key_expert_jumprelu checkpoints.",
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


@torch.no_grad()
def decode_w8_weight_only(
    sae: ProductKeyExpertJumpReLUSparseCoder,
    quant_state: QuantizedDecoderState,
    top_acts: Tensor,
    top_indices: Tensor,
) -> Tensor:
    original_shape = top_acts.shape[:-1]
    flat_acts = top_acts.reshape(-1, top_acts.shape[-1])
    flat_indices = top_indices.reshape(-1, top_indices.shape[-1])
    sparse_out = simulate_w8_sparse_decode_prequantized(
        flat_indices,
        flat_acts,
        quant_state.q_weight,
        quant_state.weight_scales,
    )
    return sparse_out.reshape(*original_shape, sae.d_in)


@torch.no_grad()
def forward_w8a8_full_encoder_w8_decoder(
    sae: ProductKeyExpertJumpReLUSparseCoder,
    quant_state: QuantizedFullStackState,
    x: Tensor,
) -> Tensor:
    top_acts, top_indices = encode_w8a8_full_encoder(sae, quant_state.encoder, x)
    sparse_out = decode_w8_weight_only(sae, quant_state.decoder, top_acts, top_indices)
    return sparse_out + sae.b_dec


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
    quant_states: dict[str, QuantizedFullStackState] = {}
    for hookpoint in hookpoints:
        sae = SparseCoder.load_any(checkpoint_paths[hookpoint], device=device_str).eval()
        sae = ensure_product_key_sae(sae, hookpoint)
        sae.requires_grad_(False)
        saes[hookpoint] = sae
        quant_states[hookpoint] = QuantizedFullStackState(
            encoder=QuantizedFullEncoderState(
                left_router=prepare_quantized_linear_state(sae.left_router),
                right_router=prepare_quantized_linear_state(sae.right_router),
                expert_encoders=prepare_quantized_expert_state(sae),
            ),
            decoder=prepare_quantized_decoder_state(sae),
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
            quant_recon = forward_w8a8_full_encoder_w8_decoder(
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
                "quantization_scope": "full_encoder_w8_decoder",
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
