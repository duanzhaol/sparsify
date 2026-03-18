"""Oracle Baseline B: Hotset Selection (C1h upper bound).

Evaluates how well a fixed global "hot set" (most frequently activated basis vectors)
covers each token's true top-K selection.

Usage:
    python -m experiments.activation_patterns.hotset.run \
        --model /root/models/Qwen3-0.6B \
        --lut_dir /root/models/Qwen3-0.6B/lut \
        --dataset /root/fineweb-edu/sample/10BT-tokenized-qwen3-2048/ \
        --num_samples 256 --seq_len 512 \
        --layers 0 7 14 21 27 \
        --output_dir results/activation_patterns/hotset/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common.data import collect_raw_activations, encode_activations, load_dataset_auto
from experiments.common.sae_utils import get_layer_hookpoints
from transformers import AutoModelForCausalLM, AutoTokenizer


def analyze_hotset(
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    K: int,
    N: int,
    hotset_pcts: list[float] | None = None,
) -> dict:
    """Run hotset selection oracle analysis.

    Args:
        top_indices: (total_tokens, 2K) int32.
        top_values: (total_tokens, 2K) float32.
        K: SAE top-K value.
        N: Total number of latents.
        hotset_pcts: Hotset sizes as fractions of N. Default [0.01, 0.05, 0.10, 0.20].
    """
    if hotset_pcts is None:
        hotset_pcts = [0.01, 0.05, 0.10, 0.20]

    # Extract top-K subset
    topk_idx = top_indices[:, :K]  # (T, K)
    topk_val = top_values[:, :K]
    T = topk_idx.shape[0]

    # Global frequency: count how often each basis vector appears in top-K
    freq = torch.zeros(N, dtype=torch.int64)
    flat_idx = topk_idx.reshape(-1).long()
    freq.scatter_add_(0, flat_idx, torch.ones_like(flat_idx))

    results = {}

    for pct in hotset_pcts:
        H_size = max(1, int(N * pct))
        # Top-H_size most frequent basis vectors
        _, H_indices = freq.topk(H_size)

        # Vectorized: build a mask of which top-K entries are in H
        H_tensor = torch.zeros(N, dtype=torch.bool)
        H_tensor[H_indices] = True

        # Per-token recall
        hits = H_tensor[topk_idx.long()]  # (T, K) bool
        per_token_hit_count = hits.sum(dim=1).float()  # (T,)
        per_token_recall = per_token_hit_count / K

        # Value-weighted recall
        hit_mass = (topk_val.abs() * hits.float()).sum(dim=1)  # (T,)
        total_mass = topk_val.abs().sum(dim=1)  # (T,)
        per_token_recall_weighted = hit_mass / total_mass.clamp(min=1e-12)

        # Hot value ratio: what fraction of total activation mass comes from hotset
        hot_value_ratio = hit_mass.sum() / total_mass.sum()

        recall_np = per_token_recall.numpy()
        recall_w_np = per_token_recall_weighted.numpy()

        results[f"|H|={H_size}({pct*100:.0f}%)"] = {
            "H_size": H_size,
            "pct": pct,
            "recall_mean": float(recall_np.mean()),
            "recall_P10": float(np.percentile(recall_np, 10)),
            "recall_P25": float(np.percentile(recall_np, 25)),
            "recall_P50": float(np.percentile(recall_np, 50)),
            "recall_P90": float(np.percentile(recall_np, 90)),
            "recall_min": float(recall_np.min()),
            "recall_weighted_mean": float(recall_w_np.mean()),
            "recall_weighted_P10": float(np.percentile(recall_w_np, 10)),
            "hot_value_ratio": float(hot_value_ratio),
            "residual_search_space": N - H_size,
            "residual_K": float(K - per_token_hit_count.mean()),
            "num_tokens": T,
        }

    # Also compute frequency distribution stats
    freq_np = freq.numpy().astype(float)
    nonzero_freq = freq_np[freq_np > 0]
    results["frequency_stats"] = {
        "num_ever_active": int((freq_np > 0).sum()),
        "num_never_active": int((freq_np == 0).sum()),
        "freq_mean": float(nonzero_freq.mean()) if len(nonzero_freq) > 0 else 0,
        "freq_P50": float(np.median(nonzero_freq)) if len(nonzero_freq) > 0 else 0,
        "freq_P90": float(np.percentile(nonzero_freq, 90)) if len(nonzero_freq) > 0 else 0,
        "freq_max": int(freq_np.max()),
        "gini": float(gini_coefficient(freq_np)),
    }

    return results


def gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient (0 = perfect equality, 1 = max inequality)."""
    values = values.flatten()
    if values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return float((2 * np.sum((np.arange(1, n + 1) * sorted_vals)) / (n * cumsum[-1])) - (n + 1) / n)


def print_summary(all_results: dict):
    """Print a concise summary table."""
    print("\n" + "=" * 80)
    print("HOTSET SELECTION ORACLE BASELINE (C1h) — SUMMARY")
    print("=" * 80)

    for layer_name, data in all_results.items():
        print(f"\n--- {layer_name} ---")

        # Frequency stats
        fs = data.get("frequency_stats", {})
        print(f"  Active/total latents: {fs.get('num_ever_active', 0)}/{fs.get('num_ever_active', 0) + fs.get('num_never_active', 0)}")
        print(f"  Gini coefficient: {fs.get('gini', 0):.4f}")

        # Hotset recall table
        print(f"  {'|H|':>14s} | {'recall':>8s} | {'recall_w':>8s} | {'P10':>6s} | {'hot_val':>7s} | {'res_K':>5s}")
        print(f"  {'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}")
        for key, d in data.items():
            if key == "frequency_stats":
                continue
            print(f"  {key:>14s} | {d['recall_mean']:>8.4f} | "
                  f"{d['recall_weighted_mean']:>8.4f} | "
                  f"{d['recall_P10']:>6.3f} | "
                  f"{d['hot_value_ratio']:>7.4f} | "
                  f"{d['residual_K']:>5.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Oracle Baseline B: Hotset Selection (C1h)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lut_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 7, 14, 21, 27])
    parser.add_argument("--op_types", type=str, nargs="+", default=["mlp"],
                        choices=["mlp", "qkv", "o"],
                        help="Op types per layer: mlp=gate_up_proj, qkv=qkv_proj, o=o_proj")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results/activation_patterns/hotset/")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map={"": device}
    )
    model.eval()

    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset_auto(args.dataset, tokenizer, max_seq_len=args.seq_len)

    # Build hookpoint list and mapping
    hookpoints = []
    hookpoint_to_lut = {}
    for layer_idx in args.layers:
        pairs = get_layer_hookpoints(layer_idx, op_types=args.op_types)
        for hp, lut in pairs:
            if hp not in hookpoint_to_lut:
                hookpoints.append(hp)
                hookpoint_to_lut[hp] = lut

    # Step 1: Collect raw activations (single forward pass for all layers)
    print(f"\nStep 1: Collecting raw activations for layers {args.layers}...")
    raw_data = collect_raw_activations(
        model=model,
        dataset=dataset,
        hookpoints=hookpoints,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    # Free model memory before SAE encoding
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2+3: Encode → analyze → free, one hookpoint at a time
    print("\n" + "=" * 80)
    print("Encoding + hotset analysis (per-hookpoint streaming)...")
    print("=" * 80)

    all_results = {}
    for hookpoint in hookpoints:
        lut_layer = hookpoint_to_lut[hookpoint]

        # Encode this single hookpoint
        single_raw = {hookpoint: raw_data[hookpoint]}
        single_lut = {hookpoint: lut_layer}
        encoded = encode_activations(
            raw_data=single_raw,
            lut_dir=args.lut_dir,
            hookpoint_to_lut=single_lut,
            device=device,
        )

        # Free raw data for this hookpoint immediately
        del raw_data[hookpoint], single_raw

        # Analyze
        data = encoded[lut_layer]
        print(f"\n  Analyzing {lut_layer}...")
        result = analyze_hotset(
            top_indices=data["top_indices"],
            top_values=data["top_values"],
            K=data["K"],
            N=data["N"],
        )
        all_results[lut_layer] = result

        # Free encoded data
        del encoded, data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print summary
    print_summary(all_results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hotset_results.json"

    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_results, default=to_serializable))
    serializable["config"] = {
        "model": args.model,
        "lut_dir": args.lut_dir,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "layers": args.layers,
        "op_types": args.op_types,
        "batch_size": args.batch_size,
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
