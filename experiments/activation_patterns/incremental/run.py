"""Oracle Baseline A: Incremental Selection (C1e upper bound).

For each consecutive token pair (t, t+1) within the same sequence, simulates
retaining the previous token's selection and oracle-replacing m positions.

Usage:
    python -m experiments.activation_patterns.incremental.run \
        --model /root/models/Qwen3-0.6B \
        --lut_dir /root/models/Qwen3-0.6B/lut \
        --dataset /root/fineweb-edu/sample/10BT-tokenized-qwen3-2048/ \
        --num_samples 256 --seq_len 512 \
        --layers 0 7 14 21 27 \
        --output_dir results/activation_patterns/incremental/
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


def _build_valid_pair_mask(seq_boundaries: list[int], T: int) -> np.ndarray:
    """Build mask for valid consecutive pairs (not crossing sequence boundaries)."""
    valid = np.ones(T - 1, dtype=bool)
    for i in range(len(seq_boundaries) - 1):
        end = seq_boundaries[i + 1]
        if 0 < end < T:
            valid[end - 1] = False  # pair (end-1, end) crosses boundary
    return valid


def _chunked_overlap(
    retained_idx: np.ndarray,
    sorted_idx_next: np.ndarray,
    sorted_abs_val_next: np.ndarray,
    N: int,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute overlap count, per-position flag, and mass using chunked dense indicators.

    Args:
        retained_idx: (P, K_ret) — retained set indices per pair.
        sorted_idx_next: (P, K) — next token's topK indices sorted by value desc.
        sorted_abs_val_next: (P, K) — corresponding |values|.
        N: Total latent dimension.
        chunk_size: Rows per chunk (trades memory for speed).

    Returns:
        overlap_count: (P,) int — number of overlapping indices per pair.
        is_overlap: (P, K) bool — True if sorted_idx_next[i,j] is in retained_idx[i,:].
        overlap_mass: (P,) float32 — sum of |values| at overlapping positions.
    """
    P, K = sorted_idx_next.shape
    is_overlap = np.empty((P, K), dtype=bool)
    overlap_mass = np.empty(P, dtype=np.float32)

    # Pre-allocate reusable buffers
    indicator = np.empty((chunk_size, N), dtype=np.uint8)

    for s in range(0, P, chunk_size):
        e = min(s + chunk_size, P)
        c = e - s
        buf = indicator[:c]
        buf[:] = 0

        # Build dense indicator for retained set
        np.put_along_axis(buf, retained_idx[s:e], 1, axis=1)

        # Check overlap: is each of next token's indices in the retained set?
        chunk_overlap = np.take_along_axis(buf, sorted_idx_next[s:e], axis=1)
        is_overlap[s:e] = chunk_overlap.view(bool)

        # Overlap mass = sum of next token's values at overlapping positions
        overlap_mass[s:e] = (sorted_abs_val_next[s:e] * chunk_overlap).sum(axis=1)

    overlap_count = is_overlap.sum(axis=1).astype(np.int32)
    return overlap_count, is_overlap, overlap_mass


def _chunked_overlap_union2(
    topk_idx: np.ndarray,
    sorted_idx_next: np.ndarray,
    sorted_abs_val_next: np.ndarray,
    seq_boundaries: list[int],
    N: int,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute overlap for union2 variant: retained = topK(t) ∪ topK(t-1).

    Returns:
        overlap_count: (P,) int
        is_overlap: (P, K) bool
        overlap_mass: (P,) float32
    """
    T = topk_idx.shape[0]
    K = sorted_idx_next.shape[1]
    P = T - 1
    is_overlap = np.empty((P, K), dtype=bool)
    overlap_mass = np.empty(P, dtype=np.float32)

    # Build set of sequence start positions for boundary handling
    seq_starts = set(seq_boundaries[:-1])

    indicator = np.empty((chunk_size, N), dtype=np.uint8)

    for s in range(0, P, chunk_size):
        e = min(s + chunk_size, P)
        c = e - s
        buf = indicator[:c]
        buf[:] = 0

        # Add topK(t) for pairs at positions s..e-1 (t = s..e-1)
        np.put_along_axis(buf, topk_idx[s:e], 1, axis=1)
        # Add topK(t-1) for t > 0
        if s > 0:
            np.put_along_axis(buf, topk_idx[s-1:e-1], 1, axis=1)
        else:
            if c > 1:
                np.put_along_axis(buf[1:], topk_idx[0:e-1], 1, axis=1)

        # Zero out t-1 contribution at sequence starts
        for start in seq_starts:
            local = start - s
            if 0 <= local < c:
                buf[local] = 0
                np.put_along_axis(buf[local:local+1], topk_idx[start:start+1], 1, axis=1)

        # Check overlap
        chunk_overlap = np.take_along_axis(buf, sorted_idx_next[s:e], axis=1)
        is_overlap[s:e] = chunk_overlap.view(bool)
        overlap_mass[s:e] = (sorted_abs_val_next[s:e] * chunk_overlap).sum(axis=1)

    overlap_count = is_overlap.sum(axis=1).astype(np.int32)
    return overlap_count, is_overlap, overlap_mass


def _compute_variant_stats(
    overlap_count: np.ndarray,
    is_overlap: np.ndarray,
    overlap_mass: np.ndarray,
    sorted_abs_val_next: np.ndarray,
    total_mass: np.ndarray,
    K: int,
    m_values: list[int],
    valid_mask: np.ndarray,
) -> dict:
    """Compute recall/mass statistics from pre-computed overlap data."""
    num_pairs = int(valid_mask.sum())
    if num_pairs == 0:
        return {}

    need_replace_count = K - overlap_count
    is_nr = ~is_overlap
    nr_cumcount = np.cumsum(is_nr, axis=1)

    safe_total = np.maximum(total_mass, 1e-12)
    nr_mass_total = total_mass - overlap_mass
    new_mass_ratio = nr_mass_total / safe_total

    # Burstiness prep
    threshold = K // 4
    is_bad = need_replace_count > threshold

    result = {}
    for m in m_values:
        recall = (overlap_count + np.minimum(m, need_replace_count)) / K

        oracle_mask = (nr_cumcount <= m) & is_nr
        oracle_mass = (sorted_abs_val_next * oracle_mask).sum(axis=1)
        recall_w = (overlap_mass + oracle_mass) / safe_total

        r_valid = recall[valid_mask]
        rw_valid = recall_w[valid_mask]
        nm_valid = new_mass_ratio[valid_mask]

        result[f"m={m}"] = {
            "recall_mean": float(r_valid.mean()),
            "recall_P50": float(np.percentile(r_valid, 50)),
            "recall_P90": float(np.percentile(r_valid, 90)),
            "recall_P99": float(np.percentile(r_valid, 99)),
            "recall_weighted_mean": float(rw_valid.mean()),
            "new_mass_ratio_mean": float(nm_valid.mean()),
            "new_mass_ratio_P90": float(np.percentile(nm_valid, 90)),
            "num_pairs": num_pairs,
        }

    # Burstiness stats — vectorized run-length encoding
    # Burstiness is m-independent (depends only on replacement count > K//4),
    # so compute once
    bad_valid = is_bad.copy()
    bad_valid[~valid_mask] = False
    # Find run lengths: transitions from True→False or at boundaries
    padded = np.concatenate([[False], bad_valid, [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    run_lengths = ends - starts

    burst_stats = {}
    for m in m_values:
        if len(run_lengths) > 0:
            burst_stats[f"m={m}"] = {
                "max_run": int(run_lengths.max()),
                "mean_run": float(run_lengths.mean()),
                "num_bursts": int(len(run_lengths)),
            }
        else:
            burst_stats[f"m={m}"] = {"max_run": 0, "mean_run": 0, "num_bursts": 0}
    result["burstiness_K4"] = burst_stats

    return result


def analyze_incremental(
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    seq_boundaries: list[int],
    K: int,
    N: int,
    m_values: list[int] | None = None,
) -> dict:
    """Run incremental selection oracle analysis (vectorized).

    Uses chunked dense indicators and sorted-array tricks for fast set operations,
    avoiding Python-level per-pair loops.

    Args:
        top_indices: (total_tokens, 2K) int32 — sorted by activation value desc.
        top_values: (total_tokens, 2K) float32.
        seq_boundaries: Cumulative token counts per sequence.
        K: SAE top-K value.
        N: Total number of latents.
        m_values: Replacement counts to sweep. Default [0, 4, 8, 16, 32, 64, K].
    """
    if m_values is None:
        m_values = [0, 4, 8, 16, 32, 64, K]

    T = top_indices.shape[0]
    topk_idx = top_indices[:, :K].numpy()   # (T, K)
    topk_val = top_values[:, :K].numpy()
    topk_abs_val = np.abs(topk_val)
    top2k_idx = top_indices.numpy()

    # Sort topK by absolute value descending (for oracle replacement ordering)
    sort_order = np.argsort(-topk_abs_val, axis=1)
    sorted_idx = np.take_along_axis(topk_idx, sort_order, axis=1)
    sorted_abs_val = np.take_along_axis(topk_abs_val, sort_order, axis=1)

    # Valid pair mask
    valid_mask = _build_valid_pair_mask(seq_boundaries, T)

    # Total mass for t+1
    total_mass = topk_abs_val[1:].sum(axis=1)  # (T-1,)

    # Next token's sorted indices/values for oracle computation
    sorted_idx_next = sorted_idx[1:]      # (T-1, K)
    sorted_abs_val_next = sorted_abs_val[1:]  # (T-1, K)

    print(f"    T={T}, K={K}, N={N}, valid_pairs={valid_mask.sum()}")

    variants = {}

    def _run_variant(name, retained_idx=None):
        """Run one variant and return results dict."""
        print(f"    Analyzing {name} variant...")
        if name == "union2":
            oc, io, om = _chunked_overlap_union2(
                topk_idx, sorted_idx_next, sorted_abs_val_next,
                seq_boundaries, N,
            )
        else:
            oc, io, om = _chunked_overlap(
                retained_idx, sorted_idx_next, sorted_abs_val_next, N,
            )
        return oc, _compute_variant_stats(
            oc, io, om, sorted_abs_val_next, total_mass, K, m_values, valid_mask,
        )

    # --- topK variant ---
    oc, topk_result = _run_variant("topK", topk_idx[:-1])
    rc = (K - oc)[valid_mask]
    topk_result["replacement_count"] = {
        "mean": float(rc.mean()),
        "P50": float(np.percentile(rc, 50)),
        "P90": float(np.percentile(rc, 90)),
        "P99": float(np.percentile(rc, 99)),
        "max": int(rc.max()),
    }
    variants["topK"] = topk_result

    # --- topL_1.5 variant ---
    L15 = int(1.5 * K)
    _, variants["topL_1.5"] = _run_variant("topL_1.5", top2k_idx[:-1, :L15])

    # --- topL_2.0 variant ---
    _, variants["topL_2.0"] = _run_variant("topL_2.0", top2k_idx[:-1])

    # --- union2 variant ---
    _, variants["union2"] = _run_variant("union2")

    return variants


def print_summary(all_results: dict):
    """Print a concise summary table."""
    print("\n" + "=" * 80)
    print("INCREMENTAL SELECTION ORACLE BASELINE (C1e) — SUMMARY")
    print("=" * 80)

    for layer_name, variants in all_results.items():
        print(f"\n--- {layer_name} ---")

        # Focus on topK variant
        topk = variants.get("topK", {})

        # Replacement count
        rc = topk.get("replacement_count", {})
        if rc:
            print(f"  Replacement count: mean={rc.get('mean', 0):.1f}, "
                  f"P90={rc.get('P90', 0):.0f}, P99={rc.get('P99', 0):.0f}, "
                  f"max={rc.get('max', 0)}")

        # Recall table
        print(f"  {'m':>6s} | {'recall':>8s} | {'recall_w':>8s} | {'new_mass':>8s}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        for key in sorted(topk.keys()):
            if not key.startswith("m="):
                continue
            d = topk[key]
            print(f"  {key:>6s} | {d['recall_mean']:>8.4f} | "
                  f"{d['recall_weighted_mean']:>8.4f} | "
                  f"{d['new_mass_ratio_mean']:>8.4f}")

        # Cross-variant comparison at fixed budget
        print(f"\n  Cross-variant comparison (total candidate budget = 1.5K):")
        for vname in ["topK", "topL_1.5", "topL_2.0", "union2"]:
            v = variants.get(vname, {})
            # For topK: budget 1.5K means retained=K + m=0.5K ≈ m=64
            # For topL_1.5: retained=1.5K + m=0
            # For topL_2.0: retained=2K + m=0 (over budget, skip)
            # For union2: retained≤2K + m=0 (over budget, skip)
            if vname == "topK":
                key = "m=64"
            elif vname == "topL_1.5":
                key = "m=0"
            elif vname == "topL_2.0":
                key = "m=0"
            elif vname == "union2":
                key = "m=0"

            d = v.get(key, {})
            recall = d.get("recall_mean", 0)
            print(f"    {vname:>12s} ({key}): recall={recall:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Oracle Baseline A: Incremental Selection (C1e)"
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
    parser.add_argument("--output_dir", type=str, default="results/activation_patterns/incremental/")
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
    print("Encoding + incremental analysis (per-hookpoint streaming)...")
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
        result = analyze_incremental(
            top_indices=data["top_indices"],
            top_values=data["top_values"],
            seq_boundaries=data["seq_boundaries"],
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
    output_path = output_dir / "incremental_results.json"

    # Convert numpy types for JSON serialization
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
