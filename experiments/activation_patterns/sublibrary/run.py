"""Oracle Baseline C: Conditional Sub-library (C1c / A2b upper bound).

Clusters token activation patterns, builds per-cluster sub-libraries, and measures
oracle routing recall — how much of each token's true top-K is covered by its
cluster's sub-library.

Usage:
    python -m experiments.activation_patterns.sublibrary.run \
        --model /root/models/Qwen3-0.6B \
        --lut_dir /root/models/Qwen3-0.6B/lut \
        --dataset /root/fineweb-edu/sample/10BT-tokenized-qwen3-2048/ \
        --num_samples 256 --seq_len 512 \
        --layers 0 7 14 21 27 \
        --output_dir results/activation_patterns/sublibrary/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common.data import collect_raw_activations, encode_activations, load_dataset_auto
from experiments.common.sae_utils import get_layer_hookpoints
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_sparse_features(topk_idx: np.ndarray, topk_val: np.ndarray, N: int) -> sp.csr_matrix:
    """Build value-weighted sparse feature matrix (T, N).

    Each row has K nonzeros at the top-K index positions, with |activation value| as weight.
    """
    T, K = topk_idx.shape
    rows = np.repeat(np.arange(T), K)
    cols = topk_idx.flatten()
    vals = np.abs(topk_val).flatten()
    return sp.csr_matrix((vals, (rows, cols)), shape=(T, N), dtype=np.float32)


def _cluster_tokens(features: sp.csr_matrix, G: int, proj_dim: int = 128,
                    random_state: int = 42) -> np.ndarray:
    """Cluster tokens using random projection + MiniBatchKMeans.

    Returns:
        labels: (T,) int — cluster assignment per token.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.random_projection import SparseRandomProjection

    projector = SparseRandomProjection(n_components=proj_dim, random_state=random_state)
    projected = projector.fit_transform(features)  # (T, proj_dim) dense

    kmeans = MiniBatchKMeans(
        n_clusters=G, batch_size=min(4096, projected.shape[0]),
        random_state=random_state, n_init=3,
    )
    labels = kmeans.fit_predict(projected)
    return labels


def _build_sublibrary(topk_idx: np.ndarray, cluster_mask: np.ndarray,
                      N: int, N_sub: int | None = None) -> np.ndarray:
    """Build sub-library for a cluster: union of top-K indices, optionally truncated.

    Args:
        topk_idx: (T, K) — all tokens' top-K indices.
        cluster_mask: (T,) bool — which tokens belong to this cluster.
        N: total latent count.
        N_sub: if given, truncate to the N_sub most frequent indices in cluster.

    Returns:
        sub_indices: 1D array of basis vector indices in the sub-library.
    """
    cluster_indices = topk_idx[cluster_mask].flatten()

    if N_sub is None:
        # Full sub-library = union of all indices
        return np.unique(cluster_indices)

    # Count frequency within cluster
    freq = np.bincount(cluster_indices, minlength=N)
    # Take top N_sub most frequent
    top_n = min(N_sub, int((freq > 0).sum()))
    sub = np.argpartition(freq, -top_n)[-top_n:]
    return sub[freq[sub] > 0]  # filter out zeros if N_sub > num_active


def _compute_recall_chunked(topk_idx: np.ndarray, topk_abs_val: np.ndarray,
                            sub_indices: np.ndarray, N: int,
                            chunk_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-token recall against a sub-library using chunked dense indicators.

    Returns:
        recall: (T,) float — fraction of top-K in sub-library.
        recall_weighted: (T,) float — value-weighted fraction.
    """
    T, K = topk_idx.shape
    recall = np.empty(T, dtype=np.float32)
    recall_w = np.empty(T, dtype=np.float32)

    # Build sub-library indicator (dense, reusable)
    sub_indicator = np.zeros(N, dtype=np.uint8)
    sub_indicator[sub_indices] = 1

    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        # Check which of each token's top-K indices are in the sub-library
        hits = sub_indicator[topk_idx[s:e]]  # (chunk, K) uint8
        hit_count = hits.sum(axis=1)
        recall[s:e] = hit_count / K

        hit_mass = (topk_abs_val[s:e] * hits).sum(axis=1)
        total_mass = topk_abs_val[s:e].sum(axis=1)
        recall_w[s:e] = hit_mass / np.maximum(total_mass, 1e-12)

    return recall, recall_w


def analyze_sublibrary(
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    K: int,
    N: int,
    num_clusters_list: list[int] | None = None,
    proj_dim: int = 128,
) -> dict:
    """Run conditional sub-library oracle analysis.

    Args:
        top_indices: (T, 2K) int32 — sorted by activation value desc.
        top_values: (T, 2K) float32.
        K: SAE top-K value.
        N: Total number of latents.
        num_clusters_list: Cluster counts to try. Default [8, 16, 32, 64].
        proj_dim: Random projection dimensionality for clustering.
    """
    if num_clusters_list is None:
        num_clusters_list = [8, 16, 32, 64]

    topk_idx = top_indices[:, :K].numpy()
    topk_val = top_values[:, :K].numpy()
    topk_abs_val = np.abs(topk_val)
    T = topk_idx.shape[0]

    print(f"    Building sparse features (T={T}, K={K}, N={N})...")
    features = _build_sparse_features(topk_idx, topk_val, N)

    results = {}

    for G in num_clusters_list:
        if G >= T:
            print(f"    Skipping G={G} (>= T={T})")
            continue

        print(f"    Clustering with G={G}...")
        labels = _cluster_tokens(features, G, proj_dim=proj_dim)

        # Cluster size stats
        cluster_sizes = np.bincount(labels, minlength=G)

        # N_sub values to test (deduplicated, sorted)
        n_sub_candidates = sorted(set([
            N // G,
            N // max(G // 2, 1),
            N // 4,
            N // 2,
        ]))
        # Filter out values that are >= N (would be trivial full library)
        n_sub_candidates = [ns for ns in n_sub_candidates if ns < N]

        g_result = {
            "num_clusters": G,
            "cluster_sizes": {
                "mean": float(cluster_sizes.mean()),
                "min": int(cluster_sizes.min()),
                "max": int(cluster_sizes.max()),
                "std": float(cluster_sizes.std()),
            },
        }

        # Build full sub-libraries and measure sizes
        full_sub_sizes = []
        full_subs = {}
        for g in range(G):
            mask = labels == g
            full_sub = _build_sublibrary(topk_idx, mask, N, N_sub=None)
            full_subs[g] = full_sub
            full_sub_sizes.append(len(full_sub))

        full_sub_sizes = np.array(full_sub_sizes)
        g_result["full_sublibrary_size"] = {
            "mean": float(full_sub_sizes.mean()),
            "min": int(full_sub_sizes.min()),
            "max": int(full_sub_sizes.max()),
            "ratio_to_N": float(full_sub_sizes.mean() / N),
        }

        # Oracle recall with full sub-library
        print(f"      Full sub-library recall...")
        all_recall = np.empty(T, dtype=np.float32)
        all_recall_w = np.empty(T, dtype=np.float32)
        for g in range(G):
            mask = labels == g
            if mask.sum() == 0:
                continue
            r, rw = _compute_recall_chunked(
                topk_idx[mask], topk_abs_val[mask], full_subs[g], N,
            )
            all_recall[mask] = r
            all_recall_w[mask] = rw

        g_result["full_sublibrary"] = {
            "recall_mean": float(all_recall.mean()),
            "recall_P10": float(np.percentile(all_recall, 10)),
            "recall_P50": float(np.percentile(all_recall, 50)),
            "recall_P90": float(np.percentile(all_recall, 90)),
            "recall_weighted_mean": float(all_recall_w.mean()),
            "num_tokens": T,
        }

        # Truncated sub-libraries
        for N_sub in n_sub_candidates:
            print(f"      N_sub={N_sub} recall...")
            trunc_recall = np.empty(T, dtype=np.float32)
            trunc_recall_w = np.empty(T, dtype=np.float32)
            for g in range(G):
                mask = labels == g
                if mask.sum() == 0:
                    continue
                trunc_sub = _build_sublibrary(topk_idx, mask, N, N_sub=N_sub)
                r, rw = _compute_recall_chunked(
                    topk_idx[mask], topk_abs_val[mask], trunc_sub, N,
                )
                trunc_recall[mask] = r
                trunc_recall_w[mask] = rw

            g_result[f"N_sub={N_sub}"] = {
                "N_sub": N_sub,
                "ratio_to_N": round(N_sub / N, 4),
                "recall_mean": float(trunc_recall.mean()),
                "recall_P10": float(np.percentile(trunc_recall, 10)),
                "recall_P50": float(np.percentile(trunc_recall, 50)),
                "recall_P90": float(np.percentile(trunc_recall, 90)),
                "recall_weighted_mean": float(trunc_recall_w.mean()),
            }

        # Cross-route analysis: wrong cluster recall
        print(f"      Cross-route analysis...")
        rng = np.random.RandomState(42)
        cross_recalls = []
        num_cross_samples = min(3, G - 1)

        for g_correct in range(G):
            mask = labels == g_correct
            n_tokens = mask.sum()
            if n_tokens == 0:
                continue

            # Sample wrong clusters
            wrong_clusters = [gc for gc in range(G) if gc != g_correct]
            if len(wrong_clusters) > num_cross_samples:
                wrong_clusters = rng.choice(wrong_clusters, num_cross_samples, replace=False)

            for g_wrong in wrong_clusters:
                r, _ = _compute_recall_chunked(
                    topk_idx[mask], topk_abs_val[mask], full_subs[g_wrong], N,
                )
                cross_recalls.extend(r.tolist())

        cross_recalls = np.array(cross_recalls) if cross_recalls else np.array([0.0])
        oracle_full_recall = float(all_recall.mean())
        cross_mean = float(cross_recalls.mean())

        g_result["cross_route"] = {
            "recall_mean": cross_mean,
            "recall_P10": float(np.percentile(cross_recalls, 10)),
            "recall_P50": float(np.percentile(cross_recalls, 50)),
            "recall_gap": round(oracle_full_recall - cross_mean, 4),
        }

        results[f"G={G}"] = g_result

    return results


def print_summary(all_results: dict):
    """Print a concise summary table."""
    print("\n" + "=" * 80)
    print("CONDITIONAL SUB-LIBRARY ORACLE BASELINE (C1c/A2b) \u2014 SUMMARY")
    print("=" * 80)

    for layer_name, data in all_results.items():
        print(f"\n--- {layer_name} ---")

        for key in sorted(data.keys()):
            if not key.startswith("G="):
                continue
            g_data = data[key]
            cs = g_data["cluster_sizes"]
            fs = g_data["full_sublibrary_size"]
            fr = g_data["full_sublibrary"]
            cr = g_data["cross_route"]

            print(f"\n  {key}: clusters size {cs['mean']:.0f} (min={cs['min']}, max={cs['max']})")
            print(f"    Full sub-lib size: {fs['mean']:.0f}/{fs['mean']/8192*100:.1f}%N, "
                  f"recall={fr['recall_mean']:.4f}, recall_w={fr['recall_weighted_mean']:.4f}")
            print(f"    Cross-route: recall={cr['recall_mean']:.4f}, gap={cr['recall_gap']:.4f}")

            # Truncated results
            for sub_key in sorted(g_data.keys()):
                if not sub_key.startswith("N_sub="):
                    continue
                sd = g_data[sub_key]
                print(f"    {sub_key} ({sd['ratio_to_N']*100:.1f}%N): "
                      f"recall={sd['recall_mean']:.4f}, P10={sd['recall_P10']:.3f}, "
                      f"recall_w={sd['recall_weighted_mean']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Oracle Baseline C: Conditional Sub-library (C1c/A2b)"
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
    parser.add_argument("--output_dir", type=str, default="results/activation_patterns/sublibrary/")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Step 1: Collect raw activations
    print(f"\nStep 1: Collecting raw activations for layers {args.layers}...")
    raw_data = collect_raw_activations(
        model=model, dataset=dataset, hookpoints=hookpoints,
        num_samples=args.num_samples, seq_len=args.seq_len,
        batch_size=args.batch_size, device=device,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2+3: Encode -> analyze -> free, one hookpoint at a time
    print("\n" + "=" * 80)
    print("Encoding + sublibrary analysis (per-hookpoint streaming)...")
    print("=" * 80)

    all_results = {}
    for hookpoint in hookpoints:
        lut_layer = hookpoint_to_lut[hookpoint]

        single_raw = {hookpoint: raw_data[hookpoint]}
        single_lut = {hookpoint: lut_layer}
        encoded = encode_activations(
            raw_data=single_raw, lut_dir=args.lut_dir,
            hookpoint_to_lut=single_lut, device=device,
        )

        del raw_data[hookpoint], single_raw

        data = encoded[lut_layer]
        print(f"\n  Analyzing {lut_layer}...")
        result = analyze_sublibrary(
            top_indices=data["top_indices"],
            top_values=data["top_values"],
            K=data["K"],
            N=data["N"],
        )
        all_results[lut_layer] = result

        del encoded, data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print summary
    print_summary(all_results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sublibrary_results.json"

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
