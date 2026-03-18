"""Oracle Baseline D: Seed Expansion (C1i upper bound).

Given a few "seed" activations (strongest top-K entries), expands candidates using
a PMI-based co-activation neighbor table. Measures how large the candidate set must
be to recover the full top-K.

Also tests the hotset-as-seeds variant (C1h + C1i combination).

Usage:
    python -m experiments.activation_patterns.seed_expand.run \
        --model /root/models/Qwen3-0.6B \
        --lut_dir /root/models/Qwen3-0.6B/lut \
        --dataset /root/fineweb-edu/sample/10BT-tokenized-qwen3-2048/ \
        --num_samples 256 --seq_len 512 \
        --layers 0 7 14 21 27 \
        --output_dir results/activation_patterns/seed_expand/
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


def _build_cooccurrence(topk_idx: np.ndarray, N: int,
                        chunk_size: int = 16384,
                        device: torch.device | None = None) -> np.ndarray:
    """Build co-occurrence matrix from top-K indices using chunked matmul.

    Uses GPU when available for orders-of-magnitude speedup on large N.
    Falls back to CPU BLAS matmul otherwise.

    Returns dense (N, N) float32 numpy matrix where entry (i,j) = number of tokens
    where both i and j appear in top-K. Diagonal is zeroed.
    """
    T, K = topk_idx.shape
    use_gpu = device is not None and device.type == 'cuda'

    if use_gpu:
        co_occur = torch.zeros((N, N), dtype=torch.float32, device=device)
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_idx = torch.from_numpy(topk_idx[start:end]).long().to(device)
            chunk_bin = torch.zeros((end - start, N), dtype=torch.float32, device=device)
            chunk_bin.scatter_(1, chunk_idx, 1.0)
            co_occur.addmm_(chunk_bin.T, chunk_bin)
            del chunk_bin, chunk_idx
        co_occur.fill_diagonal_(0)
        result = co_occur.cpu().numpy()
        del co_occur
        torch.cuda.empty_cache()
        return result
    else:
        co_occur = np.zeros((N, N), dtype=np.float32)
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            c = end - start
            chunk_bin = np.zeros((c, N), dtype=np.float32)
            np.put_along_axis(chunk_bin, topk_idx[start:end], 1.0, axis=1)
            co_occur += chunk_bin.T @ chunk_bin
        np.fill_diagonal(co_occur, 0)
        return co_occur


def _compute_pmi_neighbors(co_occur: np.ndarray, freq: np.ndarray,
                           T: int, max_neighbors: int = 64) -> np.ndarray:
    """Compute PMI and extract top neighbors per basis vector (vectorized).

    PMI(i,j) = log(P(i,j) / (P(i) * P(j)))
    Only positive PMI values are considered.

    Args:
        co_occur: (N, N) dense co-occurrence counts (float32).
        freq: (N,) per-basis-vector frequency counts.
        T: total number of tokens.
        max_neighbors: number of top-PMI neighbors to keep per basis vector.

    Returns:
        neighbor_table: (N, max_neighbors) int32 — top PMI neighbors per basis vector.
            Padded with -1 if fewer than max_neighbors valid neighbors exist.
    """
    N = co_occur.shape[0]
    neighbor_table = np.full((N, max_neighbors), -1, dtype=np.int32)

    # P(i) for all i
    p_i = np.maximum(freq.astype(np.float64) / T, 1e-12)

    # Vectorized PMI computation over the full matrix
    # PMI(i,j) = log(P(i,j) / (P(i) * P(j)))
    # Process in row chunks to limit peak memory for float64
    ROW_CHUNK = min(N, 2048)

    for row_start in range(0, N, ROW_CHUNK):
        row_end = min(row_start + ROW_CHUNK, N)
        chunk = co_occur[row_start:row_end].astype(np.float64)  # (chunk, N)

        # P(i,j) = co_count / T
        p_ij = chunk / T
        # PMI = log(P(i,j)) - log(P(i)) - log(P(j))
        # Use -inf for zero entries to get PMI = -inf (which becomes 0 after clipping)
        log_pij = np.full_like(p_ij, -np.inf)
        nonzero = p_ij > 0
        log_pij[nonzero] = np.log(p_ij[nonzero])

        pmi_chunk = log_pij - np.log(p_i[row_start:row_end])[:, None] - np.log(p_i)[None, :]
        np.maximum(pmi_chunk, 0, out=pmi_chunk)  # positive PMI only

        # Extract top-k neighbors per row
        for local_i in range(row_end - row_start):
            row = pmi_chunk[local_i]
            pos_count = int((row > 0).sum())
            if pos_count == 0:
                continue
            n_keep = min(max_neighbors, pos_count)
            top_k_idx = np.argpartition(row, -n_keep)[-n_keep:]
            sorted_within = top_k_idx[np.argsort(-row[top_k_idx])]
            neighbor_table[row_start + local_i, :n_keep] = sorted_within

    return neighbor_table


def _expand_and_recall_chunked(
    topk_idx: np.ndarray,
    topk_abs_val: np.ndarray,
    seed_indices: np.ndarray,
    neighbor_table: np.ndarray,
    n: int,
    N: int,
    chunk_size: int = 8192,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand seeds via neighbor table and compute recall.

    Uses a sentinel index (N) to handle -1 padding without any Python loops.
    The indicator array has N+1 columns; column N acts as a trash bin that
    absorbs all invalid scatter operations. Only columns 0..N-1 are used
    for recall computation.

    Uses GPU when available for fast scatter/gather operations.

    Args:
        topk_idx: (T, K) — true top-K indices.
        topk_abs_val: (T, K) — true top-K |values|.
        seed_indices: (T, s) — seed indices per token. May contain -1 for padding.
        neighbor_table: (N, max_n) — pre-built PMI neighbor table. -1 = no neighbor.
        n: number of neighbors to use per seed (from neighbor_table[:, :n]).
        N: total latent dimension.
        chunk_size: rows per chunk.
        device: torch device. Uses GPU path if CUDA device.

    Returns:
        recall: (T,) float
        recall_weighted: (T,) float
        candidate_size: (T,) int — number of unique candidates per token
    """
    T, K = topk_idx.shape
    recall = np.empty(T, dtype=np.float32)
    recall_w = np.empty(T, dtype=np.float32)
    cand_size = np.empty(T, dtype=np.int32)
    use_gpu = device is not None and device.type == 'cuda'

    # Prepare: replace -1 with sentinel N in neighbor table slice
    nb = neighbor_table[:, :n].copy()  # (N, n)
    nb[nb < 0] = N
    nb_with_sentinel = np.full((N + 1, n), N, dtype=np.int32)
    nb_with_sentinel[:N] = nb

    if use_gpu:
        nb_gpu = torch.from_numpy(nb_with_sentinel).long().to(device)  # (N+1, n)
        topk_gpu = torch.from_numpy(topk_idx).long().to(device)        # (T, K)
        topk_val_gpu = torch.from_numpy(topk_abs_val).float().to(device)

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            c = end - start

            buf = torch.zeros((c, N + 1), dtype=torch.float32, device=device)

            seeds = torch.from_numpy(seed_indices[start:end].copy()).long().to(device)
            seeds[seeds < 0] = N

            buf.scatter_(1, seeds, 1.0)

            seed_nb = nb_gpu[seeds]       # (c, s, n)
            flat_nb = seed_nb.reshape(c, -1)
            buf.scatter_(1, flat_nb, 1.0)

            valid = buf[:, :N]
            cand_size[start:end] = valid.sum(dim=1).int().cpu().numpy()

            hits = valid.gather(1, topk_gpu[start:end])  # (c, K)
            recall[start:end] = (hits.sum(dim=1) / K).cpu().numpy()

            hit_mass = (topk_val_gpu[start:end] * hits).sum(dim=1)
            total_mass = topk_val_gpu[start:end].sum(dim=1)
            recall_w[start:end] = (hit_mass / total_mass.clamp(min=1e-12)).cpu().numpy()

            del buf, seeds, seed_nb, flat_nb, valid, hits

        del nb_gpu, topk_gpu, topk_val_gpu
        torch.cuda.empty_cache()
    else:
        # CPU fallback with sentinel pattern
        indicator = np.empty((chunk_size, N + 1), dtype=np.uint8)

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            c = end - start
            buf = indicator[:c]
            buf[:] = 0

            seeds = seed_indices[start:end].copy()
            seeds[seeds < 0] = N
            np.put_along_axis(buf, seeds, 1, axis=1)

            seed_neighbors = nb_with_sentinel[seeds]
            flat_nb = seed_neighbors.reshape(c, -1)
            np.put_along_axis(buf, flat_nb, 1, axis=1)

            valid_buf = buf[:, :N]
            cand_size[start:end] = valid_buf.sum(axis=1)

            hits = np.take_along_axis(valid_buf, topk_idx[start:end], axis=1)
            hit_count = hits.sum(axis=1)
            recall[start:end] = hit_count / K

            hit_mass = (topk_abs_val[start:end] * hits).sum(axis=1)
            total_mass = topk_abs_val[start:end].sum(axis=1)
            recall_w[start:end] = hit_mass / np.maximum(total_mass, 1e-12)

    return recall, recall_w, cand_size


def analyze_seed_expand(
    top_indices: torch.Tensor,
    top_values: torch.Tensor,
    K: int,
    N: int,
    seed_counts: list[int] | None = None,
    neighbor_counts: list[int] | None = None,
    hotset_pct: float = 0.20,
    device: torch.device | None = None,
) -> dict:
    """Run seed expansion oracle analysis.

    Args:
        top_indices: (T, 2K) int32 — sorted by activation value desc.
        top_values: (T, 2K) float32.
        K: SAE top-K value.
        N: Total number of latents.
        seed_counts: Seed sizes to sweep. Default [4, 8, 16, 32].
        neighbor_counts: Neighbors per seed. Default [8, 16, 32, 64].
        hotset_pct: Hotset fraction for hotset-as-seeds variant.
        device: torch device. CUDA enables GPU-accelerated co-occurrence and recall.
    """
    if seed_counts is None:
        seed_counts = [4, 8, 16, 32]
    if neighbor_counts is None:
        neighbor_counts = [8, 16, 32, 64]

    topk_idx = top_indices[:, :K].numpy()  # (T, K), sorted by value desc
    topk_val = top_values[:, :K].numpy()
    topk_abs_val = np.abs(topk_val)
    T = topk_idx.shape[0]

    max_n = max(neighbor_counts)

    # Phase 1: Build PMI co-activation neighbor table
    print(f"    Building co-occurrence matrix (T={T}, K={K}, N={N})...")
    co_occur = _build_cooccurrence(topk_idx, N, device=device)

    freq = np.bincount(topk_idx.flatten(), minlength=N).astype(np.float64)

    print(f"    Computing PMI neighbor table (max_n={max_n})...")
    neighbor_table = _compute_pmi_neighbors(co_occur, freq, T, max_neighbors=max_n)

    # Neighbor table stats
    valid_neighbors = (neighbor_table >= 0).sum(axis=1)
    results = {
        "neighbor_table_stats": {
            "mean_valid_neighbors": float(valid_neighbors.mean()),
            "frac_with_any_neighbor": float((valid_neighbors > 0).mean()),
            "mean_freq": float(freq[freq > 0].mean()) if (freq > 0).any() else 0,
        }
    }

    del co_occur  # Free memory

    # Phase 2a: Oracle seeds (top-s strongest activations per token)
    print(f"    Oracle seeds expansion...")
    oracle_results = {}
    for s in seed_counts:
        seeds = topk_idx[:, :s]  # (T, s) — already sorted by value desc
        for n in neighbor_counts:
            print(f"      s={s}, n={n}...")
            recall, recall_w, cand_size = _expand_and_recall_chunked(
                topk_idx, topk_abs_val, seeds, neighbor_table, n, N,
                device=device,
            )
            key = f"s={s},n={n}"
            oracle_results[key] = {
                "seed_count": s,
                "neighbor_count": n,
                "candidate_size_mean": float(cand_size.mean()),
                "candidate_size_P50": float(np.percentile(cand_size, 50)),
                "candidate_size_P90": float(np.percentile(cand_size, 90)),
                "candidate_ratio_mean": float(cand_size.mean() / N),
                "recall_mean": float(recall.mean()),
                "recall_P10": float(np.percentile(recall, 10)),
                "recall_P50": float(np.percentile(recall, 50)),
                "recall_P90": float(np.percentile(recall, 90)),
                "recall_weighted_mean": float(recall_w.mean()),
                "num_tokens": T,
            }

    results["oracle_seeds"] = oracle_results

    # Phase 2b: Hotset-as-seeds variant
    print(f"    Hotset-as-seeds variant (H={hotset_pct*100:.0f}%N)...")
    H_size = max(1, int(N * hotset_pct))
    hot_indices = np.argpartition(freq, -H_size)[-H_size:]
    hot_mask = np.zeros(N, dtype=bool)
    hot_mask[hot_indices] = True

    # For each token, find which top-K entries are in hotset → those are seeds
    is_hot = hot_mask[topk_idx]  # (T, K) bool
    hot_count_per_token = is_hot.sum(axis=1)  # (T,)

    # Build variable-length seed arrays (pad with -1) — vectorized
    max_hot_seeds = int(hot_count_per_token.max())
    hotset_seeds = np.full((T, max_hot_seeds), -1, dtype=np.int32)
    # For each token, pack the hot indices to the left
    # Create a sort key: hot entries get low values (their position), non-hot get K+1
    sort_key = np.where(is_hot, np.arange(K)[None, :], K + 1)
    pack_order = np.argsort(sort_key, axis=1, kind='stable')
    packed_idx = np.take_along_axis(topk_idx, pack_order, axis=1)
    # Copy the hot portion (first max_hot_seeds columns)
    cols_to_copy = min(max_hot_seeds, K)
    hotset_seeds[:, :cols_to_copy] = packed_idx[:, :cols_to_copy]
    # Mask out non-hot entries that leaked in (tokens with fewer hot entries)
    for col in range(cols_to_copy):
        hotset_seeds[hot_count_per_token <= col, col] = -1

    hotset_results = {}
    for n in neighbor_counts:
        print(f"      H={hotset_pct*100:.0f}%, n={n}...")
        recall, recall_w, cand_size = _expand_and_recall_chunked(
            topk_idx, topk_abs_val, hotset_seeds, neighbor_table, n, N,
            device=device,
        )
        key = f"H={hotset_pct*100:.0f}%,n={n}"
        hotset_results[key] = {
            "hotset_pct": hotset_pct,
            "H_size": H_size,
            "neighbor_count": n,
            "actual_seed_count_mean": float(hot_count_per_token.mean()),
            "actual_seed_count_P10": float(np.percentile(hot_count_per_token, 10)),
            "actual_seed_count_P50": float(np.percentile(hot_count_per_token, 50)),
            "candidate_size_mean": float(cand_size.mean()),
            "candidate_ratio_mean": float(cand_size.mean() / N),
            "recall_mean": float(recall.mean()),
            "recall_P10": float(np.percentile(recall, 10)),
            "recall_P50": float(np.percentile(recall, 50)),
            "recall_P90": float(np.percentile(recall, 90)),
            "recall_weighted_mean": float(recall_w.mean()),
            "num_tokens": T,
        }

    results["hotset_seeds"] = hotset_results

    # Phase 3: Cross-validation (train/test split)
    print(f"    Cross-validation stability check...")
    rng = np.random.RandomState(42)
    perm = rng.permutation(T)
    half = T // 2
    train_mask = np.zeros(T, dtype=bool)
    train_mask[perm[:half]] = True
    test_mask = ~train_mask

    train_idx = topk_idx[train_mask]
    test_idx = topk_idx[test_mask]
    test_abs_val = topk_abs_val[test_mask]

    # Build neighbor table from train half only
    co_train = _build_cooccurrence(train_idx, N, device=device)
    freq_train = np.bincount(train_idx.flatten(), minlength=N).astype(np.float64)
    nb_train = _compute_pmi_neighbors(co_train, freq_train, int(train_mask.sum()),
                                      max_neighbors=max_n)
    del co_train

    # Test: use s=16, n=32 as representative config
    cv_results = {}
    s_cv, n_cv = 16, 32
    if s_cv <= K:
        seeds_test = test_idx[:, :s_cv]
        # Recall with full neighbor table
        r_full, _, _ = _expand_and_recall_chunked(
            test_idx, test_abs_val, seeds_test, neighbor_table, n_cv, N,
            device=device,
        )
        # Recall with train-only neighbor table
        r_train, _, _ = _expand_and_recall_chunked(
            test_idx, test_abs_val, seeds_test, nb_train, n_cv, N,
            device=device,
        )
        cv_results[f"s={s_cv},n={n_cv}"] = {
            "recall_full_table": float(r_full.mean()),
            "recall_train_table": float(r_train.mean()),
            "gap": float(abs(r_full.mean() - r_train.mean())),
        }

    results["cross_validation"] = cv_results

    return results


def print_summary(all_results: dict):
    """Print a concise summary table."""
    print("\n" + "=" * 80)
    print("SEED EXPANSION ORACLE BASELINE (C1i) \u2014 SUMMARY")
    print("=" * 80)

    for layer_name, data in all_results.items():
        print(f"\n--- {layer_name} ---")

        # Neighbor table stats
        nts = data.get("neighbor_table_stats", {})
        print(f"  Neighbor table: avg valid neighbors={nts.get('mean_valid_neighbors', 0):.1f}, "
              f"coverage={nts.get('frac_with_any_neighbor', 0)*100:.1f}%")

        # Oracle seeds
        oracle = data.get("oracle_seeds", {})
        if oracle:
            print(f"\n  Oracle seeds:")
            print(f"  {'config':>14s} | {'cand_size':>9s} | {'cand/N':>6s} | "
                  f"{'recall':>7s} | {'P10':>6s} | {'recall_w':>8s}")
            print(f"  {'-'*14}-+-{'-'*9}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}")
            for key in sorted(oracle.keys()):
                d = oracle[key]
                print(f"  {key:>14s} | {d['candidate_size_mean']:>9.0f} | "
                      f"{d['candidate_ratio_mean']:>6.3f} | "
                      f"{d['recall_mean']:>7.4f} | {d['recall_P10']:>6.3f} | "
                      f"{d['recall_weighted_mean']:>8.4f}")

        # Hotset seeds
        hotset = data.get("hotset_seeds", {})
        if hotset:
            print(f"\n  Hotset-as-seeds:")
            for key in sorted(hotset.keys()):
                d = hotset[key]
                print(f"    {key}: seeds={d['actual_seed_count_mean']:.1f}(P10={d['actual_seed_count_P10']:.0f}), "
                      f"cand={d['candidate_size_mean']:.0f}({d['candidate_ratio_mean']*100:.1f}%N), "
                      f"recall={d['recall_mean']:.4f}, P10={d['recall_P10']:.3f}")

        # Cross-validation
        cv = data.get("cross_validation", {})
        if cv:
            print(f"\n  Cross-validation:")
            for key, d in cv.items():
                print(f"    {key}: full={d['recall_full_table']:.4f}, "
                      f"train-only={d['recall_train_table']:.4f}, "
                      f"gap={d['gap']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Oracle Baseline D: Seed Expansion (C1i)"
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
    parser.add_argument("--output_dir", type=str, default="results/activation_patterns/seed_expand/")
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
    print("Encoding + seed expansion analysis (per-hookpoint streaming)...")
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
        result = analyze_seed_expand(
            top_indices=data["top_indices"],
            top_values=data["top_values"],
            K=data["K"],
            N=data["N"],
            device=device,
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
    output_path = output_dir / "seed_expand_results.json"

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
