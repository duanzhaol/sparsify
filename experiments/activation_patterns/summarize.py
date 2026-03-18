"""Summarize activation pattern experiment results into CSV.

Usage:
    python -m experiments.activation_patterns.summarize \
        --results_dir results/activation_patterns/ \
        --output results/activation_patterns/summary.csv
"""

import argparse
import csv
import json
from pathlib import Path


def format_config_value(value):
    """Format config values into stable CSV-friendly strings."""
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return value


def extract_incremental_rows(filepath: Path) -> list[dict]:
    """Extract rows from an incremental experiment JSON."""
    with open(filepath) as f:
        data = json.load(f)

    cfg = data.get("config", {})
    common = {
        "source_file": str(filepath),
        "model": cfg.get("model", ""),
        "num_samples": cfg.get("num_samples", ""),
        "seq_len": cfg.get("seq_len", ""),
        "batch_size": cfg.get("batch_size", ""),
        "layers": format_config_value(cfg.get("layers", "")),
        "op_types": format_config_value(cfg.get("op_types", "")),
    }
    rows = []

    for layer_name, variants in data.items():
        if layer_name == "config":
            continue
        topk = variants.get("topK", {})
        rc = topk.get("replacement_count", {})

        for m_key, metrics in topk.items():
            if not m_key.startswith("m="):
                continue

            row = {
                **common,
                "experiment": "incremental",
                "layer": layer_name,
                "variant": "topK",
                "param": m_key,
                "recall_mean": metrics.get("recall_mean", ""),
                "recall_P50": metrics.get("recall_P50", ""),
                "recall_P90": metrics.get("recall_P90", ""),
                "recall_P99": metrics.get("recall_P99", ""),
                "recall_weighted_mean": metrics.get("recall_weighted_mean", ""),
                "new_mass_ratio_mean": metrics.get("new_mass_ratio_mean", ""),
                "num_pairs": metrics.get("num_pairs", ""),
                "replacement_mean": rc.get("mean", ""),
                "replacement_P90": rc.get("P90", ""),
                "replacement_P99": rc.get("P99", ""),
            }
            rows.append(row)

        # Also add cross-variant rows at m=0
        for vname in ["topL_1.5", "topL_2.0", "union2"]:
            v = variants.get(vname, {})
            m0 = v.get("m=0", {})
            if m0:
                rows.append({
                    **common,
                    "experiment": "incremental",
                    "layer": layer_name,
                    "variant": vname,
                    "param": "m=0",
                    "recall_mean": m0.get("recall_mean", ""),
                    "recall_P50": m0.get("recall_P50", ""),
                    "recall_P90": m0.get("recall_P90", ""),
                    "recall_P99": m0.get("recall_P99", ""),
                    "recall_weighted_mean": m0.get("recall_weighted_mean", ""),
                    "new_mass_ratio_mean": m0.get("new_mass_ratio_mean", ""),
                    "num_pairs": m0.get("num_pairs", ""),
                })

    return rows


def extract_hotset_rows(filepath: Path) -> list[dict]:
    """Extract rows from a hotset experiment JSON."""
    with open(filepath) as f:
        data = json.load(f)

    cfg = data.get("config", {})
    common = {
        "source_file": str(filepath),
        "model": cfg.get("model", ""),
        "num_samples": cfg.get("num_samples", ""),
        "seq_len": cfg.get("seq_len", ""),
        "batch_size": cfg.get("batch_size", ""),
        "layers": format_config_value(cfg.get("layers", "")),
        "op_types": format_config_value(cfg.get("op_types", "")),
    }
    rows = []

    for layer_name, layer_data in data.items():
        if layer_name == "config":
            continue
        freq = layer_data.get("frequency_stats", {})

        for key, metrics in layer_data.items():
            if key == "frequency_stats":
                continue

            row = {
                **common,
                "experiment": "hotset",
                "layer": layer_name,
                "variant": "hotset",
                "param": key,
                "H_size": metrics.get("H_size", ""),
                "recall_mean": metrics.get("recall_mean", ""),
                "recall_P10": metrics.get("recall_P10", ""),
                "recall_P25": metrics.get("recall_P25", ""),
                "recall_P50": metrics.get("recall_P50", ""),
                "recall_P90": metrics.get("recall_P90", ""),
                "recall_min": metrics.get("recall_min", ""),
                "recall_weighted_mean": metrics.get("recall_weighted_mean", ""),
                "hot_value_ratio": metrics.get("hot_value_ratio", ""),
                "residual_K": metrics.get("residual_K", ""),
                "gini": freq.get("gini", ""),
                "num_ever_active": freq.get("num_ever_active", ""),
            }
            rows.append(row)

    return rows


def extract_sublibrary_rows(filepath: Path) -> list[dict]:
    """Extract rows from a sublibrary experiment JSON."""
    with open(filepath) as f:
        data = json.load(f)

    cfg = data.get("config", {})
    common = {
        "source_file": str(filepath),
        "model": cfg.get("model", ""),
        "num_samples": cfg.get("num_samples", ""),
        "seq_len": cfg.get("seq_len", ""),
        "batch_size": cfg.get("batch_size", ""),
        "layers": format_config_value(cfg.get("layers", "")),
        "op_types": format_config_value(cfg.get("op_types", "")),
    }
    rows = []

    for layer_name, layer_data in data.items():
        if layer_name == "config":
            continue

        for g_key, g_data in layer_data.items():
            if not g_key.startswith("G="):
                continue

            cs = g_data.get("cluster_sizes", {})
            fs = g_data.get("full_sublibrary_size", {})
            cr = g_data.get("cross_route", {})

            # Full sub-library row
            full = g_data.get("full_sublibrary", {})
            if full:
                rows.append({
                    **common,
                    "experiment": "sublibrary",
                    "layer": layer_name,
                    "variant": g_key,
                    "param": "full",
                    "sublibrary_size_mean": fs.get("mean", ""),
                    "sublibrary_ratio": fs.get("ratio_to_N", ""),
                    "recall_mean": full.get("recall_mean", ""),
                    "recall_P10": full.get("recall_P10", ""),
                    "recall_P50": full.get("recall_P50", ""),
                    "recall_P90": full.get("recall_P90", ""),
                    "recall_weighted_mean": full.get("recall_weighted_mean", ""),
                    "cross_route_recall": cr.get("recall_mean", ""),
                    "cross_route_gap": cr.get("recall_gap", ""),
                    "cluster_size_mean": cs.get("mean", ""),
                    "cluster_size_min": cs.get("min", ""),
                    "cluster_size_max": cs.get("max", ""),
                })

            # Truncated sub-library rows
            for sub_key, sub_data in g_data.items():
                if not sub_key.startswith("N_sub="):
                    continue
                rows.append({
                    **common,
                    "experiment": "sublibrary",
                    "layer": layer_name,
                    "variant": g_key,
                    "param": sub_key,
                    "sublibrary_size_mean": sub_data.get("N_sub", ""),
                    "sublibrary_ratio": sub_data.get("ratio_to_N", ""),
                    "recall_mean": sub_data.get("recall_mean", ""),
                    "recall_P10": sub_data.get("recall_P10", ""),
                    "recall_P50": sub_data.get("recall_P50", ""),
                    "recall_P90": sub_data.get("recall_P90", ""),
                    "recall_weighted_mean": sub_data.get("recall_weighted_mean", ""),
                })

    return rows


def extract_seed_expand_rows(filepath: Path) -> list[dict]:
    """Extract rows from a seed expansion experiment JSON."""
    with open(filepath) as f:
        data = json.load(f)

    cfg = data.get("config", {})
    common = {
        "source_file": str(filepath),
        "model": cfg.get("model", ""),
        "num_samples": cfg.get("num_samples", ""),
        "seq_len": cfg.get("seq_len", ""),
        "batch_size": cfg.get("batch_size", ""),
        "layers": format_config_value(cfg.get("layers", "")),
        "op_types": format_config_value(cfg.get("op_types", "")),
    }
    rows = []

    for layer_name, layer_data in data.items():
        if layer_name == "config":
            continue

        # Oracle seeds
        oracle = layer_data.get("oracle_seeds", {})
        for key, metrics in oracle.items():
            rows.append({
                **common,
                "experiment": "seed_expand",
                "layer": layer_name,
                "variant": "oracle",
                "param": key,
                "candidate_size_mean": metrics.get("candidate_size_mean", ""),
                "candidate_ratio_mean": metrics.get("candidate_ratio_mean", ""),
                "recall_mean": metrics.get("recall_mean", ""),
                "recall_P10": metrics.get("recall_P10", ""),
                "recall_P50": metrics.get("recall_P50", ""),
                "recall_P90": metrics.get("recall_P90", ""),
                "recall_weighted_mean": metrics.get("recall_weighted_mean", ""),
            })

        # Hotset seeds
        hotset = layer_data.get("hotset_seeds", {})
        for key, metrics in hotset.items():
            rows.append({
                **common,
                "experiment": "seed_expand",
                "layer": layer_name,
                "variant": "hotset",
                "param": key,
                "actual_seed_count_mean": metrics.get("actual_seed_count_mean", ""),
                "candidate_size_mean": metrics.get("candidate_size_mean", ""),
                "candidate_ratio_mean": metrics.get("candidate_ratio_mean", ""),
                "recall_mean": metrics.get("recall_mean", ""),
                "recall_P10": metrics.get("recall_P10", ""),
                "recall_P50": metrics.get("recall_P50", ""),
                "recall_P90": metrics.get("recall_P90", ""),
                "recall_weighted_mean": metrics.get("recall_weighted_mean", ""),
            })

        # Cross-validation
        cv = layer_data.get("cross_validation", {})
        for key, metrics in cv.items():
            rows.append({
                **common,
                "experiment": "seed_expand",
                "layer": layer_name,
                "variant": "cross_validation",
                "param": key,
                "recall_mean": metrics.get("recall_full_table", ""),
                "recall_train_table": metrics.get("recall_train_table", ""),
                "cv_gap": metrics.get("gap", ""),
            })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Summarize activation pattern results to CSV"
    )
    parser.add_argument("--results_dir", type=str,
                        default="results/activation_patterns/")
    parser.add_argument("--output", type=str,
                        default="results/activation_patterns/summary.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_rows = []

    # Collect incremental results
    for f in sorted(results_dir.rglob("incremental_results.json")):
        print(f"  Reading {f}")
        all_rows.extend(extract_incremental_rows(f))

    # Collect hotset results
    for f in sorted(results_dir.rglob("hotset_results.json")):
        print(f"  Reading {f}")
        all_rows.extend(extract_hotset_rows(f))

    # Collect sublibrary results
    for f in sorted(results_dir.rglob("sublibrary_results.json")):
        print(f"  Reading {f}")
        all_rows.extend(extract_sublibrary_rows(f))

    # Collect seed expansion results
    for f in sorted(results_dir.rglob("seed_expand_results.json")):
        print(f"  Reading {f}")
        all_rows.extend(extract_seed_expand_rows(f))

    if not all_rows:
        print(f"No result files found in {results_dir}")
        return

    # Write CSV with union of all columns
    all_keys = list(dict.fromkeys(k for row in all_rows for k in row.keys()))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Summary written to {output_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
