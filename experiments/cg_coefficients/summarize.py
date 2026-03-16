"""Summarize all CG experiment JSON results into a single CSV.

Usage:
    python -m experiments.cg_coefficients.summarize \
        --results_dir experiments/cg_coefficients/results \
        --output experiments/cg_coefficients/results/summary.csv
"""

import argparse
import csv
import json
from pathlib import Path


def extract_row(filepath: Path) -> dict | None:
    with open(filepath) as f:
        data = json.load(f)

    cfg = data.get("config", {})
    if not cfg:
        return None

    # Parse layer and proj type from hookpoint or checkpoint label
    hookpoint = cfg.get("hookpoint", "")
    checkpoint = cfg.get("checkpoint", "")

    row = {
        "file": filepath.name,
        "hookpoint": hookpoint,
        "checkpoint": checkpoint,
        "num_samples": cfg.get("num_samples", ""),
        "cg_max_iter": cfg.get("cg_max_iter", ""),
        "elbow_threshold": cfg.get("elbow_threshold", ""),
        "condition_number": data.get("condition_number_median", ""),
        "MSE_inner": data.get("MSE_inner", ""),
        "MSE_cg": data.get("MSE_cg_encoder_init", ""),
        "MSE_exact": data.get("MSE_exact", ""),
        "MSE_red_cg%": data.get("MSE_reduction_cg_encoder_init_%", ""),
        "MSE_red_exact%": data.get("MSE_reduction_exact_%", ""),
        "FVU_inner": data.get("FVU_inner", ""),
        "FVU_cg": data.get("FVU_cg_encoder_init", ""),
        "FVU_exact": data.get("FVU_exact", ""),
        "cg_iters_enc": data.get("cg_iters_encoder_init", ""),
        "cg_iters_zero": data.get("cg_iters_zero_init", ""),
    }

    # Add exceed ratios for all tau values found
    tau_values = sorted(set(
        float(k.split("tau=")[1])
        for k in data.keys()
        if "exceed_ratio_inner_tau=" in k
    ))

    for tau in tau_values:
        row[f"p_inner_t{tau}"] = data.get(f"exceed_ratio_inner_tau={tau}", "")
        row[f"p_cg_t{tau}"] = data.get(f"exceed_ratio_cg_tau={tau}", "")
        row[f"p_exact_t{tau}"] = data.get(f"exceed_ratio_exact_tau={tau}", "")
        row[f"p_red_cg_t{tau}%"] = data.get(f"p_reduction_cg_tau={tau}_%", "")
        row[f"p_red_exact_t{tau}%"] = data.get(f"p_reduction_exact_tau={tau}_%", "")

    return row


def main():
    parser = argparse.ArgumentParser(description="Summarize CG experiment results to CSV")
    parser.add_argument("--results_dir", type=str,
                        default="experiments/cg_coefficients/results")
    parser.add_argument("--output", type=str,
                        default="experiments/cg_coefficients/results/summary.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    json_files = sorted(results_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return

    rows = []
    for f in json_files:
        row = extract_row(f)
        if row:
            rows.append(row)

    if not rows:
        print("No valid results found")
        return

    # Collect all column names (union of all rows)
    all_keys = list(dict.fromkeys(k for row in rows for k in row.keys()))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Summary written to {output_path} ({len(rows)} experiments)")


if __name__ == "__main__":
    main()
