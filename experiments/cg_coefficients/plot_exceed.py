"""Plot exceed ratio (online computation ratio) vs tau for inner-product vs CG.

Usage:
    python -m experiments.cg_coefficients.plot_exceed \
        --csv experiments/cg_coefficients/results/summary.csv \
        --output experiments/cg_coefficients/results/exceed_ratio.png
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_layer_and_op(hookpoint: str) -> tuple[int, str]:
    parts = hookpoint.split(".")
    layer = int(parts[2])
    if "mlp" in hookpoint:
        op = "mlp"
    elif "o_proj" in hookpoint:
        op = "o_proj"
    elif "q_proj" in hookpoint:
        op = "qkv"
    else:
        op = ".".join(parts[3:])
    return layer, op


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default="experiments/cg_coefficients/results/summary.csv")
    parser.add_argument("--output", type=str,
                        default="experiments/cg_coefficients/results/exceed_ratio.png")
    parser.add_argument("--tau_min", type=float, default=0.1)
    parser.add_argument("--tau_max", type=float, default=1.0)
    parser.add_argument("--title", type=str, default=None,
                        help="Custom plot title (auto-detected if not set)")
    args = parser.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    tau_values = sorted(set(
        float(col.split("_t")[1])
        for col in rows[0].keys()
        if col.startswith("p_inner_t")
    ))
    tau_values = [t for t in tau_values if args.tau_min <= t <= args.tau_max]

    # Build: {(layer, op): {tau: (p_inner, p_cg)}}
    data = {}
    for row in rows:
        layer, op = parse_layer_and_op(row["hookpoint"])
        tau_map = {}
        for tau in tau_values:
            p_inner = float(row.get(f"p_inner_t{tau}", 0) or 0)
            p_cg = float(row.get(f"p_cg_t{tau}", 0) or 0)
            tau_map[tau] = (p_inner, p_cg)
        data[(layer, op)] = tau_map

    layers = sorted(set(k[0] for k in data))
    all_ops = set(k[1] for k in data)
    op_order = [op for op in ["mlp", "qkv", "o_proj"] if op in all_ops]
    op_labels = {"mlp": "MLP (gate_up)", "qkv": "Attn (qkv)", "o_proj": "Attn (o)"}

    n_rows = len(layers)
    n_cols = len(op_order)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    for r, layer in enumerate(layers):
        for c, op in enumerate(op_order):
            ax = axes[r, c]
            key = (layer, op)

            if key not in data:
                ax.set_visible(False)
                continue

            tau_map = data[key]
            inner_vals = [tau_map[t][0] for t in tau_values]
            cg_vals = [tau_map[t][1] for t in tau_values]

            ax.plot(tau_values, inner_vals, color="tab:blue", linewidth=2,
                    marker="o", markersize=3, label="Inner product")
            ax.plot(tau_values, cg_vals, color="tab:orange", linewidth=2,
                    marker="s", markersize=3, label="CG (10 iter)")
            ax.fill_between(tau_values, inner_vals, cg_vals,
                            alpha=0.15, color="tab:green")

            ax.grid(True, alpha=0.3)
            ax.set_xlim(args.tau_min, args.tau_max)
            ax.set_ylim(0, 0.85)

            # Column title on first row
            if r == 0:
                ax.set_title(op_labels.get(op, op), fontsize=13, fontweight="bold")

            # Row label on first column
            if c == 0:
                ax.set_ylabel(f"Layer {layer}\np (online ratio)", fontsize=11)

            # X label on last row
            if r == n_rows - 1:
                ax.set_xlabel("τ", fontsize=11)

            # Legend only on top-left
            if r == 0 and c == 0:
                ax.legend(fontsize=9, loc="upper right")

            # Annotate delta at tau=0.5
            if 0.5 in tau_values:
                idx = tau_values.index(0.5)
                pi = inner_vals[idx]
                pc = cg_vals[idx]
                if pi > 0:
                    pct = (pi - pc) / pi * 100
                    mid_y = (pi + pc) / 2
                    ax.text(0.52, mid_y, f"-{pct:.1f}%",
                            fontsize=8, color="tab:green", fontweight="bold",
                            va="center")

    title = args.title or (
        "Online Computation Ratio: Inner Product vs CG Coefficients\n"
        f"{n_rows} layers, {n_cols} ops, 4096 samples per layer"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
