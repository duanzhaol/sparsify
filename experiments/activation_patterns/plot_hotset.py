"""Hotset experiment visualization.

Usage:
    python experiments/activation_patterns/plot_hotset.py \
        --results results/activation_patterns/hotset/hotset_results.json \
        --output results/activation_patterns/hotset/

Generates:
    1. recall_by_layer_op.png   — 各层各算子在不同 hotset 大小下的 recall
    2. gini_heatmap.png         — Gini 系数热力图 (层 × 算子)
    3. recall_vs_gini.png       — Gini 与 20% hotset recall 的相关性
    4. weighted_vs_unweighted.png — weighted recall vs unweighted recall 对比
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Op type display config
OP_MAP = {
    "gate_up_proj": ("MLP", "#2196F3"),
    "qkv_proj": ("QKV", "#FF9800"),
    "o_proj": ("O", "#4CAF50"),
}


def classify_hookpoint(name: str) -> tuple[int, str, str]:
    """Extract (layer_idx, op_key, display_name) from hookpoint name."""
    # e.g. "layers.21.self_attn.o_proj" -> (21, "o_proj", "O")
    parts = name.split(".")
    layer_idx = int(parts[1])
    op_key = parts[-1]  # gate_up_proj / qkv_proj / o_proj
    display, color = OP_MAP.get(op_key, (op_key, "#999"))
    return layer_idx, op_key, display, color


def load_data(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    data.pop("config", None)
    return data


def plot_recall_by_layer_op(data: dict, output_dir: Path):
    """Fig 1: Recall curves — one subplot per op type, x=hotset%, lines=layers."""
    ops_order = ["gate_up_proj", "qkv_proj", "o_proj"]
    op_display = {"gate_up_proj": "MLP", "qkv_proj": "QKV", "o_proj": "O"}

    # Group data
    grouped = {}  # op_key -> [(layer_idx, pct_list, recall_list, recall_w_list)]
    for hp_name, hp_data in data.items():
        layer_idx, op_key, _, _ = classify_hookpoint(hp_name)
        pcts, recalls, recalls_w = [], [], []
        for key, metrics in sorted(hp_data.items()):
            if key == "frequency_stats":
                continue
            pcts.append(metrics["pct"] * 100)
            recalls.append(metrics["recall_mean"])
            recalls_w.append(metrics["recall_weighted_mean"])
        grouped.setdefault(op_key, []).append((layer_idx, pcts, recalls, recalls_w))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    layer_colors = {0: "#E91E63", 7: "#9C27B0", 14: "#3F51B5", 21: "#009688", 27: "#FF5722"}

    for ax_idx, op_key in enumerate(ops_order):
        ax = axes[ax_idx]
        entries = sorted(grouped.get(op_key, []), key=lambda x: x[0])
        for layer_idx, pcts, recalls, _ in entries:
            color = layer_colors.get(layer_idx, "#666")
            ax.plot(pcts, recalls, "o-", color=color, label=f"L{layer_idx}",
                    markersize=6, linewidth=2)
        ax.set_title(f"{op_display[op_key]}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Hotset size (% of N)", fontsize=12)
        ax.set_xticks([1, 5, 10, 20])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Recall (unweighted)", fontsize=12)

    axes[0].set_ylim(0, 1.0)
    fig.suptitle("Hotset Recall by Layer and Op Type", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "recall_by_layer_op.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved recall_by_layer_op.png")


def plot_gini_heatmap(data: dict, output_dir: Path):
    """Fig 2: Gini coefficient heatmap (layer x op)."""
    ops_order = ["gate_up_proj", "qkv_proj", "o_proj"]
    op_display = ["MLP", "QKV", "O"]
    layers = sorted(set(classify_hookpoint(k)[0] for k in data))

    gini_matrix = np.zeros((len(layers), len(ops_order)))
    for hp_name, hp_data in data.items():
        layer_idx, op_key, _, _ = classify_hookpoint(hp_name)
        row = layers.index(layer_idx)
        col = ops_order.index(op_key)
        gini_matrix[row, col] = hp_data.get("frequency_stats", {}).get("gini", 0)

    fig, ax = plt.subplots(figsize=(5, 6))
    im = ax.imshow(gini_matrix, cmap="YlOrRd", aspect="auto", vmin=0.3, vmax=0.9)

    ax.set_xticks(range(len(ops_order)))
    ax.set_xticklabels(op_display, fontsize=12)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=12)

    # Annotate cells
    for i in range(len(layers)):
        for j in range(len(ops_order)):
            val = gini_matrix[i, j]
            color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)

    fig.colorbar(im, ax=ax, label="Gini coefficient", shrink=0.8)
    ax.set_title("Gini Coefficient (Basis Usage Inequality)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "gini_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved gini_heatmap.png")


def plot_recall_vs_gini(data: dict, output_dir: Path):
    """Fig 3: Scatter — Gini vs 20% hotset recall, colored by op type."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for hp_name, hp_data in data.items():
        layer_idx, op_key, display, color = classify_hookpoint(hp_name)
        gini = hp_data.get("frequency_stats", {}).get("gini", 0)

        # Find 20% hotset recall
        recall_20 = 0
        for key, metrics in hp_data.items():
            if key == "frequency_stats":
                continue
            if abs(metrics.get("pct", 0) - 0.2) < 0.01:
                recall_20 = metrics["recall_mean"]
                break

        ax.scatter(gini, recall_20, c=color, s=120, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(f"L{layer_idx}", (gini, recall_20), fontsize=9,
                    textcoords="offset points", xytext=(6, 4))

    # Legend by op type
    for op_key in ["gate_up_proj", "qkv_proj", "o_proj"]:
        display, color = OP_MAP[op_key]
        ax.scatter([], [], c=color, s=80, label=display)
    ax.legend(fontsize=11, loc="lower right")

    ax.set_xlabel("Gini coefficient", fontsize=12)
    ax.set_ylabel("Recall @ |H|=20%N", fontsize=12)
    ax.set_title("Higher Gini = More Concentrated Usage = Better Hotset", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.3, 0.95)
    ax.set_ylim(0.35, 1.0)
    fig.tight_layout()
    fig.savefig(output_dir / "recall_vs_gini.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved recall_vs_gini.png")


def plot_weighted_vs_unweighted(data: dict, output_dir: Path):
    """Fig 4: Grouped bar chart — weighted vs unweighted recall at 20% hotset."""
    ops_order = ["gate_up_proj", "qkv_proj", "o_proj"]
    op_display = {"gate_up_proj": "MLP", "qkv_proj": "QKV", "o_proj": "O"}
    layers = sorted(set(classify_hookpoint(k)[0] for k in data))

    # Collect data
    recall_uw = {}  # (layer, op) -> unweighted
    recall_w = {}   # (layer, op) -> weighted
    for hp_name, hp_data in data.items():
        layer_idx, op_key, _, _ = classify_hookpoint(hp_name)
        for key, metrics in hp_data.items():
            if key == "frequency_stats":
                continue
            if abs(metrics.get("pct", 0) - 0.2) < 0.01:
                recall_uw[(layer_idx, op_key)] = metrics["recall_mean"]
                recall_w[(layer_idx, op_key)] = metrics["recall_weighted_mean"]

    fig, ax = plt.subplots(figsize=(14, 6))
    n_groups = len(layers) * len(ops_order)
    x = np.arange(n_groups)
    bar_width = 0.35

    uw_vals, w_vals, labels = [], [], []
    for layer_idx in layers:
        for op_key in ops_order:
            uw_vals.append(recall_uw.get((layer_idx, op_key), 0))
            w_vals.append(recall_w.get((layer_idx, op_key), 0))
            labels.append(f"L{layer_idx}\n{op_display[op_key]}")

    bars1 = ax.bar(x - bar_width / 2, uw_vals, bar_width, label="Unweighted", color="#64B5F6")
    bars2 = ax.bar(x + bar_width / 2, w_vals, bar_width, label="Weighted", color="#FF8A65")

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Recall @ |H|=20%N", fontsize=12)
    ax.set_title("Weighted vs Unweighted Recall (20% Hotset)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Add layer separators
    for i in range(1, len(layers)):
        ax.axvline(x=i * len(ops_order) - 0.5, color="gray", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_dir / "weighted_vs_unweighted.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved weighted_vs_unweighted.png")


def main():
    parser = argparse.ArgumentParser(description="Plot hotset experiment results")
    parser.add_argument("--results", type=str,
                        default="results/activation_patterns/hotset/hotset_results.json")
    parser.add_argument("--output", type=str,
                        default="results/activation_patterns/hotset/")
    args = parser.parse_args()

    data = load_data(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating hotset plots...")
    plot_recall_by_layer_op(data, output_dir)
    plot_gini_heatmap(data, output_dir)
    plot_recall_vs_gini(data, output_dir)
    plot_weighted_vs_unweighted(data, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
