"""Seed expansion experiment visualization.

Usage:
    python experiments/activation_patterns/plot_seed_expand.py \
        --results results/activation_patterns/seed_expand/seed_expand_results.json \
        --output results/activation_patterns/seed_expand/

Generates:
    1. oracle_recall_heatmap.png        — Oracle seeds: recall heatmap (s x n), per op type
    2. hotset_expand_recall.png         — Hotset+expansion: recall vs neighbor count, by op/layer
    3. candidate_vs_recall.png          — Candidate ratio vs recall (oracle + hotset), unified curve
    4. recall_weighted_vs_unweighted.png — Weighted vs unweighted recall comparison
    5. cross_validation.png             — Cross-validation gap bar chart
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Op type display config (same as plot_hotset.py)
OP_MAP = {
    "gate_up_proj": ("MLP", "#2196F3"),
    "qkv_proj": ("QKV", "#FF9800"),
    "o_proj": ("O", "#4CAF50"),
}

LAYER_COLORS = {0: "#E91E63", 7: "#9C27B0", 14: "#3F51B5", 21: "#009688", 27: "#FF5722"}
LAYER_MARKERS = {0: "o", 7: "s", 14: "D", 21: "^", 27: "v"}


def classify_hookpoint(name: str) -> tuple[int, str, str, str]:
    """Extract (layer_idx, op_key, display_name, color) from hookpoint name."""
    parts = name.split(".")
    layer_idx = int(parts[1])
    op_key = parts[-1]
    display, color = OP_MAP.get(op_key, (op_key, "#999"))
    return layer_idx, op_key, display, color


def load_data(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    data.pop("config", None)
    return data


def parse_config_key(key: str) -> dict:
    """Parse 's=4,n=8' or 'H=20%,n=8' into dict."""
    result = {}
    for part in key.split(","):
        k, v = part.split("=")
        v = v.rstrip("%")
        result[k] = int(v) if v.isdigit() else float(v)
    return result


def plot_oracle_recall_heatmap(data: dict, output_dir: Path):
    """Fig 1: Oracle seeds recall heatmap (s x n grid), one subplot per (layer, op)."""
    ops_order = ["gate_up_proj", "qkv_proj", "o_proj"]
    op_display = {"gate_up_proj": "MLP", "qkv_proj": "QKV", "o_proj": "O"}

    # Group by (layer, op)
    entries = []
    for hp_name, hp_data in data.items():
        layer_idx, op_key, _, _ = classify_hookpoint(hp_name)
        entries.append((layer_idx, op_key, hp_data))
    entries.sort(key=lambda x: (x[0], ops_order.index(x[1])))

    layers = sorted(set(e[0] for e in entries))
    n_layers = len(layers)
    n_ops = len(ops_order)

    fig, axes = plt.subplots(n_layers, n_ops, figsize=(4.5 * n_ops, 3.5 * n_layers),
                             squeeze=False)

    seed_counts = [4, 8, 16, 32]
    neighbor_counts = [8, 16, 32, 64]

    for layer_idx, op_key, hp_data in entries:
        row = layers.index(layer_idx)
        col = ops_order.index(op_key)
        ax = axes[row, col]

        oracle = hp_data.get("oracle_seeds", {})
        matrix = np.zeros((len(seed_counts), len(neighbor_counts)))

        for key, metrics in oracle.items():
            cfg = parse_config_key(key)
            s, n = int(cfg["s"]), int(cfg["n"])
            if s in seed_counts and n in neighbor_counts:
                si = seed_counts.index(s)
                ni = neighbor_counts.index(n)
                matrix[si, ni] = metrics["recall_mean"]

        im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=0.6)
        ax.set_xticks(range(len(neighbor_counts)))
        ax.set_xticklabels([str(n) for n in neighbor_counts], fontsize=10)
        ax.set_yticks(range(len(seed_counts)))
        ax.set_yticklabels([str(s) for s in seed_counts], fontsize=10)

        for i in range(len(seed_counts)):
            for j in range(len(neighbor_counts)):
                val = matrix[i, j]
                color = "white" if val > 0.35 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

        ax.set_title(f"L{layer_idx} {op_display[op_key]}", fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Seeds (s)", fontsize=11)
        if row == n_layers - 1:
            ax.set_xlabel("Neighbors (n)", fontsize=11)

    fig.suptitle("Oracle Seeds Recall (s x n)", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_recall_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved oracle_recall_heatmap.png")


def plot_hotset_expand_recall(data: dict, output_dir: Path):
    """Fig 2: Hotset+expansion recall vs neighbor count, one subplot per op type."""
    ops_order = ["gate_up_proj", "qkv_proj", "o_proj"]
    op_display = {"gate_up_proj": "MLP", "qkv_proj": "QKV", "o_proj": "O"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax_idx, op_key in enumerate(ops_order):
        ax = axes[ax_idx]
        plotted = False

        for hp_name, hp_data in sorted(data.items()):
            layer_idx, hk_op, _, _ = classify_hookpoint(hp_name)
            if hk_op != op_key:
                continue

            hotset = hp_data.get("hotset_seeds", {})
            if not hotset:
                continue

            ns, recalls, recalls_w, recalls_p10 = [], [], [], []
            for key in sorted(hotset.keys(), key=lambda k: parse_config_key(k).get("n", 0)):
                cfg = parse_config_key(key)
                ns.append(cfg["n"])
                recalls.append(hotset[key]["recall_mean"])
                recalls_w.append(hotset[key]["recall_weighted_mean"])
                recalls_p10.append(hotset[key]["recall_P10"])

            color = LAYER_COLORS.get(layer_idx, "#666")
            marker = LAYER_MARKERS.get(layer_idx, "o")
            ax.plot(ns, recalls, f"{marker}-", color=color, label=f"L{layer_idx} recall",
                    markersize=7, linewidth=2)
            ax.fill_between(ns, recalls_p10, recalls,
                            color=color, alpha=0.1)
            ax.plot(ns, recalls_w, f"{marker}--", color=color, label=f"L{layer_idx} recall_w",
                    markersize=5, linewidth=1.5, alpha=0.7)
            plotted = True

        ax.set_title(f"{op_display[op_key]}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Neighbors per seed (n)", fontsize=12)
        ax.set_xticks([8, 16, 32, 64])
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=8, loc="lower right")
        if ax_idx == 0:
            ax.set_ylabel("Recall", fontsize=12)

    axes[0].set_ylim(0, 1.0)
    fig.suptitle("Hotset-as-Seeds + PMI Expansion (H=20%N)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "hotset_expand_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved hotset_expand_recall.png")


def plot_candidate_vs_recall(data: dict, output_dir: Path):
    """Fig 3: Unified candidate_ratio vs recall curve (oracle + hotset), per op type."""
    ops_order = ["gate_up_proj", "qkv_proj", "o_proj"]
    op_display = {"gate_up_proj": "MLP", "qkv_proj": "QKV", "o_proj": "O"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax_idx, op_key in enumerate(ops_order):
        ax = axes[ax_idx]

        for hp_name, hp_data in sorted(data.items()):
            layer_idx, hk_op, _, _ = classify_hookpoint(hp_name)
            if hk_op != op_key:
                continue

            color = LAYER_COLORS.get(layer_idx, "#666")

            # Oracle seeds points
            oracle = hp_data.get("oracle_seeds", {})
            o_cand, o_recall = [], []
            for key, metrics in oracle.items():
                o_cand.append(metrics["candidate_ratio_mean"] * 100)
                o_recall.append(metrics["recall_mean"])
            ax.scatter(o_cand, o_recall, color=color, marker="x", s=30, alpha=0.5, zorder=3)

            # Hotset points (larger, connected)
            hotset = hp_data.get("hotset_seeds", {})
            h_cand, h_recall = [], []
            for key in sorted(hotset.keys(), key=lambda k: parse_config_key(k).get("n", 0)):
                h_cand.append(hotset[key]["candidate_ratio_mean"] * 100)
                h_recall.append(hotset[key]["recall_mean"])
            if h_cand:
                ax.plot(h_cand, h_recall, "o-", color=color, markersize=8, linewidth=2.5,
                        label=f"L{layer_idx} hotset+exp", zorder=5)

        # Reference lines
        ax.axhline(y=0.9, color="red", linestyle=":", alpha=0.5, label="90% recall")
        ax.axhline(y=0.8, color="orange", linestyle=":", alpha=0.4)

        ax.set_title(f"{op_display[op_key]}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Candidate set (% of N)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        if ax_idx == 0:
            ax.set_ylabel("Recall", fontsize=12)

    axes[0].set_ylim(0, 1.0)
    axes[0].set_xlim(0, 30)
    fig.suptitle("Candidate Size vs Recall (x = oracle seeds, o = hotset+expansion)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "candidate_vs_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved candidate_vs_recall.png")


def plot_weighted_vs_unweighted(data: dict, output_dir: Path):
    """Fig 4: Grouped bar — weighted vs unweighted recall for hotset+expansion (n=32)."""
    ops_order = ["gate_up_proj", "qkv_proj", "o_proj"]
    op_display = {"gate_up_proj": "MLP", "qkv_proj": "QKV", "o_proj": "O"}

    # Collect data for n=32
    entries = []
    for hp_name, hp_data in sorted(data.items()):
        layer_idx, op_key, _, _ = classify_hookpoint(hp_name)
        hotset = hp_data.get("hotset_seeds", {})
        for key, metrics in hotset.items():
            cfg = parse_config_key(key)
            if int(cfg.get("n", 0)) == 32:
                entries.append({
                    "layer": layer_idx,
                    "op": op_key,
                    "recall": metrics["recall_mean"],
                    "recall_w": metrics["recall_weighted_mean"],
                    "cand_ratio": metrics["candidate_ratio_mean"],
                })

    entries.sort(key=lambda e: (e["layer"], ops_order.index(e["op"])))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(entries))
    bar_width = 0.35

    recalls = [e["recall"] for e in entries]
    recalls_w = [e["recall_w"] for e in entries]
    labels = [f"L{e['layer']}\n{op_display[e['op']]}" for e in entries]

    bars1 = ax.bar(x - bar_width / 2, recalls, bar_width,
                   label="Unweighted", color="#64B5F6")
    bars2 = ax.bar(x + bar_width / 2, recalls_w, bar_width,
                   label="Weighted", color="#FF8A65")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)

    # Annotate candidate ratio below bars
    for i, e in enumerate(entries):
        ax.text(x[i], -0.04, f"{e['cand_ratio']*100:.1f}%N",
                ha="center", va="top", fontsize=7, color="#666")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Recall (H=20%, n=32)", fontsize=12)
    ax.set_title("Hotset+Expansion: Weighted vs Unweighted Recall (n=32)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(-0.08, 1.05)
    ax.axhline(y=0.9, color="red", linestyle=":", alpha=0.4)

    # Layer separators
    layers = sorted(set(e["layer"] for e in entries))
    for i in range(1, len(layers)):
        sep_x = i * len(ops_order) - 0.5
        if sep_x < len(entries):
            ax.axvline(x=sep_x, color="gray", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_dir / "recall_weighted_vs_unweighted.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved recall_weighted_vs_unweighted.png")


def plot_cross_validation(data: dict, output_dir: Path):
    """Fig 5: Cross-validation gap bar chart."""
    ops_order = ["gate_up_proj", "qkv_proj", "o_proj"]
    op_display = {"gate_up_proj": "MLP", "qkv_proj": "QKV", "o_proj": "O"}

    entries = []
    for hp_name, hp_data in sorted(data.items()):
        layer_idx, op_key, _, _ = classify_hookpoint(hp_name)
        cv = hp_data.get("cross_validation", {})
        for key, metrics in cv.items():
            entries.append({
                "layer": layer_idx,
                "op": op_key,
                "recall_full": metrics["recall_full_table"],
                "recall_train": metrics["recall_train_table"],
                "gap": metrics["gap"],
            })

    entries.sort(key=lambda e: (e["layer"], ops_order.index(e["op"])))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    x = np.arange(len(entries))
    bar_width = 0.35
    labels = [f"L{e['layer']}\n{op_display[e['op']]}" for e in entries]

    # Top: full vs train recall
    full_vals = [e["recall_full"] for e in entries]
    train_vals = [e["recall_train"] for e in entries]

    ax1.bar(x - bar_width / 2, full_vals, bar_width,
            label="Full table", color="#64B5F6")
    ax1.bar(x + bar_width / 2, train_vals, bar_width,
            label="Train-only table", color="#81C784")

    for i in range(len(entries)):
        ax1.text(x[i], max(full_vals[i], train_vals[i]) + 0.005,
                 f"{full_vals[i]:.4f}", ha="center", va="bottom", fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Recall (s=16, n=32)", fontsize=11)
    ax1.set_title("Cross-Validation: Full Table vs Train-Only Table",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # Bottom: gap
    gaps = [e["gap"] for e in entries]
    colors = [OP_MAP.get(e["op"], ("", "#999"))[1] for e in entries]
    ax2.bar(x, gaps, 0.6, color=colors, edgecolor="white", linewidth=1)
    for i, g in enumerate(gaps):
        ax2.text(x[i], g + 0.00005, f"{g:.4f}", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Gap", fontsize=11)
    ax2.set_title("Recall Gap (|full - train|)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "cross_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cross_validation.png")


def main():
    parser = argparse.ArgumentParser(description="Plot seed expansion experiment results")
    parser.add_argument("--results", type=str,
                        default="results/activation_patterns/seed_expand/seed_expand_results.json")
    parser.add_argument("--output", type=str,
                        default="results/activation_patterns/seed_expand/")
    args = parser.parse_args()

    data = load_data(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating seed expansion plots...")
    plot_oracle_recall_heatmap(data, output_dir)
    plot_hotset_expand_recall(data, output_dir)
    plot_candidate_vs_recall(data, output_dir)
    plot_weighted_vs_unweighted(data, output_dir)
    plot_cross_validation(data, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
