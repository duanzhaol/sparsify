#!/usr/bin/env python3
"""Plot round vs overall deployment cost from the minimal AutoResearch CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("research/history/rounds_0001_0195_minimal.csv")
DEFAULT_OUTPUT = Path("research/history/rounds_0001_0195_overall_cost.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot AutoResearch round-wise overall cost, defined as "
            "total_cost_ratio + exceed_alpha_0_50."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--title",
        default="AutoResearch Round vs Overall Cost",
        help="Plot title.",
    )
    return parser.parse_args()


def load_points(csv_path: Path) -> tuple[list[int], list[float], list[int]]:
    rounds: list[int] = []
    overall_costs: list[float] = []
    missing_rounds: list[int] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"round", "total_cost_ratio", "exceed_alpha_0_50"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            round_id = int(row["round"])
            total_cost_ratio = row.get("total_cost_ratio", "").strip()
            exceed_alpha = row.get("exceed_alpha_0_50", "").strip()
            if not total_cost_ratio or not exceed_alpha:
                missing_rounds.append(round_id)
                continue

            overall_cost = float(total_cost_ratio) + float(exceed_alpha)
            rounds.append(round_id)
            overall_costs.append(overall_cost)

    return rounds, overall_costs, missing_rounds


def main() -> int:
    args = parse_args()
    rounds, overall_costs, missing_rounds = load_points(args.input)
    if not rounds:
        raise ValueError("No valid rows found in input CSV.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(rounds, overall_costs, color="#1f77b4", linewidth=1.8, marker="o", markersize=3)
    plt.xlabel("Round")
    plt.ylabel("Overall Cost = total_cost_ratio + exceed_alpha_0_50")
    plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.35)

    if missing_rounds:
        for round_id in missing_rounds:
            plt.axvline(round_id, color="#d62728", alpha=0.08, linewidth=1)

    plt.tight_layout()
    plt.savefig(args.output, dpi=180)

    print(f"Saved plot to: {args.output}")
    print(f"Plotted {len(rounds)} rounds.")
    if missing_rounds:
        print(f"Skipped rounds with missing data: {missing_rounds}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
