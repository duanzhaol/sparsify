#!/usr/bin/env python3
"""通信分析 — 从 communication_statistic CSV 分析 DDP 通信开销。

用法:
    python analyze_communication.py <prof_dir>
"""

import argparse
import csv
import sys
from pathlib import Path


def find_csv(prof_dir: str, prefix: str) -> Path:
    output_dir = Path(prof_dir) / "mindstudio_profiler_output"
    if not output_dir.exists():
        output_dir = Path(prof_dir)
    matches = sorted(output_dir.glob(f"{prefix}*.csv"))
    if not matches:
        print(f"ERROR: 未找到 {prefix}*.csv in {output_dir}", file=sys.stderr)
        sys.exit(1)
    return matches[-1]


def analyze_communication(csv_path: Path):
    """分析通信统计。"""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("无通信数据")
        return

    total_time = sum(float(r["Total Time(us)"]) for r in rows)

    print(f"\n{'='*80}")
    print(f"通信分析 — {csv_path.name}")
    print(f"{'='*80}")
    print(f"总通信时间: {total_time/1e6:.3f}s")

    print(f"\n{'通信类型':20s} | {'次数':>8} | {'总耗时(ms)':>12} | {'平均(ms)':>10} | {'最大(ms)':>10} | {'占比':>7}")
    print("-" * 80)

    for row in sorted(rows, key=lambda r: -float(r["Total Time(us)"])):
        count = int(row["Count"])
        total = float(row["Total Time(us)"])
        avg = float(row["Avg Time(us)"])
        max_t = float(row["Max Time(us)"])
        ratio = total / total_time * 100 if total_time > 0 else 0
        op_type = row["OP Type"]
        print(
            f"{op_type:20s} | {count:8d} | {total/1e3:12.1f} | "
            f"{avg/1e3:10.3f} | {max_t/1e3:10.1f} | {ratio:6.1f}%"
        )

    # 检测异常
    print(f"\n--- 潜在问题 ---")
    for row in rows:
        max_t = float(row["Max Time(us)"])
        avg_t = float(row["Avg Time(us)"])
        if max_t > avg_t * 10 and max_t > 10000:  # 最大值远大于平均值
            print(
                f"  [长尾] {row['OP Type']}: 最大 {max_t/1e3:.1f}ms vs "
                f"平均 {avg_t/1e3:.3f}ms — 可能是同步阻塞或 logging 触发"
            )


def main():
    parser = argparse.ArgumentParser(description="通信分析")
    parser.add_argument("prof_dir", help="msprof 输出的 PROF_xxx 目录")
    args = parser.parse_args()

    csv_path = find_csv(args.prof_dir, "communication_statistic")
    analyze_communication(csv_path)


if __name__ == "__main__":
    main()
