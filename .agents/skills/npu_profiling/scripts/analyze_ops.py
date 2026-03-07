#!/usr/bin/env python3
"""算子级统计分析 — 从 op_statistic CSV 生成 Top-N 算子排名和按核心类型的汇总。

用法:
    python analyze_ops.py <prof_dir> [--top N]

其中 <prof_dir> 是 msprof 输出的 PROF_xxx 目录。
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def find_csv(prof_dir: str, prefix: str) -> Path:
    """在 mindstudio_profiler_output/ 下查找以 prefix 开头的 CSV。"""
    output_dir = Path(prof_dir) / "mindstudio_profiler_output"
    if not output_dir.exists():
        # 可能用户直接传了 mindstudio_profiler_output 路径
        output_dir = Path(prof_dir)
    matches = sorted(output_dir.glob(f"{prefix}*.csv"))
    if not matches:
        print(f"ERROR: 未找到 {prefix}*.csv in {output_dir}", file=sys.stderr)
        sys.exit(1)
    return matches[-1]  # 取最新的


def analyze_op_statistic(csv_path: Path, top_n: int = 20):
    """解析 op_statistic CSV，输出 Top-N 算子和按核心类型汇总。"""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # 按总耗时排序
    rows.sort(key=lambda r: float(r["Total Time(us)"]), reverse=True)
    total_time = sum(float(r["Total Time(us)"]) for r in rows)

    print(f"\n{'='*80}")
    print(f"算子统计分析 — {csv_path.name}")
    print(f"{'='*80}")
    print(f"总算子类型数: {len(rows)}")
    print(f"总 NPU 时间: {total_time/1e6:.3f}s")

    # Top-N 算子
    print(f"\n--- Top {top_n} 算子 (按 NPU 耗时) ---")
    print(f"{'排名':>4} | {'算子':30s} | {'核心类型':20s} | {'次数':>8} | {'总耗时(ms)':>12} | {'平均(us)':>10} | {'占比':>7}")
    print("-" * 105)
    for i, row in enumerate(rows[:top_n], 1):
        print(
            f"{i:4d} | {row['OP Type']:30s} | {row['Core Type']:20s} | "
            f"{int(row['Count']):8d} | {float(row['Total Time(us)'])/1e3:12.1f} | "
            f"{float(row['Avg Time(us)']):10.1f} | {float(row['Ratio(%)']):6.1f}%"
        )

    # 按核心类型汇总
    core_stats = defaultdict(lambda: {"count": 0, "time": 0.0, "ops": 0})
    for row in rows:
        core = row["Core Type"]
        core_stats[core]["count"] += int(row["Count"])
        core_stats[core]["time"] += float(row["Total Time(us)"])
        core_stats[core]["ops"] += 1

    print(f"\n--- 按核心类型汇总 ---")
    print(f"{'核心类型':20s} | {'算子种类':>8} | {'调用次数':>10} | {'总耗时(ms)':>12} | {'占比':>7}")
    print("-" * 70)
    for core, stats in sorted(core_stats.items(), key=lambda x: -x[1]["time"]):
        ratio = stats["time"] / total_time * 100
        print(
            f"{core:20s} | {stats['ops']:8d} | {stats['count']:10d} | "
            f"{stats['time']/1e3:12.1f} | {ratio:6.1f}%"
        )

    # 识别潜在问题
    print(f"\n--- 潜在问题检测 ---")
    issues = []
    for row in rows:
        if row["Core Type"] == "AI_CPU":
            issues.append(
                f"  [CPU Fallback] {row['OP Type']}: {int(row['Count'])} 次, "
                f"共 {float(row['Total Time(us)'])/1e3:.1f}ms"
            )
    if issues:
        print("发现 AI_CPU 回退算子:")
        for issue in issues:
            print(issue)
    else:
        print("未发现 AI_CPU 回退算子")

    # 高频低效算子检测
    for row in rows[:top_n]:
        if row["Core Type"] == "AI_VECTOR_CORE" and float(row["Ratio(%)"]) > 5:
            print(
                f"  [VECTOR_CORE 热点] {row['OP Type']}: "
                f"{float(row['Ratio(%)']):5.1f}% — 考虑是否可用 AI_CORE Cube 替代"
            )


def main():
    parser = argparse.ArgumentParser(description="算子级统计分析")
    parser.add_argument("prof_dir", help="msprof 输出的 PROF_xxx 目录")
    parser.add_argument("--top", type=int, default=20, help="显示 Top N 算子")
    args = parser.parse_args()

    csv_path = find_csv(args.prof_dir, "op_statistic")
    analyze_op_statistic(csv_path, args.top)


if __name__ == "__main__":
    main()
