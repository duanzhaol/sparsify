#!/usr/bin/env python3
"""Kernel 级时间分析 — 从 task_time CSV 分析 NPU 利用率和空闲占比。

用法:
    python analyze_kernel_time.py <prof_dir>

其中 <prof_dir> 是 msprof 输出的 PROF_xxx 目录。
"""

import argparse
import csv
import sys
from collections import defaultdict
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


def analyze_task_time(csv_path: Path):
    """分析 task_time，按 kernel_type 统计时间分布和 NPU 利用率。"""
    type_stats = defaultdict(lambda: {"count": 0, "time_us": 0})
    total_events = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_events += 1
            ktype = row.get("kernel_type", "UNKNOWN")
            try:
                dur = float(row["task_time(us)"])
            except (ValueError, KeyError):
                continue
            type_stats[ktype]["count"] += 1
            type_stats[ktype]["time_us"] += dur

    total_time = sum(s["time_us"] for s in type_stats.values())

    print(f"\n{'='*80}")
    print(f"Kernel 级时间分析 — {csv_path.name}")
    print(f"{'='*80}")
    print(f"总事件数: {total_events}")
    print(f"总时间: {total_time/1e6:.3f}s")

    # 分类: 计算 vs 等待 vs 其他
    compute_types = {"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV", "AI_CPU", "MIX_AICPU"}
    wait_types = {"NOTIFY_WAIT_SQE", "EVENT_WAIT", "FFTS_PLUS_WAIT"}
    comm_types = {"SDMA", "HCCS"}

    compute_time = sum(s["time_us"] for k, s in type_stats.items() if k in compute_types)
    wait_time = sum(s["time_us"] for k, s in type_stats.items() if k in wait_types)
    comm_time = sum(s["time_us"] for k, s in type_stats.items() if k in comm_types)
    other_time = total_time - compute_time - wait_time - comm_time

    print(f"\n--- 高层分类 ---")
    print(f"  NPU 计算:  {compute_time/1e6:.3f}s ({compute_time/total_time*100:.1f}%)")
    print(f"  NPU 等待:  {wait_time/1e6:.3f}s ({wait_time/total_time*100:.1f}%)")
    print(f"  通信 (DMA): {comm_time/1e6:.3f}s ({comm_time/total_time*100:.1f}%)")
    print(f"  其他:      {other_time/1e6:.3f}s ({other_time/total_time*100:.1f}%)")
    print(f"\n  NPU 计算利用率: {compute_time/total_time*100:.1f}%")

    # 详细按 kernel_type
    print(f"\n--- 按 Kernel Type 详细 ---")
    print(f"{'Kernel Type':25s} | {'事件数':>8} | {'总耗时(ms)':>12} | {'占比':>7} | 说明")
    print("-" * 85)

    type_notes = {
        "AI_CORE": "Cube 矩阵计算单元 (MatMul)",
        "AI_VECTOR_CORE": "向量计算单元 (EmbeddingBag, scatter 等)",
        "MIX_AIC": "混合模式 AIC (FlashAttention)",
        "MIX_AIV": "混合模式 AIV (TopK, ReduceMean)",
        "AI_CPU": "CPU 回退算子",
        "NOTIFY_WAIT_SQE": "NPU 空闲等待 (host 端延迟)",
        "EVENT_WAIT": "事件等待 (同步)",
        "SDMA": "片上 DMA 传输",
        "HCCS": "片间高速通信",
        "FFTS_PLUS_WAIT": "调度等待",
    }

    for ktype, stats in sorted(type_stats.items(), key=lambda x: -x[1]["time_us"]):
        ratio = stats["time_us"] / total_time * 100
        note = type_notes.get(ktype, "")
        print(
            f"{ktype:25s} | {stats['count']:8d} | "
            f"{stats['time_us']/1e3:12.1f} | {ratio:6.1f}% | {note}"
        )

    # AI_CORE vs AI_VECTOR_CORE 比例（关键指标）
    aic = type_stats.get("AI_CORE", {"time_us": 0})["time_us"]
    aiv = type_stats.get("AI_VECTOR_CORE", {"time_us": 0})["time_us"]
    if aic + aiv > 0:
        print(f"\n--- Cube vs Vector 比例 ---")
        print(f"  AI_CORE (Cube):        {aic/1e3:.1f}ms ({aic/(aic+aiv)*100:.1f}%)")
        print(f"  AI_VECTOR_CORE (Vector): {aiv/1e3:.1f}ms ({aiv/(aic+aiv)*100:.1f}%)")
        print(f"  → Cube 利用率偏低意味着 MatMul 类计算占比少，scatter/embedding 类计算占比大")


def main():
    parser = argparse.ArgumentParser(description="Kernel 级时间分析")
    parser.add_argument("prof_dir", help="msprof 输出的 PROF_xxx 目录")
    args = parser.parse_args()

    csv_path = find_csv(args.prof_dir, "task_time")
    analyze_task_time(csv_path)


if __name__ == "__main__":
    main()
