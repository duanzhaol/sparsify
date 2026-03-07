#!/usr/bin/env python3
"""一键全量分析 — 运行所有分析脚本并生成综合报告。

用法:
    python full_analysis.py <prof_dir> [--output report.md]
"""

import argparse
import sys
from pathlib import Path
from io import StringIO

# 把同目录的脚本导入
sys.path.insert(0, str(Path(__file__).parent))
from analyze_ops import find_csv, analyze_op_statistic
from analyze_timeline import (
    load_op_summary, detect_steps, analyze_step,
    classify_matmul_by_shape, analyze_interleaving,
)
from analyze_kernel_time import analyze_task_time
from analyze_communication import analyze_communication


def capture_output(func, *args, **kwargs):
    """捕获函数的 stdout 输出为字符串。"""
    old_stdout = sys.stdout
    sys.stdout = buf = StringIO()
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="NPU Profiling 全量分析")
    parser.add_argument("prof_dir", help="msprof 输出的 PROF_xxx 目录")
    parser.add_argument("--output", "-o", help="输出 markdown 文件路径")
    parser.add_argument("--top", type=int, default=20, help="Top N 算子")
    parser.add_argument("--skip-init-seconds", type=float, default=5.0,
                        help="跳过前 N 秒初始化")
    args = parser.parse_args()

    prof_dir = args.prof_dir
    sections = []

    print(f"NPU Profiling 全量分析: {prof_dir}")
    print("=" * 80)

    # 1. 算子统计
    print("\n[1/4] 算子统计分析...")
    try:
        csv_path = find_csv(prof_dir, "op_statistic")
        output = capture_output(analyze_op_statistic, csv_path, args.top)
        print(output)
        sections.append(("算子统计", output))
    except SystemExit:
        print("  跳过 (未找到 op_statistic CSV)")

    # 2. 时间线分析
    print("\n[2/4] 时间线分析...")
    try:
        csv_path = find_csv(prof_dir, "op_summary")
        ops = load_op_summary(csv_path)
        print(f"  加载 {len(ops)} 条算子记录")

        t0 = ops[0]["start_us"] if ops else 0
        skip_us = args.skip_init_seconds * 1e6
        train_ops = [op for op in ops if op["start_us"] - t0 > skip_us]

        buf = StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

        steps = detect_steps(train_ops)
        if len(steps) >= 2:
            print(f"检测到 {len(steps)} 个 step 起始点")
            for i in range(min(len(steps) - 1, 4)):
                step_dur = (steps[i+1] - steps[i]) / 1e3
                print(f"\n{'='*60}")
                print(f"Step {i+1} (wall time: {step_dur:.1f}ms)")
                print(f"{'='*60}")
                analyze_step(ops, steps[i], steps[i+1])

        classify_matmul_by_shape(train_ops if train_ops else ops)
        analyze_interleaving(train_ops if train_ops else ops)

        sys.stdout = old_stdout
        output = buf.getvalue()
        print(output)
        sections.append(("时间线分析", output))
    except (SystemExit, Exception) as e:
        sys.stdout = old_stdout if 'old_stdout' in dir() else sys.stdout
        print(f"  跳过 ({e})")

    # 3. Kernel 级时间
    print("\n[3/4] Kernel 级时间分析...")
    try:
        csv_path = find_csv(prof_dir, "task_time")
        output = capture_output(analyze_task_time, csv_path)
        print(output)
        sections.append(("Kernel 级时间", output))
    except SystemExit:
        print("  跳过 (未找到 task_time CSV)")

    # 4. 通信分析
    print("\n[4/4] 通信分析...")
    try:
        csv_path = find_csv(prof_dir, "communication_statistic")
        output = capture_output(analyze_communication, csv_path)
        print(output)
        sections.append(("通信分析", output))
    except SystemExit:
        print("  跳过 (未找到 communication_statistic CSV)")

    # 生成 markdown 报告
    if args.output and sections:
        with open(args.output, "w") as f:
            f.write(f"# NPU Profiling 分析报告\n\n")
            f.write(f"数据来源: `{prof_dir}`\n\n")
            for title, content in sections:
                f.write(f"## {title}\n\n```\n{content}\n```\n\n")
        print(f"\n报告已保存到: {args.output}")


if __name__ == "__main__":
    main()
