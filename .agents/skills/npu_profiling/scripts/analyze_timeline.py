#!/usr/bin/env python3
"""时间线分析 — 从 op_summary CSV 分析 step 边界、阶段划分和执行模式。

用法:
    python analyze_timeline.py <prof_dir> [--skip-init-seconds N]

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


def load_op_summary(csv_path: Path):
    """加载 op_summary，返回按时间排序的算子列表。"""
    ops = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row["Task Start Time(us)"].strip())
                duration = float(row["Task Duration(us)"])
            except (ValueError, KeyError):
                continue
            ops.append({
                "op_name": row.get("Op Name", ""),
                "op_type": row.get("OP Type", ""),
                "core_type": row.get("Core Type", ""),
                "task_type": row.get("Task Type", ""),
                "start_us": start,
                "duration_us": duration,
                "shapes": row.get("Input Shapes", ""),
            })
    ops.sort(key=lambda x: x["start_us"])
    return ops


def detect_steps(ops, gap_threshold_ms=50):
    """通过大时间间隔检测 step 边界。

    策略: 找到 FlashAttentionScore 或 allreduce 算子集群之间的大间隔。
    """
    # 找出所有 FlashAttention 算子作为 step 标记
    fa_ops = [op for op in ops if "FlashAttention" in op["op_type"]]
    if not fa_ops:
        # 回退: 用 allreduce 集群
        fa_ops = [op for op in ops if "allreduce" in op["op_type"].lower() or "allReduce" in op["op_type"]]

    if len(fa_ops) < 2:
        print("WARNING: 无法通过 FlashAttention/allreduce 检测 step 边界")
        return []

    # 找 FA 算子间的大间隔
    gaps = []
    for i in range(1, len(fa_ops)):
        gap = fa_ops[i]["start_us"] - (fa_ops[i-1]["start_us"] + fa_ops[i-1]["duration_us"])
        if gap > gap_threshold_ms * 1000:  # 转 us
            gaps.append((i, gap, fa_ops[i]["start_us"]))

    # 用间隔划分 steps
    step_boundaries = [fa_ops[0]["start_us"]]
    for idx, gap, ts in gaps:
        step_boundaries.append(ts)

    return step_boundaries


def analyze_step(ops, step_start, step_end):
    """分析单个 step 内的算子分布。"""
    step_ops = [op for op in ops if step_start <= op["start_us"] < step_end]

    wall_time_ms = (step_end - step_start) / 1e3

    # 按类别分类
    categories = defaultdict(lambda: {"time_us": 0, "count": 0, "ops": defaultdict(float)})

    for op in step_ops:
        op_type = op["op_type"]
        dur = op["duration_us"]

        # 分类逻辑
        if "FlashAttention" in op_type:
            cat = "LLM Attention"
        elif "allreduce" in op_type.lower() or "allReduce" in op_type:
            cat = "Communication"
        elif "broadcast" in op_type.lower():
            cat = "Communication"
        elif "EmbeddingBag" in op_type:
            cat = "SAE Decode (forward)"
        elif "TopK" in op_type:
            cat = "SAE TopK"
        elif op_type in ("IndexPutV2", "IndexPut"):
            cat = "SAE Backward (scatter)"
        elif "ScatterElements" in op_type:
            cat = "SAE Backward (scatter)"
        elif op_type in ("MatMulV3", "MatMulV2", "BatchMatMul"):
            # 需要按 shape 区分 LLM vs SAE — 简化处理: 看 shape 中的数字
            cat = "MatMul (需按shape分类)"
        elif op_type == "Cast":
            cat = "数据搬运 (Cast)"
        elif op_type == "Transpose":
            cat = "数据搬运 (Transpose)"
        elif op["core_type"] == "AI_CPU":
            cat = "AI_CPU Fallback"
        elif op_type in ("Lerp", "LerpV2"):
            cat = "Optimizer"
        else:
            cat = "其他计算"

        categories[cat]["time_us"] += dur
        categories[cat]["count"] += 1
        categories[cat]["ops"][op_type] += dur

    print(f"\n  Step wall time: {wall_time_ms:.1f}ms, 算子数: {len(step_ops)}")
    print(f"  {'类别':25s} | {'耗时(ms)':>10} | {'占比':>7} | {'次数':>6} | 主要算子")
    print("  " + "-" * 90)

    total_compute = sum(c["time_us"] for c in categories.values())
    for cat, stats in sorted(categories.items(), key=lambda x: -x[1]["time_us"]):
        ratio = stats["time_us"] / total_compute * 100 if total_compute > 0 else 0
        top_ops = sorted(stats["ops"].items(), key=lambda x: -x[1])[:3]
        top_ops_str = ", ".join(f"{n}({t/1e3:.1f}ms)" for n, t in top_ops)
        print(
            f"  {cat:25s} | {stats['time_us']/1e3:10.1f} | {ratio:6.1f}% | "
            f"{stats['count']:6d} | {top_ops_str}"
        )

    idle_ms = wall_time_ms - total_compute / 1e3
    if idle_ms > 0:
        idle_ratio = idle_ms / wall_time_ms * 100
        print(f"  {'NPU Idle (估算)':25s} | {idle_ms:10.1f} | {idle_ratio:6.1f}% |        | host开销+kernel间隙")


def classify_matmul_by_shape(ops):
    """将 MatMul 算子按 input shape 分类，帮助区分 LLM vs SAE。"""
    matmul_shapes = defaultdict(lambda: {"count": 0, "time_us": 0})

    for op in ops:
        if op["op_type"] not in ("MatMulV3", "MatMulV2"):
            continue
        shapes = op["shapes"].replace('"', '').strip()
        matmul_shapes[shapes]["count"] += 1
        matmul_shapes[shapes]["time_us"] += op["duration_us"]

    if not matmul_shapes:
        return

    print(f"\n--- MatMul 按 Shape 分类 ---")
    print(f"{'Shape':50s} | {'次数':>6} | {'总耗时(ms)':>10} | {'平均(us)':>10}")
    print("-" * 90)
    for shape, stats in sorted(matmul_shapes.items(), key=lambda x: -x[1]["time_us"]):
        if stats["time_us"] < 100:  # 忽略 <0.1ms 的
            continue
        avg = stats["time_us"] / stats["count"] if stats["count"] > 0 else 0
        print(
            f"{shape:50s} | {stats['count']:6d} | "
            f"{stats['time_us']/1e3:10.1f} | {avg:10.1f}"
        )


def analyze_interleaving(ops):
    """分析 LLM 和 SAE 算子的交织模式。"""
    # 提取关键算子序列
    key_ops = []
    for op in ops:
        if op["op_type"] in ("FlashAttentionScore", "EmbeddingBag", "TopKV2",
                              "IndexPutV2", "IndexPut", "ScatterElementsV2"):
            key_ops.append(op)
        elif "allreduce" in op["op_type"].lower():
            key_ops.append(op)

    if not key_ops:
        return

    print(f"\n--- 执行模式分析 (关键算子序列) ---")

    # 检测 FA 聚集 → TopK → EmbeddingBag → IndexPut 模式
    pattern_groups = []
    current_group = {"fa": 0, "topk": 0, "emb": 0, "scatter": 0}
    last_type = None

    for op in key_ops:
        op_type = op["op_type"]
        if "FlashAttention" in op_type:
            if last_type and last_type not in ("FlashAttentionScore",):
                if any(v > 0 for v in current_group.values()):
                    pattern_groups.append(current_group.copy())
                current_group = {"fa": 0, "topk": 0, "emb": 0, "scatter": 0}
            current_group["fa"] += 1
            last_type = "FlashAttentionScore"
        elif "TopK" in op_type:
            current_group["topk"] += 1
            last_type = "TopK"
        elif "EmbeddingBag" in op_type:
            current_group["emb"] += 1
            last_type = "EmbeddingBag"
        elif op_type in ("IndexPutV2", "IndexPut", "ScatterElementsV2"):
            current_group["scatter"] += 1
            last_type = "scatter"

    if any(v > 0 for v in current_group.values()):
        pattern_groups.append(current_group)

    if pattern_groups:
        print(f"检测到 {len(pattern_groups)} 个执行组:")
        for i, g in enumerate(pattern_groups[:8]):
            print(f"  组 {i}: FA×{g['fa']} → TopK×{g['topk']} → EmbBag×{g['emb']} → Scatter×{g['scatter']}")
        if len(pattern_groups) > 8:
            print(f"  ... 共 {len(pattern_groups)} 组")

        # 判断模式
        fa_per_group = [g["fa"] for g in pattern_groups if g["fa"] > 0]
        if fa_per_group:
            avg_fa = sum(fa_per_group) / len(fa_per_group)
            print(f"\n  模式: LLM 每 ~{avg_fa:.0f} 层 FA 后执行一轮 SAE 前向+反向 (交织执行)")


def main():
    parser = argparse.ArgumentParser(description="时间线分析")
    parser.add_argument("prof_dir", help="msprof 输出的 PROF_xxx 目录")
    parser.add_argument("--skip-init-seconds", type=float, default=5.0,
                        help="跳过前 N 秒的初始化阶段")
    args = parser.parse_args()

    csv_path = find_csv(args.prof_dir, "op_summary")
    print(f"加载 {csv_path}...")
    ops = load_op_summary(csv_path)
    print(f"共 {len(ops)} 条算子记录")

    if not ops:
        print("ERROR: 无数据", file=sys.stderr)
        sys.exit(1)

    t0 = ops[0]["start_us"]
    t_end = max(op["start_us"] + op["duration_us"] for op in ops)
    total_s = (t_end - t0) / 1e6
    print(f"时间线: {total_s:.3f}s")

    # 跳过初始化
    skip_us = args.skip_init_seconds * 1e6
    train_ops = [op for op in ops if op["start_us"] - t0 > skip_us]
    if train_ops:
        print(f"跳过前 {args.skip_init_seconds}s 初始化，剩余 {len(train_ops)} 条算子")

    # Step 检测
    steps = detect_steps(train_ops)
    if len(steps) >= 2:
        print(f"\n检测到 {len(steps)} 个 step 起始点")
        for i in range(min(len(steps) - 1, 4)):
            step_dur = (steps[i+1] - steps[i]) / 1e3
            print(f"\n{'='*60}")
            print(f"Step {i+1} (wall time: {step_dur:.1f}ms)")
            print(f"{'='*60}")
            analyze_step(ops, steps[i], steps[i+1])
    else:
        print("\n无法自动检测 step 边界，分析全部训练算子")
        if train_ops:
            t_start = train_ops[0]["start_us"]
            t_end2 = max(op["start_us"] + op["duration_us"] for op in train_ops)
            analyze_step(ops, t_start, t_end2)

    # MatMul shape 分类
    classify_matmul_by_shape(train_ops if train_ops else ops)

    # 交织模式分析
    analyze_interleaving(train_ops if train_ops else ops)


if __name__ == "__main__":
    main()
