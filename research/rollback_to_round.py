#!/usr/bin/env python3
"""将 AutoResearch 系统状态回滚到指定 round 结束、下一轮即将开始的状态。

用法:
    python -m research.rollback_to_round 112          # 回滚到 round 112 结束
    python -m research.rollback_to_round 112 --dry-run # 预览会删除什么
    python -m research.rollback_to_round 112 --keep-checkpoints  # 保留 checkpoint 文件

会清理的内容:
    - state.json: round_index 设为目标轮, session 关闭, counters 归零
    - frontier.json: 移除 round > 目标轮 产生的条目
    - results.tsv: 移除目标轮之后的所有行
    - timeline.jsonl: 移除 round > 目标轮 的事件
    - round_summaries/: 删除 round > 目标轮 的 summary 文件
    - logs/: 删除 round > 目标轮 的 agent_action / agent_round / round config+log 文件
    - .snapshots/: 删除 round > 目标轮 的快照目录
    - memory.json: recent_rounds 只保留 <= 目标轮的条目
    - current_status.json: 重置为 idle
    - checkpoints/: 删除 round > 目标轮 的 checkpoint 目录 (除非 --keep-checkpoints)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = REPO_ROOT / "research" / "history"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "research_agent"

STATE_PATH = HISTORY_DIR / "state.json"
FRONTIER_PATH = HISTORY_DIR / "frontier.json"
RESULTS_PATH = HISTORY_DIR / "results.tsv"
TIMELINE_PATH = HISTORY_DIR / "timeline.jsonl"
MEMORY_PATH = HISTORY_DIR / "memory.json"
STATUS_PATH = HISTORY_DIR / "current_status.json"
ROUND_SUMMARIES_DIR = HISTORY_DIR / "round_summaries"
LOG_DIR = HISTORY_DIR / "logs"
SNAPSHOTS_DIR = HISTORY_DIR / ".snapshots"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _extract_round_number(name: str) -> int | None:
    """从文件名中提取 round 编号。支持 round_0112, round0112, r112 等格式。"""
    m = re.search(r"round[_]?0*(\d+)", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.match(r"r(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def rollback(target_round: int, *, dry_run: bool = False, keep_checkpoints: bool = False) -> None:
    print(f"{'[DRY RUN] ' if dry_run else ''}回滚到 round {target_round} 结束, round {target_round + 1} 即将开始")
    print()

    # --- 1. Validate target round ---
    summary_path = ROUND_SUMMARIES_DIR / f"round_{target_round:04d}.json"
    if not summary_path.exists():
        print(f"错误: round {target_round} 的 summary 不存在 ({summary_path})")
        print(f"可用的最新 round summary:")
        summaries = sorted(ROUND_SUMMARIES_DIR.glob("round_*.json"))
        for s in summaries[-5:]:
            rn = _extract_round_number(s.name)
            print(f"  round {rn}: {s.name}")
        sys.exit(1)

    # --- 2. state.json ---
    print("=== state.json ===")
    state = _load_json(STATE_PATH)
    old_round = state.get("agent", {}).get("round_index", "?")
    print(f"  round_index: {old_round} -> {target_round}")

    if not dry_run:
        agent = state.setdefault("agent", {})
        agent["round_index"] = target_round
        agent["consecutive_crashes"] = 0
        agent["consecutive_no_improve"] = 0
        agent["rounds_since_new_family"] = 0
        agent["active_session_id"] = None
        agent["active_session_started_at"] = None
        agent["active_session_rounds"] = 0
        agent["active_session_status"] = "closed"
        agent["last_resume_ok_at"] = None
        agent["crash_resets"] = 0

    # --- 3. frontier.json ---
    print("\n=== frontier.json ===")
    frontier = _load_json(FRONTIER_PATH)
    to_remove = []
    for key in list(frontier.keys()):
        if not isinstance(frontier[key], dict):
            continue
        rn = _extract_round_number(key)
        if rn is not None and rn > target_round:
            to_remove.append(key)

    if to_remove:
        print(f"  移除 {len(to_remove)} 个条目: {to_remove}")
        if not dry_run:
            for key in to_remove:
                del frontier[key]
    else:
        print("  无需修改")

    # Sync frontier into state
    if not dry_run:
        state["frontier"] = frontier
        _save_json(STATE_PATH, state)
        _save_json(FRONTIER_PATH, frontier)

    # --- 4. results.tsv ---
    print("\n=== results.tsv ===")
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            lines = f.readlines()

        # Find the last line belonging to target_round
        last_target_idx = None
        for i, line in enumerate(lines):
            rn = _extract_round_number(line.split("\t")[0] if "\t" in line else "")
            if rn is not None and rn == target_round:
                last_target_idx = i
            # Also handle lines without round prefix (policy_reject with just timestamp)
            # These belong to the round context they follow

        # Strategy: keep everything up to and including the last target_round line.
        # Lines without a round prefix (orphan policy_rejects) belong to the round
        # context that precedes them, so they get trimmed together.
        if last_target_idx is not None:
            removed = len(lines) - (last_target_idx + 1)
            print(f"  保留 {last_target_idx + 1} 行, 移除 {removed} 行")
            if not dry_run:
                with open(RESULTS_PATH, "w") as f:
                    f.writelines(lines[:last_target_idx + 1])
        else:
            # No line for target_round found; remove lines for rounds > target
            # Also remove orphan lines (no round prefix) that follow a removed round
            kept = []
            removed_count = 0
            removing_tail = False
            for line in lines:
                fields = line.split("\t")
                rn = _extract_round_number(fields[0]) if fields else None
                if rn is not None and rn > target_round:
                    removed_count += 1
                    removing_tail = True
                    continue
                if rn is None and removing_tail:
                    # Orphan line after a removed round
                    removed_count += 1
                    continue
                removing_tail = False
                kept.append(line)
            print(f"  保留 {len(kept)} 行, 移除 {removed_count} 行")
            if not dry_run:
                with open(RESULTS_PATH, "w") as f:
                    f.writelines(kept)
    else:
        print("  文件不存在, 跳过")

    # --- 5. timeline.jsonl ---
    print("\n=== timeline.jsonl ===")
    if TIMELINE_PATH.exists():
        with open(TIMELINE_PATH) as f:
            tl_lines = f.readlines()

        kept = []
        removed_count = 0
        for line in tl_lines:
            try:
                evt = json.loads(line)
                rn = evt.get("round") or 0
                if isinstance(rn, (int, float)) and rn > target_round:
                    removed_count += 1
                    continue
            except (json.JSONDecodeError, AttributeError):
                pass
            kept.append(line)

        print(f"  保留 {len(kept)} 行, 移除 {removed_count} 行")
        if not dry_run:
            with open(TIMELINE_PATH, "w") as f:
                f.writelines(kept)
    else:
        print("  文件不存在, 跳过")

    # --- 6. round_summaries/ ---
    print("\n=== round_summaries/ ===")
    removed_summaries = []
    if ROUND_SUMMARIES_DIR.exists():
        for p in sorted(ROUND_SUMMARIES_DIR.glob("round_*.json")):
            rn = _extract_round_number(p.name)
            if rn is not None and rn > target_round:
                removed_summaries.append(p.name)
                if not dry_run:
                    p.unlink()

    if removed_summaries:
        print(f"  移除 {len(removed_summaries)} 个文件: {removed_summaries[:10]}")
        if len(removed_summaries) > 10:
            print(f"  ... 及其他 {len(removed_summaries) - 10} 个")
    else:
        print("  无需清理")

    # --- 7. logs/ ---
    print("\n=== logs/ ===")
    removed_logs = []
    if LOG_DIR.exists():
        # Patterns: agent_action_NNNN.json, agent_round_NNNN.stdout.log,
        #           roundNNNN_*.log, roundNNNN_*.config.json
        for p in sorted(LOG_DIR.iterdir()):
            rn = _extract_round_number(p.name)
            if rn is not None and rn > target_round:
                removed_logs.append(p.name)
                if not dry_run:
                    p.unlink()

    if removed_logs:
        print(f"  移除 {len(removed_logs)} 个文件: {removed_logs[:10]}")
        if len(removed_logs) > 10:
            print(f"  ... 及其他 {len(removed_logs) - 10} 个")
    else:
        print("  无需清理")

    # --- 8. .snapshots/ ---
    print("\n=== .snapshots/ ===")
    removed_snaps = []
    if SNAPSHOTS_DIR.exists():
        for p in sorted(SNAPSHOTS_DIR.iterdir()):
            if not p.is_dir():
                continue
            rn = _extract_round_number(p.name)
            if rn is not None and rn > target_round:
                removed_snaps.append(p.name)
                if not dry_run:
                    shutil.rmtree(p)

    if removed_snaps:
        print(f"  移除 {len(removed_snaps)} 个目录: {removed_snaps}")
    else:
        print("  无需清理")

    # --- 9. memory.json ---
    print("\n=== memory.json ===")
    if MEMORY_PATH.exists():
        memory = _load_json(MEMORY_PATH)
        recent = memory.get("recent_rounds", [])
        before_count = len(recent)
        filtered = [r for r in recent if r.get("round", 0) <= target_round]
        removed_count = before_count - len(filtered)
        print(f"  recent_rounds: {before_count} -> {len(filtered)} (移除 {removed_count})")

        if not dry_run:
            memory["recent_rounds"] = filtered
            _save_json(MEMORY_PATH, memory)
    else:
        print("  文件不存在, 跳过")

    # --- 10. current_status.json ---
    print("\n=== current_status.json ===")
    print("  重置为 idle")
    if not dry_run:
        _save_json(STATUS_PATH, {
            "status": "idle",
            "round": target_round,
            "message": f"Rolled back to round {target_round} completed",
        })

    # --- 11. checkpoints ---
    print("\n=== checkpoints ===")
    removed_ckpts = []
    if not keep_checkpoints and CHECKPOINT_ROOT.exists():
        for p in sorted(CHECKPOINT_ROOT.iterdir()):
            if not p.is_dir():
                continue
            rn = _extract_round_number(p.name)
            if rn is not None and rn > target_round:
                removed_ckpts.append(p.name)
                if not dry_run:
                    shutil.rmtree(p)

    if keep_checkpoints:
        print("  --keep-checkpoints: 跳过")
    elif removed_ckpts:
        print(f"  移除 {len(removed_ckpts)} 个目录: {removed_ckpts[:10]}")
        if len(removed_ckpts) > 10:
            print(f"  ... 及其他 {len(removed_ckpts) - 10} 个")
    else:
        print("  无需清理")

    # --- 12. Remove stale timeline backups ---
    removed_baks = []
    for p in HISTORY_DIR.glob("timeline.jsonl.bak_*"):
        removed_baks.append(p.name)
        if not dry_run:
            p.unlink()
    if removed_baks:
        print(f"\n=== timeline backups ===")
        print(f"  移除: {removed_baks}")

    # --- Summary ---
    print("\n" + "=" * 50)
    if dry_run:
        print(f"[DRY RUN] 以上为预览, 未做任何修改")
    else:
        print(f"已回滚到 round {target_round} 结束")
        print(f"下一轮: round {target_round + 1}")
    print(f"Frontier 条目: {list(frontier.keys()) if not dry_run else '(未修改)'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 AutoResearch 系统状态回滚到指定 round 结束的状态",
    )
    parser.add_argument("round", type=int, help="目标 round 编号 (该轮保留, 之后的全部清除)")
    parser.add_argument("--dry-run", action="store_true", help="只预览不执行")
    parser.add_argument("--keep-checkpoints", action="store_true", help="保留 checkpoint 文件不删除")
    args = parser.parse_args()

    if args.round < 1:
        print("错误: round 编号必须 >= 1")
        sys.exit(1)

    rollback(args.round, dry_run=args.dry_run, keep_checkpoints=args.keep_checkpoints)


if __name__ == "__main__":
    main()
