"""
Minimal experiment controller for SAE autoresearch.

This controller does not decide experiments or run training. It provides:
- state initialization
- result parsing and keep/archive/promote decisions
- synchronized history files for long-lived agent memory
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = REPO_ROOT / "research"
HISTORY_DIR = RESEARCH_DIR / "history"
LOG_DIR = HISTORY_DIR / "logs"
ROUND_SUMMARIES_DIR = HISTORY_DIR / "round_summaries"
STATE_PATH = HISTORY_DIR / "state.json"
RESULTS_PATH = HISTORY_DIR / "results.tsv"
FRONTIER_PATH = HISTORY_DIR / "frontier.json"
MEMORY_PATH = HISTORY_DIR / "memory.json"
HINTS_PATH = HISTORY_DIR / "operator_hints.json"
TIMELINE_PATH = HISTORY_DIR / "timeline.jsonl"

RESULT_COLUMNS = [
    "experiment_id",
    "timestamp",
    "tier",
    "status",
    "decision",
    "val_fvu",
    "k",
    "architecture",
    "expansion_factor",
    "wall_time_sec",
    "peak_memory_gb",
    "total_tokens",
    "checkpoint",
    "head_commit",
    "head_branch",
    "workspace_dirty",
    "description",
    "self_review",
    "log_path",
]

SUMMARY_PATTERNS = {
    "status": re.compile(r"^status:\s+(\S+)$", re.MULTILINE),
    "val_fvu": re.compile(r"^val_fvu:\s+(\S+)$", re.MULTILINE),
    "k": re.compile(r"^k:\s+(\d+)$", re.MULTILINE),
    "architecture": re.compile(r"^architecture:\s+(\S+)$", re.MULTILINE),
    "wall_time_sec": re.compile(r"^wall_time_sec:\s+([0-9.]+)$", re.MULTILINE),
    "peak_memory_gb": re.compile(r"^peak_memory_gb:\s+([0-9.]+)$", re.MULTILINE),
    "total_tokens": re.compile(r"^total_tokens:\s+(\d+)$", re.MULTILINE),
    "checkpoint": re.compile(r"^checkpoint:\s+(.+)$", re.MULTILINE),
    "expansion_factor": re.compile(
        r"^expansion_factor:\s+(\d+)$", re.MULTILINE
    ),
}


@dataclass
class RunResult:
    experiment_id: str
    tier: str
    status: str
    decision: str
    val_fvu: float | None
    k: int | None
    architecture: str | None
    expansion_factor: int | None
    wall_time_sec: float | None
    peak_memory_gb: float | None
    total_tokens: int | None
    checkpoint: str | None
    head_commit: str
    head_branch: str
    workspace_dirty: bool
    description: str
    self_review: str
    log_path: str


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_json_retry(path: Path) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        time.sleep(0.1)
        with open(path) as f:
            return json.load(f)


def append_timeline_event(event: str, **payload: Any) -> str:
    TIMELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TIMELINE_PATH.touch(exist_ok=True)
    event_id = f"evt_{time.time_ns()}"
    record = {
        "event_id": event_id,
        "ts": int(time.time()),
        "event": event,
        **payload,
    }
    with open(TIMELINE_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return event_id


def get_git_state() -> dict[str, Any]:
    head_commit = _run_git(["rev-parse", "HEAD"])
    head_branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(_run_git(["status", "--porcelain"]))
    return {
        "head_commit": head_commit,
        "head_branch": head_branch,
        "workspace_dirty": dirty,
    }


def normalize_frontiers(state: dict[str, Any]) -> dict[str, Any]:
    proxy_frontier = state.setdefault("proxy_frontier", {})
    full_frontier = state.setdefault("full_frontier", {})
    legacy_frontier = state.get("frontier", {})

    for key, point in legacy_frontier.items():
        if not isinstance(point, dict):
            continue
        target = proxy_frontier if point.get("tier") == "proxy" else full_frontier
        current = target.get(key)
        if current is None or float(point.get("fvu", float("inf"))) < float(current.get("fvu", float("inf"))):
            target[key] = point

    for key, point in list(full_frontier.items()):
        if isinstance(point, dict) and point.get("tier") == "proxy":
            current = proxy_frontier.get(key)
            if current is None or float(point.get("fvu", float("inf"))) < float(current.get("fvu", float("inf"))):
                proxy_frontier[key] = point
            del full_frontier[key]

    state["frontier"] = full_frontier
    return state


def ensure_history_files() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ROUND_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] | None = None

    if not RESULTS_PATH.exists():
        with open(RESULTS_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, delimiter="\t")
            writer.writeheader()

    if not STATE_PATH.exists():
        git_state = get_git_state()
        state = {
            "initialized_at": int(time.time()),
            "base_commit": git_state["head_commit"],
            "base_branch": git_state["head_branch"],
            "base_workspace_dirty": git_state["workspace_dirty"],
            "frontier": {},
            "proxy_frontier": {},
            "full_frontier": {},
            "total_experiments": 0,
            "total_keeps": 0,
            "total_promotes": 0,
            "total_incubates": 0,
            "total_discards": 0,
            "total_archives": 0,
            "total_crashes": 0,
            "observations": [],
            "next_ideas": [],
            "last_result": None,
        }
        normalize_frontiers(state)
        save_state(state)
    else:
        state = read_json_retry(STATE_PATH)
        normalize_frontiers(state)
        save_state(state)

    FRONTIER_PATH.write_text(json.dumps((state or {}).get("frontier", {}), indent=2) + "\n")

    if not MEMORY_PATH.exists():
        memory = {
            "current_focus": (
                "Treat K=128 only as the initial search anchor, not as a success criterion. "
                "The research objective is to find configurations whose FVU is dramatically better "
                "than the current K=128 baseline, with a target on the order of halving that baseline FVU "
                "before rewarding smaller K or lower cost."
            ),
            "architecture_findings": [],
            "performance_findings": [],
            "failure_patterns": [],
            "baseline_runtime": {},
            "architecture_families": {},
            "recent_rounds": [],
            "recent_insights": [],
            "next_hypotheses": [
                "Establish or refresh a trustworthy K=128 quality anchor before rewarding lower-K points.",
                "Do not treat smaller K as success unless its FVU is materially better than the current K=128 baseline.",
                "Prefer experiments that can plausibly cut the current K=128 FVU by about half, even if they keep K unchanged at first."
            ],
        }
        MEMORY_PATH.write_text(json.dumps(memory, indent=2) + "\n")

    if not HINTS_PATH.exists():
        HINTS_PATH.write_text("[]\n")
    TIMELINE_PATH.touch(exist_ok=True)


def load_state() -> dict[str, Any]:
    ensure_history_files()
    state = read_json_retry(STATE_PATH)
    normalize_frontiers(state)
    return state


def save_state(state: dict[str, Any]) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    normalize_frontiers(state)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    FRONTIER_PATH.write_text(json.dumps(state.get("frontier", {}), indent=2) + "\n")


def load_config_from_json(path: str | None) -> dict[str, Any]:
    if path is None:
        raise RuntimeError(
            "record now requires --config-json; the legacy research/train.py "
            "fallback has been removed because it could diverge from the real "
            "env-driven training configuration."
        )
    with open(path) as f:
        return json.load(f)


def parse_log_output(output: str) -> dict[str, Any]:
    """Parse the key-value summary block printed by the training launcher."""
    parsed: dict[str, Any] = {}
    for key, pattern in SUMMARY_PATTERNS.items():
        match = pattern.search(output)
        parsed[key] = match.group(1).strip() if match else None

    if parsed["val_fvu"] in (None, "nan"):
        parsed["val_fvu"] = None
    elif parsed["val_fvu"] is not None:
        parsed["val_fvu"] = float(parsed["val_fvu"])

    for key in ("k", "total_tokens", "expansion_factor"):
        if parsed[key] is not None:
            parsed[key] = int(parsed[key])

    for key in ("wall_time_sec", "peak_memory_gb"):
        if parsed[key] is not None:
            parsed[key] = float(parsed[key])

    if parsed["checkpoint"] == "none":
        parsed["checkpoint"] = None

    return parsed


def parse_checkpoint_result(checkpoint_dir: str, config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(checkpoint_dir)
    summary_path = checkpoint / "summary.json"
    manifest_path = checkpoint / "manifest.json"
    parsed: dict[str, Any] = {
        "status": "crash",
        "val_fvu": None,
        "k": config.get("k"),
        "architecture": config.get("architecture"),
        "wall_time_sec": None,
        "peak_memory_gb": None,
        "total_tokens": None,
        "checkpoint": str(checkpoint),
        "expansion_factor": config.get("expansion_factor"),
    }

    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            parsed["architecture"] = manifest.get("architecture", parsed["architecture"])
        except json.JSONDecodeError:
            pass

    if not summary_path.exists():
        return parsed

    with open(summary_path) as f:
        summary = json.load(f)

    fvu_dict = summary.get("final_fvu") or summary.get("best_fvu") or {}
    vals = [v for v in fvu_dict.values() if isinstance(v, (int, float))]
    parsed["val_fvu"] = sum(vals) / len(vals) if vals else None
    parsed["total_tokens"] = summary.get("total_tokens")
    parsed["status"] = "ok" if parsed["val_fvu"] is not None else "crash"
    return parsed


def decide(frontier: dict[str, Any], parsed: dict[str, Any], tier: str) -> str:
    status = parsed.get("status")
    fvu = parsed.get("val_fvu")
    k = parsed.get("k")
    peak_memory_gb = parsed.get("peak_memory_gb")

    if status != "ok" or fvu is None or k is None:
        return "crash"

    fvu_tol = 0.001
    mem_tol = 0.5
    key = str(k)
    current = frontier.get(key)

    improve_same_k = False
    if current is None:
        improve_same_k = True
    else:
        current_fvu = current["fvu"]
        if fvu < current_fvu - fvu_tol:
            improve_same_k = True
        elif abs(fvu - current_fvu) <= fvu_tol:
            current_mem = current.get("peak_memory_gb")
            if (
                peak_memory_gb is not None
                and current_mem is not None
                and peak_memory_gb < current_mem - mem_tol
            ):
                improve_same_k = True

    improve_smaller_k = False
    for frontier_k, frontier_point in frontier.items():
        fk = int(frontier_k)
        ffvu = frontier_point["fvu"]
        if k < fk and fvu <= ffvu + fvu_tol:
            improve_smaller_k = True
            break

    is_improvement = improve_same_k or improve_smaller_k
    if is_improvement:
        return "promote" if tier == "proxy" else "keep"

    near_same_k = (
        current is not None and abs(fvu - current["fvu"]) <= fvu_tol
    )
    return "archive" if near_same_k else "discard"


def decide_with_context(
    state: dict[str, Any],
    parsed: dict[str, Any],
    tier: str,
    config: dict[str, Any],
) -> str:
    frontier_key = "proxy_frontier" if tier == "proxy" else "full_frontier"
    decision = decide(state.get(frontier_key, {}), parsed, tier)
    if decision in {"promote", "keep", "archive", "crash"}:
        return decision
    if tier != "proxy":
        return decision

    family_name = str(config.get("family_name") or parsed.get("architecture") or "").strip().lower()
    family_stage = str(config.get("family_stage") or "").strip().lower()
    known_families = {
        str(point.get("architecture", "")).lower()
        for frontier_name in ("proxy_frontier", "full_frontier")
        for point in state.get(frontier_name, {}).values()
        if isinstance(point, dict)
    }

    if family_stage == "prototype":
        return "incubate"
    if family_name and family_name not in known_families:
        return "incubate"
    return decision


def append_result(result: RunResult) -> None:
    ensure_history_files()
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, delimiter="\t")
        writer.writerow(
            {
                "experiment_id": result.experiment_id,
                "timestamp": int(time.time()),
                "tier": result.tier,
                "status": result.status,
                "decision": result.decision,
                "val_fvu": result.val_fvu,
                "k": result.k,
                "architecture": result.architecture,
                "expansion_factor": result.expansion_factor,
                "wall_time_sec": result.wall_time_sec,
                "peak_memory_gb": result.peak_memory_gb,
                "total_tokens": result.total_tokens,
                "checkpoint": result.checkpoint,
                "head_commit": result.head_commit,
                "head_branch": result.head_branch,
                "workspace_dirty": result.workspace_dirty,
                "description": result.description,
                "self_review": result.self_review,
                "log_path": result.log_path,
            }
        )


def update_state_with_result(
    state: dict[str, Any],
    result: RunResult,
    config: dict[str, Any],
) -> None:
    state["total_experiments"] += 1
    active_frontier_key = "proxy_frontier" if result.tier == "proxy" else "full_frontier"
    active_frontier = state.setdefault(active_frontier_key, {})
    if result.decision == "keep":
        state["total_keeps"] += 1
        if result.k is not None and result.val_fvu is not None:
            active_frontier[str(result.k)] = {
                "fvu": result.val_fvu,
                "architecture": result.architecture,
                "commit": result.head_commit,
                "config": config,
                "tier": result.tier,
                "checkpoint": result.checkpoint,
                "peak_memory_gb": result.peak_memory_gb,
            }
    elif result.decision == "promote":
        state["total_promotes"] += 1
        if result.k is not None and result.val_fvu is not None:
            key = str(result.k)
            current = active_frontier.get(key)
            if current is None or result.val_fvu < current["fvu"]:
                active_frontier[key] = {
                    "fvu": result.val_fvu,
                    "architecture": result.architecture,
                    "commit": result.head_commit,
                    "config": config,
                    "tier": result.tier,
                    "checkpoint": result.checkpoint,
                    "peak_memory_gb": result.peak_memory_gb,
                }
    elif result.decision == "archive":
        state["total_archives"] += 1
    elif result.decision == "incubate":
        state["total_incubates"] = int(state.get("total_incubates", 0)) + 1
    elif result.decision == "discard":
        state["total_discards"] += 1
    elif result.decision == "crash":
        state["total_crashes"] += 1

    state["last_result"] = {
        "experiment_id": result.experiment_id,
        "tier": result.tier,
        "status": result.status,
        "decision": result.decision,
        "val_fvu": result.val_fvu,
        "description": result.description,
        "self_review": result.self_review,
    }
    state["frontier"] = state.get("full_frontier", {})


def print_status(state: dict[str, Any]) -> None:
    print(f"total_experiments: {state['total_experiments']}")
    print(f"total_keeps: {state['total_keeps']}")
    print(f"total_promotes: {state.get('total_promotes', 0)}")
    print(f"total_incubates: {state.get('total_incubates', 0)}")
    print(f"total_crashes: {state.get('total_crashes', 0)}")
    print("full_frontier:")
    for k in sorted(state.get("full_frontier", {}).keys(), key=int):
        pt = state["full_frontier"][k]
        print(f"  K={k}: FVU={pt['fvu']:.6f} arch={pt['architecture']} tier={pt['tier']}")
    print("proxy_frontier:")
    for k in sorted(state.get("proxy_frontier", {}).keys(), key=int):
        pt = state["proxy_frontier"][k]
        print(f"  K={k}: FVU={pt['fvu']:.6f} arch={pt['architecture']} tier={pt['tier']}")
    if state.get("last_result"):
        lr = state["last_result"]
        print(f"last: {lr['decision']} | {lr['description']} | fvu={lr.get('val_fvu')}")


def load_hints() -> list[dict[str, Any]]:
    ensure_history_files()
    return read_json_retry(HINTS_PATH)


def save_hints(hints: list[dict[str, Any]]) -> None:
    HINTS_PATH.write_text(json.dumps(hints, indent=2) + "\n")


def add_hint(message: str, priority: str, scope: str, tag: str | None) -> dict[str, Any]:
    hints = load_hints()
    hint = {
        "id": f"hint_{int(time.time())}",
        "message": message,
        "priority": priority,
        "scope": scope,
        "tag": tag,
        "status": "pending",
        "created_at": int(time.time()),
        "applied_at": None,
    }
    hints.append(hint)
    save_hints(hints)
    append_timeline_event(
        "hint_added",
        round=None,
        tier=None,
        run_name=None,
        family_name=None,
        family_stage=None,
        status=hint["status"],
        decision=None,
        payload=hint,
    )
    return hint


def update_hint(
    hint_id: str,
    priority: str | None,
    scope: str | None,
    tag: str | None,
    status: str | None,
) -> dict[str, Any]:
    hints = load_hints()
    for hint in hints:
        if hint.get("id") != hint_id:
            continue
        if priority is not None:
            hint["priority"] = priority
        if scope is not None:
            hint["scope"] = scope
        if tag is not None:
            hint["tag"] = tag
        if status is not None:
            hint["status"] = status
            if status == "pending":
                hint["applied_at"] = None
                hint.pop("applied_in_round", None)
            elif status in {"applied", "dismissed"}:
                hint["applied_at"] = int(time.time())
        save_hints(hints)
        append_timeline_event(
            "hint_updated",
            round=None,
            tier=None,
            run_name=None,
            family_name=None,
            family_stage=None,
            status=hint.get("status"),
            decision=None,
            payload=hint,
        )
        return hint
    raise KeyError(f"hint not found: {hint_id}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAE autoresearch controller",
        epilog="""
Usage:
  python controller.py init                          # first time setup
  python controller.py status                        # show frontier & stats
  python controller.py hint --message "..."         # enqueue an operator hint
  python controller.py hints                         # list operator hints
  python controller.py hint-update --id ...         # edit an operator hint
  python controller.py record --log run.log \\
    --description "baseline" --self-review "..."      # record experiment result
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Initialize history/state files")
    sub.add_parser("status", help="Print controller state summary")
    sub.add_parser("hints", help="Print operator hints")

    hint_parser = sub.add_parser("hint", help="Add an operator hint")
    hint_parser.add_argument("--message", required=True)
    hint_parser.add_argument("--priority", choices=["low", "normal", "high"], default="normal")
    hint_parser.add_argument("--scope", choices=["next_round", "persistent"], default="next_round")
    hint_parser.add_argument("--tag", default=None)

    hint_update = sub.add_parser("hint-update", help="Update an operator hint")
    hint_update.add_argument("--id", required=True)
    hint_update.add_argument("--priority", choices=["low", "normal", "high"])
    hint_update.add_argument("--scope", choices=["next_round", "persistent"])
    hint_update.add_argument("--tag")
    hint_update.add_argument("--status", choices=["pending", "applied", "dismissed"])

    rec = sub.add_parser("record", help="Parse a training log and record the result")
    rec.add_argument("--log", help="Path to the run.log file")
    rec.add_argument(
        "--checkpoint-dir",
        help="Checkpoint directory to parse directly (for script-based runs)",
    )
    rec.add_argument(
        "--config-json",
        help="JSON file containing the exact experiment config used for this run",
    )
    rec.add_argument("--tier", default="proxy", choices=["proxy", "full"])
    rec.add_argument("--description", default="experiment", help="Short description")
    rec.add_argument("--self-review", default="", help="Why this might be fake")

    args = parser.parse_args()
    ensure_history_files()

    if args.cmd == "init":
        print(f"initialized: {STATE_PATH}")
        print(f"results: {RESULTS_PATH}")
        return 0

    state = load_state()

    if args.cmd == "status":
        print_status(state)
        return 0

    if args.cmd == "hints":
        for hint in load_hints():
            print(json.dumps(hint, ensure_ascii=False))
        return 0

    if args.cmd == "hint":
        hint = add_hint(args.message, args.priority, args.scope, args.tag)
        print(f"hint_id: {hint['id']}")
        print(f"status: {hint['status']}")
        print(f"priority: {hint['priority']}")
        print(f"scope: {hint['scope']}")
        return 0

    if args.cmd == "hint-update":
        try:
            hint = update_hint(args.id, args.priority, args.scope, args.tag, args.status)
        except KeyError as exc:
            print(f"ERROR: {exc}")
            return 1
        print(f"hint_id: {hint['id']}")
        print(f"status: {hint['status']}")
        print(f"priority: {hint['priority']}")
        print(f"scope: {hint['scope']}")
        print(f"tag: {hint.get('tag')}")
        return 0

    # --- record ---
    if not args.log and not args.checkpoint_dir:
        print("ERROR: record requires --log or --checkpoint-dir")
        return 1

    git_state = get_git_state()
    config = load_config_from_json(args.config_json)

    full_output = ""
    archived_log: Path | None = None
    if args.log:
        log_file = Path(args.log)
        if not log_file.exists():
            print(f"ERROR: log file not found: {args.log}")
            return 1
        full_output = log_file.read_text()

    if args.checkpoint_dir:
        parsed = parse_checkpoint_result(args.checkpoint_dir, config)
    else:
        parsed = parse_log_output(full_output)

    decision = decide_with_context(state, parsed, args.tier, config)

    experiment_id = f"{args.tier}_{int(time.time())}"

    if full_output:
        archived_log = LOG_DIR / f"{experiment_id}.log"
        archived_log.write_text(full_output)

    result = RunResult(
        experiment_id=experiment_id,
        tier=args.tier,
        status=parsed.get("status") or "crash",
        decision=decision,
        val_fvu=parsed.get("val_fvu"),
        k=parsed.get("k"),
        architecture=parsed.get("architecture"),
        expansion_factor=parsed.get("expansion_factor"),
        wall_time_sec=parsed.get("wall_time_sec"),
        peak_memory_gb=parsed.get("peak_memory_gb"),
        total_tokens=parsed.get("total_tokens"),
        checkpoint=parsed.get("checkpoint"),
        head_commit=git_state["head_commit"],
        head_branch=git_state["head_branch"],
        workspace_dirty=git_state["workspace_dirty"],
        description=args.description,
        self_review=args.self_review,
        log_path=str(archived_log) if archived_log is not None else "",
    )
    append_result(result)
    update_state_with_result(state, result, config)
    save_state(state)

    print(f"experiment_id: {result.experiment_id}")
    print(f"status:        {result.status}")
    print(f"decision:      {result.decision}")
    print(f"val_fvu:       {result.val_fvu}")
    print(f"k:             {result.k}")
    print(f"architecture:  {result.architecture}")
    print(f"log_path:      {result.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
