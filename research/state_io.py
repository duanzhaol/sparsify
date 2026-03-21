"""State, memory, timeline, and results I/O for the autoresearch loop."""

from __future__ import annotations

import csv
import json
import time
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
STATUS_PATH = HISTORY_DIR / "current_status.json"
HINTS_PATH = HISTORY_DIR / "operator_hints.json"
TIMELINE_PATH = HISTORY_DIR / "timeline.jsonl"
SESSION_BRIEF_PATH = HISTORY_DIR / "session_brief.json"
OPERATOR_GUIDE_PATH = RESEARCH_DIR / "operator_guide.md"

BASE_ENV_DEFAULTS = {
    "ARCHITECTURE": "topk",
    "EXPANSION_FACTOR": "8",
    "K": "128",
    "HOOKPOINTS": "layers.[3].self_attn.o_proj",
    "OPTIMIZER": "signum",
    "LR": "8e-4",
    "BATCH_SIZE": "1",
    "GRAD_ACC_STEPS": "8",
    "MICRO_ACC_STEPS": "1",
    "AUXK_ALPHA": "0.03125",
    "DEAD_FEATURE_THRESHOLD": "10000000",
    "USE_HADAMARD": "0",
    "COMPILE_MODEL": "1",
}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_operator_guide_excerpt(max_chars: int = 6000) -> str:
    """Load a compact operator guidance excerpt for prompt injection.

    The file is optional. When present, trim it to a bounded size so one
    large operator note cannot dominate the prompt context window.
    """
    if not OPERATOR_GUIDE_PATH.exists():
        return ""
    text = OPERATOR_GUIDE_PATH.read_text().strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars - 64].rstrip()
    return head + "\n\n[truncated: operator guide excerpt clipped for prompt budget]"


def load_state() -> dict[str, Any]:
    state = load_json(STATE_PATH, {})
    agent_state = state.setdefault(
        "agent",
        {
            "round_index": 0,
            "consecutive_crashes": 0,
            "consecutive_no_improve": 0,
            "rounds_since_new_family": 0,
            "last_action_file": None,
            "last_patch_file": None,
            "active_session_id": None,
            "active_session_started_at": None,
            "active_session_rounds": 0,
            "active_session_status": "closed",
            "last_resume_ok_at": None,
            "crash_resets": 0,
            "last_meta_round": 0,
        },
    )
    agent_state.setdefault("round_index", 0)
    agent_state.setdefault("consecutive_crashes", 0)
    agent_state.setdefault("consecutive_no_improve", 0)
    agent_state.setdefault("rounds_since_new_family", 0)
    agent_state.setdefault("last_action_file", None)
    agent_state.setdefault("last_patch_file", None)
    agent_state.setdefault("active_session_id", None)
    agent_state.setdefault("active_session_started_at", None)
    agent_state.setdefault("active_session_rounds", 0)
    agent_state.setdefault("active_session_status", "closed")
    agent_state.setdefault("last_resume_ok_at", None)
    agent_state.setdefault("crash_resets", 0)
    agent_state.setdefault("last_meta_round", 0)
    state.setdefault("pareto_frontier", state.get("pareto_frontier", []))
    state.setdefault("pareto_full_frontier", state.get("pareto_full_frontier", state.get("pareto_frontier", [])))
    state.setdefault("pareto_proxy_frontier", state.get("pareto_proxy_frontier", []))
    return state


def save_state(state: dict[str, Any]) -> None:
    save_json(STATE_PATH, state)
    save_json(FRONTIER_PATH, state.get("frontier", {}))


def load_memory() -> dict[str, Any]:
    return load_json(
        MEMORY_PATH,
        {
            "current_focus": (
                "Track a Pareto frontier across reconstruction quality and sparsity/cost. "
                "Treat K=128 as one anchor point, not the only success criterion. "
                "Prefer experiments that add non-dominated tradeoff points, including smaller-K runs "
                "that accept some FVU increase when they improve the overall frontier."
            ),
            "architecture_findings": [],
            "performance_findings": [],
            "failure_patterns": [],
            "baseline_runtime": {},
            "architecture_families": {},
            "recent_rounds": [],
            "recent_insights": [],
            "recent_sanity_failures": [],
            "recent_training_failures": [],
            "next_hypotheses": [
                "Maintain a Pareto frontier over FVU and K rather than optimizing only the single best FVU point.",
                "Probe smaller K values even when FVU rises, as long as the new point may improve the tradeoff frontier.",
                "Use K=128 quality anchors to calibrate tradeoffs, not to suppress lower-K exploration."
            ],
        },
    )


def load_operator_hints() -> list[dict[str, Any]]:
    return load_json(HINTS_PATH, [])


def save_operator_hints(hints: list[dict[str, Any]]) -> None:
    save_json(HINTS_PATH, hints)


def split_hints(hints: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pending = [h for h in hints if h.get("status") == "pending"]
    non_pending = [h for h in hints if h.get("status") != "pending"]
    return pending, non_pending


def write_status(stage: str, **payload: Any) -> None:
    save_json(
        STATUS_PATH,
        {
            "timestamp": int(time.time()),
            "stage": stage,
            **payload,
        },
    )


def append_timeline_event(event: str, **payload: Any) -> str:
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


def log_round_event(round_ctx: dict[str, Any], event: str, **payload: Any) -> str:
    event_id = append_timeline_event(
        event,
        round=round_ctx.get("round"),
        session_id=payload.get("session_id", round_ctx.get("session_id")),
        tier=payload.get("tier"),
        run_name=payload.get("run_name"),
        family_name=payload.get("family_name", round_ctx.get("family_name")),
        family_stage=payload.get("family_stage", round_ctx.get("family_stage")),
        status=payload.get("status"),
        decision=payload.get("decision"),
        payload=payload,
    )
    round_ctx.setdefault("timeline_event_ids", []).append(event_id)
    round_ctx.setdefault("timeline_start_event_id", event_id)
    round_ctx["timeline_end_event_id"] = event_id
    return event_id


def load_results(limit: int | None = None) -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    with open(RESULTS_PATH, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    return rows[-limit:] if limit is not None else rows


def summarize_results(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keys = ["experiment_id", "tier", "status", "decision", "val_fvu", "k", "architecture", "description"]
    return [{k: row.get(k, "") for k in keys} for row in rows]


def recent_round_summaries(limit: int = 3) -> list[dict[str, Any]]:
    paths = sorted(ROUND_SUMMARIES_DIR.glob("round_*.json"))[-limit:]
    return [load_json(path, {}) for path in paths]


def recent_round_summaries_trimmed(limit: int = 3) -> list[dict[str, Any]]:
    """Return recent round summaries with only the essential digest fields.

    Strips bulky action/result/proxy_result/full_result to keep prompt size down.
    """
    summaries = recent_round_summaries(limit=limit)
    trimmed: list[dict[str, Any]] = []
    for s in summaries:
        action = s.get("action", {})
        result = s.get("result", {})
        trimmed.append({
            "round": s.get("round"),
            "family_name": s.get("family_name"),
            "family_stage": s.get("family_stage"),
            "duration_sec": s.get("duration_sec"),
            "hypothesis": action.get("hypothesis"),
            "change_type": action.get("change_type"),
            "experiment_tier": action.get("experiment_tier"),
            "decision": result.get("decision"),
            "val_fvu": result.get("val_fvu"),
            "run_health": result.get("run_health"),
            "termination_reason": result.get("termination_reason"),
        })
    return trimmed


def compact_failure_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "round": entry.get("round"),
        "family_name": entry.get("family_name"),
        "change_type": entry.get("change_type"),
        "error_type": entry.get("error_type"),
        "termination_reason": entry.get("termination_reason"),
        "error_summary": entry.get("error_summary") or entry.get("stderr_excerpt") or entry.get("traceback_excerpt"),
    }


def load_session_brief() -> dict[str, Any]:
    return load_json(
        SESSION_BRIEF_PATH,
        {
            "updated_at": None,
            "active_session_id": None,
            "current_focus": None,
            "best_full_frontier": {},
            "pareto_full_frontier": [],
            "recent_results": [],
            "recent_round_summaries": [],
            "incubating_families": {},
            "recent_performance_findings": [],
            "recent_training_failures": [],
            "pending_hints": [],
            "next_move_guidance": [],
        },
    )


def save_session_brief(brief: dict[str, Any]) -> None:
    save_json(SESSION_BRIEF_PATH, brief)


def build_session_brief(
    state: dict[str, Any],
    memory: dict[str, Any],
    recent_results: list[dict[str, str]],
    round_id: int,
) -> dict[str, Any]:
    operator_hints, _ = split_hints(load_operator_hints())
    families = memory.get("architecture_families", {})
    # Only include incubating families, trimmed to essentials
    incubating: dict[str, Any] = {}
    for name, value in families.items():
        if value.get("status") != "incubating":
            continue
        incubating[name] = {
            "status": value.get("status"),
            "design_hypothesis": value.get("design_hypothesis"),
            "best_proxy_fvu": value.get("best_proxy_fvu"),
            "tested_configs": value.get("tested_configs", [])[-3:],
            "next_steps": value.get("next_steps", [])[-3:],
            "last_round": value.get("last_round"),
        }
    return {
        "updated_at": int(time.time()),
        "active_session_id": state.get("agent", {}).get("active_session_id"),
        "current_focus": memory.get("current_focus"),
        "best_full_frontier": state.get("full_frontier", state.get("frontier", {})),
        "pareto_full_frontier": state.get("pareto_full_frontier", state.get("pareto_frontier", [])),
        "recent_results": summarize_results(recent_results)[-3:],
        "recent_round_summaries": recent_round_summaries_trimmed(limit=3),
        "incubating_families": incubating,
        "recent_performance_findings": memory.get("performance_findings", [])[-4:],
        "recent_sanity_failures": [compact_failure_entry(item) for item in memory.get("recent_sanity_failures", [])[-4:] if isinstance(item, dict)],
        "recent_training_failures": [compact_failure_entry(item) for item in memory.get("recent_training_failures", [])[-4:] if isinstance(item, dict)],
        "pending_hints": operator_hints[:4],
        "next_move_guidance": memory.get("next_hypotheses", [])[:5],
        "last_round": round_id,
    }


def append_memory(memory: dict[str, Any], action: dict[str, Any], result: dict[str, str], round_id: int, touched: list[str]) -> dict[str, Any]:
    overrides = action.get("env_overrides", [])
    if isinstance(overrides, dict):
        arch = overrides.get("ARCHITECTURE", BASE_ENV_DEFAULTS["ARCHITECTURE"]).lower()
    else:
        arch = next(
            (
                item.get("value", BASE_ENV_DEFAULTS["ARCHITECTURE"])
                for item in overrides
                if item.get("key") == "ARCHITECTURE"
            ),
            BASE_ENV_DEFAULTS["ARCHITECTURE"],
        ).lower()

    family_name = str(action.get("family_name") or arch).lower()
    family_stage = str(action.get("family_stage") or ("mainline" if action["change_type"] == "param_only" else "prototype"))

    entry = {
        "round": round_id,
        "hypothesis": action["hypothesis"],
        "change_type": action["change_type"],
        "expected_win": action["expected_win"],
        "family_name": family_name,
        "family_stage": family_stage,
        "decision": result.get("decision"),
        "val_fvu": result.get("val_fvu"),
        "observed_fvu": result.get("observed_fvu"),
        "k": result.get("k"),
        "architecture": arch,
        "touched_files": touched,
        "run_health": result.get("run_health"),
        "termination_reason": result.get("termination_reason"),
        "tokens_per_sec": result.get("tokens_per_sec"),
        "baseline_ratio": result.get("baseline_ratio"),
    }
    memory.setdefault("recent_rounds", []).append(entry)
    memory["recent_rounds"] = memory["recent_rounds"][-12:]

    families = memory.setdefault("architecture_families", {})
    family = families.setdefault(
        family_name,
        {
            "status": "incubating" if family_stage != "mainline" else "active",
            "goal": "new_family" if family_stage != "mainline" else "mainline",
            "design_hypothesis": action["hypothesis"],
            "tested_configs": [],
            "known_issues": [],
            "next_steps": [],
            "best_proxy_fvu": None,
            "best_full_fvu": None,
            "last_round": None,
        },
    )
    family["last_round"] = round_id
    family["design_hypothesis"] = action["hypothesis"]
    family.setdefault("tested_configs", []).append(
        {
            "round": round_id,
            "stage": family_stage,
            "k": result.get("k"),
            "decision": result.get("decision"),
            "val_fvu": result.get("val_fvu"),
            "run_health": result.get("run_health"),
        }
    )
    family["tested_configs"] = family["tested_configs"][-20:]

    try:
        val_fvu = float(result.get("val_fvu")) if result.get("val_fvu") not in (None, "") else None
    except (TypeError, ValueError):
        val_fvu = None
    if result.get("decision") == "promote":
        family["status"] = "active"
        if val_fvu is not None and (
            family.get("best_proxy_fvu") is None or val_fvu < float(family["best_proxy_fvu"])
        ):
            family["best_proxy_fvu"] = val_fvu
    elif result.get("decision") == "keep":
        family["status"] = "active" if family_stage == "mainline" else "promoted"
        if val_fvu is not None and (
            family.get("best_full_fvu") is None or val_fvu < float(family["best_full_fvu"])
        ):
            family["best_full_fvu"] = val_fvu
    elif result.get("decision") == "incubate":
        family["status"] = "incubating"
    elif result.get("decision") == "discard":
        # Do not demote families that already have proven results
        if family.get("status") not in ("active", "promoted"):
            family["status"] = "discarded"

    for note in action.get("notes_to_memory", []):
        memory.setdefault("recent_insights", []).append(note)
    outcome_note = (
        f"round {round_id}: {action['hypothesis']} -> {result.get('decision')} "
        f"(fvu={result.get('val_fvu')}, observed_fvu={result.get('observed_fvu')}, "
        f"k={result.get('k')}, health={result.get('run_health')}, "
        f"termination={result.get('termination_reason')})"
    )
    memory.setdefault("recent_insights", []).append(outcome_note)
    memory["recent_insights"] = memory["recent_insights"][-40:]

    if result.get("run_health") == "perf_regression":
        perf_note = (
            f"round {round_id}: suspected performance regression for {arch} "
            f"(tps={result.get('tokens_per_sec')}, baseline_ratio={result.get('baseline_ratio')}, "
            f"termination={result.get('termination_reason')})"
        )
        memory.setdefault("performance_findings", []).append(perf_note)
        memory["performance_findings"] = memory["performance_findings"][-20:]
        family.setdefault("known_issues", []).append(perf_note)
        family["known_issues"] = family["known_issues"][-20:]
    elif action["change_type"] == "edit_perf_code":
        memory.setdefault("performance_findings", []).append(outcome_note)
        memory["performance_findings"] = memory["performance_findings"][-20:]
    elif action["change_type"] != "param_only":
        memory.setdefault("architecture_findings", []).append(outcome_note)
        memory["architecture_findings"] = memory["architecture_findings"][-20:]

    if result.get("decision") == "crash" or result.get("run_health") in {"perf_regression", "crash"}:
        memory.setdefault("failure_patterns", []).append(
            {
                "pattern": action["hypothesis"],
                "count": 1,
                "last_round": round_id,
                "run_health": result.get("run_health"),
                "termination_reason": result.get("termination_reason"),
            }
        )
        memory["failure_patterns"] = memory["failure_patterns"][-20:]
    if result.get("decision") == "crash":
        training_failure = {
            "round": round_id,
            "tier": action.get("experiment_tier"),
            "family_name": family_name,
            "change_type": action.get("change_type"),
            "primary_variable": action.get("primary_variable"),
            "hypothesis": action.get("hypothesis"),
            "termination_reason": result.get("termination_reason"),
            "error_type": result.get("error_type", ""),
            "error_summary": result.get("error_summary", ""),
            "traceback_excerpt": result.get("traceback_excerpt", ""),
            "log_excerpt": result.get("log_excerpt", ""),
            "log_path": result.get("log_path", ""),
        }
        failures = memory.setdefault("recent_training_failures", [])
        failures.append(training_failure)
        memory["recent_training_failures"] = failures[-12:]
        if training_failure["error_summary"]:
            family.setdefault("known_issues", []).append(
                f"round {round_id}: training crash for {family_name} | {training_failure['error_summary']}"
            )
            family["known_issues"] = family["known_issues"][-20:]

    memory["next_hypotheses"] = action.get("next_hypotheses", [])[:12]
    family["next_steps"] = action.get("next_hypotheses", [])[:8]
    return memory


def write_round_summary(
    round_id: int,
    action: dict[str, Any],
    result: dict[str, str],
    touched: list[str],
    patch_path: Path | None,
    round_ctx: dict[str, Any],
) -> None:
    from research.git_ops import current_git_branch, commit_message_for_round

    tier = "full" if round_ctx.get("full_result") is not None else action.get("experiment_tier")
    summary = {
        "round": round_id,
        "timestamp": int(time.time()),
        "started_at": round_ctx.get("started_at"),
        "ended_at": round_ctx.get("ended_at"),
        "duration_sec": round_ctx.get("duration_sec"),
        "family_name": round_ctx.get("family_name"),
        "family_stage": round_ctx.get("family_stage"),
        "session_id": round_ctx.get("session_id"),
        "timeline_start_event_id": round_ctx.get("timeline_start_event_id"),
        "timeline_end_event_id": round_ctx.get("timeline_end_event_id"),
        "timeline_event_ids": round_ctx.get("timeline_event_ids", []),
        "repair_attempts": round_ctx.get("repair_attempts", []),
        "experiment_branch": current_git_branch(),
        "experiment_commit_message": commit_message_for_round(round_id, action, result, tier or "proxy"),
        "proxy_result": round_ctx.get("proxy_result"),
        "full_result": round_ctx.get("full_result"),
        "action": action,
        "result": result,
        "touched_files": touched,
        "patch_path": str(patch_path) if patch_path is not None else None,
    }
    save_json(ROUND_SUMMARIES_DIR / f"round_{round_id:04d}.json", summary)


def update_agent_state(
    state: dict[str, Any],
    result: dict[str, str],
    action: dict[str, Any],
    action_path: Path,
    patch_path: Path | None,
    is_new_family: bool,
) -> None:
    agent = state.setdefault("agent", {})
    agent["round_index"] = int(agent.get("round_index", 0)) + 1
    decision = result.get("decision", "")
    if decision == "crash":
        agent["consecutive_crashes"] = int(agent.get("consecutive_crashes", 0)) + 1
    else:
        agent["consecutive_crashes"] = 0
        agent["crash_resets"] = 0  # Reset the reset counter on any non-crash
    if decision in {"promote", "keep"}:
        agent["consecutive_no_improve"] = 0
    elif decision == "policy_reject":
        pass  # Policy blocks are not real experiments; don't count toward no-improve
    else:
        agent["consecutive_no_improve"] = int(agent.get("consecutive_no_improve", 0)) + 1
    if is_new_family:
        agent["rounds_since_new_family"] = 0
    else:
        agent["rounds_since_new_family"] = int(agent.get("rounds_since_new_family", 0)) + 1
    agent["last_action_file"] = str(action_path)
    agent["last_patch_file"] = str(patch_path) if patch_path is not None else None


def mark_hints_applied(round_id: int) -> None:
    hints = load_operator_hints()
    changed = False
    for hint in hints:
        if hint.get("status") != "pending":
            continue
        if hint.get("scope") == "next_round":
            hint["status"] = "applied"
            hint["applied_at"] = int(time.time())
            hint["applied_in_round"] = round_id
            append_timeline_event(
                "hint_applied",
                round=round_id,
                tier=None,
                run_name=None,
                family_name=None,
                family_stage=None,
                status="applied",
                decision=None,
                payload=hint,
            )
            changed = True
    if changed:
        save_operator_hints(hints)
