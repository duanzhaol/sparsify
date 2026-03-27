"""Rebuild runtime decision state from history using compatibility labels.

This script performs a "medium cleanup":
- keep raw round_summaries / logs / timeline untouched
- rebuild frontier and family runtime state from compatible families only
- mark explicitly incompatible families as filtered_incompatible
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .compatibility import (
    INCOMPATIBLE,
    is_compatible_label,
    parse_compatibility_registry,
)
from .controller import frontier_key
from .types import BASE_ENV_DEFAULTS, HISTORY_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild AutoResearch runtime state")
    parser.add_argument("--history-dir", default=str(HISTORY_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-active-session", action="store_true")
    args = parser.parse_args()

    history_dir = Path(args.history_dir)
    repo_root = Path(__file__).resolve().parents[2]
    prior_path = Path(__file__).resolve().parent / "prior_research_history.md"
    prior_text = prior_path.read_text().strip() if prior_path.exists() else ""
    registry = parse_compatibility_registry(prior_text)

    state_path = history_dir / "state.json"
    memory_path = history_dir / "memory.json"
    frontier_path = history_dir / "frontier.json"
    round_dir = history_dir / "round_summaries"
    compat_path = history_dir / "compatibility_registry.json"

    state = _load_json(state_path, {})
    memory = _load_json(memory_path, {})
    summaries = [
        _load_json(path, {})
        for path in sorted(round_dir.glob("round_*.json"))
    ]

    rebuilt = rebuild_runtime_state(summaries, registry, memory)

    state["frontier"] = rebuilt["frontier"]
    state.setdefault("agent", {})
    state["agent"]["round_index"] = rebuilt["latest_round"]
    state["agent"]["consecutive_crashes"] = 0
    state["agent"]["consecutive_no_improve"] = 0
    state["agent"]["rounds_since_new_family"] = 0
    state["agent"]["crash_resets"] = 0
    if not args.keep_active_session:
        state["agent"]["active_session_id"] = None
        state["agent"]["active_session_started_at"] = None
        state["agent"]["active_session_rounds"] = 0
        state["agent"]["active_session_status"] = "closed"

    if args.dry_run:
        print(_json({
            "frontier_entries": len(rebuilt["frontier"]),
            "latest_round": rebuilt["latest_round"],
            "active_families": rebuilt["active_families"],
            "filtered_incompatible": rebuilt["filtered_incompatible"],
        }))
        return 0

    _save_json(state_path, state)
    _save_json(frontier_path, rebuilt["frontier"])
    _save_json(memory_path, rebuilt["memory"])
    _save_json(compat_path, registry)

    print(f"Rebuilt frontier entries: {len(rebuilt['frontier'])}")
    print(f"Latest round: {rebuilt['latest_round']}")
    print(f"Active compatible families: {', '.join(rebuilt['active_families']) or '(none)'}")
    print(f"Filtered incompatible families: {', '.join(rebuilt['filtered_incompatible']) or '(none)'}")
    print(f"Wrote: {state_path}")
    print(f"Wrote: {frontier_path}")
    print(f"Wrote: {memory_path}")
    print(f"Wrote: {compat_path}")
    print(f"Repo root: {repo_root}")
    return 0


def rebuild_runtime_state(
    summaries: list[dict[str, Any]],
    registry: dict[str, str],
    existing_memory: dict[str, Any],
) -> dict[str, Any]:
    frontier: dict[str, Any] = {}
    families: dict[str, dict[str, Any]] = {}
    recent_rounds: list[dict[str, Any]] = []
    recent_insights: list[str] = []
    performance_findings: list[str] = []
    recent_training_failures: list[dict[str, Any]] = []
    active_family_names: set[str] = set()
    filtered_incompatible: set[str] = set()
    latest_round = 0

    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        round_id = int(summary.get("round") or 0)
        latest_round = max(latest_round, round_id)
        action = summary.get("action", {})
        result = summary.get("result", {})
        family_name = str(summary.get("family_name") or action.get("family_name") or result.get("architecture") or "").lower()
        compat_label = registry.get(family_name, "unknown")
        if compat_label == INCOMPATIBLE:
            filtered_incompatible.add(family_name)

        family = families.setdefault(family_name or "unknown", {
            "status": "discarded",
            "design_hypothesis": action.get("hypothesis", ""),
            "tested_configs": [],
            "known_issues": [],
            "next_steps": [],
            "best_fvu": None,
            "last_round": round_id,
            "compatibility": compat_label,
        })
        family["last_round"] = round_id
        family["design_hypothesis"] = action.get("hypothesis", family.get("design_hypothesis", ""))
        family["next_steps"] = list(action.get("next_hypotheses", []))[:8]
        family["compatibility"] = compat_label

        if result.get("decision") != "policy_reject":
            family["tested_configs"].append({
                "round": round_id,
                "stage": summary.get("family_stage"),
                "k": result.get("k"),
                "decision": result.get("decision"),
                "val_fvu": result.get("val_fvu"),
                "run_health": result.get("run_health"),
            })
            family["tested_configs"] = family["tested_configs"][-20:]

        if result.get("decision") == "crash":
            failure = {
                "round": round_id,
                "family_name": family_name,
                "change_type": action.get("change_type"),
                "primary_variable": action.get("primary_variable"),
                "hypothesis": action.get("hypothesis"),
                "termination_reason": result.get("termination_reason"),
                "error_type": result.get("error_type") or "",
                "error_summary": result.get("error_summary") or "",
                "traceback_excerpt": result.get("traceback_excerpt") or "",
                "log_excerpt": result.get("log_excerpt") or "",
                "log_path": result.get("log_path") or "",
            }
            if is_compatible_label(compat_label):
                recent_training_failures.append(failure)

        if not is_compatible_label(compat_label):
            family["status"] = "filtered_incompatible"
            continue

        if _is_successful_metric(result):
            current_best = family.get("best_fvu")
            fvu = float(result["val_fvu"])
            if current_best is None or fvu < float(current_best):
                family["best_fvu"] = fvu

            recent_rounds.append(_recent_round_entry(summary))
            recent_insights.append(_outcome_line(summary))
            if result.get("run_health") == "perf_regression":
                performance_findings.append(_outcome_line(summary))

            round_id = summary.get("round", "unknown")
            key = frontier_key(round_id)
            config = _effective_config(summary)
            arch = str(result.get("architecture") or config.get("architecture") or family_name).lower()
            k_val = int(result["k"])
            ef_val = int(result.get("expansion_factor") or 12)
            new_entry = {
                "k": k_val,
                "ef": ef_val,
                "fvu": fvu,
                "architecture": arch,
                "commit": result.get("head_commit") or "",
                "config": config,
                "checkpoint": result.get("checkpoint"),
                "peak_memory_gb": result.get("peak_memory_gb"),
            }
            # Compute selection_cost
            from .controller import _estimate_cost_from_entry
            new_entry["selection_cost"] = _estimate_cost_from_entry(new_entry)
            frontier[key] = new_entry

    # Pareto cleanup: remove dominated points
    from .controller import _pareto_dominates, _entry_to_point
    keys_to_remove = []
    frontier_items = list(frontier.items())
    for i, (ki, ei) in enumerate(frontier_items):
        if not isinstance(ei, dict):
            continue
        pi = _entry_to_point(ei)
        for j, (kj, ej) in enumerate(frontier_items):
            if i == j or not isinstance(ej, dict):
                continue
            pj = _entry_to_point(ej)
            if _pareto_dominates(pj, pi):
                keys_to_remove.append(ki)
                break
    for k in keys_to_remove:
        frontier.pop(k, None)

    for entry in frontier.values():
        family_name = str(entry.get("config", {}).get("family_name") or entry.get("architecture") or "").lower()
        active_family_names.add(family_name)

    for family_name, family in families.items():
        compat_label = family.get("compatibility", "unknown")
        if not is_compatible_label(compat_label):
            family["status"] = "filtered_incompatible"
            continue
        if family_name in active_family_names:
            family["status"] = "active"
        elif family.get("best_fvu") is not None:
            family["status"] = "archived"
        else:
            family["status"] = "discarded"

    rebuilt_memory = dict(existing_memory)
    rebuilt_memory["architecture_families"] = families
    rebuilt_memory["recent_rounds"] = recent_rounds[-12:]
    rebuilt_memory["recent_insights"] = recent_insights[-40:]
    rebuilt_memory["performance_findings"] = performance_findings[-20:]
    rebuilt_memory["recent_training_failures"] = recent_training_failures[-12:]
    rebuilt_memory["next_hypotheses"] = _derive_next_hypotheses(frontier, families)
    rebuilt_memory["current_focus"] = _derive_current_focus(frontier)

    return {
        "frontier": frontier,
        "memory": rebuilt_memory,
        "latest_round": latest_round,
        "active_families": sorted(active_family_names),
        "filtered_incompatible": sorted(filtered_incompatible),
    }


def _effective_config(summary: dict[str, Any]) -> dict[str, Any]:
    config = {
        "architecture": BASE_ENV_DEFAULTS["ARCHITECTURE"],
        "expansion_factor": int(BASE_ENV_DEFAULTS["EXPANSION_FACTOR"]),
        "k": int(BASE_ENV_DEFAULTS["K"]),
        "optimizer": BASE_ENV_DEFAULTS["OPTIMIZER"],
        "lr": BASE_ENV_DEFAULTS["LR"],
        "hookpoints": BASE_ENV_DEFAULTS["HOOKPOINTS"],
        "batch_size": int(BASE_ENV_DEFAULTS["BATCH_SIZE"]),
        "grad_acc_steps": int(BASE_ENV_DEFAULTS["GRAD_ACC_STEPS"]),
        "micro_acc_steps": int(BASE_ENV_DEFAULTS["MICRO_ACC_STEPS"]),
        "auxk_alpha": float(BASE_ENV_DEFAULTS["AUXK_ALPHA"]),
        "dead_feature_threshold": int(BASE_ENV_DEFAULTS["DEAD_FEATURE_THRESHOLD"]),
        "use_hadamard": BASE_ENV_DEFAULTS["USE_HADAMARD"] not in {"0", "false", "False"},
    }

    action = summary.get("action", {})
    for item in action.get("env_overrides", []):
        key = str(item.get("key") or "")
        value = item.get("value")
        if key == "ARCHITECTURE":
            config["architecture"] = str(value).lower()
        elif key == "EXPANSION_FACTOR":
            config["expansion_factor"] = int(value)
        elif key == "K":
            config["k"] = int(value)
        elif key == "OPTIMIZER":
            config["optimizer"] = str(value)
        elif key == "LR":
            config["lr"] = str(value)
        elif key == "HOOKPOINTS":
            config["hookpoints"] = str(value)
        elif key == "BATCH_SIZE":
            config["batch_size"] = int(value)
        elif key == "GRAD_ACC_STEPS":
            config["grad_acc_steps"] = int(value)
        elif key == "MICRO_ACC_STEPS":
            config["micro_acc_steps"] = int(value)
        elif key == "AUXK_ALPHA":
            config["auxk_alpha"] = float(value)
        elif key == "DEAD_FEATURE_THRESHOLD":
            config["dead_feature_threshold"] = int(value)
        elif key == "USE_HADAMARD":
            config["use_hadamard"] = str(value).lower() not in {"0", "false"}

    family_name = str(summary.get("family_name") or action.get("family_name") or config["architecture"]).lower()
    config["family_name"] = family_name
    config["family_stage"] = summary.get("family_stage") or action.get("family_stage") or "mainline"
    result = summary.get("result", {})
    if result.get("architecture"):
        config["architecture"] = str(result["architecture"]).lower()
    if result.get("k") not in (None, "", "None"):
        config["k"] = int(result["k"])
    if result.get("expansion_factor") not in (None, "", "None"):
        config["expansion_factor"] = int(result["expansion_factor"])
    return config


def _is_successful_metric(result: dict[str, Any]) -> bool:
    if result.get("status") != "ok":
        return False
    try:
        float(result.get("val_fvu"))
    except (TypeError, ValueError):
        return False
    return result.get("k") not in (None, "", "None")


def _recent_round_entry(summary: dict[str, Any]) -> dict[str, Any]:
    action = summary.get("action", {})
    result = summary.get("result", {})
    return {
        "round": summary.get("round"),
        "hypothesis": action.get("hypothesis"),
        "change_type": action.get("change_type"),
        "expected_win": action.get("expected_win"),
        "family_name": str(summary.get("family_name") or "").lower(),
        "family_stage": summary.get("family_stage"),
        "decision": result.get("decision"),
        "val_fvu": result.get("val_fvu"),
        "observed_fvu": result.get("observed_fvu"),
        "k": result.get("k"),
        "architecture": str(result.get("architecture") or "").lower(),
        "touched_files": summary.get("touched_files", []),
        "run_health": result.get("run_health"),
        "termination_reason": result.get("termination_reason"),
        "tokens_per_sec": result.get("tokens_per_sec"),
        "baseline_ratio": result.get("baseline_ratio"),
    }


def _outcome_line(summary: dict[str, Any]) -> str:
    action = summary.get("action", {})
    result = summary.get("result", {})
    family_name = str(summary.get("family_name") or "").lower()
    return (
        f"r{summary.get('round','?')} {family_name} "
        f"{action.get('change_type','?')} k{result.get('k','?')} "
        f"-> {result.get('decision','?')} fvu={result.get('val_fvu','?')}"
    )


def _derive_next_hypotheses(
    frontier: dict[str, Any],
    families: dict[str, dict[str, Any]],
) -> list[str]:
    best = None
    best_fvu = float("inf")
    for entry in frontier.values():
        try:
            fvu = float(entry.get("fvu", float("inf")))
        except (TypeError, ValueError):
            continue
        if fvu < best_fvu:
            best_fvu = fvu
            best = entry
    if best is None:
        return []

    family_name = str(best.get("config", {}).get("family_name") or best.get("architecture") or "").lower()
    family = families.get(family_name, {})
    next_steps = [str(s) for s in family.get("next_steps", []) if s]
    if next_steps:
        return next_steps[:8]
    return [
        f"继续围绕兼容主线 {family_name} 推进，不要重新启用被过滤的不兼容 family。",
        "优先比较兼容 family 的结构收益与运行代价，再决定是否继续扩展该分支。",
    ]


def _derive_current_focus(frontier: dict[str, Any]) -> str:
    if not frontier:
        return "兼容 frontier 已重建，但当前没有可用兼容点。"
    best = min(
        frontier.values(),
        key=lambda item: float(item.get("fvu", float("inf"))),
    )
    family_name = str(best.get("config", {}).get("family_name") or best.get("architecture") or "").lower()
    return f"兼容 frontier 已重建；当前主线优先关注 {family_name}。"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _save_json(path: Path, data: Any) -> None:
    path.write_text(_json(data) + "\n")


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
