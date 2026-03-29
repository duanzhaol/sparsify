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
from .config_resolution import config_from_round_summary, summary_invalid_reason
from .controller import compact_frontier, frontier_key, _extract_extra_config
from .target_profile import resolve_target_profile
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
        family["compatibility"] = compat_label
        invalid_reason = summary_invalid_reason(summary)

        if invalid_reason is None:
            family["last_round"] = round_id
            family["design_hypothesis"] = action.get("hypothesis", family.get("design_hypothesis", ""))
            family["next_steps"] = list(action.get("next_hypotheses", []))[:8]

        if invalid_reason is None and result.get("decision") != "policy_reject":
            family["tested_configs"].append({
                "round": round_id,
                "stage": summary.get("family_stage"),
                "k": result.get("k"),
                "decision": result.get("decision"),
                "val_fvu": result.get("val_fvu"),
                "run_health": result.get("run_health"),
            })
            family["tested_configs"] = family["tested_configs"][-20:]

        if invalid_reason is None and result.get("decision") == "crash":
            failure = {
                "round": round_id,
                "family_name": family_name,
                "change_type": action.get("change_type"),
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

        if invalid_reason is not None:
            family.setdefault("known_issues", []).append(
                f"round {round_id}: invalid result ignored | {invalid_reason}"
            )
            family["known_issues"] = family["known_issues"][-20:]
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
            # Compute cost breakdown (selection + deployment + total)
            from .compatibility import compute_selection_cost
            target_profile = resolve_target_profile(config)
            cost = compute_selection_cost(
                arch,
                k=k_val,
                ef=ef_val,
                d_in=target_profile.d_in,
                n_output=target_profile.n_output,
                extra_config=_extract_extra_config(config),
            )
            if "error" not in cost:
                new_entry["selection_cost"] = float(cost["total_accesses"])
                new_entry["deployment_accesses"] = float(cost.get("deployment_accesses", 0))
                new_entry["deployment_ratio"] = cost.get("deployment_ratio")
                new_entry["total_cost"] = float(cost.get("combined_accesses", 0))
            else:
                from .controller import _estimate_total_cost_from_entry
                new_entry["total_cost"] = _estimate_total_cost_from_entry(new_entry)
            new_entry["target_profile"] = target_profile.to_dict()
            new_entry["cost_model_label"] = target_profile.cost_model_label
            new_entry["metric_version"] = "total_cost_v1"
            frontier[key] = new_entry

    # Replay frontier entries through the same near-duplicate + dominance
    # semantics used by the live controller, so rebuilt state matches runtime.
    frontier = compact_frontier(frontier)

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
    resolved_env = config_from_round_summary(summary)
    if resolved_env is not None:
        config = {
            "architecture": str(resolved_env.get("ARCHITECTURE", BASE_ENV_DEFAULTS["ARCHITECTURE"])).lower(),
            "expansion_factor": int(resolved_env.get("EXPANSION_FACTOR", BASE_ENV_DEFAULTS["EXPANSION_FACTOR"])),
            "k": int(resolved_env.get("K", BASE_ENV_DEFAULTS["K"])),
            "optimizer": str(resolved_env.get("OPTIMIZER", BASE_ENV_DEFAULTS["OPTIMIZER"])),
            "lr": str(resolved_env.get("LR", BASE_ENV_DEFAULTS["LR"])),
            "hookpoints": str(resolved_env.get("HOOKPOINTS", BASE_ENV_DEFAULTS["HOOKPOINTS"])),
            "batch_size": int(resolved_env.get("BATCH_SIZE", BASE_ENV_DEFAULTS["BATCH_SIZE"])),
            "grad_acc_steps": int(resolved_env.get("GRAD_ACC_STEPS", BASE_ENV_DEFAULTS["GRAD_ACC_STEPS"])),
            "micro_acc_steps": int(resolved_env.get("MICRO_ACC_STEPS", BASE_ENV_DEFAULTS["MICRO_ACC_STEPS"])),
            "auxk_alpha": float(resolved_env.get("AUXK_ALPHA", BASE_ENV_DEFAULTS["AUXK_ALPHA"])),
            "dead_feature_threshold": int(resolved_env.get("DEAD_FEATURE_THRESHOLD", BASE_ENV_DEFAULTS["DEAD_FEATURE_THRESHOLD"])),
            "use_hadamard": str(resolved_env.get("USE_HADAMARD", BASE_ENV_DEFAULTS["USE_HADAMARD"])).lower() not in {"0", "false"},
        }
        optional_map = {
            "TRUNK_RANK": ("trunk_rank", int),
            "NUM_CODES": ("num_codes", int),
            "STAGE1_RATIO": ("stage1_ratio", float),
            "FACTORIZED_HIDDEN_DIM": ("factorized_hidden_dim", int),
            "NUM_EXPERTS": ("num_experts", int),
        }
        for env_key, (cfg_key, caster) in optional_map.items():
            value = resolved_env.get(env_key)
            if value not in (None, ""):
                config[cfg_key] = caster(value)
        action = summary.get("action", {})
        config["family_name"] = str(summary.get("family_name") or action.get("family_name") or config["architecture"]).lower()
        config["family_stage"] = summary.get("family_stage") or action.get("family_stage") or "mainline"
        target_profile = resolve_target_profile(config)
        config["target_profile"] = target_profile.to_dict()
        config["cost_model_label"] = target_profile.cost_model_label
        config["d_in"] = target_profile.d_in
        config["n_output"] = target_profile.n_output
        return config

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
        elif key == "NUM_EXPERTS":
            config["num_experts"] = int(value)

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
    target_profile = resolve_target_profile(config)
    config["target_profile"] = target_profile.to_dict()
    config["cost_model_label"] = target_profile.cost_model_label
    config["d_in"] = target_profile.d_in
    config["n_output"] = target_profile.n_output
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
