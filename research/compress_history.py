#!/usr/bin/env python3
"""Build an agent-driven, runtime-compatible compact history snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = REPO_ROOT / "research"
DEFAULT_SRC_HISTORY = RESEARCH_DIR / "history"
DEFAULT_OUT_ROOT = RESEARCH_DIR / "history_compacted"
SCHEMA_PATH = RESEARCH_DIR / "compaction_action.schema.json"

REQUIRED_FILES = [
    "state.json",
    "frontier.json",
    "memory.json",
    "results.tsv",
    "session_brief.json",
    "operator_hints.json",
    "timeline.jsonl",
    "current_status.json",
    "compaction_plan.json",
    "compression_manifest.json",
]

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

DEFAULT_AGENT_TIMEOUT_SEC = 10 * 60
MAX_CONTEXT_RESULTS = 10
MAX_CONTEXT_ROUNDS = 80


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_results(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in RESULT_COLUMNS})


def sorted_round_summary_paths(round_dir: Path) -> list[Path]:
    return sorted(round_dir.glob("round_*.json"))


def parse_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def dedupe_keep_order(items: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False) if not isinstance(item, str) else item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def trim_text(value: Any, limit: int = 1600) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[: limit - 32].rstrip() + "\n[truncated for compact history]"


def compact_failure_entry(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    return {
        "round": parse_int(item.get("round")),
        "family_name": item.get("family_name"),
        "change_type": item.get("change_type"),
        "error_type": item.get("error_type"),
        "termination_reason": item.get("termination_reason"),
        "error_summary": trim_text(
            item.get("error_summary") or item.get("stderr_excerpt") or item.get("traceback_excerpt"),
            limit=260,
        ),
    }


def compact_result_blob(blob: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(blob, dict):
        return blob
    compact = {
        "experiment_id": blob.get("experiment_id"),
        "status": blob.get("status"),
        "decision": blob.get("decision"),
        "val_fvu": blob.get("val_fvu"),
        "k": blob.get("k"),
        "architecture": blob.get("architecture"),
        "termination_reason": blob.get("termination_reason"),
        "run_health": blob.get("run_health"),
        "tokens_per_sec": blob.get("tokens_per_sec"),
        "baseline_ratio": blob.get("baseline_ratio"),
        "proxy_mode": blob.get("proxy_mode"),
        "evaluation_basis": blob.get("evaluation_basis"),
        "error_type": blob.get("error_type"),
        "error_summary": trim_text(blob.get("error_summary"), limit=260),
        "log_excerpt": trim_text(blob.get("log_excerpt"), limit=260),
        "traceback_excerpt": trim_text(blob.get("traceback_excerpt"), limit=260),
        "log_path": blob.get("log_path"),
        "metrics_path": blob.get("metrics_path"),
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def compact_round_summary(summary: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "round": summary.get("round"),
        "timestamp": summary.get("timestamp"),
        "started_at": summary.get("started_at"),
        "ended_at": summary.get("ended_at"),
        "duration_sec": summary.get("duration_sec"),
        "family_name": summary.get("family_name"),
        "family_stage": summary.get("family_stage"),
        "proxy_result": compact_result_blob(summary.get("proxy_result")),
        "full_result": compact_result_blob(summary.get("full_result")),
        "result": compact_result_blob(summary.get("result")),
        "touched_files": summary.get("touched_files", []),
        "patch_path": summary.get("patch_path"),
    }
    action = summary.get("action")
    if isinstance(action, dict):
        compact_action = {
            "command": action.get("command"),
            "hypothesis": trim_text(action.get("hypothesis"), limit=480),
            "summary": trim_text(action.get("summary"), limit=320),
            "change_type": action.get("change_type"),
            "experiment_tier": action.get("experiment_tier"),
            "expected_win": action.get("expected_win"),
            "family_name": action.get("family_name"),
            "family_stage": action.get("family_stage"),
            "self_review": trim_text(action.get("self_review"), limit=240),
            "needs_sanity": action.get("needs_sanity"),
            "env_overrides": action.get("env_overrides", []),
            "touched_files": action.get("touched_files", []),
            "notes_to_memory": [trim_text(note, limit=180) for note in action.get("notes_to_memory", [])[:4]],
            "next_hypotheses": [trim_text(note, limit=180) for note in action.get("next_hypotheses", [])[:3]],
            "primary_variable": action.get("primary_variable"),
        }
        compact["action"] = {key: value for key, value in compact_action.items() if value not in (None, "", [], {})}
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def frontier_digest(frontier: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    if isinstance(frontier, dict):
        iterable = frontier.values()
    else:
        iterable = frontier
    for point in iterable:
        if not isinstance(point, dict):
            continue
        config = point.get("config", {}) if isinstance(point.get("config"), dict) else {}
        points.append({
            "k": parse_int(point.get("k") if "k" in point else config.get("k")),
            "fvu": parse_float(point.get("fvu")),
            "architecture": point.get("architecture") or config.get("architecture"),
            "expansion_factor": parse_int(config.get("expansion_factor")),
            "optimizer": config.get("optimizer"),
            "lr": config.get("lr"),
            "auxk_alpha": config.get("auxk_alpha"),
            "family_name": config.get("family_name"),
            "family_stage": config.get("family_stage"),
            "tier": point.get("tier") or config.get("tier"),
            "checkpoint": point.get("checkpoint"),
        })
    return [pt for pt in points if pt.get("k") is not None and pt.get("fvu") is not None]


def frontier_context_digest(frontier: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "k": item.get("k"),
            "fvu": item.get("fvu"),
            "architecture": item.get("architecture"),
            "tier": item.get("tier"),
        }
        for item in frontier_digest(frontier)
    ]


def summarize_results(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keys = ["experiment_id", "tier", "status", "decision", "val_fvu", "k", "architecture", "description"]
    return [{k: row.get(k, "") for k in keys} for row in rows]


def extract_override_map(action: dict[str, Any]) -> dict[str, str]:
    overrides = action.get("env_overrides", [])
    out: dict[str, str] = {}
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            out[str(key)] = str(value)
        return out
    if isinstance(overrides, list):
        for item in overrides:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            value = item.get("value")
            if key is not None and value is not None:
                out[str(key)] = str(value)
    return out


def collect_frontier_checkpoints(state: dict[str, Any]) -> set[str]:
    checkpoints: set[str] = set()
    for frontier_name in ("frontier", "proxy_frontier", "full_frontier"):
        frontier = state.get(frontier_name, {})
        if not isinstance(frontier, dict):
            continue
        for point in frontier.values():
            if not isinstance(point, dict):
                continue
            checkpoint = point.get("checkpoint")
            if checkpoint:
                checkpoints.add(str(checkpoint))
    return checkpoints


def determine_best_family(memory: dict[str, Any]) -> str | None:
    best_family: str | None = None
    best_fvu = float("inf")
    families = memory.get("architecture_families", {})

    for family_name, family in families.items():
        if not isinstance(family, dict) or family.get("status") not in {"active", "promoted"}:
            continue
        fvu = parse_float(family.get("best_full_fvu"))
        if fvu is not None and fvu < best_fvu:
            best_fvu = fvu
            best_family = str(family_name)

    if best_family is not None:
        return best_family

    for family_name, family in families.items():
        if not isinstance(family, dict) or family.get("status") not in {"active", "promoted"}:
            continue
        fvu = parse_float(family.get("best_proxy_fvu"))
        if fvu is not None and fvu < best_fvu:
            best_fvu = fvu
            best_family = str(family_name)
    return best_family


def choose_candidate_rounds(
    summaries: list[dict[str, Any]],
    state: dict[str, Any],
    memory: dict[str, Any],
    keep_last_rounds: int,
) -> tuple[list[int], dict[str, list[int]]]:
    key_rounds: set[int] = set()
    reasons: dict[str, list[int]] = defaultdict(list)
    frontier_checkpoints = collect_frontier_checkpoints(state)
    best_family = determine_best_family(memory)
    family_first_round: dict[str, int] = {}
    family_best_proxy: dict[str, tuple[float, int]] = {}
    family_best_full: dict[str, tuple[float, int]] = {}
    family_last_failure: dict[str, int] = {}
    family_param_coverage: set[tuple[str, str, str]] = set()

    for summary in summaries:
        round_id = int(summary.get("round", 0))
        action = summary.get("action", {}) if isinstance(summary.get("action"), dict) else {}
        result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
        proxy_result = summary.get("proxy_result", {}) if isinstance(summary.get("proxy_result"), dict) else {}
        full_result = summary.get("full_result", {}) if isinstance(summary.get("full_result"), dict) else {}
        family = str(summary.get("family_name") or action.get("family_name") or "").lower()
        change_type = str(action.get("change_type") or "")
        decision = str(result.get("decision") or proxy_result.get("decision") or "")
        termination_reason = str(result.get("termination_reason") or proxy_result.get("termination_reason") or "")
        touched = summary.get("touched_files") or []

        if family and family not in family_first_round:
            family_first_round[family] = round_id
            key_rounds.add(round_id)
            reasons["family_first_seen"].append(round_id)

        if change_type not in ("param_only", "no_change") or touched or summary.get("patch_path"):
            key_rounds.add(round_id)
            reasons["code_or_patch"].append(round_id)

        if decision == "crash" or termination_reason in {
            "sanity_failed",
            "invalid_env_overrides",
            "identical_to_baseline",
            "first_step_timeout",
            "throughput_too_low",
        }:
            key_rounds.add(round_id)
            reasons["failure_or_terminal"].append(round_id)
            if family:
                family_last_failure[family] = round_id

        for blob in (proxy_result, full_result):
            checkpoint = blob.get("checkpoint")
            if checkpoint and str(checkpoint) in frontier_checkpoints:
                key_rounds.add(round_id)
                reasons["frontier_checkpoint"].append(round_id)
            fvu = parse_float(blob.get("val_fvu"))
            if fvu is None or not family:
                continue
            if str(blob.get("tier") or "").lower() == "full" or blob is full_result:
                current = family_best_full.get(family)
                if current is None or fvu < current[0]:
                    family_best_full[family] = (fvu, round_id)
            else:
                current = family_best_proxy.get(family)
                if current is None or fvu < current[0]:
                    family_best_proxy[family] = (fvu, round_id)

        overrides = extract_override_map(action)
        for param_key in ("LR", "AUXK_ALPHA", "EXPANSION_FACTOR", "K"):
            if param_key not in overrides or not family or family != best_family:
                continue
            token = (family, param_key, overrides[param_key])
            if token not in family_param_coverage:
                family_param_coverage.add(token)
                key_rounds.add(round_id)
                reasons[f"param_coverage_{param_key.lower()}"].append(round_id)

    for _, round_id in family_best_proxy.values():
        key_rounds.add(round_id)
        reasons["family_best_proxy"].append(round_id)
    for _, round_id in family_best_full.values():
        key_rounds.add(round_id)
        reasons["family_best_full"].append(round_id)
    for round_id in family_last_failure.values():
        key_rounds.add(round_id)
        reasons["family_last_failure"].append(round_id)

    all_rounds = sorted(int(summary.get("round", 0)) for summary in summaries if summary.get("round") is not None)
    for round_id in all_rounds[-keep_last_rounds:]:
        key_rounds.add(round_id)
        reasons["recent_rounds"].append(round_id)

    for _, family in memory.get("architecture_families", {}).items():
        if not isinstance(family, dict):
            continue
        status = family.get("status")
        if status not in {"active", "promoted", "incubating"}:
            continue
        last_round = parse_int(family.get("last_round"))
        if last_round:
            key_rounds.add(last_round)
            reasons["family_last_round"].append(last_round)

    final_rounds = sorted(key_rounds)
    deduped_reasons = {key: sorted(set(value)) for key, value in reasons.items() if value}
    return final_rounds, deduped_reasons


def summarize_recent_failures(all_summaries: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for summary in all_summaries:
        result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
        if result.get("decision") != "crash":
            continue
        failures.append({
            "round": parse_int(summary.get("round")),
            "family_name": summary.get("family_name"),
            "change_type": (summary.get("action") or {}).get("change_type") if isinstance(summary.get("action"), dict) else None,
            "termination_reason": result.get("termination_reason"),
            "error_type": result.get("error_type"),
            "error_summary": trim_text(result.get("error_summary"), limit=220),
        })
    return failures[-limit:]


def build_code_history_summary(all_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, dict[str, Any]] = {}
    repeated_backend_failures: list[int] = []
    for summary in all_summaries:
        round_id = parse_int(summary.get("round"))
        if round_id is None:
            continue
        if summary.get("error"):
            repeated_backend_failures.append(round_id)
            continue
        action = summary.get("action", {}) if isinstance(summary.get("action"), dict) else {}
        family = str(summary.get("family_name") or action.get("family_name") or "").lower()
        if not family:
            continue
        change_type = str(action.get("change_type") or "")
        if change_type in {"param_only", "no_change"} and not summary.get("touched_files"):
            continue
        entry = by_family.setdefault(family, {
            "introduced_round": round_id,
            "code_rounds": [],
            "change_types": [],
            "touched_files": [],
            "patch_rounds": [],
        })
        entry["introduced_round"] = min(entry["introduced_round"], round_id)
        entry["code_rounds"].append(round_id)
        if change_type:
            entry["change_types"].append(change_type)
        for path in summary.get("touched_files", []) or []:
            if path not in entry["touched_files"]:
                entry["touched_files"].append(path)
        if summary.get("patch_path"):
            entry["patch_rounds"].append(round_id)

    for entry in by_family.values():
        entry["code_rounds"] = sorted(set(entry["code_rounds"]))
        entry["patch_rounds"] = sorted(set(entry["patch_rounds"]))
        entry["change_types"] = sorted(set(entry["change_types"]))

    return {
        "families": by_family,
        "aggregated_backend_failures": {
            "count": len(repeated_backend_failures),
            "rounds": sorted(repeated_backend_failures),
        },
    }


def build_research_takeaways(state: dict[str, Any], memory: dict[str, Any], all_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    best_family = determine_best_family(memory)
    families = memory.get("architecture_families", {})
    exhausted = sorted(
        name for name, family in families.items()
        if isinstance(family, dict) and family.get("status") == "archived"
    )
    blocked = []
    for summary in all_summaries:
        result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
        if result.get("decision") != "crash":
            continue
        termination_reason = str(result.get("termination_reason") or "")
        error_summary = str(result.get("error_summary") or "")
        if termination_reason in {"sanity_failed", "invalid_env_overrides"} or "Unknown architecture" in error_summary:
            blocked.append({
                "round": parse_int(summary.get("round")),
                "family_name": summary.get("family_name"),
                "termination_reason": termination_reason,
                "error_summary": trim_text(error_summary, limit=220),
            })
    return {
        "best_mainline_family": best_family,
        "best_full_points": frontier_digest(state.get("pareto_full_frontier", [])),
        "best_proxy_points": frontier_digest(state.get("pareto_proxy_frontier", [])),
        "exhausted_families": exhausted,
        "blocked_families": blocked[-8:],
        "open_blockers": summarize_recent_failures(all_summaries, limit=6),
        "next_search_biases": memory.get("next_hypotheses", [])[:6],
    }


def _coerce_frontier_point(key: str, point: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(point, dict):
        return None
    try:
        fvu = float(point.get("fvu"))
        k = int(point.get("config", {}).get("k", key))
    except (TypeError, ValueError):
        return None
    return {
        "k": k,
        "fvu": fvu,
        "architecture": point.get("architecture"),
        "commit": point.get("commit"),
        "config": point.get("config", {}),
        "tier": point.get("tier"),
        "checkpoint": point.get("checkpoint"),
        "peak_memory_gb": point.get("peak_memory_gb"),
    }


def pareto_dominates(a: dict[str, Any], b: dict[str, Any], fvu_tol: float = 0.001, mem_tol: float = 0.5) -> bool:
    if a["k"] > b["k"]:
        return False
    if a["fvu"] > b["fvu"] + fvu_tol:
        return False
    a_mem = a.get("peak_memory_gb")
    b_mem = b.get("peak_memory_gb")
    mem_not_worse = True
    mem_strict = False
    if a_mem is not None and b_mem is not None:
        mem_not_worse = a_mem <= b_mem + mem_tol
        mem_strict = a_mem < b_mem - mem_tol
    if not mem_not_worse:
        return False
    return a["k"] < b["k"] or a["fvu"] < b["fvu"] - fvu_tol or mem_strict


def compute_pareto_frontier(frontier: dict[str, Any]) -> list[dict[str, Any]]:
    points = []
    for key, point in frontier.items():
        normalized = _coerce_frontier_point(key, point)
        if normalized is not None:
            points.append(normalized)
    pareto: list[dict[str, Any]] = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            if pareto_dominates(other, point):
                dominated = True
                break
        if not dominated:
            pareto.append(point)
    pareto.sort(key=lambda x: (x["k"], x["fvu"]))
    return pareto


def normalize_state_for_compaction(state: dict[str, Any], reset_round_index: bool) -> dict[str, Any]:
    compact = deepcopy(state)
    compact.setdefault("frontier", compact.get("full_frontier", compact.get("frontier", {})))
    compact.setdefault("proxy_frontier", {})
    compact.setdefault("full_frontier", compact.get("frontier", {}))
    compact["pareto_proxy_frontier"] = compute_pareto_frontier(compact.get("proxy_frontier", {}))
    compact["pareto_full_frontier"] = compute_pareto_frontier(compact.get("full_frontier", {}))
    compact["pareto_frontier"] = compact["pareto_full_frontier"]
    agent = compact.setdefault("agent", {})
    if reset_round_index:
        agent["round_index"] = 0
    agent["active_session_id"] = None
    agent["active_session_started_at"] = None
    agent["active_session_rounds"] = 0
    agent["active_session_status"] = "closed"
    agent["last_resume_ok_at"] = None
    agent["last_action_file"] = None
    agent["last_patch_file"] = None
    return compact


def summarize_family(family_name: str, family: dict[str, Any]) -> dict[str, Any]:
    tested = family.get("tested_configs", []) if isinstance(family.get("tested_configs"), list) else []
    best_test = None
    best_fvu = float("inf")
    for cfg in tested:
        fvu = parse_float(cfg.get("val_fvu"))
        if fvu is None or fvu >= best_fvu:
            continue
        best_fvu = fvu
        best_test = {
            "round": parse_int(cfg.get("round")),
            "stage": cfg.get("stage"),
            "k": parse_int(cfg.get("k")),
            "decision": cfg.get("decision"),
            "val_fvu": fvu,
            "run_health": cfg.get("run_health"),
        }
    return {
        "family_name": family_name,
        "status": family.get("status"),
        "goal": family.get("goal"),
        "best_proxy_fvu": family.get("best_proxy_fvu"),
        "best_full_fvu": family.get("best_full_fvu"),
        "last_round": parse_int(family.get("last_round")),
        "tested_config_count": len(tested),
        "known_issues": [trim_text(x, limit=110) for x in family.get("known_issues", [])[-1:]],
        "next_steps": [trim_text(x, limit=110) for x in family.get("next_steps", [])[-1:]],
        "best_tested_config": best_test,
    }


def compact_runtime_baselines(baseline_runtime: dict[str, Any]) -> dict[str, Any]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for key, value in baseline_runtime.items():
        if not isinstance(value, dict):
            continue
        entries.append((key, value))
    entries.sort(key=lambda item: item[1].get("updated_at") or 0)
    compact: dict[str, Any] = {}
    for key, value in entries[-6:]:
        compact[key] = {
            "tokens_per_sec": value.get("tokens_per_sec"),
            "round": value.get("round"),
            "tier": value.get("tier"),
            "architecture": value.get("architecture"),
            "k": value.get("k"),
        }
    return compact


def compact_architecture_finding(line: str) -> str:
    if not isinstance(line, str):
        return ""
    prefix, sep, suffix = line.partition(" hypothesis=")
    if not sep:
        return trim_text(line, limit=160)
    return f"{trim_text(prefix, limit=120)} lesson={trim_text(suffix, limit=120)}"


def build_round_digest(summary: dict[str, Any], candidate_reasons: dict[int, list[str]]) -> dict[str, Any]:
    round_id = parse_int(summary.get("round"))
    action = summary.get("action", {}) if isinstance(summary.get("action"), dict) else {}
    result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
    proxy_result = summary.get("proxy_result", {}) if isinstance(summary.get("proxy_result"), dict) else {}
    full_result = summary.get("full_result", {}) if isinstance(summary.get("full_result"), dict) else {}
    primary_result = result or full_result or proxy_result
    overrides = extract_override_map(action)
    return {
        "round": round_id,
        "family_name": summary.get("family_name"),
        "family_stage": summary.get("family_stage"),
        "change_type": action.get("change_type"),
        "primary_variable": action.get("primary_variable"),
        "decision": primary_result.get("decision"),
        "termination_reason": primary_result.get("termination_reason"),
        "val_fvu": primary_result.get("val_fvu"),
        "k": primary_result.get("k"),
        "run_health": primary_result.get("run_health"),
        "has_code_touch": bool(summary.get("touched_files")),
        "touched_file_count": len(summary.get("touched_files", [])),
        "has_patch": bool(summary.get("patch_path")),
        "has_error_only": bool(summary.get("error")) and not bool(action),
        "key_override_keys": sorted(overrides.keys())[:3],
        "candidate_keep": bool(candidate_reasons.get(round_id or -1)),
    }


def build_artifact_inventory(
    src_history: Path,
    summaries: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    logs_dir = src_history / "logs"
    inventory: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        round_id = parse_int(summary.get("round"))
        family_name = summary.get("family_name")
        refs: list[tuple[str, str]] = []
        for blob_name in ("proxy_result", "full_result", "result"):
            blob = summary.get(blob_name)
            if not isinstance(blob, dict):
                continue
            path = blob.get("log_path")
            if path:
                refs.append((Path(str(path)).name, "log"))
        patch_path = summary.get("patch_path")
        if patch_path:
            refs.append((Path(str(patch_path)).name, "patch"))
        for name, kind in refs:
            entry = inventory.setdefault(name, {
                "name": name,
                "kind": kind,
                "rounds": [],
                "size_bytes": (logs_dir / name).stat().st_size if (logs_dir / name).exists() else None,
            })
            if round_id is not None and round_id not in entry["rounds"]:
                entry["rounds"].append(round_id)
    items = sorted(inventory.values(), key=lambda item: ((item["rounds"][0] if item["rounds"] else 10**9), item["name"]))
    for item in items:
        item["size_kb"] = round((item.pop("size_bytes") or 0) / 1024, 1)
    return items[:limit] if limit is not None else items


def build_context_pack(
    src_history: Path,
    state: dict[str, Any],
    memory: dict[str, Any],
    results: list[dict[str, str]],
    all_summaries: list[dict[str, Any]],
    operator_hints: list[dict[str, Any]],
    keep_last_rounds: int,
) -> dict[str, Any]:
    candidate_rounds, retention_reasons = choose_candidate_rounds(all_summaries, state, memory, keep_last_rounds)
    candidate_reason_map: dict[int, list[str]] = defaultdict(list)
    for reason, rounds in retention_reasons.items():
        for round_id in rounds:
            candidate_reason_map[round_id].append(reason)

    families = memory.get("architecture_families", {})
    family_digests = [
        summarize_family(name, family)
        for name, family in sorted(families.items())
        if isinstance(family, dict)
    ]
    round_digests = [
        build_round_digest(summary, candidate_reason_map)
        for summary in all_summaries[-MAX_CONTEXT_ROUNDS:]
    ]
    recent_sanity_failures = []
    for item in memory.get("recent_sanity_failures", [])[-4:]:
        if not isinstance(item, dict):
            continue
        recent_sanity_failures.append({
            "round": item.get("round"),
            "family_name": item.get("family_name"),
            "error_type": item.get("error_type"),
            "message": trim_text(item.get("message"), limit=120),
        })

    takeaways = build_research_takeaways(state, memory, all_summaries)
    research_takeaways_context = {
        "best_mainline_family": takeaways.get("best_mainline_family"),
        "exhausted_families": takeaways.get("exhausted_families", []),
        "blocked_families": [
            {
                "round": item.get("round"),
                "family_name": item.get("family_name"),
                "termination_reason": item.get("termination_reason"),
            }
            for item in takeaways.get("blocked_families", [])[-6:]
        ],
        "open_blockers": [
            {
                "round": item.get("round"),
                "family_name": item.get("family_name"),
                "termination_reason": item.get("termination_reason"),
                "error_type": item.get("error_type"),
            }
            for item in takeaways.get("open_blockers", [])[-6:]
        ],
    }

    recent_results = []
    for row in summarize_results(results[-MAX_CONTEXT_RESULTS:]):
        recent_results.append({
            "experiment_id": row.get("experiment_id"),
            "tier": row.get("tier"),
            "decision": row.get("decision"),
            "val_fvu": row.get("val_fvu"),
            "k": row.get("k"),
            "architecture": row.get("architecture"),
        })

    return {
        "source_history": str(src_history),
        "round_count": len(all_summaries),
        "result_count": len(results),
        "frontier": {
            "full_frontier": frontier_context_digest(state.get("full_frontier", state.get("frontier", {}))),
            "proxy_frontier": frontier_context_digest(state.get("proxy_frontier", {})),
            "pareto_full_frontier": frontier_context_digest(state.get("pareto_full_frontier", state.get("pareto_frontier", []))),
            "pareto_proxy_frontier": frontier_context_digest(state.get("pareto_proxy_frontier", [])),
        },
        "best_family": determine_best_family(memory),
        "family_digests": family_digests,
        "round_digests": round_digests,
        "recent_results": recent_results,
        "recent_sanity_failures": recent_sanity_failures,
        "recent_training_failures": summarize_recent_failures(all_summaries, limit=10),
        "code_history_summary": build_code_history_summary(all_summaries),
        "research_takeaways": research_takeaways_context,
        "artifact_inventory": build_artifact_inventory(src_history, all_summaries),
        "candidate_keep_rounds": candidate_rounds,
        "candidate_retention_reasons": retention_reasons,
        "hard_constraints": {
            "preserve_results_tsv_rows": True,
            "preserve_state_frontier_exactly": True,
            "preserve_pareto_frontiers_exactly": True,
            "preserve_best_family_and_best_points": True,
            "preserve_blocked_family_evidence": True,
            "preserve_code_lineage": True,
            "recent_context_must_be_actionable": True,
        },
    }


def build_compaction_prompt(context_pack: dict[str, Any]) -> str:
    return f"""
You are the history compaction agent for this repository.

Goal:
- decide what to retain, what to drop, and what to summarize when compressing `research/history/`
- produce a minimal but sufficient history that can be used to restart `python -m research.agent_loop`
- avoid preserving redundant exploratory churn
- avoid losing facts that would cause the next research session to repeat known dead ends

Important constraints:
- `results.tsv` is the structured source of truth and must be preserved in full
- `state.json` frontier and pareto fields must remain exact
- your job is to decide retention and summarization, not to rewrite research conclusions arbitrarily
- every source round must be covered exactly once by either `keep` or `drop`
- if you drop a round, your round decision summary must still preserve the key lesson from that round
- prefer dropping repeated local sweeps, repeated low-information backend failures, and long logs that add no unique evidence
- prefer keeping family introduction rounds, best evidence rounds, blocker diagnosis rounds, key code lineage rounds, and recent continuity rounds
- logs and patches may be dropped if the retained summaries and manifests still preserve the needed information

Output rules:
- return only a JSON object matching the compaction schema
- `safety_checks` must stay true for all hard constraints
- default to `rebuild_digest` for `timeline_policy` unless per-event filtered history is genuinely needed
- avoid retaining artifacts or timeline detail when `round_decisions` plus compact memory already preserve the lesson
- `keep_log_artifacts` and `drop_log_artifacts` refer only to names from `artifact_inventory`; do not list core files like `state.json` or `results.tsv` there

Structured context:
{json.dumps(context_pack, indent=2, ensure_ascii=False)}
""".strip()


def read_text_safe(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise RuntimeError("Could not extract JSON object from Codex response")


def run_compaction_agent(
    prompt: str,
    out_dir: Path,
    model: str | None,
    agent_proxy: str | None,
    timeout_sec: int,
    save_debug_artifacts: bool,
) -> tuple[dict[str, Any], Path]:
    debug_dir = out_dir / ".compaction_debug"
    if save_debug_artifacts:
        debug_dir.mkdir(parents=True, exist_ok=True)
    action_path = debug_dir / "compaction_plan.raw.json" if save_debug_artifacts else out_dir / ".tmp_compaction_plan.raw.json"
    stdout_path = debug_dir / "compaction_agent.stdout.log" if save_debug_artifacts else out_dir / ".tmp_compaction_agent.stdout.log"
    cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd",
        str(REPO_ROOT),
        "--output-schema",
        str(SCHEMA_PATH),
        "-o",
        str(action_path),
    ]
    if model:
        cmd.extend(["--model", model])
    env = os.environ.copy()
    proxy_env_keys = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
    if agent_proxy is not None:
        for key in proxy_env_keys:
            env.pop(key, None)
    if agent_proxy:
        for key in proxy_env_keys:
            env[key] = agent_proxy
    try:
        result = subprocess.run(cmd, input=prompt, text=True, capture_output=True, env=env, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"compaction agent timed out after {timeout_sec}s") from exc
    stdout_path.write_text(result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else ""))
    if result.returncode != 0:
        raise RuntimeError(f"compaction agent failed: see {stdout_path}")
    plan = load_json(action_path, {})
    if not plan:
        plan = extract_json_object(read_text_safe(action_path))
    return plan, stdout_path


def run_compaction_agent_with_retry(
    prompt: str,
    out_dir: Path,
    model: str | None,
    agent_proxy: str | None,
    timeout_sec: int,
    max_retries: int,
    retry_base_sec: int,
    save_debug_artifacts: bool,
) -> tuple[dict[str, Any], Path]:
    attempt = 0
    last_error: Exception | None = None
    while attempt <= max_retries:
        try:
            return run_compaction_agent(prompt, out_dir, model, agent_proxy, timeout_sec, save_debug_artifacts)
        except Exception as exc:
            last_error = exc
            attempt += 1
            if attempt > max_retries:
                break
            time.sleep(retry_base_sec * attempt)
    assert last_error is not None
    raise last_error


def validate_compaction_plan(
    plan: dict[str, Any],
    all_summaries: list[dict[str, Any]],
    artifact_inventory: list[dict[str, Any]],
) -> None:
    required = load_json(SCHEMA_PATH, {}).get("required", [])
    missing = [key for key in required if key not in plan]
    if missing:
        raise RuntimeError(f"Compaction plan missing required keys: {', '.join(missing)}")

    all_rounds = sorted(parse_int(summary.get("round")) for summary in all_summaries if parse_int(summary.get("round")) is not None)
    keep_rounds = sorted(set(int(r) for r in plan.get("keep_rounds", [])))
    drop_rounds = sorted(set(int(r) for r in plan.get("drop_rounds", [])))
    if sorted(keep_rounds + drop_rounds) != all_rounds:
        raise RuntimeError("Compaction plan must partition all rounds into keep_rounds and drop_rounds")

    decisions = plan.get("round_decisions", [])
    decision_rounds = sorted(set(int(item.get("round")) for item in decisions if parse_int(item.get("round")) is not None))
    if decision_rounds != all_rounds:
        raise RuntimeError("Compaction plan round_decisions must cover every source round")

    decision_map = {int(item["round"]): item for item in decisions}
    for round_id in keep_rounds:
        if decision_map[round_id].get("decision") != "keep":
            raise RuntimeError(f"round {round_id} listed in keep_rounds but decision is not keep")
    for round_id in drop_rounds:
        if decision_map[round_id].get("decision") != "drop":
            raise RuntimeError(f"round {round_id} listed in drop_rounds but decision is not drop")

    safety = plan.get("safety_checks", {})
    required_safety = [
        "results_tsv_full",
        "frontier_exact",
        "pareto_exact",
        "best_family_preserved",
        "code_lineage_preserved",
        "blocked_failures_preserved",
    ]
    bad = [key for key in required_safety if safety.get(key) is not True]
    if bad:
        raise RuntimeError(f"Compaction plan rejected required safety checks: {', '.join(bad)}")

    artifact_names = {item["name"] for item in artifact_inventory}
    core_runtime_files = {
        "state.json",
        "frontier.json",
        "memory.json",
        "results.tsv",
        "session_brief.json",
        "operator_hints.json",
        "current_status.json",
        "timeline.jsonl",
        "compression_manifest.json",
        "compression_report.md",
        "compaction_plan.json",
    }
    unknown_keep = sorted(
        set(plan.get("keep_log_artifacts", []))
        - artifact_names
        - core_runtime_files
    )
    if unknown_keep:
        raise RuntimeError(f"Compaction plan references unknown artifacts: {', '.join(unknown_keep[:8])}")

    if plan.get("timeline_policy") not in {"full", "filtered", "rebuild_digest"}:
        raise RuntimeError("timeline_policy must be one of: full, filtered, rebuild_digest")


def normalize_kept_artifacts(plan: dict[str, Any], artifact_inventory: list[dict[str, Any]]) -> list[str]:
    artifact_names = {item["name"] for item in artifact_inventory}
    return sorted(set(name for name in plan.get("keep_log_artifacts", []) if name in artifact_names))


def build_dropped_rounds_digest(
    all_summaries: list[dict[str, Any]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    round_map = {int(summary["round"]): summary for summary in all_summaries if parse_int(summary.get("round")) is not None}
    decision_map = {int(item["round"]): item for item in plan.get("round_decisions", [])}
    dropped_entries = []
    for round_id in sorted(int(r) for r in plan.get("drop_rounds", [])):
        summary = round_map[round_id]
        decision = decision_map[round_id]
        action = summary.get("action", {}) if isinstance(summary.get("action"), dict) else {}
        result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
        primary_result = result or summary.get("proxy_result") or summary.get("full_result") or {}
        dropped_entries.append({
            "round": round_id,
            "family_name": summary.get("family_name"),
            "family_stage": summary.get("family_stage"),
            "change_type": action.get("change_type"),
            "decision": decision.get("decision"),
            "reason": trim_text(decision.get("reason"), limit=220),
            "compressed_summary": trim_text(decision.get("compressed_summary"), limit=300),
            "original_hypothesis": trim_text(action.get("hypothesis"), limit=280),
            "result_decision": primary_result.get("decision"),
            "termination_reason": primary_result.get("termination_reason"),
            "val_fvu": primary_result.get("val_fvu"),
            "touched_files": summary.get("touched_files", []),
        })
    return {
        "generated_at": int(time.time()),
        "dropped_round_count": len(dropped_entries),
        "dropped_rounds": dropped_entries,
    }


def build_compact_memory(
    memory: dict[str, Any],
    all_summaries: list[dict[str, Any]],
    kept_summaries: list[dict[str, Any]],
    dropped_digest: dict[str, Any],
    state: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    compact = deepcopy(memory)
    kept_rounds = [int(summary["round"]) for summary in kept_summaries if parse_int(summary.get("round")) is not None]
    kept_round_set = set(kept_rounds)
    family_summaries: dict[str, Any] = {}
    for family_name, family in compact.get("architecture_families", {}).items():
        if not isinstance(family, dict):
            continue
        family_summary = summarize_family(family_name, family)
        kept_family_rounds = [
            parse_int(cfg.get("round"))
            for cfg in family.get("tested_configs", [])
            if parse_int(cfg.get("round")) in kept_round_set
        ]
        if kept_family_rounds:
            family_summary["kept_rounds"] = kept_family_rounds[-3:]
        family_summaries[family_name] = family_summary

    architecture_findings = []
    performance_findings = []
    kept_round_entries = []
    for summary in kept_summaries:
        round_id = parse_int(summary.get("round"))
        if round_id is None:
            continue
        action = summary.get("action", {}) if isinstance(summary.get("action"), dict) else {}
        result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
        line = (
            f"round {round_id}: family={summary.get('family_name')} stage={summary.get('family_stage')} "
            f"change={action.get('change_type')} decision={result.get('decision')} "
            f"fvu={result.get('val_fvu')} termination={result.get('termination_reason')}"
        )
        kept_round_entries.append({
            "round": round_id,
            "family_name": summary.get("family_name"),
            "family_stage": summary.get("family_stage"),
            "change_type": action.get("change_type"),
            "hypothesis": trim_text(action.get("hypothesis"), limit=420),
            "decision": result.get("decision"),
            "val_fvu": result.get("val_fvu"),
            "termination_reason": result.get("termination_reason"),
            "touched_files": summary.get("touched_files", []),
        })
        if action.get("change_type") in {"edit_sae_code", "edit_perf_code"}:
            architecture_findings.append(line + f" hypothesis={trim_text(action.get('hypothesis'), limit=240)}")
        if result.get("decision") == "crash" or result.get("run_health") in {"perf_regression", "crash"}:
            performance_findings.append(line)

    compact["architecture_families"] = family_summaries
    compact["architecture_findings"] = [
        item for item in (
            compact_architecture_finding(line)
            for line in dedupe_keep_order(architecture_findings)[-8:]
        )
        if item
    ]
    compact["performance_findings"] = dedupe_keep_order(performance_findings)[-8:]
    compact["recent_training_failures"] = summarize_recent_failures(all_summaries, limit=6)
    compact["recent_sanity_failures"] = [
        item for item in (
            compact_failure_entry(entry)
            for entry in compact.get("recent_sanity_failures", [])[-4:]
        )
        if item
    ]
    compact["code_history_summary"] = build_code_history_summary(all_summaries)
    compact["research_takeaways"] = build_research_takeaways(state, compact, all_summaries)
    compact["compressed_history_summary"] = {
        "source_round_count": len(all_summaries),
        "retained_round_count": len(kept_summaries),
        "retained_rounds": kept_rounds,
        "dropped_round_count": dropped_digest.get("dropped_round_count", 0),
        "generated_at": int(time.time()),
        "strategy_summary": trim_text(plan.get("strategy_summary"), limit=500),
        "kept_round_entries": kept_round_entries[-24:],
        "dropped_round_highlights": dropped_digest.get("dropped_rounds", [])[-24:],
    }
    compact["recent_insights"] = dedupe_keep_order([
        f"Compaction agent retained {len(kept_summaries)} of {len(all_summaries)} rounds.",
        trim_text(plan.get("strategy_summary"), limit=240),
        *[trim_text(item, limit=180) for item in compact.get("recent_insights", [])[-8:]],
        *[trim_text(item.get("compressed_summary"), limit=180) for item in dropped_digest.get("dropped_rounds", [])[-6:]],
    ])[-16:]
    compact["failure_patterns"] = [
        {
            "last_round": parse_int(item.get("last_round")) if isinstance(item, dict) else None,
            "run_health": item.get("run_health") if isinstance(item, dict) else None,
            "termination_reason": item.get("termination_reason") if isinstance(item, dict) else None,
            "pattern": trim_text(item.get("pattern"), limit=140) if isinstance(item, dict) else None,
        }
        for item in compact.get("failure_patterns", [])[-5:]
        if isinstance(item, dict)
    ]
    compact["baseline_runtime"] = compact_runtime_baselines(compact.get("baseline_runtime", {}))
    return compact


def build_session_brief_from_compact(
    state: dict[str, Any],
    memory: dict[str, Any],
    results: list[dict[str, str]],
    kept_summaries: list[dict[str, Any]],
    operator_hints: list[dict[str, Any]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    incubating = {}
    for name, value in memory.get("architecture_families", {}).items():
        if not isinstance(value, dict) or value.get("status") != "incubating":
            continue
        incubating[name] = {
            "status": value.get("status"),
            "design_hypothesis": value.get("design_hypothesis"),
            "best_proxy_fvu": value.get("best_proxy_fvu"),
            "tested_configs": value.get("tested_configs", [])[-3:],
            "next_steps": value.get("next_steps", [])[-3:],
            "last_round": value.get("last_round"),
        }

    recent_round_targets = set(int(r) for r in plan.get("session_brief_focus", {}).get("recent_rounds", []))
    if recent_round_targets:
        brief_summaries = [summary for summary in kept_summaries if int(summary.get("round")) in recent_round_targets]
    else:
        brief_summaries = kept_summaries[-3:]

    trimmed_rounds = []
    for summary in brief_summaries[-3:]:
        action = summary.get("action", {}) if isinstance(summary.get("action"), dict) else {}
        result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
        trimmed_rounds.append({
            "round": summary.get("round"),
            "family_name": summary.get("family_name"),
            "family_stage": summary.get("family_stage"),
            "duration_sec": summary.get("duration_sec"),
            "hypothesis": action.get("hypothesis"),
            "change_type": action.get("change_type"),
            "experiment_tier": action.get("experiment_tier"),
            "decision": result.get("decision"),
            "val_fvu": result.get("val_fvu"),
            "run_health": result.get("run_health"),
            "termination_reason": result.get("termination_reason"),
        })

    pending_hints = [hint for hint in operator_hints if hint.get("status") == "pending"][:4]
    return {
        "updated_at": int(time.time()),
        "active_session_id": None,
        "current_focus": memory.get("current_focus"),
        "pareto_full_frontier": frontier_digest(state.get("pareto_full_frontier", state.get("pareto_frontier", []))),
        "recent_results": summarize_results(results[-3:]),
        "recent_round_summaries": trimmed_rounds,
        "incubating_families": incubating,
        "recent_performance_findings": memory.get("performance_findings", [])[-4:],
        "recent_sanity_failures": [
            item for item in (
                compact_failure_entry(entry)
                for entry in memory.get("recent_sanity_failures", [])[-4:]
            )
            if item
        ],
        "recent_training_failures": [
            item for item in (
                compact_failure_entry(entry)
                for entry in memory.get("recent_training_failures", [])[-4:]
            )
            if item
        ],
        "pending_hints": pending_hints,
        "next_move_guidance": [
            trim_text(item, limit=180)
            for item in plan.get("session_brief_focus", {}).get("next_move_guidance", memory.get("next_hypotheses", [])[:5])
        ][:5],
        "last_round": 0,
    }


def rewrite_summary_paths(
    summary: dict[str, Any],
    kept_artifacts: set[str],
    out_history: Path,
) -> dict[str, Any]:
    compact = compact_round_summary(summary)
    logs_dir = out_history / "logs"
    for blob_name in ("proxy_result", "full_result", "result"):
        blob = compact.get(blob_name)
        if not isinstance(blob, dict):
            continue
        log_path = blob.get("log_path")
        if not log_path:
            continue
        name = Path(str(log_path)).name
        blob["log_path"] = str(logs_dir / name) if name in kept_artifacts else None
    patch_path = compact.get("patch_path")
    if patch_path:
        patch_name = Path(str(patch_path)).name
        compact["patch_path"] = str(logs_dir / patch_name) if patch_name in kept_artifacts else None
    return compact


def copy_selected_artifacts(src_history: Path, out_history: Path, names: list[str]) -> list[str]:
    copied: list[str] = []
    src_logs = src_history / "logs"
    out_logs = out_history / "logs"
    out_logs.mkdir(parents=True, exist_ok=True)
    for name in sorted(set(names)):
        src = src_logs / name
        if not src.exists() or not src.is_file():
            continue
        shutil.copy2(src, out_logs / name)
        copied.append(name)
    return copied


def write_timeline(
    src: Path,
    dest: Path,
    plan: dict[str, Any],
    kept_rounds: set[int],
    dropped_digest: dict[str, Any],
) -> None:
    policy = plan.get("timeline_policy", "filtered")
    if policy == "full":
        if src.exists():
            shutil.copy2(src, dest)
        else:
            dest.write_text("")
        return

    out_lines: list[str] = []
    if policy == "filtered" and src.exists():
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                round_id = parse_int(event.get("round"))
                if round_id is None or round_id in kept_rounds:
                    out_lines.append(json.dumps(event, ensure_ascii=False))
    else:
        out_lines.append(json.dumps({
            "event": "compaction_snapshot_created",
            "ts": int(time.time()),
            "retained_rounds": sorted(kept_rounds),
            "dropped_round_count": dropped_digest.get("dropped_round_count", 0),
            "strategy_summary": trim_text(plan.get("strategy_summary"), limit=260),
        }, ensure_ascii=False))
        for entry in dropped_digest.get("dropped_rounds", [])[-24:]:
            out_lines.append(json.dumps({
                "event": "dropped_round_digest",
                "ts": int(time.time()),
                "round": entry.get("round"),
                "family_name": entry.get("family_name"),
                "summary": entry.get("compressed_summary"),
                "reason": entry.get("reason"),
            }, ensure_ascii=False))
    dest.write_text(("\n".join(out_lines) + "\n") if out_lines else "")


def write_current_status(dest: Path, kept_rounds: list[int]) -> None:
    save_json(dest, {
        "timestamp": int(time.time()),
        "stage": "idle",
        "message": "agent-compacted history snapshot ready",
        "retained_rounds": kept_rounds,
    })


def build_file_roles_manifest(copied_logs: list[str], kept_rounds: list[int]) -> dict[str, Any]:
    round_summary_paths = [f"round_summaries/round_{round_id:04d}.json" for round_id in kept_rounds]
    log_paths = [f"logs/{name}" for name in copied_logs]
    return {
        "record_files": [
            "timeline.jsonl",
            "results.tsv",
            "compression_report.md",
            "compression_manifest.json",
            "compaction_plan.json",
            *log_paths,
        ],
        "agent_context_files": [
            "state.json",
            "frontier.json",
            "memory.json",
            "results.tsv",
            "session_brief.json",
            "operator_hints.json",
            "current_status.json",
            *round_summary_paths,
        ],
        "prompt_primary_files": [
            "state.json",
            "memory.json",
            "results.tsv",
            "session_brief.json",
            "operator_hints.json",
            *round_summary_paths[-3:],
        ],
    }


def build_report(
    out_path: Path,
    state: dict[str, Any],
    memory: dict[str, Any],
    manifest: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    lines = [
        "# Agent-Compacted History",
        "",
        f"- Generated at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(manifest['generated_at']))}",
        f"- Source rounds: {manifest['source_round_count']}",
        f"- Retained round summaries: {manifest['retained_round_count']}",
        f"- Timeline policy: {manifest['timeline_policy']}",
        "",
        "## Strategy",
        "",
        f"- {trim_text(plan.get('strategy_summary'), limit=400)}",
        "",
        "## Current Pareto Frontier",
        "",
    ]
    for pt in state.get("pareto_full_frontier", []):
        lines.append(f"- Full K={pt['k']} FVU={pt['fvu']:.6f} arch={pt.get('architecture')}")
    for pt in state.get("pareto_proxy_frontier", []):
        lines.append(f"- Proxy K={pt['k']} FVU={pt['fvu']:.6f} arch={pt.get('architecture')}")

    lines.extend([
        "",
        "## Family Summaries",
        "",
    ])
    for family_name, summary in sorted(memory.get("architecture_families", {}).items()):
        lines.append(
            f"- `{family_name}`: status={summary.get('status')} best_proxy={summary.get('best_proxy_fvu')} "
            f"best_full={summary.get('best_full_fvu')} last_round={summary.get('last_round')}"
        )
    out_path.write_text("\n".join(lines) + "\n")


def validate_compacted_history(history_dir: Path) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (history_dir / name).exists():
            errors.append(f"missing required file: {name}")

    state = load_json(history_dir / "state.json", {})
    frontier = load_json(history_dir / "frontier.json", {})
    memory = load_json(history_dir / "memory.json", {})
    session_brief = load_json(history_dir / "session_brief.json", {})
    plan = load_json(history_dir / "compaction_plan.json", {})

    if frontier != state.get("frontier", {}):
        errors.append("frontier.json does not match state.json frontier")

    agent = state.get("agent", {})
    if agent.get("active_session_id") is not None:
        errors.append("active_session_id should be null in compacted state")
    if agent.get("active_session_status") != "closed":
        errors.append("active_session_status should be 'closed' in compacted state")

    if "pareto_full_frontier" not in state or "pareto_proxy_frontier" not in state:
        errors.append("state missing pareto frontier fields")
    if "compressed_history_summary" not in memory:
        errors.append("memory missing compressed_history_summary")
    if session_brief.get("active_session_id") is not None:
        errors.append("session_brief active_session_id should be null")
    if not plan:
        errors.append("compaction_plan.json is empty")

    try:
        _ = load_results(history_dir / "results.tsv")
    except Exception as exc:
        errors.append(f"results.tsv failed to parse: {exc}")

    round_dir = history_dir / "round_summaries"
    if not round_dir.exists():
        errors.append("round_summaries directory missing")
    else:
        for path in sorted_round_summary_paths(round_dir):
            try:
                summary = load_json(path, {})
                if "round" not in summary:
                    errors.append(f"round summary missing round field: {path.name}")
                elif not any(key in summary for key in ("action", "result", "error")):
                    errors.append(f"round summary missing action/result/error payload: {path.name}")
                for blob_name in ("proxy_result", "full_result", "result"):
                    blob = summary.get(blob_name)
                    if not isinstance(blob, dict):
                        continue
                    log_path = blob.get("log_path")
                    if log_path and not Path(str(log_path)).exists():
                        errors.append(f"round summary references missing log path: {path.name}:{blob_name}")
                patch_path = summary.get("patch_path")
                if patch_path and not Path(str(patch_path)).exists():
                    errors.append(f"round summary references missing patch path: {path.name}")
            except Exception as exc:
                errors.append(f"round summary failed to parse: {path.name}: {exc}")

    return errors


def resolve_out_dir(out_arg: str | None) -> Path:
    if out_arg:
        return Path(out_arg).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_OUT_ROOT / stamp).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compress research/history into a runtime-compatible compact snapshot")
    parser.add_argument("--src-history", default=str(DEFAULT_SRC_HISTORY))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--keep-last-rounds", type=int, default=3)
    parser.add_argument("--preserve-round-index", action="store_true")
    parser.add_argument("--validate-only", default=None, help="Validate an existing compacted history directory and exit")
    parser.add_argument("--model", default=None)
    parser.add_argument("--agent-proxy", default=None)
    parser.add_argument("--agent-timeout-sec", type=int, default=DEFAULT_AGENT_TIMEOUT_SEC)
    parser.add_argument("--agent-max-retries", type=int, default=2)
    parser.add_argument("--agent-retry-base-sec", type=int, default=5)
    parser.add_argument("--save-context-pack", action="store_true")
    parser.add_argument("--save-debug-artifacts", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        errors = validate_compacted_history(Path(args.validate_only).resolve())
        if errors:
            for err in errors:
                print(f"ERROR: {err}")
            return 1
        print("Validation OK")
        return 0

    src_history = Path(args.src_history).resolve()
    out_dir = resolve_out_dir(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "round_summaries").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    state = load_json(src_history / "state.json", {})
    memory = load_json(src_history / "memory.json", {})
    operator_hints = load_json(src_history / "operator_hints.json", [])
    results = load_results(src_history / "results.tsv")
    all_summaries = [load_json(path, {}) for path in sorted_round_summary_paths(src_history / "round_summaries")]

    full_artifact_inventory = build_artifact_inventory(src_history, all_summaries, limit=None)

    context_pack = build_context_pack(
        src_history=src_history,
        state=state,
        memory=memory,
        results=results,
        all_summaries=all_summaries,
        operator_hints=operator_hints,
        keep_last_rounds=args.keep_last_rounds,
    )
    if args.save_context_pack:
        save_json(out_dir / "compaction_context_pack.json", context_pack)

    prompt = build_compaction_prompt(context_pack)
    if args.save_debug_artifacts:
        debug_dir = out_dir / ".compaction_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "compaction_prompt.txt").write_text(prompt)
    plan, agent_stdout = run_compaction_agent_with_retry(
        prompt=prompt,
        out_dir=out_dir,
        model=args.model,
        agent_proxy=args.agent_proxy,
        timeout_sec=args.agent_timeout_sec,
        max_retries=args.agent_max_retries,
        retry_base_sec=args.agent_retry_base_sec,
        save_debug_artifacts=args.save_debug_artifacts,
    )
    if not args.save_debug_artifacts:
        for temp_name in (".tmp_compaction_plan.raw.json", ".tmp_compaction_agent.stdout.log"):
            temp_path = out_dir / temp_name
            if temp_path.exists():
                temp_path.unlink()
    validate_compaction_plan(plan, all_summaries, full_artifact_inventory)

    kept_rounds = sorted(set(int(r) for r in plan.get("keep_rounds", [])))
    kept_round_set = set(kept_rounds)
    kept_artifacts = normalize_kept_artifacts(plan, full_artifact_inventory)
    copied_logs = copy_selected_artifacts(src_history, out_dir, kept_artifacts)

    rewritten_summaries = []
    for summary in all_summaries:
        round_id = parse_int(summary.get("round"))
        if round_id is None or round_id not in kept_round_set:
            continue
        rewritten = rewrite_summary_paths(summary, set(copied_logs), out_dir)
        rewritten_summaries.append(rewritten)
        save_json(out_dir / "round_summaries" / f"round_{round_id:04d}.json", rewritten)

    dropped_digest = build_dropped_rounds_digest(all_summaries, plan)

    compact_state = normalize_state_for_compaction(state, reset_round_index=not args.preserve_round_index)
    compact_memory = build_compact_memory(memory, all_summaries, rewritten_summaries, dropped_digest, compact_state, plan)
    compact_session_brief = build_session_brief_from_compact(
        compact_state,
        compact_memory,
        results,
        rewritten_summaries,
        operator_hints,
        plan,
    )

    save_json(out_dir / "state.json", compact_state)
    save_json(out_dir / "frontier.json", compact_state.get("frontier", {}))
    save_json(out_dir / "memory.json", compact_memory)
    save_json(out_dir / "session_brief.json", compact_session_brief)
    save_json(out_dir / "operator_hints.json", operator_hints)
    save_json(out_dir / "compaction_plan.json", plan)
    write_results(out_dir / "results.tsv", results)
    write_timeline(src_history / "timeline.jsonl", out_dir / "timeline.jsonl", plan, kept_round_set, dropped_digest)
    write_current_status(out_dir / "current_status.json", kept_rounds)

    manifest = {
        "generated_at": int(time.time()),
        "source_history": str(src_history),
        "output_history": str(out_dir),
        "source_round_count": len(all_summaries),
        "retained_round_count": len(rewritten_summaries),
        "retained_rounds": kept_rounds,
        "dropped_rounds": sorted(set(parse_int(s.get("round")) for s in all_summaries if parse_int(s.get("round")) is not None) - kept_round_set),
        "timeline_policy": plan.get("timeline_policy"),
        "copied_logs": copied_logs,
        "dropped_log_artifacts": plan.get("drop_log_artifacts", []),
        "reset_round_index": not args.preserve_round_index,
        "strategy_summary": plan.get("strategy_summary"),
        "context_pack_saved": args.save_context_pack,
        "debug_artifacts_saved": args.save_debug_artifacts,
    }
    manifest.update(build_file_roles_manifest(copied_logs, kept_rounds))
    if args.save_debug_artifacts:
        manifest["agent_stdout_log"] = str(agent_stdout)
    save_json(out_dir / "compression_manifest.json", manifest)
    build_report(out_dir / "compression_report.md", compact_state, compact_memory, manifest, plan)

    errors = validate_compacted_history(out_dir)
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1

    print(f"Compacted history written to: {out_dir}")
    print(f"Retained {len(rewritten_summaries)} / {len(all_summaries)} round summaries")
    print(f"Retained rounds: {', '.join(str(r) for r in kept_rounds)}")
    print(f"Copied logs: {len(copied_logs)}")
    print("Validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
