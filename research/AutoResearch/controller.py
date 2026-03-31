"""Lightweight result evaluation for the autoresearch framework.

The runtime leaderboard is now a single-objective ranking over:

    objective_score = total_cost_ratio + exceed_alpha_0_50

where total_cost_ratio uses the current fused-QKV cost proxy and
``exceed_alpha_0_50`` comes from the final training-step metrics.

FVU remains a diagnostic field and only acts as a late tie-break.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .compatibility import COST_METRIC_VERSION, compute_selection_cost, extract_cost_extra_config
from .objective import (
    EXCEED_FIELD,
    LEADERBOARD_LIMIT,
    candidate_objective_metrics,
    entry_objective_metrics,
    is_objective_near_duplicate,
    objective_rank_tuple,
)
from .target_profile import TargetProfile, resolve_target_profile


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_SUMMARY_PATTERNS = {
    "status": re.compile(r"^status:\s+(\S+)$", re.MULTILINE),
    "val_fvu": re.compile(r"^val_fvu:\s+(\S+)$", re.MULTILINE),
    "k": re.compile(r"^k:\s+(\d+)$", re.MULTILINE),
    "architecture": re.compile(r"^architecture:\s+(\S+)$", re.MULTILINE),
    "wall_time_sec": re.compile(r"^wall_time_sec:\s+([0-9.]+)$", re.MULTILINE),
    "peak_memory_gb": re.compile(r"^peak_memory_gb:\s+([0-9.]+)$", re.MULTILINE),
    "total_tokens": re.compile(r"^total_tokens:\s+(\d+)$", re.MULTILINE),
    "checkpoint": re.compile(r"^checkpoint:\s+(.+)$", re.MULTILINE),
    "expansion_factor": re.compile(r"^expansion_factor:\s+(\d+)$", re.MULTILINE),
}


def parse_log(log_path: Path) -> dict[str, Any]:
    """Extract key metrics from a training log file."""
    if not log_path.exists():
        return {"status": "crash"}

    text = log_path.read_text(errors="replace")
    parsed: dict[str, Any] = {}

    for key, pattern in _SUMMARY_PATTERNS.items():
        matches = list(pattern.finditer(text))
        parsed[key] = matches[-1].group(1).strip() if matches else None

    if parsed.get("val_fvu") in (None, "nan"):
        parsed["val_fvu"] = None
    elif parsed["val_fvu"] is not None:
        parsed["val_fvu"] = float(parsed["val_fvu"])

    for key in ("k", "total_tokens", "expansion_factor"):
        if parsed.get(key) is not None:
            parsed[key] = int(parsed[key])

    for key in ("wall_time_sec", "peak_memory_gb"):
        if parsed.get(key) is not None:
            parsed[key] = float(parsed[key])

    if parsed.get("checkpoint") == "none":
        parsed["checkpoint"] = None

    if parsed.get("status") is None:
        parsed["status"] = "ok" if parsed.get("val_fvu") is not None else "crash"

    return parsed


# ---------------------------------------------------------------------------
# Decision logic — single-objective leaderboard
# ---------------------------------------------------------------------------


def frontier_key(round_id: int | str) -> str:
    return f"r{round_id}"


def decide(
    frontier: dict[str, Any],
    parsed: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> str:
    """Compare a result against the single-objective leaderboard.

    Returns one of:
    - keep: enters the feasible top-k objective leaderboard
    - archive: objective-near-duplicate of an existing retained point
    - discard: valid run, but not leaderboard-worthy
    - crash: missing essential metrics
    """
    status = parsed.get("status")
    fvu = parsed.get("val_fvu")
    if status != "ok" or fvu is None:
        return "crash"

    candidate = _candidate_metrics(parsed, config)
    if candidate["objective_score"] is None:
        return "crash"
    if not candidate["feasible"]:
        return "discard"

    target_profile = _resolve_cost_profile(config or {})
    entries = _frontier_entries(frontier, target_profile.profile_id)
    if not entries:
        return "keep"

    for entry in entries:
        existing = _entry_to_metrics(entry)
        if existing["objective_score"] is None:
            continue
        if is_objective_near_duplicate(
            existing["objective_score"],
            candidate["objective_score"],
        ):
            if _rank_tuple(candidate) < _rank_tuple(existing):
                return "keep"
            return "archive"

    ranked_existing = sorted(
        (_entry_to_metrics(entry) for entry in entries),
        key=_rank_tuple,
    )
    if len(ranked_existing) < LEADERBOARD_LIMIT:
        return "keep"

    worst = ranked_existing[-1]
    return "keep" if _rank_tuple(candidate) < _rank_tuple(worst) else "discard"


def update_frontier(
    frontier: dict[str, Any],
    parsed: dict[str, Any],
    decision: str,
    config: dict[str, Any],
    commit: str,
    round_id: int | str | None = None,
) -> None:
    """If decision is 'keep', update the retained objective leaderboard."""
    if decision != "keep":
        return

    metrics = _candidate_metrics(parsed, config)
    if metrics["objective_score"] is None or not metrics["feasible"]:
        return

    k = parsed.get("k")
    ef = parsed.get("expansion_factor")
    target_profile = _resolve_cost_profile(config)

    key = frontier_key(round_id or "unknown")
    frontier[key] = {
        "objective_score": metrics["objective_score"],
        "total_cost": metrics["total_cost"],
        "total_cost_ratio": metrics["total_cost_ratio"],
        "selection_cost": metrics["selection_cost"],
        "selection_cost_ratio": metrics["selection_cost_ratio"],
        "deployment_accesses": metrics["deployment_accesses"],
        "deployment_ratio": metrics["deployment_ratio"],
        EXCEED_FIELD: metrics[EXCEED_FIELD],
        "fvu": parsed.get("val_fvu"),
        "k": int(k) if k is not None else None,
        "ef": int(ef) if ef is not None else None,
        "architecture": parsed.get("architecture"),
        "commit": commit,
        "config": config,
        "checkpoint": parsed.get("checkpoint"),
        "peak_memory_gb": parsed.get("peak_memory_gb"),
        "target_profile": target_profile.to_dict(),
        "cost_model_label": target_profile.cost_model_label,
        "metric_version": COST_METRIC_VERSION,
    }

    compacted = compact_frontier(frontier)
    frontier.clear()
    frontier.update(compacted)


def compact_frontier(frontier: dict[str, Any]) -> dict[str, Any]:
    """Compact stored entries into a per-target top-k objective leaderboard."""
    grouped: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = {}
    for key, entry in frontier.items():
        if not isinstance(entry, dict):
            continue
        try:
            target_profile = _entry_target_profile(entry)
            metrics = _entry_to_metrics(entry)
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if metrics["objective_score"] is None or not metrics["feasible"]:
            continue
        grouped.setdefault(target_profile.profile_id, []).append((key, entry, metrics))

    compacted: dict[str, Any] = {}
    for _profile_id, items in grouped.items():
        items.sort(key=lambda item: (_rank_tuple(item[2]), _frontier_sort_key(item[0])))
        kept_metrics: list[dict[str, Any]] = []
        kept_count = 0
        for key, entry, metrics in items:
            if any(
                is_objective_near_duplicate(
                    kept["objective_score"],
                    metrics["objective_score"],
                )
                for kept in kept_metrics
            ):
                continue
            compacted[key] = dict(entry)
            kept_metrics.append(metrics)
            kept_count += 1
            if kept_count >= LEADERBOARD_LIMIT:
                break
    return compacted


def _frontier_entries(
    frontier: dict[str, Any],
    target_profile_id: str | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        if target_profile_id is not None and _entry_target_profile(entry).profile_id != target_profile_id:
            continue
        entries.append(entry)
    return entries


def _entry_to_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert a stored entry to normalized objective metrics."""
    return entry_objective_metrics(entry)


def _rank_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return objective_rank_tuple(
        metrics.get("objective_score"),
        metrics.get("total_cost_ratio"),
        metrics.get(EXCEED_FIELD),
        metrics.get("fvu"),
    )


def _frontier_sort_key(key: str) -> tuple[int, int, str]:
    if key.startswith("r") and key[1:].isdigit():
        return (0, int(key[1:]), key)
    m = re.search(r"(\d+)$", key)
    if m:
        return (1, int(m.group(1)), key)
    return (2, 0, key)


# ---------------------------------------------------------------------------
# Cost / objective computation helpers
# ---------------------------------------------------------------------------


def _compute_candidate_cost_full(
    parsed: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute full cost dict (selection + deployment + combined) for a candidate."""
    cfg = config or {}
    arch = parsed.get("architecture") or cfg.get("architecture", "topk")
    k = int(parsed.get("k") or cfg.get("k") or 128)
    ef = int(parsed.get("expansion_factor") or cfg.get("expansion_factor") or 12)
    target_profile = _resolve_cost_profile(cfg)

    extra_config = extract_cost_extra_config(cfg)
    cost = compute_selection_cost(
        arch,
        k=k,
        ef=ef,
        d_in=target_profile.d_in,
        n_output=target_profile.n_output,
        extra_config=extra_config or None,
    )
    if "error" not in cost:
        return cost

    fallback = float(target_profile.d_in * target_profile.d_in * ef)
    return {"total_accesses": fallback, "combined_accesses": fallback, "error": "fallback"}


def _candidate_metrics(
    parsed: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    target_profile = _resolve_cost_profile(cfg)
    arch = parsed.get("architecture") or cfg.get("architecture", "topk")
    k = int(parsed.get("k") or cfg.get("k") or 128)
    ef = int(parsed.get("expansion_factor") or cfg.get("expansion_factor") or 12)
    return candidate_objective_metrics(
        str(arch).lower(),
        k,
        ef,
        target_profile,
        extra_config=extract_cost_extra_config(cfg),
        exceed_alpha_0_50=parsed.get(EXCEED_FIELD),
        fvu=parsed.get("val_fvu"),
    )


def _estimate_total_cost_from_entry(entry: dict[str, Any]) -> float:
    """Estimate total_cost (selection + deployment) from an entry."""
    metrics = entry_objective_metrics(entry)
    if metrics["total_cost"] is not None:
        return float(metrics["total_cost"])
    target_profile = _entry_target_profile(entry)
    ef = int(entry.get("ef") or entry.get("config", {}).get("expansion_factor") or 12)
    return float(target_profile.d_in * target_profile.d_in * ef)


def _entry_target_profile(entry: dict[str, Any]) -> TargetProfile:
    cfg = dict(entry.get("config", {}) or {})
    if entry.get("target_profile") is not None and "target_profile" not in cfg:
        cfg["target_profile"] = entry["target_profile"]
    return _resolve_cost_profile(cfg)


def _resolve_cost_profile(config: dict[str, Any]) -> TargetProfile:
    return resolve_target_profile(config)
