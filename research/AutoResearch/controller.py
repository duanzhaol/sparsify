"""Lightweight result evaluation for the autoresearch framework.

Replaces the old controller.py CLI.  Called as Python functions, not
as a subprocess.  No proxy/full tier distinction — single frontier.

The frontier is a 2D Pareto front over (total_cost, FVU).
total_cost = encoder selection cost + deployment lookup cost.
Lower total_cost and lower FVU are both better.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .compatibility import compute_selection_cost
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
    """Extract key metrics from a training log file.

    Returns a dict with: status, val_fvu, k, architecture, expansion_factor,
    wall_time_sec, peak_memory_gb, total_tokens, checkpoint.
    """
    if not log_path.exists():
        return {"status": "crash"}

    text = log_path.read_text(errors="replace")
    parsed: dict[str, Any] = {}

    for key, pattern in _SUMMARY_PATTERNS.items():
        matches = list(pattern.finditer(text))
        parsed[key] = matches[-1].group(1).strip() if matches else None

    # Type coercion
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
# Decision logic — 2D Pareto front: (total_cost, FVU)
# ---------------------------------------------------------------------------

FVU_TOL = 0.001
COST_REL_TOL = 0.05  # 5% relative tolerance for total_cost near-duplicate


def frontier_key(round_id: int | str) -> str:
    """Canonical frontier key based on round id."""
    return f"r{round_id}"


def decide(
    frontier: dict[str, Any],
    parsed: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> str:
    """Compare a training result against the 2D Pareto frontier.

    The frontier is two-dimensional: **(total_cost, FVU)**.
    total_cost = encoder selection cost + deployment lookup cost.
    Lower total_cost and lower FVU are both better.

    Returns one of: "keep", "crash", "archive", "discard".
    - keep:    result is Pareto non-dominated (expands or improves frontier)
    - crash:   training failed, no usable metric
    - archive: result is near-duplicate of an existing frontier point
    - discard: result is Pareto-dominated by existing frontier points
    """
    status = parsed.get("status")
    fvu = parsed.get("val_fvu")

    if status != "ok" or fvu is None:
        return "crash"

    target_profile = _resolve_cost_profile(config or {})
    total_cost = _compute_candidate_total_cost(parsed, config)
    candidate = {"total_cost": total_cost, "fvu": fvu}

    points = _frontier_points(frontier, target_profile.profile_id)
    if not points:
        return "keep"

    # Check for near-duplicate (both dimensions within tolerance)
    for pt in points:
        cost_close = (
            abs(pt["total_cost"] - total_cost) / max(total_cost, 1) <= COST_REL_TOL
        )
        fvu_close = abs(pt["fvu"] - fvu) <= FVU_TOL
        if cost_close and fvu_close:
            if fvu < pt["fvu"] - FVU_TOL:
                return "keep"  # same cost region but strictly better FVU
            return "archive"

    # Pareto dominance check
    if any(_pareto_dominates(pt, candidate) for pt in points):
        return "discard"

    return "keep"


def update_frontier(
    frontier: dict[str, Any],
    parsed: dict[str, Any],
    decision: str,
    config: dict[str, Any],
    commit: str,
    round_id: int | str | None = None,
) -> None:
    """If decision is 'keep', add to frontier and remove dominated points."""
    if decision != "keep":
        return
    fvu = parsed.get("val_fvu")
    if fvu is None:
        return

    k = parsed.get("k")
    ef = parsed.get("expansion_factor")
    full_cost = _compute_candidate_cost_full(parsed, config)
    sel_cost = float(full_cost["total_accesses"])
    deploy_accesses = full_cost.get("deployment_accesses", 0)
    combined = float(full_cost.get("combined_accesses", sel_cost + (deploy_accesses or 0)))
    target_profile = _resolve_cost_profile(config)

    key = frontier_key(round_id or "unknown")
    frontier[key] = {
        "total_cost": combined,
        "selection_cost": sel_cost,
        "fvu": fvu,
        "k": int(k) if k is not None else None,
        "ef": int(ef) if ef is not None else None,
        "architecture": parsed.get("architecture"),
        "commit": commit,
        "config": config,
        "checkpoint": parsed.get("checkpoint"),
        "peak_memory_gb": parsed.get("peak_memory_gb"),
        "deployment_accesses": deploy_accesses,
        "deployment_ratio": full_cost.get("deployment_ratio"),
        "target_profile": target_profile.to_dict(),
        "cost_model_label": target_profile.cost_model_label,
        "metric_version": "total_cost_v1",
    }

    # Remove points dominated by the new entry
    new_pt = {"total_cost": combined, "fvu": fvu}
    to_remove = [
        fk
        for fk, entry in frontier.items()
        if fk != key
        and isinstance(entry, dict)
        and _entry_target_profile(entry).profile_id == target_profile.profile_id
        and _pareto_dominates(new_pt, _entry_to_point(entry))
    ]
    for fk in to_remove:
        del frontier[fk]


# ---------------------------------------------------------------------------
# Pareto helpers — 2D (total_cost, FVU)
# ---------------------------------------------------------------------------


def compute_pareto_frontier(frontier: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute the non-dominated 2D Pareto frontier."""
    points = _frontier_points(frontier)
    pareto: list[dict[str, Any]] = []
    for candidate in points:
        dominated = any(
            _pareto_dominates(other, candidate) and other is not candidate
            for other in points
        )
        if not dominated:
            pareto.append(candidate)
    pareto.sort(key=lambda x: x["total_cost"])
    return pareto


def compact_frontier(frontier: dict[str, Any]) -> dict[str, Any]:
    """Compact a stored frontier using the same semantics as live updates.

    This removes:
    - malformed entries that cannot be interpreted as (total_cost, fvu)
    - near-duplicate archive-like points within current tolerance
    - points dominated by earlier kept points

    The surviving representative entry keeps its original stored metadata.
    Entries are replayed in round-key order so the result matches live
    controller behavior as closely as possible.
    """
    compacted: dict[str, Any] = {}
    for key, entry in sorted(frontier.items(), key=lambda kv: _frontier_sort_key(kv[0])):
        if not isinstance(entry, dict):
            continue
        try:
            candidate = _entry_to_point(entry)
            candidate_profile = _entry_target_profile(entry)
        except (TypeError, ValueError, KeyError, IndexError):
            continue

        duplicate = False
        kept_points: list[tuple[str, dict[str, Any]]] = []
        for kept_key, kept_entry in compacted.items():
            try:
                if _entry_target_profile(kept_entry).profile_id != candidate_profile.profile_id:
                    continue
                kept_pt = _entry_to_point(kept_entry)
            except (TypeError, ValueError, KeyError, IndexError):
                continue
            kept_points.append((kept_key, kept_pt))
            cost_close = (
                abs(kept_pt["total_cost"] - candidate["total_cost"])
                / max(candidate["total_cost"], 1)
                <= COST_REL_TOL
            )
            fvu_close = abs(kept_pt["fvu"] - candidate["fvu"]) <= FVU_TOL
            if cost_close and fvu_close:
                if candidate["fvu"] < kept_pt["fvu"] - FVU_TOL:
                    break
                duplicate = True
                break
        if duplicate:
            continue

        if any(_pareto_dominates(kept_pt, candidate) for _, kept_pt in kept_points):
            continue

        compacted[key] = dict(entry)
        to_remove = [
            kept_key
            for kept_key, kept_pt in kept_points
            if _pareto_dominates(candidate, kept_pt)
        ]
        for kept_key in to_remove:
            compacted.pop(kept_key, None)

    return compacted


def _frontier_points(
    frontier: dict[str, Any],
    target_profile_id: str | None = None,
) -> list[dict[str, Any]]:
    """Extract {total_cost, fvu} from all frontier entries."""
    points: list[dict[str, Any]] = []
    for _key, entry in frontier.items():
        if not isinstance(entry, dict):
            continue
        if target_profile_id is not None and _entry_target_profile(entry).profile_id != target_profile_id:
            continue
        try:
            points.append(_entry_to_point(entry))
        except (TypeError, ValueError, KeyError, IndexError):
            continue
    return points


def _entry_to_point(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert a frontier entry to a {total_cost, fvu} point."""
    fvu = float(entry["fvu"])
    # Prefer stored total_cost
    tc = entry.get("total_cost")
    if tc is not None:
        return {"total_cost": float(tc), "fvu": fvu}
    # Backward compat: selection_cost + deployment_accesses
    sel_cost = entry.get("selection_cost")
    deploy = entry.get("deployment_accesses", 0) or 0
    if sel_cost is not None:
        return {"total_cost": float(sel_cost) + float(deploy), "fvu": fvu}
    # Last resort: recompute
    return {"total_cost": _estimate_total_cost_from_entry(entry), "fvu": fvu}


def _pareto_dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return True if point a dominates point b in 2D (total_cost, FVU).

    a dominates b if a is at least as good in both dimensions and strictly
    better in at least one.
    """
    cost_ok = a["total_cost"] <= b["total_cost"]
    fvu_ok = a["fvu"] <= b["fvu"] + FVU_TOL
    strictly_better = (
        (a["total_cost"] < b["total_cost"])
        or (a["fvu"] < b["fvu"] - FVU_TOL)
    )
    return cost_ok and fvu_ok and strictly_better


def _frontier_sort_key(key: str) -> tuple[int, int, str]:
    """Sort round-based keys chronologically, then keep legacy keys last."""
    if key.startswith("r") and key[1:].isdigit():
        return (0, int(key[1:]), key)
    m = re.search(r"(\d+)$", key)
    if m:
        return (1, int(m.group(1)), key)
    return (2, 0, key)


# ---------------------------------------------------------------------------
# Cost computation helpers
# ---------------------------------------------------------------------------


def _compute_candidate_cost_full(
    parsed: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute full cost dict (encoder + deployment + combined) for a candidate."""
    cfg = config or {}
    arch = parsed.get("architecture") or cfg.get("architecture", "topk")
    k = int(parsed.get("k") or cfg.get("k") or 128)
    ef = int(parsed.get("expansion_factor") or cfg.get("expansion_factor") or 12)
    target_profile = _resolve_cost_profile(cfg)

    extra_config = _extract_extra_config(cfg)
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

    # Fallback: rough estimate (encoder-only, no deployment info)
    fallback = float(target_profile.d_in * target_profile.d_in * ef)
    return {"total_accesses": fallback, "combined_accesses": fallback, "error": "fallback"}


def _compute_candidate_total_cost(
    parsed: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> float:
    """Compute total_cost (encoder + deployment) for a candidate result."""
    cost = _compute_candidate_cost_full(parsed, config)
    if "combined_accesses" in cost:
        return float(cost["combined_accesses"])
    # Fallback: encoder + deployment if available
    sel = float(cost.get("total_accesses", 0))
    deploy = float(cost.get("deployment_accesses", 0) or 0)
    return sel + deploy


def _extract_extra_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract architecture-specific params from config for cost estimation."""
    extra: dict[str, Any] = {}
    for env_key, cfg_key in [
        ("trunk_rank", "trunk_rank"),
        ("num_codes", "num_codes"),
        ("stage1_ratio", "stage1_ratio"),
        ("factorized_hidden_dim", "factorized_hidden_dim"),
        ("TRUNK_RANK", "trunk_rank"),
        ("NUM_CODES", "num_codes"),
        ("STAGE1_RATIO", "stage1_ratio"),
        ("FACTORIZED_HIDDEN_DIM", "factorized_hidden_dim"),
        ("NUM_EXPERTS", "num_experts"),
        ("ACTIVE_EXPERTS", "active_experts"),
    ]:
        val = config.get(env_key)
        if val is not None and val != "":
            try:
                extra[cfg_key] = float(val) if "." in str(val) else int(val)
            except (ValueError, TypeError):
                pass
    return extra


def _estimate_total_cost_from_entry(entry: dict[str, Any]) -> float:
    """Estimate total_cost (encoder + deployment) from a frontier entry."""
    cfg = entry.get("config", {})
    arch = str(
        entry.get("architecture")
        or cfg.get("architecture")
        or cfg.get("family_name")
        or "topk"
    ).lower()
    k = int(entry.get("k") or cfg.get("k") or 128)
    ef = int(entry.get("ef") or cfg.get("expansion_factor") or 12)
    target_profile = _entry_target_profile(entry)

    extra_config = _extract_extra_config(cfg)
    cost = compute_selection_cost(
        arch,
        k=k,
        ef=ef,
        d_in=target_profile.d_in,
        n_output=target_profile.n_output,
        extra_config=extra_config or None,
    )
    if "error" not in cost:
        if "combined_accesses" in cost:
            return float(cost["combined_accesses"])
        sel = float(cost.get("total_accesses", 0))
        deploy = float(cost.get("deployment_accesses", 0) or 0)
        return sel + deploy

    return float(target_profile.d_in * target_profile.d_in * ef)


def _entry_target_profile(entry: dict[str, Any]) -> TargetProfile:
    cfg = dict(entry.get("config", {}) or {})
    if entry.get("target_profile") is not None and "target_profile" not in cfg:
        cfg["target_profile"] = entry["target_profile"]
    return _resolve_cost_profile(cfg)


def _resolve_cost_profile(config: dict[str, Any]) -> TargetProfile:
    return resolve_target_profile(config)
