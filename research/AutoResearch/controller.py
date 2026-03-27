"""Lightweight result evaluation for the autoresearch framework.

Replaces the old controller.py CLI.  Called as Python functions, not
as a subprocess.  No proxy/full tier distinction — single frontier.

The frontier is a 2D Pareto front over (selection_cost, FVU).
Lower selection_cost and lower FVU are both better.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .compatibility import compute_selection_cost


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
# Decision logic — 2D Pareto front: (selection_cost, FVU)
# ---------------------------------------------------------------------------

FVU_TOL = 0.001
COST_REL_TOL = 0.05  # 5% relative tolerance for selection_cost near-duplicate
D_IN_DEFAULT = 1024   # default d_in for cost estimation (fallback)

# Known hookpoint → d_in mapping for quick lookup
_HOOKPOINT_DIN: dict[str, int] = {
    # Qwen3-0.6B / Pythia-160m: o_proj output dim = 1024
    "layers.[3].self_attn.o_proj": 1024,
}


def frontier_key(round_id: int | str) -> str:
    """Canonical frontier key based on round id."""
    return f"r{round_id}"


def decide(
    frontier: dict[str, Any],
    parsed: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> str:
    """Compare a training result against the 2D Pareto frontier.

    The frontier is two-dimensional: **(selection_cost, FVU)**.
    Lower selection_cost and lower FVU are both better.

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

    sel_cost = _compute_candidate_cost(parsed, config)
    candidate = {"selection_cost": sel_cost, "fvu": fvu}

    points = _frontier_points(frontier)
    if not points:
        return "keep"

    # Check for near-duplicate (both dimensions within tolerance)
    for pt in points:
        cost_close = (
            abs(pt["selection_cost"] - sel_cost) / max(sel_cost, 1) <= COST_REL_TOL
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
    sel_cost = _compute_candidate_cost(parsed, config)

    key = frontier_key(round_id or "unknown")
    frontier[key] = {
        "selection_cost": sel_cost,
        "fvu": fvu,
        "k": int(k) if k is not None else None,
        "ef": int(ef) if ef is not None else None,
        "architecture": parsed.get("architecture"),
        "commit": commit,
        "config": config,
        "checkpoint": parsed.get("checkpoint"),
        "peak_memory_gb": parsed.get("peak_memory_gb"),
    }

    # Remove points dominated by the new entry
    new_pt = {"selection_cost": sel_cost, "fvu": fvu}
    to_remove = [
        fk
        for fk, entry in frontier.items()
        if fk != key and isinstance(entry, dict) and _pareto_dominates(new_pt, _entry_to_point(entry))
    ]
    for fk in to_remove:
        del frontier[fk]


# ---------------------------------------------------------------------------
# Pareto helpers — 2D (selection_cost, FVU)
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
    pareto.sort(key=lambda x: x["selection_cost"])
    return pareto


def _frontier_points(frontier: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract {selection_cost, fvu} from all frontier entries."""
    points: list[dict[str, Any]] = []
    for _key, entry in frontier.items():
        if not isinstance(entry, dict):
            continue
        try:
            points.append(_entry_to_point(entry))
        except (TypeError, ValueError, KeyError, IndexError):
            continue
    return points


def _entry_to_point(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert a frontier entry to a {selection_cost, fvu} point."""
    fvu = float(entry["fvu"])
    # Prefer stored selection_cost; fall back to estimation
    sel_cost = entry.get("selection_cost")
    if sel_cost is None:
        sel_cost = _estimate_cost_from_entry(entry)
    return {"selection_cost": float(sel_cost), "fvu": fvu}


def _pareto_dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return True if point a dominates point b in 2D (selection_cost, FVU).

    a dominates b if a is at least as good in both dimensions and strictly
    better in at least one.
    """
    cost_ok = a["selection_cost"] <= b["selection_cost"]
    fvu_ok = a["fvu"] <= b["fvu"] + FVU_TOL
    strictly_better = (
        (a["selection_cost"] < b["selection_cost"])
        or (a["fvu"] < b["fvu"] - FVU_TOL)
    )
    return cost_ok and fvu_ok and strictly_better


# ---------------------------------------------------------------------------
# Cost computation helpers
# ---------------------------------------------------------------------------


def _compute_candidate_cost(
    parsed: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> float:
    """Compute selection_cost for a candidate result."""
    cfg = config or {}
    arch = parsed.get("architecture") or cfg.get("architecture", "topk")
    k = int(parsed.get("k") or cfg.get("k") or 128)
    ef = int(parsed.get("expansion_factor") or cfg.get("expansion_factor") or 12)
    d_in = _resolve_d_in(cfg)

    extra_config = _extract_extra_config(cfg)
    cost = compute_selection_cost(arch, k=k, ef=ef, d_in=d_in, extra_config=extra_config or None)
    if "error" not in cost:
        return float(cost["total_accesses"])

    # Fallback: rough estimate as d_in * N where N = d_in * ef
    return float(d_in * d_in * ef)


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
    ]:
        val = config.get(env_key)
        if val is not None and val != "":
            try:
                extra[cfg_key] = float(val) if "." in str(val) else int(val)
            except (ValueError, TypeError):
                pass
    return extra


def _estimate_cost_from_entry(entry: dict[str, Any]) -> float:
    """Estimate selection_cost from a legacy frontier entry without stored cost."""
    cfg = entry.get("config", {})
    arch = str(
        entry.get("architecture")
        or cfg.get("architecture")
        or cfg.get("family_name")
        or "topk"
    ).lower()
    k = int(entry.get("k") or cfg.get("k") or 128)
    ef = int(entry.get("ef") or cfg.get("expansion_factor") or 12)
    d_in = _resolve_d_in(cfg)

    extra_config = _extract_extra_config(cfg)
    cost = compute_selection_cost(arch, k=k, ef=ef, d_in=d_in, extra_config=extra_config or None)
    if "error" not in cost:
        return float(cost["total_accesses"])

    return float(d_in * d_in * ef)


def _resolve_d_in(config: dict[str, Any]) -> int:
    """Resolve d_in from config, falling back to D_IN_DEFAULT."""
    # Explicit d_in in config takes priority
    d_in = config.get("d_in") or config.get("D_IN")
    if d_in is not None:
        try:
            return int(d_in)
        except (ValueError, TypeError):
            pass
    # Try hookpoint lookup
    hookpoints = str(config.get("hookpoints") or config.get("HOOKPOINTS") or "")
    if hookpoints in _HOOKPOINT_DIN:
        return _HOOKPOINT_DIN[hookpoints]
    return D_IN_DEFAULT
