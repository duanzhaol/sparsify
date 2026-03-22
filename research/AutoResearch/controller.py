"""Lightweight result evaluation for the autoresearch framework.

Replaces the old controller.py CLI.  Called as Python functions, not
as a subprocess.  No proxy/full tier distinction — single frontier.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


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
        match = pattern.search(text)
        parsed[key] = match.group(1).strip() if match else None

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
# Decision logic
# ---------------------------------------------------------------------------

FVU_TOL = 0.001


def frontier_key(k: int, ef: int) -> str:
    """Canonical frontier key: ``'{k}_{ef}'``."""
    return f"{k}_{ef}"


def decide(
    frontier: dict[str, Any],
    parsed: dict[str, Any],
) -> str:
    """Compare a training result against the frontier.

    The frontier is three-dimensional: **(K, EF, FVU)**.
    Lower K, lower EF, and lower FVU are all better.

    Returns one of: "keep", "crash", "archive", "discard".
    - keep:    result improves the frontier (same slot or Pareto non-dominated)
    - crash:   training failed, no usable metric
    - archive: result is within FVU_TOL of current best at same (K, EF)
    - discard: result is dominated by existing frontier points
    """
    status = parsed.get("status")
    fvu = parsed.get("val_fvu")
    k = parsed.get("k")
    ef = parsed.get("expansion_factor")

    if status != "ok" or fvu is None or k is None:
        return "crash"

    if ef is None:
        ef = 8  # fallback default

    key = frontier_key(int(k), int(ef))
    current = frontier.get(key)

    # Check: improvement at same (K, EF) slot?
    improve_same_slot = False
    if current is None:
        improve_same_slot = True
    else:
        cur_fvu = float(current.get("fvu", float("inf")))
        if fvu < cur_fvu - FVU_TOL:
            improve_same_slot = True

    # Check: Pareto non-dominated across all (K, EF, FVU)?
    candidate = {"k": int(k), "ef": int(ef), "fvu": fvu}
    current_points = _frontier_points(frontier)
    pareto_non_dominated = not current_points or not any(
        _pareto_dominates(pt, candidate) for pt in current_points
    )

    if improve_same_slot or pareto_non_dominated:
        return "keep"

    if current is not None and abs(fvu - float(current.get("fvu", float("inf")))) <= FVU_TOL:
        return "archive"

    return "discard"


def update_frontier(
    frontier: dict[str, Any],
    parsed: dict[str, Any],
    decision: str,
    config: dict[str, Any],
    commit: str,
) -> None:
    """If decision is 'keep', update the frontier in-place."""
    if decision != "keep":
        return
    k = parsed.get("k")
    fvu = parsed.get("val_fvu")
    ef = parsed.get("expansion_factor")
    if k is None or fvu is None:
        return
    if ef is None:
        ef = 8

    key = frontier_key(int(k), int(ef))
    frontier[key] = {
        "k": int(k),
        "ef": int(ef),
        "fvu": fvu,
        "architecture": parsed.get("architecture"),
        "commit": commit,
        "config": config,
        "checkpoint": parsed.get("checkpoint"),
        "peak_memory_gb": parsed.get("peak_memory_gb"),
    }


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------


def compute_pareto_frontier(frontier: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute the non-dominated Pareto frontier from a (K,EF)→point dict."""
    points = _frontier_points(frontier)
    pareto: list[dict[str, Any]] = []
    for candidate in points:
        dominated = any(
            _pareto_dominates(other, candidate) and other is not candidate
            for other in points
        )
        if not dominated:
            pareto.append(candidate)
    pareto.sort(key=lambda x: (x["k"], x["ef"], x["fvu"]))
    return pareto


def _frontier_points(frontier: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for key, entry in frontier.items():
        if not isinstance(entry, dict):
            continue
        try:
            # New format stores k and ef explicitly in entry
            k = int(entry.get("k", key.split("_")[0] if "_" in key else key))
            ef = int(entry.get("ef", key.split("_")[1] if "_" in key else 8))
            points.append({
                "k": k,
                "ef": ef,
                "fvu": float(entry["fvu"]),
            })
        except (TypeError, ValueError, KeyError, IndexError):
            continue
    return points


def _pareto_dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return True if point a dominates point b.

    Three dimensions: lower K, lower EF, lower FVU are all better.
    a dominates b if a is at least as good in all dimensions and strictly
    better in at least one.
    """
    fvu_ok = a["fvu"] <= b["fvu"] + FVU_TOL
    k_ok = a["k"] <= b["k"]
    ef_ok = a["ef"] <= b["ef"]
    strictly_better = (
        (a["fvu"] < b["fvu"] - FVU_TOL) or (a["k"] < b["k"]) or (a["ef"] < b["ef"])
    )
    return fvu_ok and k_ok and ef_ok and strictly_better
