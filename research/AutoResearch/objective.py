"""Single-objective utilities for AutoResearch.

The current proxy objective is:

    objective_score = total_cost_ratio + exceed_alpha_0_50

Lower is better. FVU remains a diagnostic / tie-break metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .compatibility import compute_selection_cost, extract_cost_extra_config
from .target_profile import TargetProfile, resolve_target_profile

EXCEED_ALPHA = 0.50
EXCEED_FIELD = "exceed_alpha_0_50"
OBJECTIVE_FIELD = "objective_score"
OBJECTIVE_SCORE_TOL = 0.005
LEADERBOARD_LIMIT = 10


def normalize_hookpoint_name(hookpoint: str | None) -> str:
    """Normalize hookpoint naming differences like ``layers.[3]`` vs ``layers.3``."""
    if not hookpoint:
        return ""
    return str(hookpoint).replace("[", "").replace("]", "")


def exceed_metric_suffix(alpha: float = EXCEED_ALPHA) -> str:
    return f"exceed_alpha_{alpha:.2f}"


def safe_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_metric_from_step(
    step_record: dict[str, Any] | None,
    metric_suffix: str,
    hookpoint: str | None = None,
) -> float | None:
    """Extract a metric like ``fvu`` or ``exceed_alpha_0.50`` from one step record."""
    if not isinstance(step_record, dict):
        return None

    if hookpoint:
        candidates = [
            f"{hookpoint}/{metric_suffix}",
            f"{normalize_hookpoint_name(hookpoint)}/{metric_suffix}",
        ]
        for key in candidates:
            value = safe_float(step_record.get(key))
            if value is not None:
                return value

    suffix = f"/{metric_suffix}"
    matches: list[tuple[str, float]] = []
    target_norm = normalize_hookpoint_name(hookpoint)
    for key, raw_value in step_record.items():
        if not str(key).endswith(suffix):
            continue
        value = safe_float(raw_value)
        if value is None:
            continue
        matches.append((str(key), value))

    if not matches:
        return None

    if target_norm:
        for key, value in matches:
            prefix = key[: -len(suffix)]
            if normalize_hookpoint_name(prefix) == target_norm:
                return value

    if len(matches) == 1:
        return matches[0][1]
    return None


def extract_step_fvu(
    step_record: dict[str, Any] | None,
    hookpoint: str | None = None,
) -> float | None:
    return extract_metric_from_step(step_record, "fvu", hookpoint=hookpoint)


def extract_step_exceed_alpha_0_50(
    step_record: dict[str, Any] | None,
    hookpoint: str | None = None,
) -> float | None:
    return extract_metric_from_step(
        step_record,
        exceed_metric_suffix(EXCEED_ALPHA),
        hookpoint=hookpoint,
    )


def read_step_records(metrics_path: str | Path | None) -> list[dict[str, Any]]:
    path = Path(metrics_path) if metrics_path is not None else None
    if path is None or not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "step":
                records.append(record)
    return records


def read_latest_step(metrics_path: str | Path | None) -> dict[str, Any] | None:
    records = read_step_records(metrics_path)
    return records[-1] if records else None


def compute_objective_score(
    total_cost_ratio: Any,
    exceed_alpha_0_50: Any,
) -> float | None:
    cost = safe_float(total_cost_ratio)
    exceed = safe_float(exceed_alpha_0_50)
    if cost is None or exceed is None:
        return None
    return cost + exceed


def objective_rank_tuple(
    objective_score: Any,
    total_cost_ratio: Any,
    exceed_alpha_0_50: Any,
    fvu: Any,
) -> tuple[float, float, float, float]:
    """Ranking tuple: lower is better in every component."""
    score = safe_float(objective_score)
    cost = safe_float(total_cost_ratio)
    exceed = safe_float(exceed_alpha_0_50)
    fvu_value = safe_float(fvu)
    inf = float("inf")
    return (
        score if score is not None else inf,
        cost if cost is not None else inf,
        exceed if exceed is not None else inf,
        fvu_value if fvu_value is not None else inf,
    )


def is_objective_near_duplicate(
    score_a: Any,
    score_b: Any,
    tol: float = OBJECTIVE_SCORE_TOL,
) -> bool:
    a = safe_float(score_a)
    b = safe_float(score_b)
    if a is None or b is None:
        return False
    return abs(a - b) < tol


def entry_config_with_target(entry: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(entry.get("config", {}) or {})
    if entry.get("target_profile") is not None and "target_profile" not in cfg:
        cfg["target_profile"] = entry["target_profile"]
    return cfg


def cached_selection_cost(
    architecture: str,
    k: int,
    ef: int,
    target_profile: TargetProfile,
    *,
    extra_config: dict[str, Any] | None = None,
    cost_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        "architecture": architecture,
        "k": int(k),
        "ef": int(ef),
        "d_in": int(target_profile.d_in),
        "n_output": int(target_profile.n_output),
        "extra_config": extra_config or {},
    }
    cache_key = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    if cost_cache is None:
        return compute_selection_cost(
            architecture,
            k=k,
            ef=ef,
            d_in=target_profile.d_in,
            n_output=target_profile.n_output,
            extra_config=extra_config or None,
        )
    if cache_key not in cost_cache:
        cost_cache[cache_key] = compute_selection_cost(
            architecture,
            k=k,
            ef=ef,
            d_in=target_profile.d_in,
            n_output=target_profile.n_output,
            extra_config=extra_config or None,
        )
    return cost_cache[cache_key]


def candidate_objective_metrics(
    architecture: str,
    k: int,
    ef: int,
    target_profile: TargetProfile,
    *,
    extra_config: dict[str, Any] | None = None,
    exceed_alpha_0_50: Any = None,
    fvu: Any = None,
    cost_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    original = float(target_profile.original_matmul_accesses)
    cost = cached_selection_cost(
        architecture,
        k,
        ef,
        target_profile,
        extra_config=extra_config,
        cost_cache=cost_cache,
    )

    if "error" not in cost:
        selection_cost = float(cost.get("total_accesses", 0) or 0)
        selection_cost_ratio = safe_float(cost.get("ratio"))
        deployment_accesses = float(cost.get("deployment_accesses", 0) or 0)
        deployment_ratio = safe_float(cost.get("deployment_ratio"))
        total_cost = float(cost.get("combined_accesses", selection_cost + deployment_accesses) or 0)
        total_cost_ratio = safe_float(cost.get("combined_ratio"))
    else:
        selection_cost = float(target_profile.d_in * target_profile.d_in * ef)
        selection_cost_ratio = selection_cost / original if original > 0 else None
        deployment_accesses = 0.0
        deployment_ratio = 0.0 if original > 0 else None
        total_cost = selection_cost
        total_cost_ratio = total_cost / original if original > 0 else None

    exceed = safe_float(exceed_alpha_0_50)
    objective_score = compute_objective_score(total_cost_ratio, exceed)
    return {
        "selection_cost": selection_cost,
        "selection_cost_ratio": selection_cost_ratio,
        "deployment_accesses": deployment_accesses,
        "deployment_ratio": deployment_ratio,
        "total_cost": total_cost,
        "total_cost_ratio": total_cost_ratio,
        EXCEED_FIELD: exceed,
        "objective_score": objective_score,
        "fvu": safe_float(fvu),
        "feasible": bool(total_cost <= target_profile.budget_accesses()),
    }


def entry_objective_metrics(
    entry: dict[str, Any],
    *,
    cost_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cfg = entry_config_with_target(entry)
    target_profile = resolve_target_profile(cfg)
    original = float(target_profile.original_matmul_accesses)

    selection_cost = safe_float(entry.get("selection_cost"))
    selection_cost_ratio = safe_float(entry.get("selection_cost_ratio"))
    deployment_accesses = safe_float(entry.get("deployment_accesses"))
    deployment_ratio = safe_float(entry.get("deployment_ratio"))
    total_cost = safe_float(entry.get("total_cost"))
    total_cost_ratio = safe_float(entry.get("total_cost_ratio"))

    needs_cost = any(
        value is None
        for value in (
            selection_cost_ratio,
            deployment_ratio,
            total_cost,
            total_cost_ratio,
        )
    )
    if needs_cost:
        arch = str(entry.get("architecture") or cfg.get("architecture") or "topk").lower()
        k = int(entry.get("k") or cfg.get("k") or 128)
        ef = int(entry.get("ef") or cfg.get("expansion_factor") or 12)
        computed = candidate_objective_metrics(
            arch,
            k,
            ef,
            target_profile,
            extra_config=extract_cost_extra_config(cfg),
            exceed_alpha_0_50=entry.get(EXCEED_FIELD),
            fvu=entry.get("fvu"),
            cost_cache=cost_cache,
        )
        selection_cost = computed["selection_cost"]
        selection_cost_ratio = computed["selection_cost_ratio"]
        deployment_accesses = computed["deployment_accesses"]
        deployment_ratio = computed["deployment_ratio"]
        total_cost = computed["total_cost"]
        total_cost_ratio = computed["total_cost_ratio"]

    if selection_cost is None and selection_cost_ratio is not None and original > 0:
        selection_cost = selection_cost_ratio * original
    if deployment_accesses is None and deployment_ratio is not None and original > 0:
        deployment_accesses = deployment_ratio * original
    if total_cost is None:
        if total_cost_ratio is not None and original > 0:
            total_cost = total_cost_ratio * original
        elif selection_cost is not None:
            total_cost = selection_cost + (deployment_accesses or 0.0)
    if total_cost_ratio is None and total_cost is not None and original > 0:
        total_cost_ratio = total_cost / original
    if selection_cost_ratio is None and selection_cost is not None and original > 0:
        selection_cost_ratio = selection_cost / original
    if deployment_accesses is None:
        deployment_accesses = 0.0
    if deployment_ratio is None and original > 0:
        deployment_ratio = deployment_accesses / original

    exceed = safe_float(entry.get(EXCEED_FIELD))
    objective_score = safe_float(entry.get(OBJECTIVE_FIELD))
    if objective_score is None:
        objective_score = compute_objective_score(total_cost_ratio, exceed)

    return {
        "selection_cost": selection_cost,
        "selection_cost_ratio": selection_cost_ratio,
        "deployment_accesses": deployment_accesses,
        "deployment_ratio": deployment_ratio,
        "total_cost": total_cost,
        "total_cost_ratio": total_cost_ratio,
        EXCEED_FIELD: exceed,
        "objective_score": objective_score,
        "fvu": safe_float(entry.get("fvu")),
        "feasible": bool(total_cost is not None and total_cost <= target_profile.budget_accesses()),
    }
