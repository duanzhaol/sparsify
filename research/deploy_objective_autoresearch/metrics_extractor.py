from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_COST_RE = re.compile(r"total_ratio=([0-9]+(?:\.[0-9]+)?)x")


@dataclass(slots=True)
class TrialSnapshot:
    total_cost_ratio: float | None
    latest_exceed_alpha_0_50: float | None
    best_exceed_alpha_0_50: float | None
    latest_fvu: float | None
    best_fvu: float | None
    best_objective: float | None
    delta_best_exceed: float | None
    delta_best_fvu: float | None
    delta_best_objective: float | None
    tokens_seen: int
    latest_step: int | None
    checkpoint_count: int
    invalid_reason: str | None = None


def read_step_records(metrics_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not metrics_path.exists():
        return records
    with metrics_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "step":
                records.append(row)
    return records


def read_latest_step_record(metrics_path: Path) -> dict[str, Any] | None:
    records = read_step_records(metrics_path)
    return records[-1] if records else None


def extract_total_cost_ratio(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    matches = _COST_RE.findall(log_path.read_text(errors="replace"))
    if not matches:
        return None
    return float(matches[-1])


def _best_value(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in records if key in row]
    if not values:
        return None
    return min(values)


def _latest_value(records: list[dict[str, Any]], key: str) -> float | None:
    for row in reversed(records):
        if key in row:
            return float(row[key])
    return None


def _best_objective(cost_ratio: float | None, exceed: float | None) -> float | None:
    if cost_ratio is None or exceed is None:
        return None
    return cost_ratio + exceed


def _improvement(previous: float | None, current: float | None) -> float | None:
    if previous is None or current is None:
        return None
    return previous - current


def extract_trial_snapshot(
    log_path: Path,
    metrics_path: Path,
    hook_metric_prefix: str,
    *,
    checkpoint_interval_tokens: int = 15_000_000,
    window_start_tokens: int = 0,
) -> TrialSnapshot:
    total_cost_ratio = extract_total_cost_ratio(log_path)
    if not metrics_path.exists():
        return TrialSnapshot(
            total_cost_ratio=total_cost_ratio,
            latest_exceed_alpha_0_50=None,
            best_exceed_alpha_0_50=None,
            latest_fvu=None,
            best_fvu=None,
            best_objective=None,
            delta_best_exceed=None,
            delta_best_fvu=None,
            delta_best_objective=None,
            tokens_seen=0,
            latest_step=None,
            checkpoint_count=0,
            invalid_reason="metrics_missing",
        )

    exceed_key = f"{hook_metric_prefix}/exceed_alpha_0.50"
    fvu_key = f"{hook_metric_prefix}/fvu"
    records = read_step_records(metrics_path)
    if not records:
        return TrialSnapshot(
            total_cost_ratio=total_cost_ratio,
            latest_exceed_alpha_0_50=None,
            best_exceed_alpha_0_50=None,
            latest_fvu=None,
            best_fvu=None,
            best_objective=None,
            delta_best_exceed=None,
            delta_best_fvu=None,
            delta_best_objective=None,
            tokens_seen=0,
            latest_step=None,
            checkpoint_count=0,
            invalid_reason="step_records_missing",
        )

    latest = records[-1]
    tokens_seen = int(latest.get("total_tokens") or 0)
    latest_step = latest.get("step")
    latest_exceed = _latest_value(records, exceed_key)
    best_exceed = _best_value(records, exceed_key)
    latest_fvu = _latest_value(records, fvu_key)
    best_fvu = _best_value(records, fvu_key)
    best_objective = _best_objective(total_cost_ratio, best_exceed)

    prev_records = [
        row for row in records if int(row.get("total_tokens") or 0) <= window_start_tokens
    ]
    prev_best_exceed = _best_value(prev_records, exceed_key)
    prev_best_fvu = _best_value(prev_records, fvu_key)
    prev_best_objective = _best_objective(total_cost_ratio, prev_best_exceed)

    checkpoint_count = 0
    if checkpoint_interval_tokens > 0:
        checkpoint_count = tokens_seen // checkpoint_interval_tokens

    return TrialSnapshot(
        total_cost_ratio=total_cost_ratio,
        latest_exceed_alpha_0_50=latest_exceed,
        best_exceed_alpha_0_50=best_exceed,
        latest_fvu=latest_fvu,
        best_fvu=best_fvu,
        best_objective=best_objective,
        delta_best_exceed=_improvement(prev_best_exceed, best_exceed),
        delta_best_fvu=_improvement(prev_best_fvu, best_fvu),
        delta_best_objective=_improvement(prev_best_objective, best_objective),
        tokens_seen=tokens_seen,
        latest_step=int(latest_step) if latest_step is not None else None,
        checkpoint_count=checkpoint_count,
        invalid_reason=None,
    )
