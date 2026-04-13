from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SchedulerDecision:
    action: str
    rationale: str
    needs_agent: bool = False


def should_force_stop(
    *,
    incumbent_objective: float | None,
    current_best_objective: float | None,
    last_window_delta: float | None,
    checkpoints_seen: int,
) -> bool:
    if current_best_objective is None:
        return checkpoints_seen >= 2
    if incumbent_objective is None or checkpoints_seen < 2:
        return False
    if current_best_objective - incumbent_objective >= 0.05 and (
        last_window_delta is None or last_window_delta <= 0.001
    ):
        return True
    if checkpoints_seen >= 4 and (last_window_delta is None or last_window_delta <= 0.0002):
        return True
    return False


def stop_action(can_spawn_more: bool) -> str:
    return (
        "stop_current_and_spawn_next"
        if can_spawn_more
        else "stop_current_and_finish_run"
    )


def evaluate_checkpoint(
    *,
    current_trial: Any,
    incumbent_trial: Any | None,
    can_spawn_more: bool,
) -> SchedulerDecision:
    if getattr(current_trial, "invalid_reason", None):
        return SchedulerDecision(
            action=stop_action(can_spawn_more),
            rationale=f"invalid checkpoint: {current_trial.invalid_reason}",
        )

    current_best = getattr(current_trial, "best_objective", None)
    checkpoints_seen = int(getattr(current_trial, "checkpoint_decisions", 0))
    delta_best = getattr(current_trial, "delta_best_objective", None)
    current_cost = getattr(current_trial, "total_cost_ratio", None)
    incumbent_best = getattr(incumbent_trial, "best_objective", None)
    incumbent_cost = getattr(incumbent_trial, "total_cost_ratio", None)

    if should_force_stop(
        incumbent_objective=incumbent_best,
        current_best_objective=current_best,
        last_window_delta=delta_best,
        checkpoints_seen=checkpoints_seen,
    ):
        return SchedulerDecision(
            action=stop_action(can_spawn_more),
            rationale="programmatic stop: weak objective and no meaningful recent gain",
        )

    if incumbent_best is None:
        if checkpoints_seen < 3:
            return SchedulerDecision(
                action="continue_current",
                rationale="programmatic continue: still seeding first incumbent",
            )
        return SchedulerDecision(
            action="continue_current",
            rationale="programmatic continue: no incumbent exists yet",
        )

    if current_best is not None and current_best <= incumbent_best - 0.003:
        return SchedulerDecision(
            action="continue_current",
            rationale="programmatic continue: already beating incumbent",
        )

    if delta_best is not None and delta_best >= 0.0015:
        return SchedulerDecision(
            action="continue_current",
            rationale="programmatic continue: last 15M still improved objective",
        )

    if (
        current_cost is not None
        and incumbent_cost is not None
        and current_cost <= incumbent_cost - 0.015
        and current_best is not None
        and current_best <= incumbent_best + 0.02
    ):
        return SchedulerDecision(
            action="continue_current",
            rationale="programmatic continue: much cheaper structure still has deployment potential",
        )

    return SchedulerDecision(
        action="continue_current",
        rationale="gray-zone: needs agent-style decision",
        needs_agent=True,
    )
