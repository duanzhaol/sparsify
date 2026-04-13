from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import SearchConfig


@dataclass(slots=True)
class AgentDecision:
    action: str
    rationale: str
    next_params: dict[str, int | float] | None = None
    source: str = "heuristic"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


ALLOWED_ACTIONS = {
    "continue_current",
    "stop_current_and_spawn_next",
    "stop_current_and_finish_run",
}


def parse_agent_decision(payload: str | dict[str, Any]) -> AgentDecision:
    raw = json.loads(payload) if isinstance(payload, str) else dict(payload)
    action = str(raw.get("action") or "").strip()
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported agent action: {action!r}")
    next_params = raw.get("next_params")
    if next_params is not None and not isinstance(next_params, dict):
        raise ValueError("next_params must be a dict when provided")
    return AgentDecision(
        action=action,
        rationale=str(raw.get("rationale") or ""),
        next_params=next_params,
        source=str(raw.get("source") or "external"),
    )


def build_agent_context(
    cfg: SearchConfig,
    current_trial: Any,
    incumbent_trial: Any | None,
    recent_trials: list[Any],
) -> dict[str, Any]:
    def trial_summary(trial: Any | None) -> dict[str, Any] | None:
        if trial is None:
            return None
        return {
            "trial_id": trial.trial_id,
            "status": trial.status,
            "params": trial.params,
            "tokens_seen": trial.tokens_seen,
            "checkpoint_decisions": trial.checkpoint_decisions,
            "total_cost_ratio": trial.total_cost_ratio,
            "best_exceed_alpha_0.50": trial.best_exceed_alpha_0_50,
            "latest_exceed_alpha_0.50": trial.latest_exceed_alpha_0_50,
            "best_fvu": trial.best_fvu,
            "latest_fvu": trial.latest_fvu,
            "best_objective": trial.best_objective,
            "delta_best_exceed": trial.delta_best_exceed,
            "delta_best_fvu": trial.delta_best_fvu,
            "delta_best_objective": trial.delta_best_objective,
        }

    return {
        "target": {
            "model_path": cfg.model_path,
            "hookpoints": cfg.hookpoints,
            "architecture": cfg.architecture,
            "objective": "total_cost_ratio + latest_exceed_alpha_0.50",
            "checkpoint_interval_tokens": cfg.checkpoint_interval_tokens,
        },
        "current_trial": trial_summary(current_trial),
        "incumbent_trial": trial_summary(incumbent_trial),
        "recent_trials": [trial_summary(trial) for trial in recent_trials[-5:]],
        "instruction": (
            "Return JSON only with action in {'continue_current', "
            "'stop_current_and_spawn_next', 'stop_current_and_finish_run'}; "
            "optionally include next_params. Prefer conservative continue."
        ),
    }


def write_agent_context(run_root: Path, trial_id: str, context: dict[str, Any]) -> Path:
    trial_context_path = run_root / "trials" / trial_id / "agent_context.json"
    trial_context_path.parent.mkdir(parents=True, exist_ok=True)
    trial_context_path.write_text(json.dumps(context, indent=2))
    current_context_path = run_root / "current_agent_context.json"
    current_context_path.write_text(json.dumps(context, indent=2))
    return trial_context_path


def maybe_load_external_decision(path: Path) -> AgentDecision | None:
    if not path.exists():
        return None
    decision = parse_agent_decision(json.loads(path.read_text()))
    path.unlink()
    return decision


def heuristic_gray_zone_decision(
    *,
    current_trial: Any,
    incumbent_trial: Any | None,
    can_spawn_more: bool,
) -> AgentDecision:
    checkpoints_seen = int(getattr(current_trial, "checkpoint_decisions", 0))
    delta_objective = getattr(current_trial, "delta_best_objective", None)
    current_objective = getattr(current_trial, "best_objective", None)
    incumbent_objective = getattr(incumbent_trial, "best_objective", None)

    if checkpoints_seen < 3:
        return AgentDecision(
            action="continue_current",
            rationale="gray-zone fallback: conservative continue before 3 checkpoints",
        )

    if delta_objective is not None and delta_objective > 0.0005:
        return AgentDecision(
            action="continue_current",
            rationale="gray-zone fallback: recent window still improves objective",
        )

    if (
        incumbent_objective is not None
        and current_objective is not None
        and current_objective <= incumbent_objective + 0.01
    ):
        return AgentDecision(
            action="continue_current",
            rationale="gray-zone fallback: objective is close to incumbent",
        )

    action = "stop_current_and_spawn_next" if can_spawn_more else "stop_current_and_finish_run"
    return AgentDecision(
        action=action,
        rationale="gray-zone fallback: plateaued and not close enough to incumbent",
    )
