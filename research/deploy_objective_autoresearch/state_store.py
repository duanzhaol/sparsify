from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import SearchConfig


TRIAL_ACTIVE_STATUSES = {
    "pending",
    "running",
    "checkpoint_ready",
    "continue_scheduled",
}
TRIAL_FINISHED_STATUSES = {"stopped", "finished"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class TrialRecord:
    trial_id: str
    params: dict[str, int | float]
    status: str
    created_at: str
    updated_at: str
    save_dir: str
    log_path: str
    checkpoint_root: str | None = None
    metrics_path: str | None = None
    target_tokens: int = 0
    tokens_seen: int = 0
    checkpoint_decisions: int = 0
    returncode: int | None = None
    total_cost_ratio: float | None = None
    latest_exceed_alpha_0_50: float | None = None
    best_exceed_alpha_0_50: float | None = None
    latest_fvu: float | None = None
    best_fvu: float | None = None
    best_objective: float | None = None
    delta_best_exceed: float | None = None
    delta_best_fvu: float | None = None
    delta_best_objective: float | None = None
    invalid_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "TrialRecord":
        return cls(**payload)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StateStore:
    root: Path

    @classmethod
    def bootstrap(cls, cfg: SearchConfig) -> "StateStore":
        store = cls(cfg.run_root)
        cfg.run_root.mkdir(parents=True, exist_ok=True)
        cfg.trials_dir.mkdir(parents=True, exist_ok=True)
        cfg.run_config_path.write_text(json.dumps(cfg.to_json(), indent=2))
        if not cfg.state_path.exists():
            cfg.state_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "status": "idle",
                        "active_trial_id": None,
                        "new_trial_count": 0,
                        "next_trial_index": 1,
                        "max_new_trials": cfg.max_new_trials,
                        "checkpoint_interval_tokens": cfg.checkpoint_interval_tokens,
                        "updated_at": utc_now_iso(),
                    },
                    indent=2,
                )
            )
        if not cfg.leaderboard_path.exists():
            cfg.leaderboard_path.write_text(json.dumps({"entries": []}, indent=2))
        cfg.trials_jsonl_path.touch(exist_ok=True)
        cfg.agent_decisions_path.touch(exist_ok=True)
        return store

    @classmethod
    def from_run_root(cls, run_root: str | Path) -> "StateStore":
        return cls(Path(run_root))

    def load_config(self) -> SearchConfig:
        return SearchConfig.from_json(json.loads((self.root / "run_config.json").read_text()))

    def load_state(self) -> dict[str, Any]:
        return json.loads((self.root / "state.json").read_text())

    def save_state(self, state: dict[str, Any]) -> None:
        state = dict(state)
        state["updated_at"] = utc_now_iso()
        (self.root / "state.json").write_text(json.dumps(state, indent=2))

    def trial_dir(self, trial_id: str) -> Path:
        return self.root / "trials" / trial_id

    def trial_path(self, trial_id: str) -> Path:
        return self.trial_dir(trial_id) / "trial.json"

    def load_trial(self, trial_id: str) -> TrialRecord:
        return TrialRecord.from_json(json.loads(self.trial_path(trial_id).read_text()))

    def save_trial(self, trial: TrialRecord) -> None:
        trial.updated_at = utc_now_iso()
        self.trial_dir(trial.trial_id).mkdir(parents=True, exist_ok=True)
        self.trial_path(trial.trial_id).write_text(json.dumps(trial.to_json(), indent=2))

    def list_trials(self) -> list[TrialRecord]:
        trials: list[TrialRecord] = []
        trials_dir = self.root / "trials"
        if not trials_dir.exists():
            return trials
        for trial_path in sorted(trials_dir.glob("*/trial.json")):
            trials.append(TrialRecord.from_json(json.loads(trial_path.read_text())))
        return trials

    def append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a") as fh:
            fh.write(json.dumps(payload) + "\n")

    def append_trial_event(
        self,
        trial_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.append_jsonl(
            self.root / "trials.jsonl",
            {
                "timestamp": utc_now_iso(),
                "trial_id": trial_id,
                "event_type": event_type,
                "payload": payload or {},
            },
        )

    def append_agent_decision(
        self,
        trial_id: str,
        decision: dict[str, Any],
    ) -> None:
        self.append_jsonl(
            self.root / "agent_decisions.jsonl",
            {
                "timestamp": utc_now_iso(),
                "trial_id": trial_id,
                **decision,
            },
        )

    def save_leaderboard(self, entries: list[dict[str, Any]]) -> None:
        (self.root / "leaderboard.json").write_text(
            json.dumps({"entries": entries}, indent=2)
        )

    def load_leaderboard(self) -> list[dict[str, Any]]:
        payload = json.loads((self.root / "leaderboard.json").read_text())
        return list(payload.get("entries", []))

    def refresh_leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for trial in self.list_trials():
            if trial.best_objective is None:
                continue
            ranked.append(
                {
                    "trial_id": trial.trial_id,
                    "status": trial.status,
                    "best_objective": trial.best_objective,
                    "best_exceed_alpha_0.50": trial.best_exceed_alpha_0_50,
                    "total_cost_ratio": trial.total_cost_ratio,
                    "best_fvu": trial.best_fvu,
                    "tokens_seen": trial.tokens_seen,
                    "params": trial.params,
                }
            )
        ranked.sort(
            key=lambda item: (
                float(item["best_objective"]),
                float(item.get("best_fvu") or 1e9),
                -int(item.get("tokens_seen") or 0),
            )
        )
        entries = ranked[:limit]
        self.save_leaderboard(entries)
        return entries

    def latest_unfinished_trial(self) -> TrialRecord | None:
        unfinished = [
            trial for trial in self.list_trials() if trial.status in TRIAL_ACTIVE_STATUSES
        ]
        if not unfinished:
            return None
        unfinished.sort(key=lambda trial: trial.trial_id)
        return unfinished[-1]

    def get_active_trial(self) -> TrialRecord | None:
        state = self.load_state()
        active_trial_id = state.get("active_trial_id")
        if not active_trial_id:
            return None
        path = self.trial_path(str(active_trial_id))
        if not path.exists():
            return None
        return self.load_trial(str(active_trial_id))

    def incumbent_trial(self) -> TrialRecord | None:
        entries = self.load_leaderboard()
        if not entries:
            return None
        trial_id = entries[0].get("trial_id")
        if not trial_id:
            return None
        path = self.trial_path(str(trial_id))
        if not path.exists():
            return None
        return self.load_trial(str(trial_id))

    def create_trial(self, cfg: SearchConfig, params: dict[str, int | float]) -> TrialRecord:
        state = self.load_state()
        trial_index = int(state.get("next_trial_index") or 1)
        trial_id = f"trial_{trial_index:04d}"
        trial_dir = self.trial_dir(trial_id)
        save_dir = cfg.checkpoint_root / trial_id
        record = TrialRecord(
            trial_id=trial_id,
            params=params,
            status="pending",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            save_dir=str(save_dir),
            log_path=str(trial_dir / "train.log"),
        )
        self.save_trial(record)
        state["active_trial_id"] = trial_id
        state["status"] = "pending"
        state["new_trial_count"] = int(state.get("new_trial_count") or 0) + 1
        state["next_trial_index"] = trial_index + 1
        self.save_state(state)
        self.append_trial_event(trial_id, "created", {"params": params})
        return record

    def set_active_trial(self, trial_id: str | None, *, status: str | None = None) -> None:
        state = self.load_state()
        state["active_trial_id"] = trial_id
        if status is not None:
            state["status"] = status
        self.save_state(state)

    def mark_run_finished(self, reason: str) -> None:
        state = self.load_state()
        state["active_trial_id"] = None
        state["status"] = "finished"
        state["finish_reason"] = reason
        self.save_state(state)

    def params_signature(self, params: dict[str, int | float]) -> str:
        ordered = sorted((key, str(value)) for key, value in params.items())
        return "|".join(f"{key}={value}" for key, value in ordered)

    def attempted_signatures(self) -> set[str]:
        return {self.params_signature(trial.params) for trial in self.list_trials()}
