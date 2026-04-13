from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .agent_interface import (
    AgentDecision,
    build_agent_context,
    heuristic_gray_zone_decision,
    maybe_load_external_decision,
    write_agent_context,
)
from .config import DEFAULT_RUN_ROOT, SearchConfig
from .metrics_extractor import extract_trial_snapshot
from .proposal_policy import normalize_candidate_params, propose_next_params
from .scheduler import evaluate_checkpoint
from .state_store import StateStore, TrialRecord
from .trial_runner import latest_checkpoint_dir, run_trial_segment, update_trial_from_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("deploy-objective-autoresearch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))

    status = subparsers.add_parser("status")
    status.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))

    metric = subparsers.add_parser("metric")
    metric.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    metric.add_argument("--json", action="store_true")
    metric.add_argument("--bootstrap-if-missing", action="store_true")

    step = subparsers.add_parser("step")
    step.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    step.add_argument("--resume-latest", action="store_true")
    step.add_argument("--dry-run-agent", action="store_true")
    step.add_argument("--decision-file")

    write_decision = subparsers.add_parser("write-decision")
    write_decision.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    write_decision.add_argument(
        "--action",
        required=True,
        choices=[
            "continue_current",
            "stop_current_and_spawn_next",
            "stop_current_and_finish_run",
        ],
    )
    write_decision.add_argument("--rationale", required=True)
    write_decision.add_argument("--next-params-json")
    write_decision.add_argument("--source", default="codex-autoresearch")
    return parser


def _store_for_run_root(run_root: str | Path) -> tuple[StateStore, SearchConfig]:
    root = Path(run_root)
    run_config_path = root / "run_config.json"
    if run_config_path.exists():
        store = StateStore.from_run_root(root)
        return store, store.load_config()
    cfg = SearchConfig.default(root)
    store = StateStore.bootstrap(cfg)
    return store, cfg


def _trial_payload(trial: TrialRecord | None) -> dict[str, Any] | None:
    if trial is None:
        return None
    return trial.to_json()


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _repair_trial_metrics_from_artifacts(
    store: StateStore,
    cfg: SearchConfig,
    trial: TrialRecord,
) -> bool:
    log_path = Path(trial.log_path)
    if not log_path.exists():
        return False

    checkpoint_root = (
        Path(trial.checkpoint_root)
        if trial.checkpoint_root
        else latest_checkpoint_dir(Path(trial.save_dir), trial.trial_id)
    )
    metrics_path = Path(trial.metrics_path) if trial.metrics_path else None
    if (metrics_path is None or not metrics_path.exists()) and checkpoint_root is not None:
        candidate_metrics = checkpoint_root / "metrics.jsonl"
        if candidate_metrics.exists():
            metrics_path = candidate_metrics
    if metrics_path is None or not metrics_path.exists():
        return False

    snapshot = extract_trial_snapshot(
        log_path=log_path,
        metrics_path=metrics_path,
        hook_metric_prefix=cfg.hookpoints,
        checkpoint_interval_tokens=cfg.checkpoint_interval_tokens,
        window_start_tokens=0,
    )
    changed = False

    for field_name, value in (
        ("checkpoint_root", str(checkpoint_root) if checkpoint_root else None),
        ("metrics_path", str(metrics_path)),
        ("tokens_seen", snapshot.tokens_seen),
        ("checkpoint_decisions", snapshot.checkpoint_count),
        ("total_cost_ratio", snapshot.total_cost_ratio),
        ("latest_exceed_alpha_0_50", snapshot.latest_exceed_alpha_0_50),
        ("best_exceed_alpha_0_50", snapshot.best_exceed_alpha_0_50),
        ("latest_fvu", snapshot.latest_fvu),
        ("best_fvu", snapshot.best_fvu),
        ("best_objective", snapshot.best_objective),
        ("delta_best_exceed", snapshot.delta_best_exceed),
        ("delta_best_fvu", snapshot.delta_best_fvu),
        ("delta_best_objective", snapshot.delta_best_objective),
        ("invalid_reason", snapshot.invalid_reason),
    ):
        if getattr(trial, field_name) != value:
            setattr(trial, field_name, value)
            changed = True

    if changed:
        store.save_trial(trial)
    return changed


def _repair_trials_from_artifacts(store: StateStore, cfg: SearchConfig) -> bool:
    changed = False
    for trial in store.list_trials():
        changed = _repair_trial_metrics_from_artifacts(store, cfg, trial) or changed
    return changed


def _current_metric_payload(store: StateStore) -> dict[str, Any]:
    cfg = store.load_config()
    if _repair_trials_from_artifacts(store, cfg):
        leaderboard = store.refresh_leaderboard()
    else:
        leaderboard = store.refresh_leaderboard()
    incumbent = store.incumbent_trial()
    active_trial = store.get_active_trial() or store.latest_unfinished_trial()
    metric = None
    metric_source = "missing"
    if incumbent is not None and incumbent.best_objective is not None:
        metric = incumbent.best_objective
        metric_source = "incumbent_best_objective"
    elif active_trial is not None and active_trial.best_objective is not None:
        metric = active_trial.best_objective
        metric_source = "active_trial_best_objective"
    return {
        "metric": metric,
        "metric_source": metric_source,
        "leaderboard": leaderboard,
        "incumbent_trial": _trial_payload(incumbent),
        "active_trial": _trial_payload(active_trial),
        "state": store.load_state(),
    }


def _dry_run_preview(store: StateStore, cfg: SearchConfig) -> dict[str, Any]:
    state = store.load_state()
    active_trial = store.get_active_trial() or store.latest_unfinished_trial()
    incumbent = store.incumbent_trial()
    attempted = store.attempted_signatures()
    if active_trial is None:
        next_params = propose_next_params(
            cfg,
            attempted_signatures=attempted,
            incumbent_trial=incumbent,
        )
        next_trial_index = int(state.get("next_trial_index") or 1)
        return {
            "mode": "dry_run",
            "would_create_trial_id": f"trial_{next_trial_index:04d}",
            "would_use_params": normalize_candidate_params(next_params, cfg),
            "active_trial": None,
            "incumbent_trial": _trial_payload(incumbent),
        }
    return {
        "mode": "dry_run",
        "active_trial": _trial_payload(active_trial),
        "incumbent_trial": _trial_payload(incumbent),
        "message": "would resume latest unfinished trial to the next 15M checkpoint",
    }


def _maybe_create_next_trial(store: StateStore, cfg: SearchConfig) -> TrialRecord | None:
    state = store.load_state()
    if int(state.get("new_trial_count") or 0) >= cfg.max_new_trials:
        return None
    pending_spawn_params = state.pop("pending_spawn_params", None)
    if pending_spawn_params is not None:
        store.save_state(state)
        params = normalize_candidate_params(pending_spawn_params, cfg)
        return store.create_trial(cfg, params)
    attempted = store.attempted_signatures()
    incumbent = store.incumbent_trial()
    params = propose_next_params(
        cfg,
        attempted_signatures=attempted,
        incumbent_trial=incumbent,
    )
    params = normalize_candidate_params(params, cfg)
    return store.create_trial(cfg, params)


def _resolve_agent_decision(
    *,
    args: argparse.Namespace,
    store: StateStore,
    cfg: SearchConfig,
    current_trial: TrialRecord,
    incumbent_trial: TrialRecord | None,
) -> AgentDecision:
    recent_trials = store.list_trials()
    context = build_agent_context(cfg, current_trial, incumbent_trial, recent_trials)
    write_agent_context(cfg.run_root, current_trial.trial_id, context)
    decision_path = (
        Path(args.decision_file)
        if args.decision_file
        else cfg.pending_agent_decision_path
    )
    external = maybe_load_external_decision(decision_path)
    if external is not None:
        return external
    can_spawn_more = int(store.load_state().get("new_trial_count") or 0) < cfg.max_new_trials
    return heuristic_gray_zone_decision(
        current_trial=current_trial,
        incumbent_trial=incumbent_trial,
        can_spawn_more=can_spawn_more,
    )


def _decision_anchor_params(store: StateStore, cfg: SearchConfig) -> dict[str, int | float]:
    trial = store.get_active_trial() or store.latest_unfinished_trial() or store.incumbent_trial()
    if trial is not None:
        return dict(trial.params)
    return dict(cfg.baseline_params())


def command_bootstrap(args: argparse.Namespace) -> int:
    cfg = SearchConfig.default(Path(args.run_root))
    StateStore.bootstrap(cfg)
    _emit(
        {
            "command": "bootstrap",
            "run_root": str(cfg.run_root),
            "checkpoint_root": str(cfg.checkpoint_root),
            "status": "ok",
        }
    )
    return 0


def command_status(args: argparse.Namespace) -> int:
    store, cfg = _store_for_run_root(args.run_root)
    _repair_trials_from_artifacts(store, cfg)
    leaderboard = store.refresh_leaderboard()
    _emit(
        {
            "command": "status",
            "run_root": str(cfg.run_root),
            "state": store.load_state(),
            "leaderboard": leaderboard,
            "active_trial": _trial_payload(store.get_active_trial()),
        }
    )
    return 0


def _execute_step(
    args: argparse.Namespace,
    *,
    emit_payload: bool,
) -> dict[str, Any]:
    store, cfg = _store_for_run_root(args.run_root)
    if args.dry_run_agent:
        payload = _dry_run_preview(store, cfg)
        if emit_payload:
            _emit(payload)
        return payload

    trial = store.get_active_trial() if args.resume_latest else None
    if trial is None:
        trial = store.latest_unfinished_trial()
    if trial is None:
        trial = _maybe_create_next_trial(store, cfg)
        if trial is None:
            store.mark_run_finished("max_new_trials_reached")
            payload = {
                "command": "step",
                "status": "finished",
                "reason": "max_new_trials_reached",
            }
            if emit_payload:
                _emit(payload)
            return payload

    if trial.status in {"pending", "continue_scheduled", "running"}:
        trial.status = "running"
        store.save_trial(trial)
        store.set_active_trial(trial.trial_id, status="running")
        store.append_trial_event(trial.trial_id, "segment_start", {"tokens_seen": trial.tokens_seen})
        result = run_trial_segment(cfg, trial)
        trial = update_trial_from_result(cfg, trial, result)
        store.save_trial(trial)
        store.append_trial_event(
            trial.trial_id,
            "segment_end",
            {
                "returncode": result.returncode,
                "target_tokens": result.target_tokens,
                "tokens_seen": trial.tokens_seen,
                "checkpoint_root": trial.checkpoint_root,
                "invalid_reason": trial.invalid_reason,
            },
        )

    leaderboard = store.refresh_leaderboard()
    incumbent = store.incumbent_trial()

    if trial.status == "checkpoint_ready":
        can_spawn_more = int(store.load_state().get("new_trial_count") or 0) < cfg.max_new_trials
        base_decision = evaluate_checkpoint(
            current_trial=trial,
            incumbent_trial=incumbent,
            can_spawn_more=can_spawn_more,
        )
        if base_decision.needs_agent:
            decision = _resolve_agent_decision(
                args=args,
                store=store,
                cfg=cfg,
                current_trial=trial,
                incumbent_trial=incumbent,
            )
        else:
            decision = AgentDecision(
                action=base_decision.action,
                rationale=base_decision.rationale,
                source="programmatic",
            )

        store.append_agent_decision(trial.trial_id, decision.to_json())
        if decision.action == "continue_current":
            trial.status = "continue_scheduled"
            store.set_active_trial(trial.trial_id, status="continue_scheduled")
        elif decision.action == "stop_current_and_spawn_next":
            trial.status = "stopped"
            if decision.next_params is not None:
                state = store.load_state()
                state["pending_spawn_params"] = decision.next_params
                store.save_state(state)
            store.set_active_trial(None, status="idle")
        else:
            trial.status = "finished"
            state = store.load_state()
            state.pop("pending_spawn_params", None)
            store.save_state(state)
            store.mark_run_finished("explicit_finish_decision")
        if decision.next_params:
            trial.notes.append(f"next_params_hint={decision.next_params}")
        store.save_trial(trial)
        leaderboard = store.refresh_leaderboard()
        payload = {
            "command": "step",
            "decision": decision.to_json(),
            "trial": _trial_payload(trial),
            "leaderboard": leaderboard,
            "state": store.load_state(),
        }
        if emit_payload:
            _emit(payload)
        return payload

    if trial.status in {"stopped", "finished"}:
        store.set_active_trial(None, status="idle")

    payload = {
        "command": "step",
        "trial": _trial_payload(trial),
        "leaderboard": leaderboard,
        "state": store.load_state(),
    }
    if emit_payload:
        _emit(payload)
    return payload


def command_step(args: argparse.Namespace) -> int:
    _execute_step(args, emit_payload=True)
    return 0


def command_metric(args: argparse.Namespace) -> int:
    store, _cfg = _store_for_run_root(args.run_root)
    payload = _current_metric_payload(store)
    if payload["metric"] is None and args.bootstrap_if_missing:
        step_args = argparse.Namespace(
            run_root=args.run_root,
            resume_latest=True,
            dry_run_agent=False,
            decision_file=None,
        )
        _execute_step(step_args, emit_payload=False)
        store, _cfg = _store_for_run_root(args.run_root)
        payload = _current_metric_payload(store)

    if payload["metric"] is None:
        if args.json:
            _emit(
                {
                    "command": "metric",
                    "status": "missing_metric",
                    **payload,
                }
            )
        else:
            print("metric_missing", file=sys.stderr)
        return 2

    if args.json:
        _emit(
            {
                "command": "metric",
                "status": "ok",
                **payload,
            }
        )
    else:
        print(f"{float(payload['metric']):.12f}")
    return 0


def command_write_decision(args: argparse.Namespace) -> int:
    store, cfg = _store_for_run_root(args.run_root)
    next_params = None
    if args.next_params_json:
        merged = _decision_anchor_params(store, cfg)
        merged.update(json.loads(args.next_params_json))
        next_params = normalize_candidate_params(merged, cfg)
    decision = AgentDecision(
        action=args.action,
        rationale=args.rationale,
        next_params=next_params,
        source=args.source,
    )
    cfg.pending_agent_decision_path.write_text(
        json.dumps(decision.to_json(), indent=2)
    )
    _emit(
        {
            "command": "write-decision",
            "status": "ok",
            "path": str(cfg.pending_agent_decision_path),
            "decision": decision.to_json(),
            "active_trial": _trial_payload(
                store.get_active_trial() or store.latest_unfinished_trial()
            ),
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "bootstrap":
        return command_bootstrap(args)
    if args.command == "status":
        return command_status(args)
    if args.command == "metric":
        return command_metric(args)
    if args.command == "step":
        return command_step(args)
    if args.command == "write-decision":
        return command_write_decision(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
