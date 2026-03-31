"""AutoResearch 主循环。

这个文件只保留流程编排：
- 读取 state
- 调用 policy 生成本轮策略说明
- 调 agent 提案
- 校验 action
- 训练 / repair
- 落盘结果
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .agent import Agent, coerce_stop_action
from .config_resolution import resolve_action_configs
from .git_ops import (
    REPO_ROOT,
    assert_allowed_changes,
    build_patch,
    capture_before_files,
    cleanup_round_snapshots,
    commit_round_state,
    ensure_clean_worktree_for_auto_commit,
    init_tracked_paths,
    snapshot_paths,
    touched_files,
)
from .policy import (
    auto_archive_stale_families,
    behavioral_diff_test,
    build_policy_guidance,
    detect_stagnation,
    validate_action,
)
from .runner import SanityCheckError, budget_remaining_sec, run_sanity, run_training
from .state import StateManager
from .types import (
    BASE_ENV_DEFAULTS,
    FRONTIER_PATH,
    HINTS_PATH,
    LOG_DIR,
    MEMORY_PATH,
    REPAIRABLE_ERROR_TYPES,
    RESULTS_PATH,
    ROUND_SUMMARIES_DIR,
    Action,
    LoopConfig,
    Result,
    RoundContext,
    SAVE_ROOT,
    SESSION_BRIEF_PATH,
    STATE_PATH,
    TIMELINE_PATH,
    failure_signature,
)

_RUNTIME_RELOAD_PLAN: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("research.AutoResearch.override_registry", ()),
    ("research.AutoResearch.compatibility", ()),
    ("research.AutoResearch.config_resolution", ("resolve_action_configs",)),
    ("research.AutoResearch.policy", ("validate_action", "behavioral_diff_test")),
    (
        "research.AutoResearch.runner",
        ("SanityCheckError", "budget_remaining_sec", "run_sanity", "run_training"),
    ),
)

_EXECUTION_LAYER_ERROR_MARKERS = (
    "Disallowed env override keys",
    "Unknown architecture",
    "unknown architecture",
    "Invalid architecture",
    "invalid_env_overrides",
    "requested structural parameters did not reach runtime",
)


def run(config: LoopConfig) -> int:
    """Main entry point. Returns exit code."""
    state, agent = _bootstrap_runtime(config, allow_reset_failure_counters=True)

    state.append_timeline_event("loop_started", rounds=config.rounds, budget_hours=config.budget_hours)
    print(f"Loop started: rounds={config.rounds}, budget_hours={config.budget_hours}")
    start_time = time.time()

    for _ in range(config.rounds):
        if budget_remaining_sec(start_time, config.budget_hours) <= 0:
            print("Budget exhausted, stopping loop")
            break

        if _exit_conditions_met(state, config):
            break

        round_id = state.round_index + 1
        ctx = RoundContext(round_id=round_id, started_at=int(time.time()))

        try:
            _run_round(round_id, ctx, state, agent, config, start_time)
        except Exception as exc:
            print(f"Round {round_id}: unhandled error: {exc}")
            state.append_timeline_event("round_error", round=round_id, error=str(exc))

    state.append_timeline_event("loop_finished")
    state.write_status("loop_finished")
    return 0


def run_one_round(config: LoopConfig, *, loop_start_time: float) -> int:
    """Run exactly one round in a fresh worker process."""
    state, agent = _bootstrap_runtime(config, allow_reset_failure_counters=True)

    if budget_remaining_sec(loop_start_time, config.budget_hours) <= 0:
        print("Budget exhausted before starting worker round")
        return 0

    if _exit_conditions_met(state, config):
        return 0

    round_id = state.round_index + 1
    ctx = RoundContext(round_id=round_id, started_at=int(time.time()))

    try:
        _run_round(round_id, ctx, state, agent, config, loop_start_time)
    except Exception as exc:
        print(f"Round {round_id}: unhandled error: {exc}")
        state.append_timeline_event("round_error", round=round_id, error=str(exc))

    return 0


# ---------------------------------------------------------------------------
# Round execution
# ---------------------------------------------------------------------------


def _run_round(
    round_id: int,
    ctx: RoundContext,
    state: StateManager,
    agent: Agent,
    config: LoopConfig,
    start_time: float,
) -> None:
    """Execute a single round: propose -> validate -> train/repair -> record."""
    state.write_status("round_started", round=round_id)
    state.log_round_event(ctx, "round_started")

    # 1. 判定本轮策略模式，并生成给 agent 的策略说明
    registry = state.load_compatibility_registry()
    policy_state = detect_stagnation(
        state.consecutive_no_improve,
        state.consecutive_crashes,
        frontier=state.frontier,
        registry=registry,
    )

    # Auto-archive stale families
    archived = auto_archive_stale_families(state.families)
    if archived:
        print(f"Round {round_id}: auto-archived stale families: {archived}")

    policy_guidance = build_policy_guidance(round_id, state, policy_state)

    # 2. Snapshot code before agent edits
    print(f"Round {round_id}: snapshotting files...")
    before = snapshot_paths()
    if before:
        capture_before_files(list(before.keys()), round_id)

    # 3. Agent proposal
    print(f"Round {round_id}: invoking agent...")
    try:
        action, _stdout_path = agent.propose(state, round_id, policy_guidance)
    except Exception as exc:
        print(f"Round {round_id}: agent invocation failed: {exc}")
        state.log_round_event(ctx, "agent_failed", error=str(exc))
        return

    # 4. Coerce stop -> run
    if action.command != "run":
        action = coerce_stop_action(action, state, round_id)

    resolved = resolve_action_configs(action, state)
    _apply_resolved_context(ctx, action, state, resolved)

    # 5. Detect code changes
    after = snapshot_paths()
    changed = touched_files(before, after) if before else []
    if changed:
        try:
            assert_allowed_changes(changed)
        except RuntimeError as exc:
            print(f"Round {round_id}: disallowed changes: {exc}")
            state.record_round_outcome(round_id, action, Result.policy_reject(str(exc)), [], None, ctx)
            cleanup_round_snapshots(round_id)
            return
    ctx.touched_files = changed
    ctx.patch_path = build_patch(before, changed, round_id) if changed else None
    _refresh_runtime_after_controller_edits(
        round_id,
        action,
        state,
        ctx,
        changed,
        reason="agent edit",
    )
    resolved = resolve_action_configs(action, state)

    # 6. Policy validation
    action, rejection = validate_action(action, state)
    if rejection:
        print(f"Round {round_id}: policy rejected: {rejection}")
        state.record_round_outcome(round_id, action, Result.policy_reject(rejection), changed, ctx.patch_path, ctx)
        cleanup_round_snapshots(round_id)
        return

    # 7. Behavioral diff test (for code edits with architecture changes)
    if action.is_code_edit and action.change_type == "edit_sae_code":
        cfg = ctx.resolved_candidate_env_config or resolved.candidate_env_config
        arch = cfg.get("ARCHITECTURE", "topk").lower()
        if arch != "topk":
            diff_result = behavioral_diff_test(
                arch,
                int(cfg.get("K", "128")),
                int(cfg.get("EXPANSION_FACTOR", "12")),
            )
            if diff_result.get("identical") and action.family_stage != "prototype":
                msg = f"Behavioral diff: {arch} encode() identical to topk — blocked"
                print(f"Round {round_id}: {msg}")
                state.record_round_outcome(round_id, action, Result.policy_reject(msg), changed, ctx.patch_path, ctx)
                cleanup_round_snapshots(round_id)
                return

    # 8. Train with repair loop
    pipeline_retry_used = False
    while True:
        try:
            result = _train_with_repair(round_id, action, ctx, state, agent, config, start_time, before)
            break
        except Exception as exc:
            if _should_retry_in_fresh_worker(exc, ctx.touched_files, pipeline_retry_used):
                pipeline_retry_used = True
                print(
                    f"Round {round_id}: controller edit hit execution-layer error; "
                    "retrying same action in fresh worker"
                )
                state.log_round_event(
                    ctx,
                    "runtime_replay_retry",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                replay_result = _retry_action_in_fresh_worker(
                    round_id,
                    action,
                    ctx,
                    config,
                    start_time,
                )
                if replay_result is not None:
                    after_replay = snapshot_paths()
                    ctx.touched_files = touched_files(before, after_replay) if before else []
                    ctx.patch_path = build_patch(before, ctx.touched_files, round_id) if before else None
                    result = replay_result
                    break

            print(f"Round {round_id}: training pipeline error: {exc}")
            state.log_round_event(
                ctx,
                "training_pipeline_error",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            result = Result.crash(
                "training_pipeline_error",
                error_type=type(exc).__name__,
                error_summary=str(exc),
                traceback_excerpt=None,
                description=f"{action.summary} [PIPELINE ERROR]",
                self_review=action.self_review,
            )
            break

    # 9. Record outcome
    state.record_round_outcome(round_id, action, result, ctx.touched_files, ctx.patch_path, ctx)

    # 10. Git commit
    if config.auto_commit:
        _commit_round(round_id, action, result, ctx, state)

    cleanup_round_snapshots(round_id)
    print(
        f"Round {round_id}: completed | decision={result.decision} "
        f"objective={result.objective_score} cost={result.total_cost_ratio} "
        f"exceed={result.exceed_alpha_0_50} fvu={result.val_fvu}"
    )


def _bootstrap_runtime(
    config: LoopConfig,
    *,
    allow_reset_failure_counters: bool,
    skip_clean_worktree_check: bool = False,
) -> tuple[StateManager, Agent]:
    """Initialize state, tracked paths, and preflight checks for a worker."""
    agent = Agent(config)

    init_tracked_paths(
        STATE_PATH, RESULTS_PATH, FRONTIER_PATH, MEMORY_PATH,
        TIMELINE_PATH, SESSION_BRIEF_PATH, HINTS_PATH,
    )

    print("Preflight: checking backend...")
    agent.check_backend_reachable()
    print("Preflight: backend OK")
    if config.auto_commit and not skip_clean_worktree_check:
        print("Preflight: checking clean worktree...")
        ensure_clean_worktree_for_auto_commit()
        print("Preflight: worktree clean")

    # Do not let startup normalization mutate tracked history before the
    # preflight clean-worktree gate. Any in-memory sanitization will be
    # persisted later as part of a normal round outcome.
    state = StateManager(persist_load_fixes=False)
    state.ensure_directories()
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    if allow_reset_failure_counters and config.reset_failure_counters:
        state.reset_crash_counters()

    return state, agent


# ---------------------------------------------------------------------------
# Training with repair loop
# ---------------------------------------------------------------------------


def _train_with_repair(
    round_id: int,
    action: Action,
    ctx: RoundContext,
    state: StateManager,
    agent: Agent,
    config: LoopConfig,
    start_time: float,
    before: dict[str, str],
) -> Result:
    """Run training with up to max_repair_attempts repair cycles."""
    base_action = action

    for attempt in range(config.max_repair_attempts + 1):
        # Budget check
        remaining = budget_remaining_sec(start_time, config.budget_hours)
        if remaining < config.timeout_sec * 0.5:
            return Result.crash("insufficient_budget")

        # Sanity check
        if action.needs_sanity and action.is_code_edit:
            cfg = resolve_action_configs(action, state).candidate_env_config
            try:
                print(f"Round {round_id}: running sanity check")
                run_sanity(
                    cfg.get("ARCHITECTURE", "topk").lower(),
                    int(cfg.get("K", "128")),
                    int(cfg.get("EXPANSION_FACTOR", "12")),
                )
                print(f"Round {round_id}: sanity passed")
            except SanityCheckError as exc:
                payload = exc.to_payload()
                state.log_round_event(ctx, "sanity_failed", error_type=payload.get("error_type"))
                if _should_repair(action, payload, attempt, config, ctx):
                    ctx.repair_attempts.append({
                        "attempt": attempt + 1, "failure_kind": "sanity_failed",
                        "error_type": payload.get("error_type"),
                    })
                    action, _ = agent.request_repair(
                        round_id, base_action, "sanity_failed", payload,
                        attempt + 1, ctx.session_id,
                    )
                    # Re-detect changes
                    after = snapshot_paths()
                    ctx.touched_files = touched_files(before, after) if before is not None else []
                    _refresh_runtime_after_controller_edits(
                        round_id,
                        action,
                        state,
                        ctx,
                        ctx.touched_files,
                        reason=f"repair {attempt + 1}",
                    )
                    continue
                return Result.crash("sanity_failed", error_type=payload.get("error_type"),
                                     error_summary=payload.get("stderr_excerpt", "")[:500],
                                     traceback_excerpt=payload.get("traceback_excerpt", ""))

        # Training
        result = run_training(action, config, round_id, state, ctx)

        if result.decision != "crash":
            return result

        # Attempt repair on crash
        if _should_repair(action, result.to_dict(), attempt, config, ctx):
            ctx.repair_attempts.append({
                "attempt": attempt + 1, "failure_kind": "training_crash",
                "error_type": result.error_type,
            })
            action, _ = agent.request_repair(
                round_id, base_action, "training_crash", result.to_dict(),
                attempt + 1, ctx.session_id,
            )
            after = snapshot_paths()
            ctx.touched_files = touched_files(before, after) if before is not None else []
            _refresh_runtime_after_controller_edits(
                round_id,
                action,
                state,
                ctx,
                ctx.touched_files,
                reason=f"repair {attempt + 1}",
            )
            continue

        return result

    return Result.crash("max_repair_attempts_exhausted")


def run_replayed_action(
    config: LoopConfig,
    *,
    round_id: int,
    action_path: Path,
    result_path: Path,
    started_at: int,
    loop_start_time: float,
) -> int:
    """Replay a single already-proposed action in a fresh worker process."""
    state, agent = _bootstrap_runtime(
        config,
        allow_reset_failure_counters=False,
        skip_clean_worktree_check=True,
    )

    raw = json.loads(action_path.read_text())
    action = Action.from_dict(raw)
    ctx = RoundContext(round_id=round_id, started_at=started_at)
    resolved = resolve_action_configs(action, state)
    _apply_resolved_context(ctx, action, state, resolved)
    before = snapshot_paths()

    try:
        result = _train_with_repair(round_id, action, ctx, state, agent, config, loop_start_time, before=before)
    except Exception as exc:
        result = Result.crash(
            "training_pipeline_error",
            error_type=type(exc).__name__,
            error_summary=str(exc),
            traceback_excerpt=None,
            description=f"{action.summary} [PIPELINE ERROR]",
            self_review=action.self_review,
        )

    payload = {
        "result": result.to_dict(),
        "round_context": {
            "family_name": ctx.family_name,
            "family_stage": ctx.family_stage,
            "resolved_reference_env_config": ctx.resolved_reference_env_config,
            "resolved_candidate_env_config": ctx.resolved_candidate_env_config,
            "changed_keys": ctx.changed_keys,
            "reference_source": ctx.reference_source,
            "runtime_config_json": ctx.runtime_config_json,
            "runtime_env_config": ctx.runtime_env_config,
        },
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return 0


def _apply_resolved_context(
    ctx: RoundContext,
    action: Action,
    state: StateManager,
    resolved: Any,
) -> None:
    ctx.family_name = (
        action.family_name
        or resolved.candidate_env_config.get("ARCHITECTURE", BASE_ENV_DEFAULTS["ARCHITECTURE"]).lower()
    )
    ctx.family_stage = action.family_stage
    ctx.session_id = state.agent.get("active_session_id")
    ctx.resolved_reference_env_config = resolved.reference_env_config
    ctx.resolved_candidate_env_config = resolved.candidate_env_config
    ctx.changed_keys = resolved.changed_keys
    ctx.reference_source = resolved.reference_source


def _controller_python_changes(paths: list[str]) -> list[str]:
    return [
        path
        for path in paths
        if path.startswith("research/AutoResearch/") and path.endswith(".py")
    ]


def _refresh_runtime_modules() -> list[str]:
    importlib.invalidate_caches()
    refreshed: list[str] = []
    for module_name, rebinds in _RUNTIME_RELOAD_PLAN:
        module = importlib.import_module(module_name)
        module = importlib.reload(module)
        refreshed.append(module_name.rsplit(".", 1)[-1])
        for name in rebinds:
            globals()[name] = getattr(module, name)
    return refreshed


def _refresh_runtime_after_controller_edits(
    round_id: int,
    action: Action,
    state: StateManager,
    ctx: RoundContext,
    changed_paths: list[str],
    *,
    reason: str,
) -> bool:
    controller_changes = _controller_python_changes(changed_paths)
    if not controller_changes:
        return False
    refreshed = _refresh_runtime_modules()
    resolved = resolve_action_configs(action, state)
    _apply_resolved_context(ctx, action, state, resolved)
    print(
        f"Round {round_id}: refreshed runtime after {reason} | "
        f"changed={', '.join(controller_changes)}"
    )
    state.log_round_event(
        ctx,
        "runtime_refreshed",
        reason=reason,
        changed_files=controller_changes,
        refreshed_modules=refreshed,
    )
    return True


def _should_retry_in_fresh_worker(
    exc: Exception,
    changed_paths: list[str],
    already_retried: bool,
) -> bool:
    if already_retried:
        return False
    if not _controller_python_changes(changed_paths):
        return False
    message = str(exc)
    return any(marker in message for marker in _EXECUTION_LAYER_ERROR_MARKERS)


def _retry_action_in_fresh_worker(
    round_id: int,
    action: Action,
    ctx: RoundContext,
    config: LoopConfig,
    loop_start_time: float,
) -> Result | None:
    replay_action_path = LOG_DIR / f"round_{round_id:04d}_replay_action.json"
    replay_result_path = LOG_DIR / f"round_{round_id:04d}_replay_result.json"
    replay_config_path = LOG_DIR / f"round_{round_id:04d}_replay_config.json"
    if replay_result_path.exists():
        replay_result_path.unlink()
    replay_action_path.write_text(json.dumps(action.to_dict(), ensure_ascii=False, indent=2) + "\n")
    replay_config_path.write_text(json.dumps(config.__dict__, ensure_ascii=False, indent=2) + "\n")

    cmd = [
        sys.executable,
        "-m",
        "research.AutoResearch",
        "--_replay-action-path",
        str(replay_action_path),
        "--_replay-result-path",
        str(replay_result_path),
        "--_replay-config-path",
        str(replay_config_path),
        "--_replay-round-id",
        str(round_id),
        "--_replay-started-at",
        str(ctx.started_at),
        "--_loop-start-time",
        str(loop_start_time),
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0 or not replay_result_path.exists():
        return None

    payload = json.loads(replay_result_path.read_text())
    round_ctx = payload.get("round_context", {})
    if isinstance(round_ctx, dict):
        ctx.runtime_config_json = round_ctx.get("runtime_config_json")
        ctx.runtime_env_config = round_ctx.get("runtime_env_config")
        ctx.resolved_reference_env_config = round_ctx.get("resolved_reference_env_config")
        ctx.resolved_candidate_env_config = round_ctx.get("resolved_candidate_env_config")
        ctx.changed_keys = round_ctx.get("changed_keys", []) or []
        ctx.reference_source = round_ctx.get("reference_source")
        ctx.family_name = round_ctx.get("family_name") or ctx.family_name
        ctx.family_stage = round_ctx.get("family_stage") or ctx.family_stage

    result_payload = payload.get("result", {})
    if not isinstance(result_payload, dict) or not result_payload:
        return None
    return Result(**result_payload)


def _should_repair(
    action: Action,
    failure_payload: dict[str, Any],
    attempt: int,
    config: LoopConfig,
    ctx: RoundContext,
) -> bool:
    """Determine whether to attempt a repair cycle."""
    if attempt >= config.max_repair_attempts:
        return False
    if not action.is_code_edit:
        return False

    error_type = str(failure_payload.get("error_type") or "")
    if error_type not in REPAIRABLE_ERROR_TYPES:
        # Also allow repair if there's no signal and there's an error
        no_signal = failure_payload.get("k") in (None, "", "None")
        has_error = bool(failure_payload.get("error_summary") or failure_payload.get("traceback_excerpt"))
        if not (no_signal and has_error):
            return False

    # Same-failure cycle detection
    sig = failure_signature(failure_payload)
    if sig == ctx.last_repair_signature:
        ctx.same_repair_streak += 1
        if ctx.same_repair_streak >= 2:
            print(f"Repair loop: same failure repeated {ctx.same_repair_streak}x, aborting")
            return False
    else:
        ctx.same_repair_streak = 1
    ctx.last_repair_signature = sig
    return True


# ---------------------------------------------------------------------------
# Exit conditions
# ---------------------------------------------------------------------------


def _exit_conditions_met(state: StateManager, config: LoopConfig) -> bool:
    if config.max_consecutive_crashes > 0 and state.consecutive_crashes >= config.max_consecutive_crashes:
        print(f"Stopping: {state.consecutive_crashes} consecutive crashes")
        return True
    if config.max_consecutive_no_improve > 0 and state.consecutive_no_improve >= config.max_consecutive_no_improve:
        print(f"Stopping: {state.consecutive_no_improve} rounds without improvement")
        return True
    return False


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------


def _commit_round(
    round_id: int,
    action: Action,
    result: Result,
    ctx: RoundContext,
    state: StateManager,
) -> None:
    """Auto-commit round state to git."""
    try:
        round_summary_path = ROUND_SUMMARIES_DIR / f"round_{round_id:04d}.json"
        commit_round_state(
            round_id,
            action.to_dict(),
            result.to_dict(),
            ctx.touched_files,
            round_summary_path,
            "standard",
        )
    except Exception as exc:
        print(f"Round {round_id}: git commit failed: {exc}")
