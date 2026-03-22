"""Main orchestration loop for the autoresearch framework.

This module contains ONLY flow control logic. All business logic
lives in agent, policy, runner, and state modules.
"""

from __future__ import annotations

import time
from typing import Any

from .agent import Agent, coerce_stop_action
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


def run(config: LoopConfig) -> int:
    """Main entry point. Returns exit code."""
    state = StateManager()
    state.ensure_directories()
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    agent = Agent(config)

    # Register tracked paths for git ops
    init_tracked_paths(
        STATE_PATH, RESULTS_PATH, FRONTIER_PATH, MEMORY_PATH,
        TIMELINE_PATH, SESSION_BRIEF_PATH, HINTS_PATH,
    )

    # Preflight
    agent.check_backend_reachable()
    if config.auto_commit:
        ensure_clean_worktree_for_auto_commit()
    if config.reset_failure_counters:
        state.reset_crash_counters()

    state.append_timeline_event("loop_started", rounds=config.rounds, budget_hours=config.budget_hours)
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

    # 1. Detect stagnation and build policy guidance
    stagnation = detect_stagnation(state.consecutive_no_improve, state.consecutive_crashes)
    force_param_only = stagnation["recommended_mode"] == "stabilize_after_crashes"

    # Auto-archive stale families
    archived = auto_archive_stale_families(state.families)
    if archived:
        print(f"Round {round_id}: auto-archived stale families: {archived}")

    guidance = build_policy_guidance(round_id, state, stagnation)

    # 2. Snapshot code before agent edits
    before = snapshot_paths()
    if before:
        capture_before_files(list(before.keys()), round_id)

    # 3. Agent proposal
    try:
        action, stdout_path = agent.propose(state, round_id, guidance)
    except RuntimeError as exc:
        print(f"Round {round_id}: agent invocation failed: {exc}")
        state.log_round_event(ctx, "agent_failed", error=str(exc))
        return

    # 4. Coerce stop -> run
    if action.command != "run":
        action = coerce_stop_action(action, state, round_id)

    # 5. Force param_only if crash recovery
    if force_param_only and action.is_code_edit:
        print(f"Round {round_id}: forcing param_only (crash recovery mode)")
        d = action.to_dict()
        d["change_type"] = "param_only"
        d["needs_sanity"] = False
        action = Action.from_dict(d)

    ctx.family_name = action.family_name or BASE_ENV_DEFAULTS["ARCHITECTURE"]
    ctx.family_stage = action.family_stage
    ctx.session_id = state.agent.get("active_session_id")

    # 6. Detect code changes
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

    # 7. Policy validation
    action, rejection = validate_action(action, state, force_param_only)
    if rejection:
        print(f"Round {round_id}: policy rejected: {rejection}")
        state.record_round_outcome(round_id, action, Result.policy_reject(rejection), changed, ctx.patch_path, ctx)
        cleanup_round_snapshots(round_id)
        return

    # 8. Behavioral diff test (for code edits with architecture changes)
    if action.is_code_edit and action.change_type == "edit_sae_code":
        cfg = action.effective_config()
        arch = cfg.get("ARCHITECTURE", "topk").lower()
        if arch != "topk":
            diff_result = behavioral_diff_test(arch, int(cfg.get("K", "128")), int(cfg.get("EXPANSION_FACTOR", "8")))
            if diff_result.get("identical") and action.family_stage != "prototype":
                msg = f"Behavioral diff: {arch} encode() identical to topk — blocked"
                print(f"Round {round_id}: {msg}")
                state.record_round_outcome(round_id, action, Result.policy_reject(msg), changed, ctx.patch_path, ctx)
                cleanup_round_snapshots(round_id)
                return

    # 9. Train with repair loop
    result = _train_with_repair(round_id, action, ctx, state, agent, config, start_time, before)

    # 10. Record outcome
    state.record_round_outcome(round_id, action, result, ctx.touched_files, ctx.patch_path, ctx)

    # 11. Git commit
    if config.auto_commit:
        _commit_round(round_id, action, result, ctx, state)

    cleanup_round_snapshots(round_id)
    print(f"Round {round_id}: completed | decision={result.decision} fvu={result.val_fvu}")


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
            cfg = action.effective_config()
            try:
                print(f"Round {round_id}: running sanity check")
                run_sanity(
                    cfg.get("ARCHITECTURE", "topk").lower(),
                    int(cfg.get("K", "128")),
                    int(cfg.get("EXPANSION_FACTOR", "8")),
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
                    ctx.touched_files = touched_files(before, after) if before else []
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
            ctx.touched_files = touched_files(before, after) if before else []
            continue

        return result

    return Result.crash("max_repair_attempts_exhausted")


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
