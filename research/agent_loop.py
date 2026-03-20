"""
Nightly SAE autoresearch loop driven by Codex CLI.

This loop keeps the execution layer fixed while delegating experiment choice
and optional code edits to a short-lived model call each round.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from research.git_ops import (
    REPO_ROOT,
    commit_message_for_round,
    commit_round_state,
    current_git_branch,
    current_git_commit,
    ensure_clean_worktree_for_auto_commit,
    worktree_dirty,
    stage_paths,
    snapshot_paths,
    touched_files,
    assert_allowed_changes,
    build_patch,
    capture_before_files,
    cleanup_all_snapshots,
    cleanup_round_snapshots,
    init_tracked_paths,
)
from research.state_io import (
    HISTORY_DIR,
    LOG_DIR,
    ROUND_SUMMARIES_DIR,
    STATE_PATH,
    RESULTS_PATH,
    FRONTIER_PATH,
    MEMORY_PATH,
    HINTS_PATH,
    TIMELINE_PATH,
    SESSION_BRIEF_PATH,
    BASE_ENV_DEFAULTS,
    load_json,
    save_json,
    load_state,
    save_state,
    load_memory,
    load_results,
    load_session_brief,
    save_session_brief,
    build_session_brief,
    append_timeline_event,
    log_round_event,
    write_status,
    append_memory,
    write_round_summary,
    update_agent_state,
    mark_hints_applied,
)
from research.training import (
    SAVE_ROOT,
    DEFAULT_PROXY_MAX_TOKENS,
    DEFAULT_FULL_MAX_TOKENS,
    DEFAULT_PROXY_TIMEOUT_SEC,
    DEFAULT_FULL_TIMEOUT_SEC,
    DEFAULT_STALL_TIMEOUT_SEC,
    DEFAULT_POLL_INTERVAL_SEC,
    DEFAULT_FIRST_STEP_TIMEOUT_SEC,
    DEFAULT_SLOW_RUN_GRACE_SEC,
    DEFAULT_MIN_TOKENS_PER_SEC_RATIO,
    DEFAULT_MIN_PROGRESS_STEPS,
    validate_env_overrides,
    run_training_round,
    _run_sanity_for_round,
    budget_remaining_sec,
    check_agent_backend_reachable,
)
from research.prompts import (
    build_prompt,
    build_resume_prompt,
)
from research.policy import (
    behavioral_diff_test,
    build_policy_guidance,
    classify_changes,
    check_variable_isolation,
    compute_proxy_budget,
    detect_stagnation,
    enforce_incubation_limits,
    auto_archive_stale_families,
)

RESEARCH_DIR = REPO_ROOT / "research"
CONTROLLER_PATH = RESEARCH_DIR / "controller.py"
PROGRAM_PATH = RESEARCH_DIR / "program.md"
SCHEMA_PATH = RESEARCH_DIR / "agent_action.schema.json"

DEFAULT_AGENT_PROXY = "http://127.0.0.1:23234"
DEFAULT_MAX_SESSION_ROUNDS = 8
DEFAULT_MAX_SESSION_HOURS = 4.0
DEFAULT_AGENT_MAX_RETRIES = 3
DEFAULT_AGENT_RETRY_BASE_SEC = 10
DEFAULT_MAX_SESSION_FAILURES = 3
DEFAULT_AGENT_TIMEOUT_SEC = 10 * 60  # 10 minutes for agent to produce an action

# Wire up tracked history paths for git_ops
TRACKED_HISTORY_PATHS = (
    STATE_PATH,
    RESULTS_PATH,
    FRONTIER_PATH,
    MEMORY_PATH,
    TIMELINE_PATH,
    SESSION_BRIEF_PATH,
    HINTS_PATH,
)
init_tracked_paths(
    STATE_PATH, RESULTS_PATH, FRONTIER_PATH, MEMORY_PATH,
    TIMELINE_PATH, SESSION_BRIEF_PATH, HINTS_PATH,
)


def ensure_setup() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ROUND_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    TIMELINE_PATH.touch(exist_ok=True)
    cleanup_all_snapshots()
    from research.git_ops import run
    run(["python", "-m", "research.controller", "init"], cwd=REPO_ROOT)


def read_text_safe(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def extract_session_id(output: str) -> str | None:
    match = re.search(r"session id:\s*([0-9a-fA-F-]{8,})", output)
    return match.group(1) if match else None


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise RuntimeError("Could not extract JSON object from Codex response")


def validate_action_shape(action: dict[str, Any]) -> None:
    required = load_json(SCHEMA_PATH, {}).get("required", [])
    missing = [key for key in required if key not in action]
    if missing:
        raise RuntimeError(f"Action missing required keys: {', '.join(missing)}")


def run_agent_round(
    prompt: str,
    round_id: int,
    model: str | None,
    agent_proxy: str | None,
    session_id: str | None,
    timeout_sec: int = DEFAULT_AGENT_TIMEOUT_SEC,
) -> tuple[dict[str, Any], Path, str | None, bool]:
    """Run a single agent invocation. No retries — caller handles that."""
    action_path = LOG_DIR / f"agent_action_{round_id:04d}.json"
    stdout_path = LOG_DIR / f"agent_round_{round_id:04d}.stdout.log"
    resumed = session_id is not None
    if resumed:
        cmd = [
            "codex",
            "exec",
            "resume",
            session_id,
            "-",
            "--dangerously-bypass-approvals-and-sandbox",
            "-o",
            str(action_path),
        ]
    else:
        cmd = [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--cd",
            str(REPO_ROOT),
            "--output-schema",
            str(SCHEMA_PATH),
            "-o",
            str(action_path),
        ]
    if model:
        cmd.extend(["--model", model])
    env = os.environ.copy()
    if agent_proxy:
        env["http_proxy"] = agent_proxy
        env["https_proxy"] = agent_proxy
    try:
        result = subprocess.run(cmd, input=prompt, text=True, capture_output=True, env=env, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"codex exec timed out after {timeout_sec}s for round {round_id}")
    stdout_path.write_text(result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else ""))
    if result.returncode != 0:
        raise RuntimeError(f"codex exec failed: see {stdout_path}")
    if resumed:
        action = extract_json_object(read_text_safe(action_path))
        validate_action_shape(action)
        active_session_id = session_id
    else:
        action = load_json(action_path, {})
        validate_action_shape(action)
        active_session_id = extract_session_id(result.stdout + "\n" + result.stderr)
    return action, stdout_path, active_session_id, resumed


def run_agent_round_with_retry(
    state: dict[str, Any],
    memory: dict[str, Any],
    recent_results: list[dict[str, str]],
    brief: dict[str, Any],
    round_id: int,
    args: argparse.Namespace,
    round_ctx: dict[str, Any],
    session_id: str | None,
    need_new_session: bool,
    policy_context: str = "",
) -> tuple[dict[str, Any], Path, str | None, bool]:
    max_retries = args.agent_max_retries
    retry_base = args.agent_retry_base_sec
    attempt = 0
    last_error: Exception | None = None
    current_session_id = session_id if not need_new_session else None

    while attempt <= max_retries:
        if need_new_session or current_session_id is None:
            prompt = build_prompt(state, memory, recent_results, policy_context=policy_context)
            try_session_id = None
        else:
            prompt = build_resume_prompt(state, memory, recent_results, round_id, brief, policy_context=policy_context)
            try_session_id = current_session_id

        try:
            result = run_agent_round(
                prompt, round_id, args.model, args.agent_proxy, try_session_id,
            )
            if attempt > 0:
                log_round_event(
                    round_ctx,
                    "agent_retry_succeeded",
                    attempt=attempt,
                    was_resume=try_session_id is not None,
                )
            return result
        except Exception as exc:
            last_error = exc
            attempt += 1
            log_round_event(
                round_ctx,
                "agent_retry",
                attempt=attempt,
                max_retries=max_retries,
                was_resume=try_session_id is not None,
                error=str(exc),
            )
            print(
                f"Round {round_id}: agent invocation failed (attempt {attempt}/{max_retries + 1}): {exc}"
            )
            if try_session_id is not None:
                print(f"Round {round_id}: session resume failed, will rebuild session")
                append_timeline_event(
                    "session_broken",
                    round=round_id,
                    tier=None,
                    run_name=None,
                    family_name=None,
                    family_stage=None,
                    status="broken",
                    decision=None,
                    payload={
                        "session_id": try_session_id,
                        "error": str(exc),
                    },
                )
                current_session_id = None
                need_new_session = True

            if attempt <= max_retries:
                delay = retry_base * (2 ** (attempt - 1))
                print(f"Round {round_id}: retrying in {delay}s...")
                time.sleep(delay)

    raise RuntimeError(
        f"Agent invocation failed after {max_retries + 1} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Helper functions extracted from main() to reduce its size
# ---------------------------------------------------------------------------


def _close_round(
    round_id: int,
    action: dict[str, Any],
    result: dict[str, str],
    round_ctx: dict[str, Any],
    memory: dict[str, Any],
    stdout_path: Path | None,
    touched: list[str],
    patch_path: Path | None,
    auto_commit: bool,
    effective_session_mode: str,
    apply_hint_updates: bool,
    refresh_session_brief: bool,
    emit_commit_prepared_event: bool,
    print_result: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Close a round and persist all derived state."""
    round_ctx["ended_at"] = int(time.time())
    round_ctx["duration_sec"] = round_ctx["ended_at"] - int(round_ctx["started_at"])
    effective_tier = (
        "full" if round_ctx.get("full_result") is not None
        else action.get("experiment_tier")
    )
    log_round_event(
        round_ctx,
        "round_finished",
        tier=effective_tier,
        family_name=round_ctx.get("family_name"),
        family_stage=round_ctx.get("family_stage"),
        status=result.get("status"),
        decision=result.get("decision"),
        val_fvu=result.get("val_fvu"),
    )
    write_round_summary(round_id, action, result, touched, patch_path, round_ctx)

    round_summary_path = ROUND_SUMMARIES_DIR / f"round_{round_id:04d}.json"
    existing_families = set(memory.get("architecture_families", {}).keys())
    memory = append_memory(memory, action, result, round_id, touched)
    save_json(MEMORY_PATH, memory)
    if apply_hint_updates:
        mark_hints_applied(round_id)

    state = load_state()
    family_name = str(action.get("family_name") or BASE_ENV_DEFAULTS["ARCHITECTURE"]).lower()
    is_new_family = family_name not in existing_families
    update_agent_state(state, result, action, stdout_path, patch_path, is_new_family)
    if refresh_session_brief and effective_session_mode == "resume-session" and state["agent"].get("active_session_status") == "active":
        state["agent"]["active_session_rounds"] = int(state["agent"].get("active_session_rounds", 0)) + 1
    save_state(state)
    if refresh_session_brief:
        save_session_brief(build_session_brief(state, memory, load_results(limit=8), round_id))

    if auto_commit:
        if emit_commit_prepared_event:
            append_timeline_event(
                "experiment_commit_prepared",
                round=round_id,
                tier=effective_tier,
                run_name=None,
                family_name=round_ctx.get("family_name"),
                family_stage=round_ctx.get("family_stage"),
                status=result.get("status"),
                decision=result.get("decision"),
                payload={"branch": current_git_branch()},
            )
        commit_hash, branch_name = commit_round_state(
            round_id,
            action,
            result,
            touched,
            round_summary_path,
            effective_tier or "proxy",
        )
        if emit_commit_prepared_event:
            write_status(
                "experiment_committed",
                round=round_id,
                commit=commit_hash,
                branch=branch_name,
            )

    if print_result:
        print(
            f"Round {round_id}: result recorded | "
            f"decision={result.get('decision')} val_fvu={result.get('val_fvu')} "
            f"log={result.get('log_path', '')}"
        )

    return result, memory


def _abort_round(
    round_id: int,
    action: dict[str, Any],
    decision: str,
    termination_reason: str,
    round_ctx: dict[str, Any],
    memory: dict[str, Any],
    stdout_path: Path | None,
    touched: list[str],
    patch_path: Path | None,
    auto_commit: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Abort a round early and finalize state without mutating the codebase."""
    result: dict[str, str] = {
        "decision": decision,
        "status": decision,
        "termination_reason": termination_reason,
        "run_health": decision,
    }
    return _close_round(
        round_id=round_id,
        action=action,
        result=result,
        round_ctx=round_ctx,
        memory=memory,
        stdout_path=stdout_path,
        touched=touched,
        patch_path=patch_path,
        auto_commit=auto_commit,
        effective_session_mode="fresh-each-round",
        apply_hint_updates=False,
        refresh_session_brief=False,
        emit_commit_prepared_event=False,
        print_result=False,
    )


def _resolve_session(
    round_id: int,
    args: argparse.Namespace,
    agent_state: dict[str, Any],
    effective_session_mode: str,
) -> tuple[str | None, bool]:
    """Determine session_id and whether a new session is needed.

    Returns (session_id, need_new_session).
    """
    session_id = (
        None if effective_session_mode == "fresh-each-round"
        else agent_state.get("active_session_id")
    )
    session_status = agent_state.get("active_session_status", "closed")
    session_started_at = agent_state.get("active_session_started_at")
    session_rounds = int(agent_state.get("active_session_rounds", 0))
    session_age_hours = (
        (time.time() - float(session_started_at)) / 3600 if session_started_at else 0.0
    )
    need_new_session = (
        effective_session_mode == "fresh-each-round"
        or not session_id
        or session_status in {"broken", "closed", "stale"}
        or session_rounds >= args.max_session_rounds
        or session_age_hours >= args.max_session_hours
    )
    if need_new_session and session_id and effective_session_mode != "fresh-each-round":
        append_timeline_event(
            "session_rebuilt",
            round=round_id,
            tier=None,
            run_name=None,
            family_name=None,
            family_stage=None,
            status="recreated",
            decision=None,
            payload={
                "previous_session_id": session_id,
                "previous_status": session_status,
                "previous_rounds": session_rounds,
                "previous_age_hours": round(session_age_hours, 4),
            },
        )
    return session_id, need_new_session


def _handle_agent_failure(
    round_id: int,
    exc: Exception,
    round_ctx: dict[str, Any],
    memory: dict[str, Any],
    effective_session_mode: str,
    auto_commit: bool,
) -> None:
    """Handle agent invocation failure: log, update state, optionally commit."""
    error_path = LOG_DIR / f"agent_round_{round_id:04d}.error.log"
    error_path.write_text(str(exc) + "\n")
    print(f"Round {round_id}: agent invocation exhausted all retries: {exc}")
    log_round_event(
        round_ctx, "round_finished",
        status="crash", decision="crash", error=str(exc),
    )
    memory.setdefault("recent_insights", []).append(
        f"round {round_id}: agent invocation failed after retries: {exc}"
    )
    memory["recent_insights"] = memory["recent_insights"][-40:]
    save_json(MEMORY_PATH, memory)

    state = load_state()
    agent = state.setdefault("agent", {})
    agent["round_index"] = round_id
    agent["consecutive_crashes"] = int(agent.get("consecutive_crashes", 0)) + 1
    agent["last_action_file"] = str(error_path)
    if effective_session_mode == "resume-session" and agent.get("active_session_id"):
        agent["active_session_status"] = "broken"
    save_state(state)

    if auto_commit:
        crash_action = {
            "family_name": "agent_invocation",
            "change_type": "no_change",
            "experiment_tier": "proxy",
        }
        crash_result = {"decision": "crash"}
        round_summary_path = ROUND_SUMMARIES_DIR / f"round_{round_id:04d}.json"
        save_json(
            round_summary_path,
            {
                "round": round_id,
                "timestamp": int(time.time()),
                "experiment_branch": current_git_branch(),
                "experiment_commit_message": commit_message_for_round(
                    round_id, crash_action, crash_result, "proxy",
                ),
                "error": str(exc),
            },
        )
        commit_round_state(
            round_id, crash_action, crash_result, [], round_summary_path, "proxy",
        )


def _update_session_after_action(
    round_id: int,
    state: dict[str, Any],
    returned_session_id: str | None,
    resumed: bool,
    effective_session_mode: str,
    round_ctx: dict[str, Any],
    session_failure_count: int,
) -> int:
    """Update session state after a successful agent action.

    Returns updated session_failure_count.
    """
    agent = state.setdefault("agent", {})
    if effective_session_mode != "resume-session":
        return session_failure_count

    if not returned_session_id:
        print(f"Round {round_id}: WARNING: Codex did not return a session id, falling back to fresh")
        session_failure_count += 1
        # Mark current session stale so _resolve_session won't keep reusing it
        if agent.get("active_session_id"):
            agent["active_session_status"] = "stale"
    else:
        event = "session_resumed" if resumed else "session_started"
        if resumed:
            agent["last_resume_ok_at"] = int(time.time())
        else:
            agent["active_session_started_at"] = int(time.time())
            agent["active_session_rounds"] = 0
        append_timeline_event(
            event,
            round=round_id,
            tier=None,
            run_name=None,
            family_name=None,
            family_stage=None,
            status="active",
            decision=None,
            payload={"session_id": returned_session_id},
        )
        agent["active_session_id"] = returned_session_id
        agent["active_session_status"] = "active"
        round_ctx["session_id"] = returned_session_id
    save_state(state)
    return session_failure_count


def _finalize_round(
    round_id: int,
    action: dict[str, Any],
    result: dict[str, str],
    round_ctx: dict[str, Any],
    memory: dict[str, Any],
    touched: list[str],
    patch_path: Path | None,
    stdout_path: Path,
    auto_commit: bool,
    effective_session_mode: str,
) -> dict[str, Any]:
    """Finalize a successfully-trained round."""
    _, memory = _close_round(
        round_id=round_id,
        action=action,
        result=result,
        round_ctx=round_ctx,
        memory=memory,
        stdout_path=stdout_path,
        touched=touched,
        patch_path=patch_path,
        auto_commit=auto_commit,
        effective_session_mode=effective_session_mode,
        apply_hint_updates=True,
        refresh_session_brief=True,
        emit_commit_prepared_event=True,
        print_result=True,
    )
    return memory


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Nightly SAE autoresearch loop")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--budget-hours", type=float, default=8.0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--agent-proxy", default=DEFAULT_AGENT_PROXY)
    parser.add_argument("--proxy-max-tokens", default=DEFAULT_PROXY_MAX_TOKENS)
    parser.add_argument("--full-max-tokens", default=DEFAULT_FULL_MAX_TOKENS)
    parser.add_argument("--proxy-timeout-sec", type=int, default=DEFAULT_PROXY_TIMEOUT_SEC)
    parser.add_argument("--full-timeout-sec", type=int, default=DEFAULT_FULL_TIMEOUT_SEC)
    parser.add_argument("--stall-timeout-sec", type=int, default=DEFAULT_STALL_TIMEOUT_SEC)
    parser.add_argument("--poll-interval-sec", type=int, default=DEFAULT_POLL_INTERVAL_SEC)
    parser.add_argument("--first-step-timeout-sec", type=int, default=DEFAULT_FIRST_STEP_TIMEOUT_SEC)
    parser.add_argument("--slow-run-grace-sec", type=int, default=DEFAULT_SLOW_RUN_GRACE_SEC)
    parser.add_argument("--min-tokens-per-sec-ratio", type=float, default=DEFAULT_MIN_TOKENS_PER_SEC_RATIO)
    parser.add_argument("--min-progress-steps", type=int, default=DEFAULT_MIN_PROGRESS_STEPS)
    parser.add_argument("--max-consecutive-crashes", type=int, default=3)
    parser.add_argument("--max-consecutive-no-improve", type=int, default=15)
    parser.add_argument("--session-mode", choices=["resume-session", "fresh-each-round"], default="resume-session")
    parser.add_argument("--max-session-rounds", type=int, default=DEFAULT_MAX_SESSION_ROUNDS)
    parser.add_argument("--max-session-hours", type=float, default=DEFAULT_MAX_SESSION_HOURS)
    parser.add_argument("--no-commit-experiments", action="store_true")
    parser.add_argument("--allow-direct-full", action="store_true")
    parser.add_argument("--reset-failure-counters", action="store_true")
    parser.add_argument("--agent-max-retries", type=int, default=DEFAULT_AGENT_MAX_RETRIES)
    parser.add_argument("--agent-retry-base-sec", type=int, default=DEFAULT_AGENT_RETRY_BASE_SEC)
    parser.add_argument("--max-session-failures", type=int, default=DEFAULT_MAX_SESSION_FAILURES)
    args = parser.parse_args()

    auto_commit = not args.no_commit_experiments
    if auto_commit:
        ensure_clean_worktree_for_auto_commit()

    ensure_setup()
    start_time = time.time()
    append_timeline_event(
        "loop_started", round=None, tier=None, run_name=None,
        family_name=None, family_stage=None, status=None, decision=None,
        payload={"rounds": args.rounds, "budget_hours": args.budget_hours},
    )
    write_status("loop_starting", rounds=args.rounds, budget_hours=args.budget_hours)
    print("Starting SAE agent loop")
    print(f"round_budget: {args.rounds}")
    print(f"time_budget_hours: {args.budget_hours}")
    print(f"agent_proxy: {args.agent_proxy}")
    print(f"session_mode: {args.session_mode}")
    print(f"auto_commit: {auto_commit}")
    print(
        "watchdog: "
        f"first_step_timeout={args.first_step_timeout_sec}s "
        f"slow_run_grace={args.slow_run_grace_sec}s "
        f"min_ratio={args.min_tokens_per_sec_ratio:.2f} "
        f"min_progress_steps={args.min_progress_steps}"
    )

    if args.reset_failure_counters:
        state = load_state()
        agent = state.setdefault("agent", {})
        agent["consecutive_crashes"] = 0
        agent["consecutive_no_improve"] = 0
        save_state(state)
        print("Reset agent failure counters")

    try:
        check_agent_backend_reachable(args.agent_proxy)
        print("Agent backend preflight: OK")
    except RuntimeError as exc:
        print(f"Agent backend preflight FAILED: {exc}")
        append_timeline_event(
            "backend_unavailable", round=None, tier=None, run_name=None,
            family_name=None, family_stage=None, status="failed", decision=None,
            payload={"error": str(exc)},
        )
        return 1

    if auto_commit:
        state = load_state()
        research_branch = current_git_branch()
        state["research_branch"] = research_branch
        state["base_branch"] = state.get("base_branch", research_branch)
        state["base_commit"] = state.get("base_commit", current_git_commit())
        save_state(state)
        append_timeline_event(
            "research_branch_ready", round=None, tier=None, run_name=None,
            family_name=None, family_stage=None, status="ready", decision=None,
            payload={"branch": research_branch, "base_branch": state.get("base_branch"), "base_commit": state.get("base_commit")},
        )
        print(f"research_branch: {research_branch}")

    session_failure_count = 0

    for _ in range(args.rounds):
        if (time.time() - start_time) / 3600 > args.budget_hours:
            print("Stopping: budget-hours limit reached")
            break

        state = load_state()
        memory = load_memory()
        agent_state = state["agent"]

        # --- Stagnation detection & crash recovery ---
        stagnation = detect_stagnation(agent_state)
        force_param_only = False
        if stagnation["crash_streak"] and int(agent_state.get("consecutive_crashes", 0)) >= 2:
            crash_resets = int(agent_state.get("crash_resets", 0))
            if crash_resets >= 2:
                print(
                    f"Stopping: crash streak persists after {crash_resets} resets "
                    f"({agent_state['consecutive_crashes']} consecutive crashes)"
                )
                break
            print("Crash streak detected; forcing param_only without modifying the worktree")
            agent_state["consecutive_crashes"] = 0
            agent_state["crash_resets"] = crash_resets + 1
            save_state(state)
            stagnation["recommended_mode"] = "stabilize_after_crashes"
            force_param_only = True

        if agent_state["consecutive_crashes"] >= args.max_consecutive_crashes:
            print(f"Stopping: consecutive crash limit ({agent_state['consecutive_crashes']} >= {args.max_consecutive_crashes})")
            break
        if agent_state["consecutive_no_improve"] >= args.max_consecutive_no_improve:
            print(f"Stopping: consecutive no-improve limit ({agent_state['consecutive_no_improve']} >= {args.max_consecutive_no_improve})")
            break

        round_id = int(agent_state["round_index"]) + 1
        round_ctx: dict[str, Any] = {
            "round": round_id,
            "started_at": int(time.time()),
            "family_name": None,
            "family_stage": None,
            "session_id": agent_state.get("active_session_id"),
            "timeline_event_ids": [],
            "proxy_result": None,
            "full_result": None,
        }
        print(f"Starting round {round_id}")
        log_round_event(round_ctx, "round_started", status="started")
        write_status("agent_deciding", round=round_id)
        log_round_event(round_ctx, "agent_deciding")
        recent_results = load_results(limit=8)
        brief = load_session_brief()

        all_results = load_results()
        policy_context, wrote_meta = build_policy_guidance(
            round_id,
            state,
            memory,
            all_results,
            stagnation,
        )
        if wrote_meta:
            state.setdefault("agent", {})["last_meta_round"] = round_id
            save_state(state)
        archived = auto_archive_stale_families(memory)
        if archived:
            print(f"Round {round_id}: auto-archived stale families: {', '.join(archived)}")
            save_json(MEMORY_PATH, memory)

        # --- Session lifecycle ---
        effective_session_mode = args.session_mode
        if session_failure_count >= args.max_session_failures and effective_session_mode == "resume-session":
            effective_session_mode = "fresh-each-round"
            print(f"Round {round_id}: degraded to fresh-each-round after {session_failure_count} session failures")
            append_timeline_event(
                "session_fallback_to_fresh", round=round_id, tier=None, run_name=None,
                family_name=None, family_stage=None, status="degraded", decision=None,
                payload={"session_failure_count": session_failure_count},
            )

        session_id, need_new_session = _resolve_session(round_id, args, agent_state, effective_session_mode)

        # --- Snapshot code before agent edits ---
        before = snapshot_paths()
        if before:
            capture_before_files(list(before.keys()), round_id)
        # --- Agent invocation ---
        try:
            action, stdout_path, returned_session_id, resumed = run_agent_round_with_retry(
                state, memory, recent_results, brief,
                round_id, args, round_ctx,
                session_id if not need_new_session else None,
                need_new_session,
                policy_context=policy_context,
            )
            session_failure_count = 0
        except Exception as exc:
            session_failure_count += 1
            _handle_agent_failure(round_id, exc, round_ctx, memory, effective_session_mode, auto_commit)
            continue

        state = load_state()
        session_failure_count = _update_session_after_action(
            round_id, state, returned_session_id, resumed,
            effective_session_mode, round_ctx, session_failure_count,
        )

        if action.get("command") == "stop":
            print(f"Round {round_id}: agent requested stop: {action.get('summary', '')}")
            log_round_event(round_ctx, "round_finished", status="stop", decision="stop", summary=action.get("summary", ""))
            memory.setdefault("recent_insights", []).append(f"round {round_id}: agent stopped: {action.get('summary', '')}")
            save_json(MEMORY_PATH, memory)
            break

        if action.get("experiment_tier") == "full" and not args.allow_direct_full:
            print(f"Round {round_id}: direct full request coerced to proxy")
            memory.setdefault("recent_insights", []).append(
                f"round {round_id}: direct full request coerced to proxy for hypothesis: {action.get('hypothesis')}"
            )
            memory["recent_insights"] = memory["recent_insights"][-40:]
            action["experiment_tier"] = "proxy"

        print(
            f"Round {round_id}: agent action received | "
            f"change_type={action.get('change_type')} tier={action.get('experiment_tier')} "
            f"expected_win={action.get('expected_win')}"
        )
        round_ctx["family_name"] = action.get("family_name") or BASE_ENV_DEFAULTS["ARCHITECTURE"]
        round_ctx["family_stage"] = action.get("family_stage") or ("mainline" if action.get("change_type") == "param_only" else "prototype")
        log_round_event(
            round_ctx, "agent_action_received",
            session_id=round_ctx.get("session_id"),
            tier=action.get("experiment_tier"),
            family_name=round_ctx["family_name"],
            family_stage=round_ctx["family_stage"],
            change_type=action.get("change_type"),
            expected_win=action.get("expected_win"),
            hypothesis=action.get("hypothesis"),
        )
        print(f"Round {round_id}: hypothesis -> {action.get('hypothesis')}")
        if action.get("family_name"):
            print(f"Round {round_id}: family -> {action['family_name']} stage={action.get('family_stage', 'unspecified')}")
        print(f"Round {round_id}: action log -> {stdout_path}")

        # --- Detect code changes ---
        touched: list[str] = []
        patch_path: Path | None = None
        is_code_edit = action.get("change_type") not in ("param_only", "no_change")

        if force_param_only and is_code_edit:
            original_change_type = action.get("change_type")
            print(f"Round {round_id}: BLOCKING code edit ({original_change_type}) — crash recovery mode requires param_only")
            log_round_event(round_ctx, "code_edit_blocked", reason="crash_recovery_force_param_only", original_change_type=original_change_type)
            action["change_type"] = "param_only"
            action["needs_sanity"] = False
            action["touched_files"] = []
            action["primary_variable"] = "other_param"
            action.setdefault("notes_to_memory", []).append(
                f"Runtime coerced {original_change_type} to param_only due to crash recovery mode"
            )
            is_code_edit = False

        if is_code_edit:
            after = snapshot_paths()
            touched = touched_files(before, after)
        if touched:
            assert_allowed_changes(touched)
            print(f"Round {round_id}: touched files -> {', '.join(touched)}")
            patch_text = build_patch(before, touched, round_id)
            patch_path = LOG_DIR / f"round_{round_id:04d}.patch"
            patch_path.write_text(patch_text)
            print(f"Round {round_id}: patch saved -> {patch_path}")
        else:
            print(f"Round {round_id}: no code files changed")
        cleanup_round_snapshots(round_id)

        # --- Validate env_overrides ---
        try:
            overrides = action.get("env_overrides", [])
            rejected_keys = validate_env_overrides(overrides)
            if rejected_keys:
                raise RuntimeError(f"env_overrides contain disallowed keys: {', '.join(rejected_keys)}")
        except RuntimeError as exc:
            print(f"Round {round_id}: env_overrides rejected: {exc}")
            log_round_event(round_ctx, "env_overrides_rejected", rejected_keys=rejected_keys, error=str(exc))
            _, memory = _abort_round(
                round_id, action, "crash", "invalid_env_overrides",
                round_ctx, memory, stdout_path, touched, patch_path,
                auto_commit,
            )
            continue

        # --- Variable isolation check (soft enforcement) ---
        changes = classify_changes(action.get("env_overrides", []), BASE_ENV_DEFAULTS)
        isolation_ok, violation_desc = check_variable_isolation(changes, action.get("change_type", ""))
        if not isolation_ok:
            warning = f"Round {round_id}: variable isolation warning: {violation_desc}"
            print(warning)
            memory.setdefault("recent_insights", []).append(warning)
            log_round_event(round_ctx, "variable_isolation_warning", violation=violation_desc)

        # --- Incubation limits check ---
        incubation_ok, incubation_msg = enforce_incubation_limits(memory, action)
        if not incubation_ok:
            print(f"Round {round_id}: incubation limit hit: {incubation_msg}")
            newly_archived = auto_archive_stale_families(memory)
            if newly_archived:
                print(f"Round {round_id}: freed incubation slots by archiving: {', '.join(newly_archived)}")
                save_json(MEMORY_PATH, memory)
            incubation_ok2, incubation_msg2 = enforce_incubation_limits(memory, action)
            if not incubation_ok2:
                print(f"Round {round_id}: BLOCKING round — incubation limit still exceeded: {incubation_msg2}")
                log_round_event(round_ctx, "incubation_limit_blocked", reason=incubation_msg2)
                memory.setdefault("recent_insights", []).append(
                    f"round {round_id}: BLOCKED by incubation limit: {incubation_msg2}"
                )
                _abort_round(
                    round_id, action, "policy_reject", "incubation_limit_exceeded",
                    round_ctx, memory, stdout_path, touched, patch_path,
                    auto_commit,
                )
                continue

        # --- Behavioral diff test for code edits ---
        if is_code_edit and action.get("change_type") == "edit_sae_code" and touched:
            config = dict(BASE_ENV_DEFAULTS)
            overrides = action.get("env_overrides", [])
            if isinstance(overrides, dict):
                config.update({k: str(v) for k, v in overrides.items()})
            else:
                for item in overrides:
                    key = item.get("key")
                    value = item.get("value")
                    if key:
                        config[key] = str(value)
            arch = config.get("ARCHITECTURE", "topk").lower()
            k_val = int(config.get("K", "128"))
            ef_val = int(config.get("EXPANSION_FACTOR", "8"))
            diff_result = behavioral_diff_test(arch, k_val, ef_val)
            if diff_result.get("identical"):
                print(f"Round {round_id}: behavioral diff test FAILED — identical to baseline topk")
                log_round_event(round_ctx, "behavioral_diff_identical", architecture=arch)
                _abort_round(
                    round_id, action, "crash", "identical_to_baseline",
                    round_ctx, memory, stdout_path, touched, patch_path,
                    auto_commit,
                )
                continue
            else:
                print(f"Round {round_id}: behavioral diff test passed (max_diff={diff_result.get('max_diff', 'N/A')})")

        # --- Dynamic proxy budget ---
        round_proxy_max_tokens = args.proxy_max_tokens
        if action.get("experiment_tier") == "proxy":
            adjusted_budget = compute_proxy_budget(action.get("change_type", ""), args.proxy_max_tokens)
            if adjusted_budget != args.proxy_max_tokens:
                print(f"Round {round_id}: dynamic proxy budget: {args.proxy_max_tokens} -> {adjusted_budget}")
                round_proxy_max_tokens = adjusted_budget

        # --- Budget check before training ---
        remaining_sec = budget_remaining_sec(start_time, args.budget_hours)
        tier = action["experiment_tier"]
        tier_timeout = args.full_timeout_sec if tier == "full" else args.proxy_timeout_sec
        if remaining_sec < tier_timeout * 0.5:
            if tier == "full":
                print(f"Round {round_id}: insufficient budget for full ({remaining_sec:.0f}s remaining), downgrading to proxy")
                action["experiment_tier"] = "proxy"
                tier = "proxy"
                tier_timeout = args.proxy_timeout_sec
            if remaining_sec < args.proxy_timeout_sec * 0.5:
                print(f"Round {round_id}: insufficient budget even for proxy ({remaining_sec:.0f}s remaining), stopping")
                break

        # --- Sanity check for code edits ---
        if action.get("needs_sanity") and is_code_edit:
            try:
                _run_sanity_for_round(action, tier, round_id, round_ctx)
            except Exception as sanity_exc:
                print(f"Round {round_id}: sanity check FAILED: {sanity_exc}")
                log_round_event(round_ctx, "sanity_failed", tier=tier, error=str(sanity_exc))
                _abort_round(
                    round_id, action, "crash", "sanity_failed",
                    round_ctx, memory, stdout_path, touched, patch_path,
                    auto_commit,
                )
                continue

        # --- Training ---
        proxy_override = round_proxy_max_tokens if tier == "proxy" else None
        result = run_training_round(action, tier, args, round_id, memory, round_ctx, proxy_max_tokens_override=proxy_override)
        round_ctx["proxy_result"] = result
        print(
            f"Round {round_id} proxy result: "
            f"decision={result.get('decision')} fvu={result.get('val_fvu')} "
            f"k={result.get('k')} health={result.get('run_health')} "
            f"termination={result.get('termination_reason')}"
        )

        # --- Full promotion ---
        if result.get("decision") == "promote" and tier == "proxy":
            remaining_sec = budget_remaining_sec(start_time, args.budget_hours)
            if remaining_sec < args.full_timeout_sec * 0.5:
                print(f"Round {round_id}: skipping full promotion, insufficient budget ({remaining_sec:.0f}s remaining)")
            else:
                print(f"Round {round_id}: promoted to full")
                full_result = run_training_round(action, "full", args, round_id, memory, round_ctx)
                result = full_result
                round_ctx["full_result"] = full_result
                print(
                    f"Round {round_id} full result: "
                    f"decision={result.get('decision')} fvu={result.get('val_fvu')} "
                    f"k={result.get('k')} health={result.get('run_health')} "
                    f"termination={result.get('termination_reason')}"
                )
        memory = _finalize_round(
            round_id, action, result, round_ctx, memory, touched,
            patch_path, stdout_path,
            auto_commit, effective_session_mode,
        )

    # --- Loop finished ---
    write_status("idle")
    state = load_state()
    if args.session_mode == "resume-session" and state.get("agent", {}).get("active_session_id"):
        state["agent"]["active_session_status"] = "closed"
        save_state(state)
        append_timeline_event(
            "session_closed", round=None, tier=None, run_name=None,
            family_name=None, family_stage=None, status="closed", decision=None,
            payload={"session_id": state["agent"].get("active_session_id")},
        )
    append_timeline_event(
        "loop_finished", round=None, tier=None, run_name=None,
        family_name=None, family_stage=None, status="finished", decision=None,
        payload=None,
    )
    if auto_commit and worktree_dirty():
        stage_paths(list(TRACKED_HISTORY_PATHS))
        from research.git_ops import git
        if git(["diff", "--cached", "--quiet"], check=False).returncode != 0:
            git(["commit", "-m", "experiment-session: close nightly session"])
    print("Agent loop finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
