"""CLI entry point: python -m research.AutoResearch.

Default mode runs a lightweight parent scheduler that launches a fresh worker
process for each round. The worker imports the current AutoResearch modules
from disk and executes exactly one round, avoiding stale in-memory Python
state after code edits under ``research/AutoResearch/``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = REPO_ROOT / "research" / "history" / "state.json"

# Keep parser defaults local so the parent scheduler stays stdlib-only.
DEFAULT_TIMEOUT_SEC = 30 * 60
DEFAULT_STALL_TIMEOUT_SEC = 15 * 60
DEFAULT_POLL_INTERVAL_SEC = 30
DEFAULT_FIRST_STEP_TIMEOUT_SEC = 180
DEFAULT_SLOW_RUN_GRACE_SEC = 120
DEFAULT_MIN_TOKENS_PER_SEC_RATIO = 0.25
DEFAULT_MIN_PROGRESS_STEPS = 4
DEFAULT_INITIAL_MAX_TOKENS = 500000
DEFAULT_CONTINUATION_STEP_TOKENS = 100000
DEFAULT_CONTINUATION_MAX_TOKENS = 1000000
DEFAULT_CONTINUATION_MIN_FVU_DROP = 0.002
DEFAULT_AGENT_PROXY: str | None = None
DEFAULT_MAX_SESSION_ROUNDS = 8
DEFAULT_MAX_SESSION_HOURS = 4.0
DEFAULT_AGENT_RETRY_BASE_SEC = 10
DEFAULT_MAX_SESSION_FAILURES = 3
DEFAULT_AGENT_TIMEOUT_SEC = 10 * 60
DEFAULT_MAX_REPAIR_ATTEMPTS = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAE autoresearch loop")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--budget-hours", type=float, default=8.0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--agent-proxy", default=DEFAULT_AGENT_PROXY)
    parser.add_argument("--initial-max-tokens", type=int, default=DEFAULT_INITIAL_MAX_TOKENS)
    parser.add_argument("--continuation-step-tokens", type=int, default=DEFAULT_CONTINUATION_STEP_TOKENS)
    parser.add_argument("--continuation-max-tokens", type=int, default=DEFAULT_CONTINUATION_MAX_TOKENS)
    parser.add_argument("--continuation-min-fvu-drop", type=float, default=DEFAULT_CONTINUATION_MIN_FVU_DROP)
    parser.add_argument("--disable-auto-continuation", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--stall-timeout-sec", type=int, default=DEFAULT_STALL_TIMEOUT_SEC)
    parser.add_argument("--poll-interval-sec", type=int, default=DEFAULT_POLL_INTERVAL_SEC)
    parser.add_argument("--first-step-timeout-sec", type=int, default=DEFAULT_FIRST_STEP_TIMEOUT_SEC)
    parser.add_argument("--slow-run-grace-sec", type=int, default=DEFAULT_SLOW_RUN_GRACE_SEC)
    parser.add_argument("--min-tokens-per-sec-ratio", type=float, default=DEFAULT_MIN_TOKENS_PER_SEC_RATIO)
    parser.add_argument("--min-progress-steps", type=int, default=DEFAULT_MIN_PROGRESS_STEPS)
    parser.add_argument("--max-consecutive-crashes", type=int, default=0)
    parser.add_argument("--max-consecutive-no-improve", type=int, default=0)
    parser.add_argument("--max-repair-attempts", type=int, default=DEFAULT_MAX_REPAIR_ATTEMPTS)
    parser.add_argument("--agent-max-retries", type=int, default=3)
    parser.add_argument("--agent-retry-base-sec", type=int, default=DEFAULT_AGENT_RETRY_BASE_SEC)
    parser.add_argument("--agent-timeout-sec", type=int, default=DEFAULT_AGENT_TIMEOUT_SEC)
    parser.add_argument("--session-mode", choices=["resume-session", "fresh-each-round"], default="resume-session")
    parser.add_argument("--max-session-rounds", type=int, default=DEFAULT_MAX_SESSION_ROUNDS)
    parser.add_argument("--max-session-hours", type=float, default=DEFAULT_MAX_SESSION_HOURS)
    parser.add_argument("--max-session-failures", type=int, default=DEFAULT_MAX_SESSION_FAILURES)
    parser.add_argument("--no-commit-experiments", action="store_true")
    parser.add_argument("--reset-failure-counters", action="store_true")

    parser.add_argument("--_single-round-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_loop-start-time", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_replay-action-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_replay-result-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_replay-config-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_replay-round-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_replay-started-at", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def _load_agent_state() -> dict[str, object]:
    try:
        with open(STATE_PATH) as f:
            import json

            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    agent = payload.get("agent", {})
    return agent if isinstance(agent, dict) else {}


def _parent_should_stop(args: argparse.Namespace, loop_start_time: float) -> bool:
    elapsed_hours = (time.time() - loop_start_time) / 3600.0
    if elapsed_hours >= args.budget_hours:
        print("Budget exhausted, stopping loop")
        return True

    agent = _load_agent_state()
    crashes = int(agent.get("consecutive_crashes", 0) or 0)
    no_improve = int(agent.get("consecutive_no_improve", 0) or 0)
    if args.max_consecutive_crashes > 0 and crashes >= args.max_consecutive_crashes:
        print(f"Stopping: {crashes} consecutive crashes")
        return True
    if args.max_consecutive_no_improve > 0 and no_improve >= args.max_consecutive_no_improve:
        print(f"Stopping: {no_improve} rounds without improvement")
        return True
    return False


def _strip_flag(argv: list[str], flag: str) -> list[str]:
    return [arg for arg in argv if arg != flag]


def _strip_internal_args(raw_argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for arg in raw_argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--_single-round-worker":
            continue
        if arg == "--_loop-start-time":
            skip_next = True
            continue
        cleaned.append(arg)
    return cleaned


def _run_parent(args: argparse.Namespace, raw_argv: list[str]) -> int:
    loop_start_time = time.time()
    print(f"Loop started: rounds={args.rounds}, budget_hours={args.budget_hours}")

    child_base_argv = _strip_internal_args(raw_argv)

    for worker_index in range(args.rounds):
        if _parent_should_stop(args, loop_start_time):
            break

        child_argv = list(child_base_argv)
        if worker_index > 0:
            child_argv = _strip_flag(child_argv, "--reset-failure-counters")
        child_argv.extend([
            "--_single-round-worker",
            "--_loop-start-time",
            str(loop_start_time),
        ])

        result = subprocess.run(
            [sys.executable, "-m", "research.AutoResearch", *child_argv],
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            print(f"Worker exited with code {result.returncode}, stopping loop")
            return result.returncode

    return 0


def _run_single_round_worker(args: argparse.Namespace) -> int:
    from .types import LoopConfig
    from .loop import run_one_round

    config = LoopConfig.from_args(args)
    loop_start_time = args._loop_start_time if args._loop_start_time is not None else time.time()
    return run_one_round(config, loop_start_time=loop_start_time)


def _run_replay_worker(args: argparse.Namespace) -> int:
    from .types import LoopConfig
    from .loop import run_replayed_action

    if not args._replay_config_path or not args._replay_action_path or not args._replay_result_path:
        raise RuntimeError("Replay worker requires config, action, and result paths")

    config_data = json.loads(Path(args._replay_config_path).read_text())
    config = LoopConfig(**config_data)
    loop_start_time = args._loop_start_time if args._loop_start_time is not None else time.time()
    return run_replayed_action(
        config,
        round_id=int(args._replay_round_id),
        action_path=Path(args._replay_action_path),
        result_path=Path(args._replay_result_path),
        started_at=args._replay_started_at or int(time.time()),
        loop_start_time=loop_start_time,
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args._replay_action_path:
        return _run_replay_worker(args)
    if args._single_round_worker:
        return _run_single_round_worker(args)
    return _run_parent(args, sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
