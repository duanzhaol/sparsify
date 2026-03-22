"""CLI entry point: python -m research.AutoResearch"""

from __future__ import annotations

import argparse
import sys

from .types import (
    DEFAULT_AGENT_PROXY,
    DEFAULT_AGENT_RETRY_BASE_SEC,
    DEFAULT_AGENT_TIMEOUT_SEC,
    DEFAULT_FIRST_STEP_TIMEOUT_SEC,
    DEFAULT_MAX_REPAIR_ATTEMPTS,
    DEFAULT_MAX_SESSION_FAILURES,
    DEFAULT_MAX_SESSION_HOURS,
    DEFAULT_MAX_SESSION_ROUNDS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_PROGRESS_STEPS,
    DEFAULT_MIN_TOKENS_PER_SEC_RATIO,
    DEFAULT_POLL_INTERVAL_SEC,
    DEFAULT_SLOW_RUN_GRACE_SEC,
    DEFAULT_STALL_TIMEOUT_SEC,
    DEFAULT_TIMEOUT_SEC,
    LoopConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="SAE autoresearch loop")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--budget-hours", type=float, default=8.0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--agent-proxy", default=DEFAULT_AGENT_PROXY)
    parser.add_argument("--max-tokens", default=DEFAULT_MAX_TOKENS)
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

    args = parser.parse_args()
    config = LoopConfig.from_args(args)

    from .loop import run
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
