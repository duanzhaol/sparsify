"""Data types, constants, and configuration for the autoresearch framework."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research"
HISTORY_DIR = RESEARCH_DIR / "history"
LOG_DIR = HISTORY_DIR / "logs"
ROUND_SUMMARIES_DIR = HISTORY_DIR / "round_summaries"
SAVE_ROOT = REPO_ROOT / "checkpoints" / "research_agent"

STATE_PATH = HISTORY_DIR / "state.json"
RESULTS_PATH = HISTORY_DIR / "results.tsv"
FRONTIER_PATH = HISTORY_DIR / "frontier.json"
MEMORY_PATH = HISTORY_DIR / "memory.json"
STATUS_PATH = HISTORY_DIR / "current_status.json"
HINTS_PATH = HISTORY_DIR / "operator_hints.json"
TIMELINE_PATH = HISTORY_DIR / "timeline.jsonl"
SESSION_BRIEF_PATH = HISTORY_DIR / "session_brief.json"
OPERATOR_GUIDE_PATH = RESEARCH_DIR / "operator_guide.md"
PRIOR_RESEARCH_PATH = Path(__file__).resolve().parent / "prior_research_history.md"
SCHEMA_PATH = RESEARCH_DIR / "agent_action.schema.json"

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

BASE_ENV_DEFAULTS: dict[str, str] = {
    "ARCHITECTURE": "topk",
    "EXPANSION_FACTOR": "8",
    "K": "128",
    "HOOKPOINTS": "layers.[3].self_attn.o_proj",
    "OPTIMIZER": "signum",
    "LR": "8e-4",
    "BATCH_SIZE": "1",
    "GRAD_ACC_STEPS": "8",
    "MICRO_ACC_STEPS": "1",
    "AUXK_ALPHA": "0.03125",
    "DEAD_FEATURE_THRESHOLD": "10000000",
    "USE_HADAMARD": "0",
    "COMPILE_MODEL": "1",
}

REPAIRABLE_ERROR_TYPES: set[str] = {
    "SanityCheckError",
    "sanity_check_failed",
    "SyntaxError",
    "NameError",
    "AttributeError",
    "TypeError",
    "ValueError",
    "RuntimeError",
    "AssertionError",
    "KeyError",
    "IndexError",
    "NotImplementedError",
}

# Timeout defaults (seconds)
DEFAULT_TIMEOUT_SEC = 30 * 60
DEFAULT_STALL_TIMEOUT_SEC = 15 * 60
DEFAULT_POLL_INTERVAL_SEC = 30
DEFAULT_FIRST_STEP_TIMEOUT_SEC = 180
DEFAULT_SLOW_RUN_GRACE_SEC = 120
DEFAULT_MIN_TOKENS_PER_SEC_RATIO = 0.25
DEFAULT_MIN_PROGRESS_STEPS = 4
DEFAULT_PROCESS_TERM_TIMEOUT_SEC = 15
DEFAULT_MAX_TOKENS = "50000000"

# Agent defaults
DEFAULT_AGENT_PROXY: str | None = None
DEFAULT_MAX_SESSION_ROUNDS = 8
DEFAULT_MAX_SESSION_HOURS = 4.0
DEFAULT_AGENT_MAX_RETRIES = 3
DEFAULT_AGENT_RETRY_BASE_SEC = 10
DEFAULT_MAX_SESSION_FAILURES = 3
DEFAULT_AGENT_TIMEOUT_SEC = 10 * 60
DEFAULT_MAX_REPAIR_ATTEMPTS = 5

# Results TSV columns
RESULTS_COLUMNS = [
    "experiment_id", "timestamp", "status", "decision",
    "val_fvu", "k", "architecture", "expansion_factor",
    "wall_time_sec", "peak_memory_gb", "total_tokens",
    "checkpoint", "head_commit", "head_branch", "workspace_dirty",
    "description", "self_review", "log_path",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Action:
    """Parsed and validated agent action."""

    command: str  # "run"
    hypothesis: str
    summary: str
    change_type: str  # param_only | edit_sae_code | edit_perf_code | no_change
    expected_win: str  # lower_fvu | smaller_k | lower_cost | explore_unknown
    family_name: str
    family_stage: str  # mainline | prototype | stabilize | promote_to_mainline
    self_review: str
    needs_sanity: bool
    env_overrides: list[dict[str, str]]
    touched_files: list[str]
    notes_to_memory: list[str]
    next_hypotheses: list[str]
    primary_variable: str  # architecture | optimizer | lr | k | ... | code_fix

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Action:
        """Construct from raw agent JSON. Validates required keys."""
        required = _load_schema_required()
        # experiment_tier removed in single-tier mode
        required = [k for k in required if k != "experiment_tier"]
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"Action missing required keys: {', '.join(missing)}")

        # Normalize env_overrides: accept both list-of-dicts and dict formats
        raw_overrides = d.get("env_overrides", [])
        if isinstance(raw_overrides, dict):
            raw_overrides = [{"key": k, "value": str(v)} for k, v in raw_overrides.items()]

        return cls(
            command=d.get("command", "run"),
            hypothesis=d.get("hypothesis", ""),
            summary=d.get("summary", ""),
            change_type=d.get("change_type", "param_only"),
            expected_win=d.get("expected_win", "explore_unknown"),
            family_name=d.get("family_name", BASE_ENV_DEFAULTS["ARCHITECTURE"]),
            family_stage=d.get("family_stage", "mainline"),
            self_review=d.get("self_review", ""),
            needs_sanity=bool(d.get("needs_sanity", False)),
            env_overrides=raw_overrides,
            touched_files=d.get("touched_files", []),
            notes_to_memory=d.get("notes_to_memory", []),
            next_hypotheses=d.get("next_hypotheses", []),
            primary_variable=d.get("primary_variable", "other_param"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "hypothesis": self.hypothesis,
            "summary": self.summary,
            "change_type": self.change_type,
            "expected_win": self.expected_win,
            "family_name": self.family_name,
            "family_stage": self.family_stage,
            "self_review": self.self_review,
            "needs_sanity": self.needs_sanity,
            "env_overrides": self.env_overrides,
            "touched_files": self.touched_files,
            "notes_to_memory": self.notes_to_memory,
            "next_hypotheses": self.next_hypotheses,
            "primary_variable": self.primary_variable,
        }

    @property
    def is_code_edit(self) -> bool:
        return self.change_type in ("edit_sae_code", "edit_perf_code")

    def env_dict(self) -> dict[str, str]:
        """Return env_overrides as a flat dict."""
        return {item["key"]: item["value"] for item in self.env_overrides}

    def effective_config(self) -> dict[str, str]:
        """Return BASE_ENV_DEFAULTS merged with env_overrides."""
        cfg = dict(BASE_ENV_DEFAULTS)
        cfg.update(self.env_dict())
        return cfg


@dataclass
class Result:
    """Outcome of a training run or policy rejection."""

    decision: str  # keep | archive | discard | crash | policy_reject
    status: str
    timestamp: str | None = None
    val_fvu: str | None = None
    observed_fvu: str | None = None
    k: str | None = None
    architecture: str | None = None
    expansion_factor: str | None = None
    run_health: str = "normal"  # normal | perf_regression | crash
    termination_reason: str = "completed"
    wall_time_sec: str | None = None
    peak_memory_gb: str | None = None
    total_tokens: str | None = None
    tokens_per_sec: str | None = None
    baseline_ratio: str | None = None
    checkpoint: str | None = None
    experiment_id: str | None = None
    head_commit: str | None = None
    head_branch: str | None = None
    workspace_dirty: str | None = None
    description: str = ""
    self_review: str = ""
    log_path: str | None = None
    metrics_path: str | None = None
    error_type: str | None = None
    error_summary: str | None = None
    traceback_excerpt: str | None = None
    log_excerpt: str | None = None
    # Curve metrics
    curve_start_fvu: str | None = None
    curve_mid_fvu: str | None = None
    curve_end_fvu: str | None = None
    curve_late_slope: str | None = None
    curve_still_improving: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Convert to flat string dict (for results.tsv compatibility)."""
        d = {}
        for k, v in self.__dict__.items():
            if v is not None:
                d[k] = str(v)
        return d

    @classmethod
    def crash(cls, reason: str, **kwargs: Any) -> Result:
        import time as _time
        return cls(
            decision="crash",
            status="crash",
            timestamp=str(int(_time.time())),
            run_health="crash",
            termination_reason=reason,
            **kwargs,
        )

    @classmethod
    def policy_reject(cls, reason: str) -> Result:
        import time as _time
        return cls(
            decision="policy_reject",
            status="policy_reject",
            timestamp=str(int(_time.time())),
            termination_reason=reason,
            description=reason,
        )


@dataclass
class RoundContext:
    """Mutable context accumulated during a single round."""

    round_id: int
    started_at: int
    family_name: str | None = None
    family_stage: str | None = None
    session_id: str | None = None
    repair_attempts: list[dict[str, Any]] = field(default_factory=list)
    timeline_event_ids: list[str] = field(default_factory=list)
    touched_files: list[str] = field(default_factory=list)
    patch_path: Path | None = None

    # Repair cycle tracking
    last_repair_signature: str | None = None
    same_repair_streak: int = 0


@dataclass
class LoopConfig:
    """All runtime configuration, parsed from CLI args."""

    rounds: int = 20
    budget_hours: float = 8.0
    model: str | None = None
    agent_proxy: str | None = DEFAULT_AGENT_PROXY
    max_tokens: str = DEFAULT_MAX_TOKENS
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    stall_timeout_sec: int = DEFAULT_STALL_TIMEOUT_SEC
    poll_interval_sec: int = DEFAULT_POLL_INTERVAL_SEC
    first_step_timeout_sec: int = DEFAULT_FIRST_STEP_TIMEOUT_SEC
    slow_run_grace_sec: int = DEFAULT_SLOW_RUN_GRACE_SEC
    min_tokens_per_sec_ratio: float = DEFAULT_MIN_TOKENS_PER_SEC_RATIO
    min_progress_steps: int = DEFAULT_MIN_PROGRESS_STEPS
    max_consecutive_crashes: int = 0
    max_consecutive_no_improve: int = 0
    max_repair_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS
    agent_max_retries: int = DEFAULT_AGENT_MAX_RETRIES
    agent_retry_base_sec: int = DEFAULT_AGENT_RETRY_BASE_SEC
    agent_timeout_sec: int = DEFAULT_AGENT_TIMEOUT_SEC
    session_mode: str = "resume-session"
    max_session_rounds: int = DEFAULT_MAX_SESSION_ROUNDS
    max_session_hours: float = DEFAULT_MAX_SESSION_HOURS
    max_session_failures: int = DEFAULT_MAX_SESSION_FAILURES
    auto_commit: bool = True
    reset_failure_counters: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> LoopConfig:
        return cls(
            rounds=args.rounds,
            budget_hours=args.budget_hours,
            model=args.model,
            agent_proxy=args.agent_proxy,
            max_tokens=args.max_tokens,
            timeout_sec=args.timeout_sec,
            stall_timeout_sec=args.stall_timeout_sec,
            poll_interval_sec=args.poll_interval_sec,
            first_step_timeout_sec=args.first_step_timeout_sec,
            slow_run_grace_sec=args.slow_run_grace_sec,
            min_tokens_per_sec_ratio=args.min_tokens_per_sec_ratio,
            min_progress_steps=args.min_progress_steps,
            max_consecutive_crashes=args.max_consecutive_crashes,
            max_consecutive_no_improve=args.max_consecutive_no_improve,
            max_repair_attempts=args.max_repair_attempts,
            agent_max_retries=args.agent_max_retries,
            agent_retry_base_sec=args.agent_retry_base_sec,
            agent_timeout_sec=getattr(args, "agent_timeout_sec", DEFAULT_AGENT_TIMEOUT_SEC),
            session_mode=args.session_mode,
            max_session_rounds=args.max_session_rounds,
            max_session_hours=args.max_session_hours,
            max_session_failures=getattr(args, "max_session_failures", DEFAULT_MAX_SESSION_FAILURES),
            auto_commit=not args.no_commit_experiments,
            reset_failure_counters=args.reset_failure_counters,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def failure_signature(payload: dict[str, Any]) -> str:
    """Compute a signature string for deduplicating repeated failures."""
    family = str(payload.get("family_name") or "")
    error_type = str(payload.get("error_type") or "")
    termination_reason = str(payload.get("termination_reason") or "")
    summary = str(
        payload.get("error_summary")
        or payload.get("traceback_excerpt")
        or payload.get("stderr_excerpt")
        or payload.get("message")
        or ""
    ).strip()
    summary = " ".join(summary.split())[:240]
    return " | ".join((family, error_type, termination_reason, summary))


def _load_schema_required() -> list[str]:
    """Load required keys from the agent action JSON schema."""
    try:
        with open(SCHEMA_PATH) as f:
            return json.load(f).get("required", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []
