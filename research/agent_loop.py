"""
Nightly SAE autoresearch loop driven by Codex CLI.

This loop keeps the execution layer fixed while delegating experiment choice
and optional code edits to a short-lived model call each round.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import signal
import re
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = REPO_ROOT / "research"
SCRIPT_PATH = REPO_ROOT / "scripts" / "autoresearch_test.sh"
CONTROLLER_PATH = RESEARCH_DIR / "controller.py"
PROGRAM_PATH = RESEARCH_DIR / "program.md"
SCHEMA_PATH = RESEARCH_DIR / "agent_action.schema.json"
HISTORY_DIR = RESEARCH_DIR / "history"
LOG_DIR = HISTORY_DIR / "logs"
ROUND_SUMMARIES_DIR = HISTORY_DIR / "round_summaries"
STATE_PATH = HISTORY_DIR / "state.json"
RESULTS_PATH = HISTORY_DIR / "results.tsv"
FRONTIER_PATH = HISTORY_DIR / "frontier.json"
MEMORY_PATH = HISTORY_DIR / "memory.json"
STATUS_PATH = HISTORY_DIR / "current_status.json"
HINTS_PATH = HISTORY_DIR / "operator_hints.json"
TIMELINE_PATH = HISTORY_DIR / "timeline.jsonl"
SESSION_BRIEF_PATH = HISTORY_DIR / "session_brief.json"
SAVE_ROOT = REPO_ROOT / "checkpoints" / "research_agent"

DEFAULT_PROXY_MAX_TOKENS = "20000000"
DEFAULT_FULL_MAX_TOKENS = "200000000"
DEFAULT_PROXY_TIMEOUT_SEC = 30 * 60
DEFAULT_FULL_TIMEOUT_SEC = 2 * 60 * 60
DEFAULT_STALL_TIMEOUT_SEC = 15 * 60
DEFAULT_POLL_INTERVAL_SEC = 30
DEFAULT_AGENT_PROXY = "http://127.0.0.1:23234"
DEFAULT_FIRST_STEP_TIMEOUT_SEC = 180
DEFAULT_SLOW_RUN_GRACE_SEC = 120
DEFAULT_MIN_TOKENS_PER_SEC_RATIO = 0.25
DEFAULT_MIN_PROGRESS_STEPS = 4
DEFAULT_MAX_SESSION_ROUNDS = 20
DEFAULT_MAX_SESSION_HOURS = 8.0
DEFAULT_RESEARCH_BRANCH_PREFIX = "research/nightly-"
DEFAULT_AGENT_MAX_RETRIES = 3
DEFAULT_AGENT_RETRY_BASE_SEC = 10
DEFAULT_MAX_SESSION_FAILURES = 3
DEFAULT_PROCESS_TERM_TIMEOUT_SEC = 15

BASE_ENV_DEFAULTS = {
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

ALLOWED_EDIT_PREFIXES = ("sparsify/",)

# Whitelist of env keys the agent is allowed to set via env_overrides.
# Anything not in this set is rejected to prevent accidental system damage.
ALLOWED_ENV_OVERRIDE_KEYS = set(BASE_ENV_DEFAULTS.keys()) | {
    "NUM_GROUPS",
    "ACTIVE_GROUPS",
    "JUMPRELU_INIT_THRESHOLD",
    "JUMPRELU_BANDWIDTH",
    "ORTHO_LAMBDA",
    "RESIDUAL_FROM",
    "MATRYOSHKA_KS",
    "MATRYOSHKA_WEIGHTS",
}
SNAPSHOT_ROOTS = ("sparsify", "research", "scripts")
SNAPSHOT_EXCLUDES = (
    "research/history/",
    "sparsify/__pycache__/",
    "research/__pycache__/",
    "scripts/__pycache__/",
    "research/agent_loop.py",
    "research/controller.py",
    "research/orchestrator.py",
    "research/prepare.py",
    "research/program.md",
    "research/agent_action.schema.json",
    "scripts/autoresearch_test.sh",
)
SNAPSHOT_EXCLUDE_SUFFIXES = (".pyc", ".pyo")
TRACKED_HISTORY_PATHS = (
    STATE_PATH,
    RESULTS_PATH,
    FRONTIER_PATH,
    MEMORY_PATH,
    TIMELINE_PATH,
    SESSION_BRIEF_PATH,
    HINTS_PATH,
)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=check, text=True, capture_output=True)


def git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=check, text=True, capture_output=True)


def ensure_setup() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ROUND_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    TIMELINE_PATH.touch(exist_ok=True)
    run(["python", str(CONTROLLER_PATH), "init"], cwd=RESEARCH_DIR)


def current_git_branch() -> str:
    return git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def current_git_commit() -> str:
    return git(["rev-parse", "HEAD"]).stdout.strip()


def worktree_dirty() -> bool:
    return bool(git(["status", "--porcelain"]).stdout.strip())


def default_research_branch_name() -> str:
    return f"{DEFAULT_RESEARCH_BRANCH_PREFIX}{time.strftime('%Y%m%d')}"


def branch_exists(branch: str) -> bool:
    return git(["rev-parse", "--verify", branch], check=False).returncode == 0


def ensure_clean_worktree_for_auto_commit() -> None:
    if worktree_dirty():
        raise RuntimeError(
            "Auto-commit mode requires a clean git worktree before starting the nightly loop."
        )


def ensure_research_branch(branch: str, state: dict[str, Any]) -> None:
    base_branch = current_git_branch()
    base_commit = current_git_commit()
    if base_branch != branch:
        if branch_exists(branch):
            git(["checkout", branch])
        else:
            git(["checkout", "-b", branch])
    state["research_branch"] = branch
    state["base_branch"] = state.get("base_branch", base_branch)
    state["base_commit"] = state.get("base_commit", base_commit)
    state["research_branch_created_at"] = state.get("research_branch_created_at", int(time.time()))


def stage_paths(paths: list[Path]) -> None:
    rel_paths = []
    for path in paths:
        if path.exists():
            rel_paths.append(path.relative_to(REPO_ROOT).as_posix())
    if rel_paths:
        git(["add", "--", *rel_paths])


def commit_message_for_round(round_id: int, action: dict[str, Any], result: dict[str, str], tier: str) -> str:
    family = str(action.get("family_name") or action.get("change_type") or "unknown")
    decision = str(result.get("decision") or "unknown")
    return f"experiment: round {round_id:04d} {tier} {family} {decision}"


def commit_round_state(
    round_id: int,
    action: dict[str, Any],
    result: dict[str, str],
    touched: list[str],
    round_summary_path: Path,
    tier: str,
) -> tuple[str | None, str]:
    paths = [REPO_ROOT / path for path in touched]
    paths.extend(TRACKED_HISTORY_PATHS)
    paths.append(round_summary_path)
    stage_paths(paths)
    if git(["diff", "--cached", "--quiet"], check=False).returncode == 0:
        return None, current_git_branch()
    message = commit_message_for_round(round_id, action, result, tier)
    git(["commit", "-m", message])
    return current_git_commit(), current_git_branch()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_state() -> dict[str, Any]:
    state = load_json(STATE_PATH, {})
    agent_state = state.setdefault(
        "agent",
        {
            "round_index": 0,
            "consecutive_crashes": 0,
            "consecutive_no_improve": 0,
            "rounds_since_new_family": 0,
            "last_action_file": None,
            "last_patch_file": None,
            "active_session_id": None,
            "active_session_started_at": None,
            "active_session_rounds": 0,
            "active_session_status": "closed",
            "last_resume_ok_at": None,
        },
    )
    agent_state.setdefault("round_index", 0)
    agent_state.setdefault("consecutive_crashes", 0)
    agent_state.setdefault("consecutive_no_improve", 0)
    agent_state.setdefault("rounds_since_new_family", 0)
    agent_state.setdefault("last_action_file", None)
    agent_state.setdefault("last_patch_file", None)
    agent_state.setdefault("active_session_id", None)
    agent_state.setdefault("active_session_started_at", None)
    agent_state.setdefault("active_session_rounds", 0)
    agent_state.setdefault("active_session_status", "closed")
    agent_state.setdefault("last_resume_ok_at", None)
    return state


def save_state(state: dict[str, Any]) -> None:
    save_json(STATE_PATH, state)
    save_json(FRONTIER_PATH, state.get("frontier", {}))


def load_memory() -> dict[str, Any]:
    return load_json(
        MEMORY_PATH,
        {
            "current_focus": (
                "Treat K=128 only as the initial search anchor, not as a success criterion. "
                "The research objective is to find configurations whose FVU is dramatically better "
                "than the current K=128 baseline, with a target on the order of halving that baseline FVU "
                "before rewarding smaller K or lower cost."
            ),
            "architecture_findings": [],
            "performance_findings": [],
            "failure_patterns": [],
            "baseline_runtime": {},
            "architecture_families": {},
            "recent_rounds": [],
            "recent_insights": [],
            "next_hypotheses": [
                "Establish or refresh a trustworthy K=128 quality anchor before rewarding lower-K points.",
                "Do not treat smaller K as success unless its FVU is materially better than the current K=128 baseline.",
                "Prefer experiments that can plausibly cut the current K=128 FVU by about half, even if they keep K unchanged at first."
            ],
        },
    )


def load_operator_hints() -> list[dict[str, Any]]:
    return load_json(HINTS_PATH, [])


def save_operator_hints(hints: list[dict[str, Any]]) -> None:
    save_json(HINTS_PATH, hints)


def split_hints(hints: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pending = [h for h in hints if h.get("status") == "pending"]
    non_pending = [h for h in hints if h.get("status") != "pending"]
    return pending, non_pending


def write_status(stage: str, **payload: Any) -> None:
    save_json(
        STATUS_PATH,
        {
            "timestamp": int(time.time()),
            "stage": stage,
            **payload,
        },
    )


def append_timeline_event(event: str, **payload: Any) -> str:
    event_id = f"evt_{time.time_ns()}"
    record = {
        "event_id": event_id,
        "ts": int(time.time()),
        "event": event,
        **payload,
    }
    with open(TIMELINE_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return event_id


def log_round_event(round_ctx: dict[str, Any], event: str, **payload: Any) -> str:
    event_id = append_timeline_event(
        event,
        round=round_ctx.get("round"),
        session_id=payload.get("session_id", round_ctx.get("session_id")),
        tier=payload.get("tier"),
        run_name=payload.get("run_name"),
        family_name=payload.get("family_name", round_ctx.get("family_name")),
        family_stage=payload.get("family_stage", round_ctx.get("family_stage")),
        status=payload.get("status"),
        decision=payload.get("decision"),
        payload=payload,
    )
    round_ctx.setdefault("timeline_event_ids", []).append(event_id)
    round_ctx.setdefault("timeline_start_event_id", event_id)
    round_ctx["timeline_end_event_id"] = event_id
    return event_id


def load_results(limit: int | None = None) -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    with open(RESULTS_PATH, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    return rows[-limit:] if limit is not None else rows


def summarize_results(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keys = ["experiment_id", "tier", "status", "decision", "val_fvu", "k", "architecture", "description"]
    return [{k: row.get(k, "") for k in keys} for row in rows]


def recent_round_summaries(limit: int = 3) -> list[dict[str, Any]]:
    paths = sorted(ROUND_SUMMARIES_DIR.glob("round_*.json"))[-limit:]
    return [load_json(path, {}) for path in paths]


def load_session_brief() -> dict[str, Any]:
    return load_json(
        SESSION_BRIEF_PATH,
        {
            "updated_at": None,
            "active_session_id": None,
            "current_focus": None,
            "best_full_frontier": {},
            "recent_results": [],
            "recent_round_summaries": [],
            "incubating_families": {},
            "recent_performance_findings": [],
            "pending_hints": [],
            "next_move_guidance": [],
        },
    )


def save_session_brief(brief: dict[str, Any]) -> None:
    save_json(SESSION_BRIEF_PATH, brief)


def snapshot_paths() -> dict[str, str]:
    snapshots: dict[str, str] = {}
    for root in SNAPSHOT_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if any(rel.startswith(prefix) for prefix in SNAPSHOT_EXCLUDES):
                continue
            if rel.endswith(SNAPSHOT_EXCLUDE_SUFFIXES):
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            snapshots[rel] = digest
    return snapshots


def touched_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    paths = set(before) | set(after)
    return sorted(path for path in paths if before.get(path) != after.get(path))


def assert_allowed_changes(paths: list[str]) -> None:
    disallowed = [path for path in paths if not path.startswith(ALLOWED_EDIT_PREFIXES)]
    if disallowed:
        joined = ", ".join(disallowed)
        raise RuntimeError(f"Agent touched files outside allowed prefixes: {joined}")


def read_text_safe(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def build_patch(before_snapshot: dict[str, str], after_paths: list[str], round_id: int) -> str:
    patch_parts: list[str] = []
    temp_root = HISTORY_DIR / ".snapshots"
    before_root = temp_root / f"round_{round_id:04d}_before"
    after_root = temp_root / f"round_{round_id:04d}_after"
    before_root.mkdir(parents=True, exist_ok=True)
    after_root.mkdir(parents=True, exist_ok=True)

    for rel in after_paths:
        if rel.endswith(SNAPSHOT_EXCLUDE_SUFFIXES) or "/__pycache__/" in rel:
            continue
        source = REPO_ROOT / rel
        before_path = before_root / rel
        after_path = after_root / rel
        before_path.parent.mkdir(parents=True, exist_ok=True)
        after_path.parent.mkdir(parents=True, exist_ok=True)
        before_text = ""
        if rel in before_snapshot and source.exists():
            # Caller only needs textual diff for changed files. Best-effort current text.
            pass
        if source.exists():
            try:
                after_text = source.read_text()
            except UnicodeDecodeError:
                continue
            after_path.write_text(after_text)
        else:
            after_text = ""
        if before_path.exists():
            try:
                before_text = before_path.read_text()
            except UnicodeDecodeError:
                continue
        before_lines = before_text.splitlines(keepends=True)
        after_lines = after_text.splitlines(keepends=True)
        patch = "".join(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )
        if patch:
            patch_parts.append(patch)
    return "\n".join(patch_parts)


def capture_before_files(paths: list[str], round_id: int) -> None:
    temp_root = HISTORY_DIR / ".snapshots"
    before_root = temp_root / f"round_{round_id:04d}_before"
    for rel in paths:
        src = REPO_ROOT / rel
        dst = before_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            dst.write_bytes(src.read_bytes())


def build_prompt(state: dict[str, Any], memory: dict[str, Any], results: list[dict[str, str]]) -> str:
    frontier = load_json(FRONTIER_PATH, state.get("frontier", {}))
    operator_hints, _ = split_hints(load_operator_hints())
    context = {
        "frontier": frontier,
        "proxy_frontier": state.get("proxy_frontier", {}),
        "full_frontier": state.get("full_frontier", frontier),
        "agent_state": state.get("agent", {}),
        "rounds_since_new_family": state.get("agent", {}).get("rounds_since_new_family", 0),
        "current_focus": memory.get("current_focus"),
        "architecture_families": memory.get("architecture_families", {}),
        "recent_insights": memory.get("recent_insights", [])[-8:],
        "recent_performance_findings": memory.get("performance_findings", [])[-8:],
        "baseline_runtime": memory.get("baseline_runtime", {}),
        "operator_hints": operator_hints[:8],
        "next_hypotheses": memory.get("next_hypotheses", [])[:8],
        "recent_results": summarize_results(results),
        "recent_round_summaries": recent_round_summaries(),
    }
    return f"""
You are the nightly SAE research agent for this repository.

Primary objective:
- reduce FVU
- K=128 is only an initial quality anchor, not a success criterion
- only reward smaller K after quality has materially beaten the current K=128 baseline
- the quality target is aggressive: aim for configurations that can drive FVU to roughly half of the current K=128 baseline before preferring smaller K or lower cost
- then reduce cost / improve throughput / memory

Execution layer is fixed:
- training entrypoint: scripts/autoresearch_test.sh
- results recorder: research/controller.py
- memory files: research/history/state.json, frontier.json, memory.json, results.tsv

Important rules:
- Read research/program.md before deciding.
- Read and respect any operator_hints in the structured context.
- You may edit ONLY files under sparsify/.
- Do not edit research/history/*, research/*.py, or scripts/autoresearch_test.sh.
- For parameter-only experiments, use env_overrides instead of editing launch code.
- Make at most ONE coherent hypothesis this round.
- Fill family_name and family_stage when the round is about a specific architecture family.
- Prefer proxy unless there is a strong reason to go straight to full.
- Direct full requests may be coerced back to proxy by the runtime.
- You are allowed to explore the SAE architecture space broadly, not just tune existing defaults.
- Architectural examples such as Sparse-ReLU, Gated, JumpReLU, GroupTopK, MoE-style encoders, factorized encoders, routed encoders, multi-branch encoders, or new sparse activation mechanisms are examples only, not limits.
- If a new architecture family seems promising, you may add it under sparsify/ as long as the change is coherent and compatible with the existing execution layer.
- Encoder design is part of the search space. You may change routing, gating, intermediate width, branch structure, activation form, grouping strategy, or other internal encoder mechanisms when justified.
- When exploring a new architecture family, also consider its own important hyperparameters, for example intermediate width or routing width in an ICE/MoE-style encoder.
- Slow runs may indicate implementation bottlenecks rather than bad architectures.
- If a recent run was a performance regression, prefer an edit_perf_code follow-up over drawing a negative architecture conclusion.
- If parameter-only search is not closing the quality gap, escalate to architecture exploration rather than continuing a weak local search.
- A new architecture family may require several rounds of incubation. Do not expect a first prototype to beat the frontier immediately.
- Use family_stage to reflect whether the round is a prototype, stabilization pass, mainline comparison, or promotion attempt.
- If several rounds have passed without proposing a new family or advancing an incubating family, bias toward architecture exploration rather than more local tuning.
- If there is no promising next move, return command="stop".
- Return a final JSON object matching the schema exactly.

Current structured context:
{json.dumps(context, indent=2)}
""".strip()


def build_session_bootstrap_prompt(state: dict[str, Any], memory: dict[str, Any], results: list[dict[str, str]]) -> str:
    return build_prompt(state, memory, results)


def build_resume_prompt(
    state: dict[str, Any],
    memory: dict[str, Any],
    results: list[dict[str, str]],
    round_id: int,
    brief: dict[str, Any],
) -> str:
    operator_hints, _ = split_hints(load_operator_hints())
    payload = {
        "round": round_id,
        "current_focus": brief.get("current_focus") or memory.get("current_focus"),
        "best_full_frontier": brief.get("best_full_frontier", state.get("full_frontier", {})),
        "recent_results": brief.get("recent_results", summarize_results(results)),
        "recent_round_summaries": brief.get("recent_round_summaries", recent_round_summaries()),
        "incubating_families": brief.get("incubating_families", {}),
        "recent_performance_findings": brief.get("recent_performance_findings", memory.get("performance_findings", [])[-8:]),
        "pending_hints": brief.get("pending_hints", operator_hints[:8]),
        "next_move_guidance": brief.get("next_move_guidance", memory.get("next_hypotheses", [])[:8]),
        "rounds_since_new_family": state.get("agent", {}).get("rounds_since_new_family", 0),
    }
    return f"""
Continue the same nightly SAE research session.

This is round {round_id}. Use the existing session context plus the structured update below.
Do not restate policy or explain your reasoning. Return one final JSON object only, with no markdown fences.
The JSON must match the same shape as the established action contract:
- command, hypothesis, summary, change_type, experiment_tier, expected_win
- family_name, family_stage, self_review, needs_sanity
- env_overrides, touched_files, notes_to_memory, next_hypotheses

Structured update:
{json.dumps(payload, indent=2)}
""".strip()


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
            "--full-auto",
            "-o",
            str(action_path),
        ]
    else:
        cmd = [
            "codex",
            "exec",
            "--full-auto",
            "--sandbox",
            "workspace-write",
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
    result = subprocess.run(cmd, input=prompt, text=True, capture_output=True, env=env)
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
) -> tuple[dict[str, Any], Path, str | None, bool]:
    """Try agent invocation with retries and session degradation.

    Strategy:
    1. If resuming, try resume first.
    2. On failure, retry with exponential backoff up to max_retries.
    3. If resume fails, rebuild session (fresh prompt, no session_id).
    4. If fresh also fails repeatedly, raise to caller.
    """
    max_retries = args.agent_max_retries
    retry_base = args.agent_retry_base_sec
    attempt = 0
    last_error: Exception | None = None
    current_session_id = session_id if not need_new_session else None

    while attempt <= max_retries:
        if need_new_session or current_session_id is None:
            prompt = build_session_bootstrap_prompt(state, memory, recent_results)
            try_session_id = None
        else:
            prompt = build_resume_prompt(state, memory, recent_results, round_id, brief)
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
                # Resume failed — rebuild session on next attempt
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


def build_session_brief(
    state: dict[str, Any],
    memory: dict[str, Any],
    recent_results: list[dict[str, str]],
    round_id: int,
) -> dict[str, Any]:
    operator_hints, _ = split_hints(load_operator_hints())
    families = memory.get("architecture_families", {})
    incubating = {
        name: value for name, value in families.items()
        if value.get("status") == "incubating"
    }
    return {
        "updated_at": int(time.time()),
        "active_session_id": state.get("agent", {}).get("active_session_id"),
        "current_focus": memory.get("current_focus"),
        "best_full_frontier": state.get("full_frontier", state.get("frontier", {})),
        "recent_results": summarize_results(recent_results)[-3:],
        "recent_round_summaries": recent_round_summaries(limit=3),
        "incubating_families": incubating,
        "recent_performance_findings": memory.get("performance_findings", [])[-6:],
        "pending_hints": operator_hints[:8],
        "next_move_guidance": memory.get("next_hypotheses", [])[:8],
        "last_round": round_id,
    }


def validate_env_overrides(overrides: list | dict) -> list[str]:
    """Validate env_overrides against the whitelist. Return list of rejected keys."""
    rejected: list[str] = []
    if isinstance(overrides, dict):
        for key in overrides:
            if key not in ALLOWED_ENV_OVERRIDE_KEYS:
                rejected.append(key)
    else:
        for item in overrides:
            key = item.get("key")
            if key and key not in ALLOWED_ENV_OVERRIDE_KEYS:
                rejected.append(key)
    return rejected


def build_env(action: dict[str, Any], tier: str, run_name: str, save_dir: Path, args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    env = os.environ.copy()
    config = dict(BASE_ENV_DEFAULTS)
    overrides = action.get("env_overrides", [])
    rejected = validate_env_overrides(overrides)
    if rejected:
        raise RuntimeError(
            f"Agent env_overrides contain disallowed keys: {', '.join(rejected)}. "
            f"Allowed keys: {sorted(ALLOWED_ENV_OVERRIDE_KEYS)}"
        )
    if isinstance(overrides, dict):
        config.update({k: str(v) for k, v in overrides.items()})
    else:
        for item in overrides:
            key = item.get("key")
            value = item.get("value")
            if key:
                config[key] = str(value)
    env.update(config)
    env["RUN_NAME"] = run_name
    env["SAVE_DIR"] = str(save_dir)
    env["WANDB_PROJECT"] = env.get("WANDB_PROJECT", "qwen3-0.6B-auto")
    env["MAX_TOKENS"] = args.full_max_tokens if tier == "full" else args.proxy_max_tokens

    config_json = {
        "architecture": config.get("ARCHITECTURE", "topk").lower(),
        "expansion_factor": int(config.get("EXPANSION_FACTOR", "8")),
        "k": int(config.get("K", "128")),
        "optimizer": config.get("OPTIMIZER", "signum"),
        "lr": config.get("LR", "8e-4"),
        "hookpoints": config.get("HOOKPOINTS", "layers.[3].self_attn.o_proj"),
        "batch_size": int(config.get("BATCH_SIZE", "1")),
        "grad_acc_steps": int(config.get("GRAD_ACC_STEPS", "8")),
        "micro_acc_steps": int(config.get("MICRO_ACC_STEPS", "1")),
        "auxk_alpha": float(config.get("AUXK_ALPHA", "0.03125")),
        "dead_feature_threshold": int(config.get("DEAD_FEATURE_THRESHOLD", "10000000")),
        "use_hadamard": config.get("USE_HADAMARD", "0") == "1",
        "tier": tier,
        "family_name": action.get("family_name") or config.get("ARCHITECTURE", "topk").lower(),
        "family_stage": action.get("family_stage") or ("mainline" if action.get("change_type") == "param_only" else "prototype"),
    }
    optional_keys = {
        "NUM_GROUPS": ("num_groups", int),
        "ACTIVE_GROUPS": ("active_groups", int),
        "JUMPRELU_INIT_THRESHOLD": ("jumprelu_init_threshold", float),
        "JUMPRELU_BANDWIDTH": ("jumprelu_bandwidth", float),
        "ORTHO_LAMBDA": ("ortho_lambda", float),
        "RESIDUAL_FROM": ("residual_from", str),
        "MATRYOSHKA_KS": ("matryoshka_ks", lambda x: [int(v) for v in x.split(",") if v]),
        "MATRYOSHKA_WEIGHTS": ("matryoshka_weights", lambda x: [float(v) for v in x.split(",") if v]),
    }
    for env_key, (cfg_key, caster) in optional_keys.items():
        value = config.get(env_key)
        if value:
            config_json[cfg_key] = caster(value)
    return env, config_json


def runtime_baseline_key(config_json: dict[str, Any], tier: str) -> str:
    return "|".join(
        [
            tier,
            str(config_json.get("architecture")),
            str(config_json.get("hookpoints")),
        ]
    )


def latest_checkpoint_dir(save_dir: Path, run_name: str) -> Path | None:
    matches = sorted(save_dir.glob(f"{run_name}*"))
    return matches[-1] if matches else None


def metrics_last_update(checkpoint_dir: Path | None) -> float | None:
    if checkpoint_dir is None:
        return None
    metrics = checkpoint_dir / "metrics.jsonl"
    return metrics.stat().st_mtime if metrics.exists() else None


def read_latest_step_record(checkpoint_dir: Path | None) -> dict[str, Any] | None:
    if checkpoint_dir is None:
        return None
    metrics = checkpoint_dir / "metrics.jsonl"
    if not metrics.exists():
        return None
    latest: dict[str, Any] | None = None
    with open(metrics) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "step":
                latest = rec
    return latest


def extract_step_fvu(step_record: dict[str, Any] | None) -> float | None:
    if not step_record:
        return None
    vals = [
        v for k, v in step_record.items()
        if k.endswith("/fvu") and isinstance(v, (int, float))
    ]
    return sum(vals) / len(vals) if vals else None


def run_sanity(config: dict[str, Any]) -> None:
    arch = config["architecture"]
    k = config["k"]
    ef = config["expansion_factor"]
    cmd = [
        "python",
        "-c",
        (
            "import sys; sys.path.insert(0, '.'); "
            "from sparsify import SparseCoder; "
            "from sparsify.config import SparseCoderConfig; "
            "import torch; "
            f"cfg = SparseCoderConfig(architecture='{arch}', k={k}, expansion_factor={ef}); "
            "sae = SparseCoder(1024, cfg, device='cuda', dtype=torch.float32); "
            "x = torch.randn(4, 1024, device='cuda'); "
            "out = sae(x); out.fvu.backward(); print('sanity: OK')"
        ),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)


def rollback_code_changes(pre_edit_commit: str, round_id: int, reason: str) -> None:
    """Restore sparsify/ code to its state at pre_edit_commit.

    Uses ``git checkout <commit> -- sparsify/`` so only the code area is
    reverted while history files stay untouched.
    """
    print(f"Round {round_id}: rolling back code to {pre_edit_commit[:12]} ({reason})")
    git(["checkout", pre_edit_commit, "--", "sparsify/"])
    append_timeline_event(
        "code_rollback",
        round=round_id,
        tier=None,
        run_name=None,
        family_name=None,
        family_stage=None,
        status="rolled_back",
        decision=None,
        payload={
            "pre_edit_commit": pre_edit_commit,
            "reason": reason,
        },
    )


def terminate_process_group(process: subprocess.Popen, timeout: int = DEFAULT_PROCESS_TERM_TIMEOUT_SEC) -> None:
    """Gracefully terminate a process and its entire process group.

    1. Send SIGTERM to the process group.
    2. Wait up to *timeout* seconds.
    3. If still alive, send SIGKILL to the process group.
    """
    pgid = None
    try:
        pgid = os.getpgid(process.pid)
    except OSError:
        pass

    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
    else:
        try:
            process.terminate()
        except OSError:
            pass

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
        else:
            try:
                process.kill()
            except OSError:
                pass
        process.wait(timeout=5)


def budget_remaining_sec(start_time: float, budget_hours: float) -> float:
    """Return remaining budget in seconds."""
    return budget_hours * 3600 - (time.time() - start_time)


def check_agent_backend_reachable(agent_proxy: str | None) -> None:
    """Quick connectivity check before entering the main loop.

    If a local proxy is configured, verify it accepts connections.
    Also verify the ``codex`` CLI is available on PATH.
    """
    import shutil

    if not shutil.which("codex"):
        raise RuntimeError(
            "codex CLI not found on PATH. Install it or adjust PATH before running."
        )

    if agent_proxy:
        import urllib.request
        import urllib.error

        # Just open a TCP connection to the proxy; don't actually send traffic.
        try:
            req = urllib.request.Request(agent_proxy)
            req.method = "HEAD"
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": agent_proxy, "https": agent_proxy})
            )
            opener.open(req, timeout=5)
        except urllib.error.URLError:
            # Proxy may refuse HEAD but still be listening — that's OK.
            # We only fail on ConnectionRefusedError / unreachable.
            pass
        except (ConnectionRefusedError, OSError) as exc:
            raise RuntimeError(
                f"Agent proxy {agent_proxy} is not reachable: {exc}. "
                f"Start the proxy or pass --agent-proxy '' to disable."
            ) from exc


def record_result(
    log_path: Path,
    checkpoint_dir: Path | None,
    config_path: Path,
    tier: str,
    description: str,
    self_review: str,
) -> dict[str, str]:
    cmd = [
        "python",
        str(CONTROLLER_PATH),
        "record",
        "--log",
        str(log_path),
        "--tier",
        tier,
        "--description",
        description,
        "--self-review",
        self_review,
        "--config-json",
        str(config_path),
    ]
    if checkpoint_dir is not None:
        cmd.extend(["--checkpoint-dir", str(checkpoint_dir)])
    result = run(cmd, cwd=RESEARCH_DIR)
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
    return parsed


def append_memory(memory: dict[str, Any], action: dict[str, Any], result: dict[str, str], round_id: int, touched: list[str]) -> dict[str, Any]:
    overrides = action.get("env_overrides", [])
    if isinstance(overrides, dict):
        arch = overrides.get("ARCHITECTURE", BASE_ENV_DEFAULTS["ARCHITECTURE"]).lower()
    else:
        arch = next(
            (
                item.get("value", BASE_ENV_DEFAULTS["ARCHITECTURE"])
                for item in overrides
                if item.get("key") == "ARCHITECTURE"
            ),
            BASE_ENV_DEFAULTS["ARCHITECTURE"],
        ).lower()

    family_name = str(action.get("family_name") or arch).lower()
    family_stage = str(action.get("family_stage") or ("mainline" if action["change_type"] == "param_only" else "prototype"))

    entry = {
        "round": round_id,
        "hypothesis": action["hypothesis"],
        "change_type": action["change_type"],
        "expected_win": action["expected_win"],
        "family_name": family_name,
        "family_stage": family_stage,
        "decision": result.get("decision"),
        "val_fvu": result.get("val_fvu"),
        "observed_fvu": result.get("observed_fvu"),
        "k": result.get("k"),
        "architecture": arch,
        "touched_files": touched,
        "run_health": result.get("run_health"),
        "termination_reason": result.get("termination_reason"),
        "tokens_per_sec": result.get("tokens_per_sec"),
        "baseline_ratio": result.get("baseline_ratio"),
    }
    memory.setdefault("recent_rounds", []).append(entry)
    memory["recent_rounds"] = memory["recent_rounds"][-12:]

    families = memory.setdefault("architecture_families", {})
    family = families.setdefault(
        family_name,
        {
            "status": "incubating" if family_stage != "mainline" else "active",
            "goal": "new_family" if family_stage != "mainline" else "mainline",
            "design_hypothesis": action["hypothesis"],
            "tested_configs": [],
            "known_issues": [],
            "next_steps": [],
            "best_proxy_fvu": None,
            "best_full_fvu": None,
            "last_round": None,
        },
    )
    family["last_round"] = round_id
    family["design_hypothesis"] = action["hypothesis"]
    family.setdefault("tested_configs", []).append(
        {
            "round": round_id,
            "stage": family_stage,
            "k": result.get("k"),
            "decision": result.get("decision"),
            "val_fvu": result.get("val_fvu"),
            "run_health": result.get("run_health"),
        }
    )
    family["tested_configs"] = family["tested_configs"][-20:]

    try:
        val_fvu = float(result.get("val_fvu")) if result.get("val_fvu") not in (None, "") else None
    except (TypeError, ValueError):
        val_fvu = None
    if result.get("decision") == "promote":
        family["status"] = "active"
        if val_fvu is not None and (
            family.get("best_proxy_fvu") is None or val_fvu < float(family["best_proxy_fvu"])
        ):
            family["best_proxy_fvu"] = val_fvu
    elif result.get("decision") == "keep":
        family["status"] = "active" if family_stage == "mainline" else "promoted"
        if val_fvu is not None and (
            family.get("best_full_fvu") is None or val_fvu < float(family["best_full_fvu"])
        ):
            family["best_full_fvu"] = val_fvu
    elif result.get("decision") == "incubate":
        family["status"] = "incubating"
    elif result.get("decision") == "discard":
        family["status"] = "discarded"

    for note in action.get("notes_to_memory", []):
        memory.setdefault("recent_insights", []).append(note)
    outcome_note = (
        f"round {round_id}: {action['hypothesis']} -> {result.get('decision')} "
        f"(fvu={result.get('val_fvu')}, observed_fvu={result.get('observed_fvu')}, "
        f"k={result.get('k')}, health={result.get('run_health')}, "
        f"termination={result.get('termination_reason')})"
    )
    memory.setdefault("recent_insights", []).append(outcome_note)
    memory["recent_insights"] = memory["recent_insights"][-40:]

    if result.get("run_health") == "perf_regression":
        perf_note = (
            f"round {round_id}: suspected performance regression for {arch} "
            f"(tps={result.get('tokens_per_sec')}, baseline_ratio={result.get('baseline_ratio')}, "
            f"termination={result.get('termination_reason')})"
        )
        memory.setdefault("performance_findings", []).append(perf_note)
        memory["performance_findings"] = memory["performance_findings"][-20:]
        family.setdefault("known_issues", []).append(perf_note)
        family["known_issues"] = family["known_issues"][-20:]
    elif action["change_type"] == "edit_perf_code":
        memory.setdefault("performance_findings", []).append(outcome_note)
        memory["performance_findings"] = memory["performance_findings"][-20:]
    elif action["change_type"] != "param_only":
        memory.setdefault("architecture_findings", []).append(outcome_note)
        memory["architecture_findings"] = memory["architecture_findings"][-20:]

    if result.get("decision") == "crash" or result.get("run_health") in {"perf_regression", "crash"}:
        memory.setdefault("failure_patterns", []).append(
            {
                "pattern": action["hypothesis"],
                "count": 1,
                "last_round": round_id,
                "run_health": result.get("run_health"),
                "termination_reason": result.get("termination_reason"),
            }
        )
        memory["failure_patterns"] = memory["failure_patterns"][-20:]

    memory["next_hypotheses"] = action.get("next_hypotheses", [])[:12]
    family["next_steps"] = action.get("next_hypotheses", [])[:8]
    return memory


def write_round_summary(
    round_id: int,
    action: dict[str, Any],
    result: dict[str, str],
    touched: list[str],
    patch_path: Path | None,
    round_ctx: dict[str, Any],
) -> None:
    tier = "full" if round_ctx.get("full_result") is not None else action.get("experiment_tier")
    summary = {
        "round": round_id,
        "timestamp": int(time.time()),
        "started_at": round_ctx.get("started_at"),
        "ended_at": round_ctx.get("ended_at"),
        "duration_sec": round_ctx.get("duration_sec"),
        "family_name": round_ctx.get("family_name"),
        "family_stage": round_ctx.get("family_stage"),
        "session_id": round_ctx.get("session_id"),
        "timeline_start_event_id": round_ctx.get("timeline_start_event_id"),
        "timeline_end_event_id": round_ctx.get("timeline_end_event_id"),
        "timeline_event_ids": round_ctx.get("timeline_event_ids", []),
        "experiment_branch": current_git_branch(),
        "experiment_commit_message": commit_message_for_round(round_id, action, result, tier or "proxy"),
        "proxy_result": round_ctx.get("proxy_result"),
        "full_result": round_ctx.get("full_result"),
        "action": action,
        "result": result,
        "touched_files": touched,
        "patch_path": str(patch_path) if patch_path is not None else None,
    }
    save_json(ROUND_SUMMARIES_DIR / f"round_{round_id:04d}.json", summary)


def update_agent_state(
    state: dict[str, Any],
    result: dict[str, str],
    action: dict[str, Any],
    action_path: Path,
    patch_path: Path | None,
    is_new_family: bool,
) -> None:
    agent = state.setdefault("agent", {})
    agent["round_index"] = int(agent.get("round_index", 0)) + 1
    decision = result.get("decision", "")
    if decision == "crash":
        agent["consecutive_crashes"] = int(agent.get("consecutive_crashes", 0)) + 1
    else:
        agent["consecutive_crashes"] = 0
    if decision in {"promote", "keep"}:
        agent["consecutive_no_improve"] = 0
    else:
        agent["consecutive_no_improve"] = int(agent.get("consecutive_no_improve", 0)) + 1
    if is_new_family:
        agent["rounds_since_new_family"] = 0
    else:
        agent["rounds_since_new_family"] = int(agent.get("rounds_since_new_family", 0)) + 1
    agent["last_action_file"] = str(action_path)
    agent["last_patch_file"] = str(patch_path) if patch_path is not None else None


def mark_hints_applied(round_id: int) -> None:
    hints = load_operator_hints()
    changed = False
    for hint in hints:
        if hint.get("status") != "pending":
            continue
        if hint.get("scope") == "next_round":
            hint["status"] = "applied"
            hint["applied_at"] = int(time.time())
            hint["applied_in_round"] = round_id
            append_timeline_event(
                "hint_applied",
                round=round_id,
                tier=None,
                run_name=None,
                family_name=None,
                family_stage=None,
                status="applied",
                decision=None,
                payload=hint,
            )
            changed = True
    if changed:
        save_operator_hints(hints)


def _run_sanity_for_round(
    action: dict[str, Any],
    tier: str,
    round_id: int,
    round_ctx: dict[str, Any],
) -> None:
    """Run sanity check and log events. Raises on failure."""
    # Build a minimal config_json for sanity from action env_overrides
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
    sanity_config = {
        "architecture": config.get("ARCHITECTURE", "topk").lower(),
        "k": int(config.get("K", "128")),
        "expansion_factor": int(config.get("EXPANSION_FACTOR", "8")),
    }
    print(f"Round {round_id}: running sanity check before {tier}")
    log_round_event(round_ctx, "sanity_started", tier=tier)
    run_sanity(sanity_config)
    print(f"Round {round_id}: sanity check passed")
    log_round_event(round_ctx, "sanity_finished", tier=tier, status="ok")


def run_training_round(
    action: dict[str, Any],
    tier: str,
    args: argparse.Namespace,
    round_id: int,
    memory: dict[str, Any],
    round_ctx: dict[str, Any],
) -> dict[str, str]:
    run_stamp = int(time.time())
    run_name = f"round{round_id:04d}_{tier}_{run_stamp}"
    save_dir = SAVE_ROOT / f"round_{round_id:04d}_{tier}"
    save_dir.mkdir(parents=True, exist_ok=True)
    env, config_json = build_env(action, tier, run_name, save_dir, args)
    baseline_key = runtime_baseline_key(config_json, tier)
    baseline_tps = memory.get("baseline_runtime", {}).get(baseline_key, {}).get("tokens_per_sec")
    config_path = LOG_DIR / f"{run_name}.config.json"
    save_json(config_path, config_json)
    log_path = LOG_DIR / f"{run_name}.log"

    print(
        f"Round {round_id}: starting {tier} training | "
        f"arch={config_json['architecture']} k={config_json['k']} "
        f"ef={config_json['expansion_factor']} run_name={run_name}"
    )
    print(f"Round {round_id}: training log -> {log_path}")
    if baseline_tps is not None:
        print(f"Round {round_id}: runtime baseline -> {baseline_tps:.2f} tokens/s")
    log_round_event(
        round_ctx,
        "training_started",
        tier=tier,
        run_name=run_name,
        family_name=config_json.get("family_name"),
        family_stage=config_json.get("family_stage"),
        checkpoint_dir=str(save_dir),
        log_path=str(log_path),
        config=config_json,
        baseline_tokens_per_sec=baseline_tps,
    )
    write_status(
        "training_started",
        round=round_id,
        tier=tier,
        run_name=run_name,
        log_path=str(log_path),
        config=config_json,
        baseline_tokens_per_sec=baseline_tps,
    )

    timeout_sec = args.full_timeout_sec if tier == "full" else args.proxy_timeout_sec
    start = time.time()
    termination_reason = "completed"
    latest_tps: float | None = None
    baseline_ratio: float | None = None
    last_step_record: dict[str, Any] | None = None
    first_step_seen = False
    slow_trigger_count = 0
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            ["/bin/bash", str(SCRIPT_PATH)],
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        last_progress = start
        while process.poll() is None:
            time.sleep(args.poll_interval_sec)
            checkpoint_dir = latest_checkpoint_dir(save_dir, run_name)
            mtime = metrics_last_update(checkpoint_dir)
            if mtime is not None:
                last_progress = max(last_progress, mtime)
            last_step_record = read_latest_step_record(checkpoint_dir)
            if last_step_record is not None:
                first_step_seen = True
                step = int(last_step_record.get("step", 0))
                total_tokens = float(last_step_record.get("total_tokens", 0))
                elapsed = max(time.time() - start, 1e-6)
                latest_tps = total_tokens / elapsed if total_tokens > 0 else None
                if latest_tps is not None and baseline_tps is not None and baseline_tps > 0:
                    baseline_ratio = latest_tps / baseline_tps
                write_status(
                    "training_running",
                    round=round_id,
                    tier=tier,
                    run_name=run_name,
                    checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
                    metrics_path=str(checkpoint_dir / "metrics.jsonl") if checkpoint_dir is not None else None,
                    latest_step=step,
                    latest_tokens_per_sec=latest_tps,
                    baseline_ratio=baseline_ratio,
                )
                log_round_event(
                    round_ctx,
                    "training_heartbeat",
                    tier=tier,
                    run_name=run_name,
                    family_name=config_json.get("family_name"),
                    family_stage=config_json.get("family_stage"),
                    checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
                    metrics_path=str(checkpoint_dir / "metrics.jsonl") if checkpoint_dir is not None else None,
                    latest_step=step,
                    latest_tokens_per_sec=latest_tps,
                    baseline_ratio=baseline_ratio,
                    latest_total_tokens=total_tokens,
                )
                if time.time() - start >= args.slow_run_grace_sec:
                    if latest_tps is not None:
                        print(
                            f"Round {round_id}: throughput watchdog active | "
                            f"step={step} tokens_per_sec={latest_tps:.2f}"
                        )
                    else:
                        print(f"Round {round_id}: throughput watchdog active | step={step}")
                if (
                    time.time() - start >= args.slow_run_grace_sec
                    and step >= args.min_progress_steps
                    and latest_tps is not None
                    and baseline_tps is not None
                    and baseline_tps > 0
                    and latest_tps < baseline_tps * args.min_tokens_per_sec_ratio
                ):
                    slow_trigger_count += 1
                    if slow_trigger_count >= 2:
                        termination_reason = "throughput_too_low"
                        print(
                            f"Round {round_id}: early stopping for low throughput | "
                            f"{latest_tps:.2f} tokens/s vs baseline {baseline_tps:.2f} "
                            f"({latest_tps / baseline_tps:.2%})"
                        )
                        terminate_process_group(process)
                        break
                else:
                    slow_trigger_count = 0
            elif not first_step_seen and time.time() - start > args.first_step_timeout_sec:
                termination_reason = "first_step_timeout"
                print(
                    f"Round {round_id}: early stopping before first step "
                    f"after {args.first_step_timeout_sec}s"
                )
                terminate_process_group(process)
                break
            elif not first_step_seen:
                log_round_event(
                    round_ctx,
                    "training_heartbeat",
                    tier=tier,
                    run_name=run_name,
                    family_name=config_json.get("family_name"),
                    family_stage=config_json.get("family_stage"),
                    checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
                    metrics_path=str(checkpoint_dir / "metrics.jsonl") if checkpoint_dir is not None else None,
                    latest_step=None,
                    latest_tokens_per_sec=None,
                    baseline_ratio=None,
                    latest_total_tokens=None,
                )
            if time.time() - start > timeout_sec:
                termination_reason = "hard_timeout"
                print(f"Round {round_id}: hard timeout after {timeout_sec}s")
                terminate_process_group(process)
                break
            if time.time() - last_progress > args.stall_timeout_sec:
                termination_reason = "stall_timeout"
                print(
                    f"Round {round_id}: stall timeout after "
                    f"{args.stall_timeout_sec}s without metrics update"
                )
                terminate_process_group(process)
                break
    process.wait()
    checkpoint_dir = latest_checkpoint_dir(save_dir, run_name)
    last_step_record = read_latest_step_record(checkpoint_dir) or last_step_record
    observed_fvu = extract_step_fvu(last_step_record)
    print(
        f"Round {round_id}: {tier} training finished | "
        f"checkpoint={checkpoint_dir if checkpoint_dir is not None else 'none'}"
    )
    result = record_result(
        log_path=log_path,
        checkpoint_dir=checkpoint_dir,
        config_path=config_path,
        tier=tier,
        description=action["summary"] if tier == "proxy" else f"{action['summary']} full",
        self_review=action["self_review"],
    )
    run_health = "normal"
    if termination_reason == "throughput_too_low":
        run_health = "perf_regression"
    elif termination_reason != "completed":
        run_health = "crash"
    result["termination_reason"] = termination_reason
    result["run_health"] = run_health
    result["tokens_per_sec"] = f"{latest_tps:.6f}" if latest_tps is not None else ""
    result["baseline_tokens_per_sec"] = f"{baseline_tps:.6f}" if baseline_tps is not None else ""
    result["baseline_ratio"] = f"{baseline_ratio:.6f}" if baseline_ratio is not None else ""
    result["observed_fvu"] = f"{observed_fvu:.6f}" if observed_fvu is not None else ""
    result["metrics_path"] = (
        str(checkpoint_dir / "metrics.jsonl")
        if checkpoint_dir is not None and (checkpoint_dir / "metrics.jsonl").exists()
        else ""
    )
    write_status(
        "training_finished",
        round=round_id,
        tier=tier,
        run_name=run_name,
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
        termination_reason=termination_reason,
        run_health=run_health,
        tokens_per_sec=latest_tps,
        baseline_ratio=baseline_ratio,
        decision=result.get("decision"),
    )
    log_round_event(
        round_ctx,
        "training_finished",
        tier=tier,
        run_name=run_name,
        family_name=config_json.get("family_name"),
        family_stage=config_json.get("family_stage"),
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
        metrics_path=result.get("metrics_path"),
        run_health=run_health,
        termination_reason=termination_reason,
        tokens_per_sec=latest_tps,
        baseline_ratio=baseline_ratio,
        status=result.get("status"),
        decision=result.get("decision"),
    )
    log_round_event(
        round_ctx,
        "result_recorded",
        tier=tier,
        run_name=run_name,
        family_name=config_json.get("family_name"),
        family_stage=config_json.get("family_stage"),
        status=result.get("status"),
        decision=result.get("decision"),
        val_fvu=result.get("val_fvu"),
        observed_fvu=result.get("observed_fvu"),
        k=result.get("k"),
        log_path=result.get("log_path"),
        checkpoint=result.get("checkpoint"),
    )
    if run_health == "normal" and latest_tps is not None:
        baseline_entry = {
            "tokens_per_sec": latest_tps,
            "round": round_id,
            "tier": tier,
            "architecture": config_json["architecture"],
            "k": config_json["k"],
            "updated_at": int(time.time()),
        }
        current = memory.setdefault("baseline_runtime", {}).get(baseline_key)
        if current is None or latest_tps > float(current.get("tokens_per_sec", 0)):
            memory.setdefault("baseline_runtime", {})[baseline_key] = baseline_entry
    return result


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
    parser.add_argument("--max-consecutive-no-improve", type=int, default=8)
    parser.add_argument("--session-mode", choices=["resume-session", "fresh-each-round"], default="resume-session")
    parser.add_argument("--max-session-rounds", type=int, default=DEFAULT_MAX_SESSION_ROUNDS)
    parser.add_argument("--max-session-hours", type=float, default=DEFAULT_MAX_SESSION_HOURS)
    parser.add_argument("--research-branch", default=None)
    parser.add_argument("--no-commit-experiments", action="store_true")
    parser.add_argument("--allow-direct-full", action="store_true")
    parser.add_argument("--reset-failure-counters", action="store_true")
    parser.add_argument("--agent-max-retries", type=int, default=DEFAULT_AGENT_MAX_RETRIES)
    parser.add_argument("--agent-retry-base-sec", type=int, default=DEFAULT_AGENT_RETRY_BASE_SEC)
    parser.add_argument("--max-session-failures", type=int, default=DEFAULT_MAX_SESSION_FAILURES)
    args = parser.parse_args()

    auto_commit_enabled = not args.no_commit_experiments

    # Clean-worktree check must run BEFORE ensure_setup(), which creates/modifies
    # git-tracked history files (state.json, frontier.json, timeline.jsonl, etc.).
    if auto_commit_enabled:
        ensure_clean_worktree_for_auto_commit()

    ensure_setup()
    start_time = time.time()
    append_timeline_event(
        "loop_started",
        round=None,
        tier=None,
        run_name=None,
        family_name=None,
        family_stage=None,
        status=None,
        decision=None,
        payload={"rounds": args.rounds, "budget_hours": args.budget_hours},
    )
    write_status("loop_starting", rounds=args.rounds, budget_hours=args.budget_hours)
    print("Starting SAE agent loop")
    print(f"round_budget: {args.rounds}")
    print(f"time_budget_hours: {args.budget_hours}")
    print(f"agent_proxy: {args.agent_proxy}")
    print(f"session_mode: {args.session_mode}")
    print(f"auto_commit: {auto_commit_enabled}")
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

    # --- Preflight: verify agent backend is reachable ---
    try:
        check_agent_backend_reachable(args.agent_proxy)
        print("Agent backend preflight: OK")
    except RuntimeError as exc:
        print(f"Agent backend preflight FAILED: {exc}")
        append_timeline_event(
            "backend_unavailable",
            round=None,
            tier=None,
            run_name=None,
            family_name=None,
            family_stage=None,
            status="failed",
            decision=None,
            payload={"error": str(exc)},
        )
        return 1

    if auto_commit_enabled:
        state = load_state()
        research_branch = args.research_branch or default_research_branch_name()
        ensure_research_branch(research_branch, state)
        save_state(state)
        append_timeline_event(
            "research_branch_ready",
            round=None,
            tier=None,
            run_name=None,
            family_name=None,
            family_stage=None,
            status="ready",
            decision=None,
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
        if agent_state["consecutive_crashes"] >= args.max_consecutive_crashes:
            print(
                "Stopping: consecutive crash limit reached "
                f"({agent_state['consecutive_crashes']} >= {args.max_consecutive_crashes})"
            )
            break
        if agent_state["consecutive_no_improve"] >= args.max_consecutive_no_improve:
            print(
                "Stopping: consecutive no-improve limit reached "
                f"({agent_state['consecutive_no_improve']} >= {args.max_consecutive_no_improve})"
            )
            break

        round_id = int(agent_state["round_index"]) + 1
        round_ctx = {
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

        # --- Session lifecycle ---
        effective_session_mode = args.session_mode
        if session_failure_count >= args.max_session_failures and effective_session_mode == "resume-session":
            effective_session_mode = "fresh-each-round"
            print(
                f"Round {round_id}: degraded to fresh-each-round after "
                f"{session_failure_count} session failures"
            )
            append_timeline_event(
                "session_fallback_to_fresh",
                round=round_id,
                tier=None,
                run_name=None,
                family_name=None,
                family_stage=None,
                status="degraded",
                decision=None,
                payload={"session_failure_count": session_failure_count},
            )

        session_id = None if effective_session_mode == "fresh-each-round" else agent_state.get("active_session_id")
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

        # --- Snapshot code before agent edits ---
        before = snapshot_paths()
        if before:
            capture_before_files(list(before.keys()), round_id)
        # Record restore point for code rollback.  Works both with and without
        # auto-commit: git HEAD is always available in a git repo.
        pre_edit_commit = current_git_commit()

        # --- Agent invocation with retry ---
        try:
            action, stdout_path, returned_session_id, resumed = run_agent_round_with_retry(
                state, memory, recent_results, brief,
                round_id, args, round_ctx,
                session_id if not need_new_session else None,
                need_new_session,
            )
            session_failure_count = 0
        except Exception as exc:
            session_failure_count += 1
            error_path = LOG_DIR / f"agent_round_{round_id:04d}.error.log"
            error_path.write_text(str(exc) + "\n")
            print(f"Round {round_id}: agent invocation exhausted all retries: {exc}")
            log_round_event(
                round_ctx,
                "round_finished",
                status="crash",
                decision="crash",
                error=str(exc),
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
            if auto_commit_enabled:
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
                        "experiment_commit_message": commit_message_for_round(round_id, crash_action, crash_result, "proxy"),
                        "error": str(exc),
                    },
                )
                commit_round_state(
                    round_id, crash_action, crash_result, [], round_summary_path, "proxy",
                )
            # Continue to next round instead of breaking the entire loop
            continue

        state = load_state()
        agent = state.setdefault("agent", {})
        if effective_session_mode == "resume-session":
            if not returned_session_id:
                print(f"Round {round_id}: WARNING: Codex did not return a session id, falling back to fresh")
                session_failure_count += 1
            else:
                if resumed:
                    agent["last_resume_ok_at"] = int(time.time())
                    append_timeline_event(
                        "session_resumed",
                        round=round_id,
                        tier=None,
                        run_name=None,
                        family_name=None,
                        family_stage=None,
                        status="active",
                        decision=None,
                        payload={"session_id": returned_session_id},
                    )
                else:
                    agent["active_session_started_at"] = int(time.time())
                    agent["active_session_rounds"] = 0
                    append_timeline_event(
                        "session_started",
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

        if action.get("command") == "stop":
            print(f"Round {round_id}: agent requested stop: {action.get('summary', '')}")
            log_round_event(
                round_ctx,
                "round_finished",
                status="stop",
                decision="stop",
                summary=action.get("summary", ""),
            )
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
            round_ctx,
            "agent_action_received",
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
            print(
                f"Round {round_id}: family -> {action.get('family_name')} "
                f"stage={action.get('family_stage', 'unspecified')}"
            )
        print(f"Round {round_id}: action log -> {stdout_path}")

        # --- Detect code changes ---
        touched: list[str] = []
        patch_path: Path | None = None
        is_code_edit = action.get("change_type") not in ("param_only", "no_change")
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

        # --- Validate env_overrides ---
        try:
            overrides = action.get("env_overrides", [])
            rejected_keys = validate_env_overrides(overrides)
            if rejected_keys:
                raise RuntimeError(
                    f"env_overrides contain disallowed keys: {', '.join(rejected_keys)}"
                )
        except RuntimeError as exc:
            print(f"Round {round_id}: env_overrides rejected: {exc}")
            log_round_event(
                round_ctx,
                "env_overrides_rejected",
                rejected_keys=rejected_keys,
                error=str(exc),
            )
            result = {"decision": "crash", "status": "crash"}
            result["termination_reason"] = "invalid_env_overrides"
            result["run_health"] = "crash"
            if is_code_edit and touched and pre_edit_commit:
                rollback_code_changes(pre_edit_commit, round_id, "invalid_env_overrides")
            round_ctx["ended_at"] = int(time.time())
            round_ctx["duration_sec"] = round_ctx["ended_at"] - int(round_ctx["started_at"])
            log_round_event(round_ctx, "round_finished", status="crash", decision="crash")
            write_round_summary(round_id, action, result, touched, patch_path, round_ctx)
            round_summary_path = ROUND_SUMMARIES_DIR / f"round_{round_id:04d}.json"
            existing_families = set(memory.get("architecture_families", {}).keys())
            memory = append_memory(memory, action, result, round_id, touched)
            save_json(MEMORY_PATH, memory)
            state = load_state()
            family_name = str(action.get("family_name") or BASE_ENV_DEFAULTS["ARCHITECTURE"]).lower()
            is_new_family = family_name not in existing_families
            update_agent_state(state, result, action, stdout_path, patch_path, is_new_family)
            save_state(state)
            if auto_commit_enabled:
                commit_round_state(round_id, action, result, touched, round_summary_path, "proxy")
            continue

        # --- Budget check before training ---
        remaining_sec = budget_remaining_sec(start_time, args.budget_hours)
        tier = action["experiment_tier"]
        tier_timeout = args.full_timeout_sec if tier == "full" else args.proxy_timeout_sec
        if remaining_sec < tier_timeout * 0.5:
            if tier == "full":
                # Not enough budget for full — downgrade to proxy
                print(
                    f"Round {round_id}: insufficient budget for full run "
                    f"({remaining_sec:.0f}s remaining vs {tier_timeout}s timeout), "
                    f"downgrading to proxy"
                )
                action["experiment_tier"] = "proxy"
                tier = "proxy"
                tier_timeout = args.proxy_timeout_sec
            if remaining_sec < args.proxy_timeout_sec * 0.5:
                print(
                    f"Round {round_id}: insufficient budget even for proxy "
                    f"({remaining_sec:.0f}s remaining), stopping"
                )
                if is_code_edit and touched and pre_edit_commit:
                    rollback_code_changes(pre_edit_commit, round_id, "budget_exhausted_before_training")
                break

        # --- Sanity check for code edits (with rollback on failure) ---
        sanity_failed = False
        if action.get("needs_sanity") and is_code_edit:
            try:
                run_sanity_result = _run_sanity_for_round(action, tier, round_id, round_ctx)
            except Exception as sanity_exc:
                sanity_failed = True
                print(f"Round {round_id}: sanity check FAILED: {sanity_exc}")
                log_round_event(
                    round_ctx,
                    "sanity_failed",
                    tier=tier,
                    error=str(sanity_exc),
                )
                if pre_edit_commit:
                    rollback_code_changes(pre_edit_commit, round_id, f"sanity_failed: {sanity_exc}")
                result = {"decision": "crash", "status": "crash"}
                result["termination_reason"] = "sanity_failed"
                result["run_health"] = "crash"
                round_ctx["ended_at"] = int(time.time())
                round_ctx["duration_sec"] = round_ctx["ended_at"] - int(round_ctx["started_at"])
                log_round_event(round_ctx, "round_finished", status="crash", decision="crash")
                write_round_summary(round_id, action, result, touched, patch_path, round_ctx)
                round_summary_path = ROUND_SUMMARIES_DIR / f"round_{round_id:04d}.json"
                existing_families = set(memory.get("architecture_families", {}).keys())
                memory = append_memory(memory, action, result, round_id, touched)
                save_json(MEMORY_PATH, memory)
                state = load_state()
                family_name = str(action.get("family_name") or BASE_ENV_DEFAULTS["ARCHITECTURE"]).lower()
                is_new_family = family_name not in existing_families
                update_agent_state(state, result, action, stdout_path, patch_path, is_new_family)
                save_state(state)
                if auto_commit_enabled:
                    commit_round_state(round_id, action, result, touched, round_summary_path, "proxy")
                continue

        if sanity_failed:
            continue

        result = run_training_round(action, action["experiment_tier"], args, round_id, memory, round_ctx)
        round_ctx["proxy_result"] = result
        print(
            f"Round {round_id} proxy result: "
            f"decision={result.get('decision')} fvu={result.get('val_fvu')} "
            f"k={result.get('k')} health={result.get('run_health')} "
            f"termination={result.get('termination_reason')}"
        )

        # --- Code rollback on crash (not perf_regression) ---
        if (
            is_code_edit
            and touched
            and pre_edit_commit
            and result.get("run_health") == "crash"
        ):
            rollback_code_changes(pre_edit_commit, round_id, f"training_crash: {result.get('termination_reason')}")

        if result.get("decision") == "promote" and action["experiment_tier"] == "proxy":
            # Budget check before full promotion
            remaining_sec = budget_remaining_sec(start_time, args.budget_hours)
            if remaining_sec < args.full_timeout_sec * 0.5:
                print(
                    f"Round {round_id}: skipping full promotion, insufficient budget "
                    f"({remaining_sec:.0f}s remaining)"
                )
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
                # Rollback code on full crash too
                if (
                    is_code_edit
                    and touched
                    and pre_edit_commit
                    and full_result.get("run_health") == "crash"
                ):
                    rollback_code_changes(
                        pre_edit_commit, round_id,
                        f"full_training_crash: {full_result.get('termination_reason')}",
                    )

        round_ctx["ended_at"] = int(time.time())
        round_ctx["duration_sec"] = round_ctx["ended_at"] - int(round_ctx["started_at"])
        log_round_event(
            round_ctx,
            "round_finished",
            tier="full" if round_ctx.get("full_result") is not None else action.get("experiment_tier"),
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
        mark_hints_applied(round_id)

        state = load_state()
        family_name = str(action.get("family_name") or BASE_ENV_DEFAULTS["ARCHITECTURE"]).lower()
        is_new_family = family_name not in existing_families
        update_agent_state(state, result, action, stdout_path, patch_path, is_new_family)
        if effective_session_mode == "resume-session" and state["agent"].get("active_session_status") == "active":
            state["agent"]["active_session_rounds"] = int(state["agent"].get("active_session_rounds", 0)) + 1
        save_state(state)
        save_session_brief(build_session_brief(state, memory, load_results(limit=8), round_id))
        if auto_commit_enabled:
            append_timeline_event(
                "experiment_commit_prepared",
                round=round_id,
                tier="full" if round_ctx.get("full_result") is not None else action.get("experiment_tier"),
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
                "full" if round_ctx.get("full_result") is not None else action.get("experiment_tier", "proxy"),
            )
            write_status(
                "experiment_committed",
                round=round_id,
                commit=commit_hash,
                branch=branch_name,
            )
        print(
            f"Round {round_id}: result recorded | "
            f"decision={result.get('decision')} val_fvu={result.get('val_fvu')} "
            f"log={result.get('log_path', '')}"
        )

    write_status("idle")
    state = load_state()
    if args.session_mode == "resume-session" and state.get("agent", {}).get("active_session_id"):
        state["agent"]["active_session_status"] = "closed"
        save_state(state)
        append_timeline_event(
            "session_closed",
            round=None,
            tier=None,
            run_name=None,
            family_name=None,
            family_stage=None,
            status="closed",
            decision=None,
            payload={"session_id": state["agent"].get("active_session_id")},
        )
    append_timeline_event(
        "loop_finished",
        round=None,
        tier=None,
        run_name=None,
        family_name=None,
        family_stage=None,
        status="finished",
        decision=None,
        payload=None,
    )
    if auto_commit_enabled and worktree_dirty():
        stage_paths(list(TRACKED_HISTORY_PATHS))
        if git(["diff", "--cached", "--quiet"], check=False).returncode != 0:
            git(["commit", "-m", "experiment-session: close nightly session"])
    print("Agent loop finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
