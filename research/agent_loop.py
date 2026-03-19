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
SNAPSHOT_ROOTS = ("sparsify", "research", "scripts")
SNAPSHOT_EXCLUDES = (
    "research/history/",
    "research/agent_loop.py",
    "research/controller.py",
    "research/orchestrator.py",
    "research/prepare.py",
    "research/program.md",
    "research/train.py",
    "research/agent_action.schema.json",
    "scripts/autoresearch_test.sh",
)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=check, text=True, capture_output=True)


def ensure_setup() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ROUND_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    run(["python", str(CONTROLLER_PATH), "init"], cwd=RESEARCH_DIR)


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
            "last_action_file": None,
            "last_patch_file": None,
        },
    )
    agent_state.setdefault("round_index", 0)
    agent_state.setdefault("consecutive_crashes", 0)
    agent_state.setdefault("consecutive_no_improve", 0)
    agent_state.setdefault("last_action_file", None)
    agent_state.setdefault("last_patch_file", None)
    return state


def save_state(state: dict[str, Any]) -> None:
    save_json(STATE_PATH, state)
    save_json(FRONTIER_PATH, state.get("frontier", {}))


def load_memory() -> dict[str, Any]:
    return load_json(
        MEMORY_PATH,
        {
            "current_focus": "Establish a stable low-K SAE frontier with the cheapest possible runs.",
            "architecture_findings": [],
            "performance_findings": [],
            "failure_patterns": [],
            "baseline_runtime": {},
            "recent_rounds": [],
            "recent_insights": [],
            "next_hypotheses": [],
        },
    )


def write_status(stage: str, **payload: Any) -> None:
    save_json(
        STATUS_PATH,
        {
            "timestamp": int(time.time()),
            "stage": stage,
            **payload,
        },
    )


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
            after_text = source.read_text()
            after_path.write_text(after_text)
        else:
            after_text = ""
        if before_path.exists():
            before_text = before_path.read_text()
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
    context = {
        "frontier": frontier,
        "agent_state": state.get("agent", {}),
        "current_focus": memory.get("current_focus"),
        "recent_insights": memory.get("recent_insights", [])[-8:],
        "recent_performance_findings": memory.get("performance_findings", [])[-8:],
        "baseline_runtime": memory.get("baseline_runtime", {}),
        "next_hypotheses": memory.get("next_hypotheses", [])[:8],
        "recent_results": summarize_results(results),
        "recent_round_summaries": recent_round_summaries(),
    }
    return f"""
You are the nightly SAE research agent for this repository.

Primary objective:
- reduce FVU
- then reduce K at similar FVU
- then reduce cost / improve throughput / memory

Execution layer is fixed:
- training entrypoint: scripts/autoresearch_test.sh
- results recorder: research/controller.py
- memory files: research/history/state.json, frontier.json, memory.json, results.tsv

Important rules:
- Read research/program.md before deciding.
- You may edit ONLY files under sparsify/.
- Do not edit research/history/*, research/*.py, or scripts/autoresearch_test.sh.
- For parameter-only experiments, use env_overrides instead of editing launch code.
- Make at most ONE coherent hypothesis this round.
- Prefer proxy unless there is a strong reason to go straight to full.
- Direct full requests may be coerced back to proxy by the runtime.
- Slow runs may indicate implementation bottlenecks rather than bad architectures.
- If a recent run was a performance regression, prefer an edit_perf_code follow-up over drawing a negative architecture conclusion.
- If there is no promising next move, return command="stop".
- Return a final JSON object matching the schema exactly.

Current structured context:
{json.dumps(context, indent=2)}
""".strip()


def run_agent_round(
    prompt: str,
    round_id: int,
    model: str | None,
    agent_proxy: str | None,
) -> tuple[dict[str, Any], Path]:
    action_path = LOG_DIR / f"agent_action_{round_id:04d}.json"
    stdout_path = LOG_DIR / f"agent_round_{round_id:04d}.stdout.log"
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
    action = load_json(action_path, {})
    return action, stdout_path


def build_env(action: dict[str, Any], tier: str, run_name: str, save_dir: Path, args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    env = os.environ.copy()
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

    entry = {
        "round": round_id,
        "hypothesis": action["hypothesis"],
        "change_type": action["change_type"],
        "expected_win": action["expected_win"],
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
    return memory


def write_round_summary(round_id: int, action: dict[str, Any], result: dict[str, str], touched: list[str], patch_path: Path | None) -> None:
    summary = {
        "round": round_id,
        "timestamp": int(time.time()),
        "action": action,
        "result": result,
        "touched_files": touched,
        "patch_path": str(patch_path) if patch_path is not None else None,
    }
    save_json(ROUND_SUMMARIES_DIR / f"round_{round_id:04d}.json", summary)


def update_agent_state(state: dict[str, Any], result: dict[str, str], action_path: Path, patch_path: Path | None) -> None:
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
    agent["last_action_file"] = str(action_path)
    agent["last_patch_file"] = str(patch_path) if patch_path is not None else None


def run_training_round(
    action: dict[str, Any],
    tier: str,
    args: argparse.Namespace,
    round_id: int,
    memory: dict[str, Any],
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
    write_status(
        "training_started",
        round=round_id,
        tier=tier,
        run_name=run_name,
        log_path=str(log_path),
        config=config_json,
        baseline_tokens_per_sec=baseline_tps,
    )

    if action.get("needs_sanity"):
        print(f"Round {round_id}: running sanity check before {tier}")
        run_sanity(config_json)
        print(f"Round {round_id}: sanity check passed")

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
                        process.kill()
                        break
                else:
                    slow_trigger_count = 0
            elif not first_step_seen and time.time() - start > args.first_step_timeout_sec:
                termination_reason = "first_step_timeout"
                print(
                    f"Round {round_id}: early stopping before first step "
                    f"after {args.first_step_timeout_sec}s"
                )
                process.kill()
                break
            if time.time() - start > timeout_sec:
                termination_reason = "hard_timeout"
                print(f"Round {round_id}: hard timeout after {timeout_sec}s")
                process.kill()
                break
            if time.time() - last_progress > args.stall_timeout_sec:
                termination_reason = "stall_timeout"
                print(
                    f"Round {round_id}: stall timeout after "
                    f"{args.stall_timeout_sec}s without metrics update"
                )
                process.kill()
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
    parser.add_argument("--allow-direct-full", action="store_true")
    parser.add_argument("--reset-failure-counters", action="store_true")
    args = parser.parse_args()

    ensure_setup()
    start_time = time.time()
    write_status("loop_starting", rounds=args.rounds, budget_hours=args.budget_hours)
    print("Starting SAE agent loop")
    print(f"round_budget: {args.rounds}")
    print(f"time_budget_hours: {args.budget_hours}")
    print(f"agent_proxy: {args.agent_proxy}")
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
        print(f"Starting round {round_id}")
        write_status("agent_deciding", round=round_id)
        recent_results = load_results(limit=8)
        prompt = build_prompt(state, memory, recent_results)

        before = snapshot_paths()
        capture_before_files(list(before.keys()), round_id)
        try:
            action, stdout_path = run_agent_round(
                prompt,
                round_id,
                args.model,
                args.agent_proxy,
            )
        except Exception as exc:
            error_path = LOG_DIR / f"agent_round_{round_id:04d}.error.log"
            error_path.write_text(str(exc) + "\n")
            print(f"Round {round_id} failed during agent invocation: {exc}")
            memory.setdefault("recent_insights", []).append(
                f"round {round_id}: agent invocation failed: {exc}"
            )
            memory["recent_insights"] = memory["recent_insights"][-40:]
            save_json(MEMORY_PATH, memory)
            state = load_state()
            agent = state.setdefault("agent", {})
            agent["round_index"] = round_id
            agent["consecutive_crashes"] = int(agent.get("consecutive_crashes", 0)) + 1
            agent["last_action_file"] = str(error_path)
            save_state(state)
            break

        if action.get("command") == "stop":
            print(f"Round {round_id}: agent requested stop: {action.get('summary', '')}")
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
        print(f"Round {round_id}: hypothesis -> {action.get('hypothesis')}")
        print(f"Round {round_id}: action log -> {stdout_path}")

        after = snapshot_paths()
        touched = touched_files(before, after)
        if touched:
            assert_allowed_changes(touched)
            print(f"Round {round_id}: touched files -> {', '.join(touched)}")
        else:
            print(f"Round {round_id}: no code files changed")

        patch_path: Path | None = None
        if touched:
            patch_text = build_patch(before, touched, round_id)
            patch_path = LOG_DIR / f"round_{round_id:04d}.patch"
            patch_path.write_text(patch_text)
            print(f"Round {round_id}: patch saved -> {patch_path}")

        result = run_training_round(action, action["experiment_tier"], args, round_id, memory)
        print(
            f"Round {round_id} proxy result: "
            f"decision={result.get('decision')} fvu={result.get('val_fvu')} "
            f"k={result.get('k')} health={result.get('run_health')} "
            f"termination={result.get('termination_reason')}"
        )
        if result.get("decision") == "promote" and action["experiment_tier"] == "proxy":
            print(f"Round {round_id}: promoted to full")
            full_result = run_training_round(action, "full", args, round_id, memory)
            result = full_result
            print(
                f"Round {round_id} full result: "
                f"decision={result.get('decision')} fvu={result.get('val_fvu')} "
                f"k={result.get('k')} health={result.get('run_health')} "
                f"termination={result.get('termination_reason')}"
            )

        write_round_summary(round_id, action, result, touched, patch_path)
        memory = append_memory(memory, action, result, round_id, touched)
        save_json(MEMORY_PATH, memory)

        state = load_state()
        update_agent_state(state, result, stdout_path, patch_path)
        save_state(state)
        print(
            f"Round {round_id}: result recorded | "
            f"decision={result.get('decision')} val_fvu={result.get('val_fvu')} "
            f"log={result.get('log_path', '')}"
        )

    write_status("idle")
    print("Agent loop finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
