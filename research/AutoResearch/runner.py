"""Training execution and monitoring for the autoresearch framework."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .git_ops import REPO_ROOT
from .override_registry import config_from_overrides, validate_env_overrides
from .controller import decide, parse_log, update_frontier
from .types import (
    Action,
    BASE_ENV_DEFAULTS,
    DEFAULT_PROCESS_TERM_TIMEOUT_SEC,
    LOG_DIR,
    SAVE_ROOT,
    LoopConfig,
    Result,
    RoundContext,
)

SCRIPT_PATH = REPO_ROOT / "scripts" / "autoresearch_test.sh"


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


class SanityCheckError(RuntimeError):
    """Structured sanity-check failure with captured subprocess output."""

    def __init__(self, cmd: list[str], returncode: int, stdout: str, stderr: str) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(f"Sanity check failed (exit {returncode})")

    def to_payload(self) -> dict[str, Any]:
        stderr_lines = [l for l in self.stderr.splitlines() if l.strip()]
        stdout_lines = [l for l in self.stdout.splitlines() if l.strip()]
        tb_lines = [
            l for l in stderr_lines
            if l.startswith("Traceback") or l.startswith("  File ")
            or l.endswith("Error") or l.endswith("Exception")
        ]
        return {
            "error_type": "sanity_check_failed",
            "returncode": self.returncode,
            "cmd": self.cmd,
            "stdout_excerpt": "\n".join(stdout_lines[-20:]),
            "stderr_excerpt": "\n".join(stderr_lines[-40:]),
            "traceback_excerpt": "\n".join(tb_lines[-40:]),
        }


def run_sanity(architecture: str, k: int, ef: int) -> None:
    """Run a fast forward+backward pass to validate the architecture."""
    script = "\n".join([
        "import sys; sys.path.insert(0, '.')",
        "from sparsify import SparseCoder",
        "from sparsify.config import SparseCoderConfig",
        "import torch",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "try:",
        "    import torch_npu",
        "    device = 'npu' if torch.npu.is_available() else device",
        "except ImportError: pass",
        f"cfg = SparseCoderConfig(architecture={architecture!r}, k={k}, expansion_factor={ef})",
        "sae = SparseCoder(1024, cfg, device=device, dtype=torch.float32)",
        "x = torch.randn(4, 1024, device=device)",
        "out = sae(x); out.fvu.backward()",
        "print('sanity: OK')",
    ])
    cmd = ["python", "-c", script]
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise SanityCheckError(cmd, exc.returncode, exc.stdout or "", exc.stderr or "") from exc


# ---------------------------------------------------------------------------
# Training execution
# ---------------------------------------------------------------------------


def run_training(
    action: Action,
    config: LoopConfig,
    round_id: int,
    state: Any,  # StateManager (avoid circular import)
    ctx: RoundContext,
) -> Result:
    """Execute training subprocess with watchdog monitoring."""
    run_stamp = int(time.time())
    run_name = f"round{round_id:04d}_standard_{run_stamp}"
    save_dir = SAVE_ROOT / f"round_{round_id:04d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    env, config_json = _build_env(action, run_name, save_dir, config, state.frontier)
    baseline_key = _baseline_key(config_json)
    baseline_tps = state.get_baseline_tps(
        config_json["architecture"], config_json["hookpoints"]
    )
    config_path = LOG_DIR / f"{run_name}.config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_json, indent=2) + "\n")
    log_path = LOG_DIR / f"{run_name}.log"

    print(
        f"Round {round_id}: starting training | "
        f"arch={config_json['architecture']} k={config_json['k']} "
        f"ef={config_json['expansion_factor']} run_name={run_name}"
    )
    if baseline_tps is not None:
        print(f"Round {round_id}: baseline -> {baseline_tps:.2f} tokens/s")

    state.log_round_event(ctx, "training_started", run_name=run_name, config=config_json)
    state.write_status("training_started", round=round_id, run_name=run_name)

    # Run subprocess with watchdog
    termination_reason, latest_tps, baseline_ratio, last_step_record, step_records = (
        _run_with_watchdog(log_path, save_dir, run_name, config, round_id, baseline_tps, env)
    )

    # Finalize
    checkpoint_dir = _latest_checkpoint_dir(save_dir, run_name)
    curve_metrics = _summarize_curve(step_records)
    observed_fvu = _extract_step_fvu(last_step_record)

    # Parse log and decide via new controller (direct call, no subprocess)
    parsed = parse_log(log_path)
    # Fill in config-derived fields if log didn't have them
    if parsed.get("k") is None:
        parsed["k"] = config_json.get("k")
    if parsed.get("architecture") is None:
        parsed["architecture"] = config_json.get("architecture")
    if parsed.get("expansion_factor") is None:
        parsed["expansion_factor"] = config_json.get("expansion_factor")
    if parsed.get("checkpoint") is None and checkpoint_dir is not None:
        parsed["checkpoint"] = str(checkpoint_dir)
    # If log didn't contain val_fvu summary line, use observed_fvu from metrics
    if parsed.get("val_fvu") is None and observed_fvu is not None:
        parsed["val_fvu"] = observed_fvu
        if parsed.get("status") in (None, "crash"):
            parsed["status"] = "ok"

    decision = decide(state.frontier, parsed)

    # Update frontier if improved
    from .git_ops import current_git_commit
    commit = current_git_commit()
    update_frontier(state.frontier, parsed, decision, config_json, commit)

    # Determine health
    run_health = "normal"
    if termination_reason == "throughput_too_low":
        run_health = "perf_regression"
    elif termination_reason != "completed" or decision == "crash":
        run_health = "crash"

    # Build Result
    result = Result(
        decision=decision,
        status=parsed.get("status", ""),
        timestamp=str(int(time.time())),
        val_fvu=str(parsed["val_fvu"]) if parsed.get("val_fvu") is not None else None,
        observed_fvu=f"{observed_fvu:.6f}" if observed_fvu is not None else None,
        k=str(parsed["k"]) if parsed.get("k") is not None else None,
        architecture=parsed.get("architecture"),
        expansion_factor=str(parsed["expansion_factor"]) if parsed.get("expansion_factor") is not None else None,
        run_health=run_health,
        termination_reason=termination_reason,
        wall_time_sec=str(parsed["wall_time_sec"]) if parsed.get("wall_time_sec") is not None else None,
        peak_memory_gb=str(parsed["peak_memory_gb"]) if parsed.get("peak_memory_gb") is not None else None,
        total_tokens=str(parsed["total_tokens"]) if parsed.get("total_tokens") is not None else None,
        tokens_per_sec=f"{latest_tps:.6f}" if latest_tps is not None else None,
        baseline_ratio=f"{baseline_ratio:.6f}" if baseline_ratio is not None else None,
        checkpoint=parsed.get("checkpoint"),
        experiment_id=f"r{round_id}_{int(time.time())}",
        head_commit=commit,
        head_branch=None,
        workspace_dirty=None,
        description=action.summary,
        self_review=action.self_review,
        log_path=str(log_path),
        metrics_path=(
            str(checkpoint_dir / "metrics.jsonl")
            if checkpoint_dir and (checkpoint_dir / "metrics.jsonl").exists()
            else None
        ),
        curve_start_fvu=_fmt(curve_metrics.get("curve_start_fvu")),
        curve_mid_fvu=_fmt(curve_metrics.get("curve_mid_fvu")),
        curve_end_fvu=_fmt(curve_metrics.get("curve_end_fvu")),
        curve_late_slope=_fmt(curve_metrics.get("curve_late_slope")),
        curve_still_improving=(
            "true" if curve_metrics.get("curve_still_improving") else
            "false" if curve_metrics.get("curve_still_improving") is not None else None
        ),
    )

    if result.decision == "crash":
        details = extract_failure_details(log_path)
        result.error_type = details.get("error_type")
        result.error_summary = details.get("error_summary")
        result.traceback_excerpt = details.get("traceback_excerpt")
        result.log_excerpt = details.get("log_excerpt")

    state.log_round_event(
        ctx, "training_finished",
        run_name=run_name, run_health=run_health,
        termination_reason=termination_reason, decision=result.decision,
    )

    # Update baseline TPS on healthy runs
    if run_health == "normal" and latest_tps is not None:
        current_tps = state.get_baseline_tps(
            config_json["architecture"], config_json["hookpoints"]
        )
        if current_tps is None or latest_tps > current_tps:
            state.update_baseline_tps(baseline_key, latest_tps, round_id, config_json)

    return result


# ---------------------------------------------------------------------------
# Watchdog loop
# ---------------------------------------------------------------------------


def _run_with_watchdog(
    log_path: Path,
    save_dir: Path,
    run_name: str,
    config: LoopConfig,
    round_id: int,
    baseline_tps: float | None,
    env: dict[str, str],
) -> tuple[str, float | None, float | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """Run training subprocess with monitoring.

    Returns (termination_reason, latest_tps, baseline_ratio, last_step_record, step_records).
    """
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
            time.sleep(config.poll_interval_sec)
            checkpoint_dir = _latest_checkpoint_dir(save_dir, run_name)
            mtime = _metrics_last_update(checkpoint_dir)
            if mtime is not None:
                last_progress = max(last_progress, mtime)

            last_step_record = _read_latest_step(checkpoint_dir)
            if last_step_record is not None:
                first_step_seen = True
                total_tokens = float(last_step_record.get("total_tokens", 0))
                elapsed = max(time.time() - start, 1e-6)
                latest_tps = total_tokens / elapsed if total_tokens > 0 else None
                if latest_tps is not None and baseline_tps and baseline_tps > 0:
                    baseline_ratio = latest_tps / baseline_tps

                # Throughput watchdog
                step = int(last_step_record.get("step", 0))
                if (
                    time.time() - start >= config.slow_run_grace_sec
                    and step >= config.min_progress_steps
                    and latest_tps is not None
                    and baseline_tps is not None
                    and baseline_tps > 0
                    and latest_tps < baseline_tps * config.min_tokens_per_sec_ratio
                ):
                    slow_trigger_count += 1
                    if slow_trigger_count >= 2:
                        termination_reason = "throughput_too_low"
                        print(f"Round {round_id}: low throughput ({latest_tps:.2f} vs {baseline_tps:.2f})")
                        _terminate(process)
                        break
                else:
                    slow_trigger_count = 0

            elif not first_step_seen and time.time() - start > config.first_step_timeout_sec:
                termination_reason = "first_step_timeout"
                print(f"Round {round_id}: no first step after {config.first_step_timeout_sec}s")
                _terminate(process)
                break

            # Hard timeout
            if time.time() - start > config.timeout_sec:
                termination_reason = "hard_timeout"
                print(f"Round {round_id}: hard timeout after {config.timeout_sec}s")
                _terminate(process)
                break

            # Stall timeout
            if time.time() - last_progress > config.stall_timeout_sec:
                termination_reason = "stall_timeout"
                print(f"Round {round_id}: stall timeout ({config.stall_timeout_sec}s)")
                _terminate(process)
                break

    process.wait()
    checkpoint_dir = _latest_checkpoint_dir(save_dir, run_name)
    last_step_record = _read_latest_step(checkpoint_dir) or last_step_record
    step_records = _read_step_records(checkpoint_dir)
    return termination_reason, latest_tps, baseline_ratio, last_step_record, step_records


# ---------------------------------------------------------------------------
# Failure extraction
# ---------------------------------------------------------------------------


def extract_failure_details(log_path: Path) -> dict[str, str]:
    """Extract error_type, error_summary, traceback from crash log."""
    if not log_path.exists():
        return {"error_type": "", "error_summary": "", "log_excerpt": "", "traceback_excerpt": ""}

    lines = log_path.read_text(errors="replace").splitlines()
    excerpt = "\n".join([l.rstrip() for l in lines if l.strip()][-80:])
    tb_lines: list[str] = []
    for line in lines:
        s = line.strip()
        if any(kw in s for kw in ("Traceback", "Root Cause", "ChildFailedError")) or \
           s.startswith("File ") or s.startswith("[rank0]:   File ") or \
           s.endswith("Error") or s.endswith("Exception") or "Error:" in s:
            tb_lines.append(line)
    traceback_excerpt = "\n".join([l.rstrip() for l in tb_lines if l.strip()][-40:])

    error_type = ""
    error_summary = ""
    pattern = re.compile(r"([A-Za-z_]\w*(?:Error|Exception))\s*:\s*(.+)")
    matches = [(m.group(1), m.group(0).strip()) for line in lines if (m := pattern.search(line))]
    preferred = [m for m in matches if m[0] not in {"ChildFailedError", "CalledProcessError"}]
    chosen = preferred[-1] if preferred else (matches[-1] if matches else None)
    if chosen:
        error_type, error_summary = chosen
    if not error_summary:
        for line in reversed(lines):
            if line.strip():
                error_summary = line.strip()
                break

    return {
        "error_type": error_type,
        "error_summary": error_summary,
        "log_excerpt": excerpt,
        "traceback_excerpt": traceback_excerpt,
    }


def budget_remaining_sec(start_time: float, budget_hours: float) -> float:
    return budget_hours * 3600 - (time.time() - start_time)


def check_agent_backend_reachable(agent_proxy: str | None) -> None:
    """Preflight: verify codex CLI exists and proxy is reachable."""
    import shutil
    import socket
    import urllib.parse

    if not shutil.which("codex"):
        raise RuntimeError("codex CLI not found on PATH")
    if agent_proxy:
        parsed = urllib.parse.urlparse(agent_proxy)
        host, port = parsed.hostname, parsed.port
        if not host or not port:
            raise RuntimeError(f"Invalid proxy URL: {agent_proxy}")
        try:
            with socket.create_connection((host, port), timeout=5):
                pass
        except (ConnectionRefusedError, OSError) as exc:
            raise RuntimeError(f"Proxy {agent_proxy} unreachable: {exc}") from exc


# ---------------------------------------------------------------------------
# Environment building
# ---------------------------------------------------------------------------


def _resolve_base_config(
    action: Action,
    frontier: dict[str, Any] | None,
) -> dict[str, str]:
    """Find the best frontier config to use as base for partial overrides.

    Looks for a frontier entry matching the action's family_name. If found,
    converts its stored config dict back to env-var format. Falls back to
    BASE_ENV_DEFAULTS if no match.
    """
    if not frontier:
        return dict(BASE_ENV_DEFAULTS)

    family = (action.family_name or "").lower()
    best_entry = None
    best_fvu = float("inf")

    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        cfg = entry.get("config", {})
        arch = str(cfg.get("architecture", "")).lower()
        if arch == family:
            fvu = float(entry.get("fvu", float("inf")))
            if fvu < best_fvu:
                best_fvu = fvu
                best_entry = entry

    if best_entry is None:
        return dict(BASE_ENV_DEFAULTS)

    # Build env-var dict from frontier config
    cfg = best_entry["config"]
    base = dict(BASE_ENV_DEFAULTS)
    key_map = {
        "architecture": "ARCHITECTURE",
        "expansion_factor": "EXPANSION_FACTOR",
        "k": "K",
        "optimizer": "OPTIMIZER",
        "lr": "LR",
        "hookpoints": "HOOKPOINTS",
        "batch_size": "BATCH_SIZE",
        "grad_acc_steps": "GRAD_ACC_STEPS",
        "micro_acc_steps": "MICRO_ACC_STEPS",
        "auxk_alpha": "AUXK_ALPHA",
        "dead_feature_threshold": "DEAD_FEATURE_THRESHOLD",
        "use_hadamard": "USE_HADAMARD",
    }
    for json_key, env_key in key_map.items():
        val = cfg.get(json_key)
        if val is not None:
            if env_key == "USE_HADAMARD":
                base[env_key] = "1" if bool(val) else "0"
            else:
                base[env_key] = str(val)
    return base


def _build_env(
    action: Action,
    run_name: str,
    save_dir: Path,
    config: LoopConfig,
    frontier: dict[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build subprocess env vars and config JSON from action.

    When frontier is provided, uses the best matching frontier entry's config
    as the base (instead of BASE_ENV_DEFAULTS) so that partial overrides
    inherit the winning recipe rather than falling back to global defaults.
    """
    env = os.environ.copy()
    overrides = action.env_overrides
    rejected = validate_env_overrides(
        overrides,
        fallback_architecture=action.family_name or BASE_ENV_DEFAULTS["ARCHITECTURE"],
    )
    if rejected:
        raise RuntimeError(f"Disallowed env override keys: {', '.join(rejected)}")

    base_config = _resolve_base_config(action, frontier)
    merged = config_from_overrides(overrides, base_config=base_config)
    # If family_name set but no explicit ARCHITECTURE override, use family_name
    has_arch = any(item.get("key") == "ARCHITECTURE" for item in overrides)
    if action.family_name and not has_arch:
        merged["ARCHITECTURE"] = action.family_name.lower()

    env.update(merged)
    env["RUN_NAME"] = run_name
    env["SAVE_DIR"] = str(save_dir)
    env["WANDB_PROJECT"] = env.get("WANDB_PROJECT", "qwen3-0.6B-auto")
    env["MAX_TOKENS"] = config.max_tokens

    config_json: dict[str, Any] = {
        "architecture": merged.get("ARCHITECTURE", "topk").lower(),
        "expansion_factor": int(merged.get("EXPANSION_FACTOR", "8")),
        "k": int(merged.get("K", "128")),
        "optimizer": merged.get("OPTIMIZER", "signum"),
        "lr": merged.get("LR", "8e-4"),
        "hookpoints": merged.get("HOOKPOINTS", "layers.[3].self_attn.o_proj"),
        "batch_size": int(merged.get("BATCH_SIZE", "1")),
        "grad_acc_steps": int(merged.get("GRAD_ACC_STEPS", "8")),
        "micro_acc_steps": int(merged.get("MICRO_ACC_STEPS", "1")),
        "auxk_alpha": float(merged.get("AUXK_ALPHA", "0.03125")),
        "dead_feature_threshold": int(merged.get("DEAD_FEATURE_THRESHOLD", "10000000")),
        "use_hadamard": merged.get("USE_HADAMARD", "0") == "1",
        "family_name": action.family_name or merged.get("ARCHITECTURE", "topk").lower(),
        "family_stage": action.family_stage or "mainline",
    }
    # Optional architecture-specific keys
    _OPTIONAL = {
        "NUM_GROUPS": ("num_groups", int),
        "ACTIVE_GROUPS": ("active_groups", int),
        "JUMPRELU_INIT_THRESHOLD": ("jumprelu_init_threshold", float),
        "JUMPRELU_BANDWIDTH": ("jumprelu_bandwidth", float),
        "GATED_TEMPERATURE": ("gated_temperature", float),
        "GATED_INIT_LOGIT": ("gated_init_logit", float),
        "ORTHO_LAMBDA": ("ortho_lambda", float),
        "RESIDUAL_FROM": ("residual_from", str),
        "MATRYOSHKA_KS": ("matryoshka_ks", lambda x: [int(v) for v in x.split(",") if v]),
        "MATRYOSHKA_WEIGHTS": ("matryoshka_weights", lambda x: [float(v) for v in x.split(",") if v]),
    }
    for env_key, (cfg_key, caster) in _OPTIONAL.items():
        value = merged.get(env_key)
        if value:
            config_json[cfg_key] = caster(value)

    return env, config_json


def _baseline_key(config_json: dict[str, Any]) -> str:
    return f"standard|{config_json.get('architecture')}|{config_json.get('hookpoints')}"


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Checkpoint / metrics helpers
# ---------------------------------------------------------------------------


def _latest_checkpoint_dir(save_dir: Path, run_name: str) -> Path | None:
    matches = sorted(save_dir.glob(f"{run_name}*"))
    return matches[-1] if matches else None


def _metrics_last_update(checkpoint_dir: Path | None) -> float | None:
    if checkpoint_dir is None:
        return None
    metrics = checkpoint_dir / "metrics.jsonl"
    return metrics.stat().st_mtime if metrics.exists() else None


def _read_latest_step(checkpoint_dir: Path | None) -> dict[str, Any] | None:
    if checkpoint_dir is None:
        return None
    metrics = checkpoint_dir / "metrics.jsonl"
    if not metrics.exists():
        return None
    latest = None
    with open(metrics) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "step":
                latest = rec
    return latest


def _read_step_records(checkpoint_dir: Path | None) -> list[dict[str, Any]]:
    if checkpoint_dir is None:
        return []
    metrics = checkpoint_dir / "metrics.jsonl"
    if not metrics.exists():
        return []
    records = []
    with open(metrics) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "step":
                records.append(rec)
    return records


def _extract_step_fvu(step_record: dict[str, Any] | None) -> float | None:
    if not step_record:
        return None
    vals = [v for k, v in step_record.items() if k.endswith("/fvu") and isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def _summarize_curve(step_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not step_records:
        return {k: None for k in ("curve_start_fvu", "curve_mid_fvu", "curve_end_fvu", "curve_late_slope", "curve_still_improving")}
    n = len(step_records)
    start_fvu = _extract_step_fvu(step_records[max(0, n // 5 - 1)])
    mid_fvu = _extract_step_fvu(step_records[max(0, n // 2 - 1)])
    end_fvu = _extract_step_fvu(step_records[-1])
    tail_fvu = _extract_step_fvu(step_records[max(0, int(n * 0.8) - 1)])
    late_slope = None
    still_improving = None
    tail_start = step_records[max(0, int(n * 0.8) - 1)]
    end_rec = step_records[-1]
    st = float(tail_start.get("total_tokens", 0) or 0)
    et = float(end_rec.get("total_tokens", 0) or 0)
    if tail_fvu is not None and end_fvu is not None and et > st:
        late_slope = (end_fvu - tail_fvu) / (et - st)
        still_improving = end_fvu < tail_fvu - 1e-4
    return {
        "curve_start_fvu": start_fvu,
        "curve_mid_fvu": mid_fvu,
        "curve_end_fvu": end_fvu,
        "curve_late_slope": late_slope,
        "curve_still_improving": still_improving,
    }


def _terminate(process: subprocess.Popen) -> None:
    """Terminate a process group, escalating to SIGKILL if needed."""
    pgid = None
    try:
        pgid = os.getpgid(process.pid)
    except OSError:
        pass
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=DEFAULT_PROCESS_TERM_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass
        process.wait(timeout=5)


def _fmt(v: Any) -> str | None:
    if v is None:
        return None
    return f"{v:.6f}" if isinstance(v, float) else str(v)
