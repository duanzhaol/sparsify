"""Training execution and monitoring for the autoresearch loop."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from research.git_ops import REPO_ROOT, run
from research.override_registry import (
    config_from_overrides,
    validate_env_overrides,
)
from research.state_io import (
    BASE_ENV_DEFAULTS,
    LOG_DIR,
    save_json,
    write_status,
    log_round_event,
)

SCRIPT_PATH = REPO_ROOT / "scripts" / "autoresearch_test.sh"
SAVE_ROOT = REPO_ROOT / "checkpoints" / "research_agent"

DEFAULT_PROXY_MAX_TOKENS = "20000000"
DEFAULT_FULL_MAX_TOKENS = "200000000"
DEFAULT_PROXY_TIMEOUT_SEC = 30 * 60
DEFAULT_FULL_TIMEOUT_SEC = 2 * 60 * 60
DEFAULT_STALL_TIMEOUT_SEC = 15 * 60
DEFAULT_POLL_INTERVAL_SEC = 30
DEFAULT_FIRST_STEP_TIMEOUT_SEC = 180
DEFAULT_SLOW_RUN_GRACE_SEC = 120
DEFAULT_MIN_TOKENS_PER_SEC_RATIO = 0.25
DEFAULT_MIN_PROGRESS_STEPS = 4
DEFAULT_PROCESS_TERM_TIMEOUT_SEC = 15


class SanityCheckError(RuntimeError):
    """Structured sanity-check failure with captured subprocess output."""

    def __init__(
        self,
        cmd: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            f"Sanity check failed with exit code {returncode}: {' '.join(cmd)}"
        )

    def to_payload(self) -> dict[str, Any]:
        stderr_lines = [line for line in self.stderr.splitlines() if line.strip()]
        stdout_lines = [line for line in self.stdout.splitlines() if line.strip()]
        traceback_lines = [
            line for line in stderr_lines
            if line.startswith("Traceback")
            or line.startswith("  File ")
            or line.endswith("Error")
            or line.endswith("Exception")
        ]
        return {
            "error_type": "sanity_check_failed",
            "returncode": self.returncode,
            "cmd": self.cmd,
            "stdout_excerpt": "\n".join(stdout_lines[-20:]),
            "stderr_excerpt": "\n".join(stderr_lines[-40:]),
            "traceback_excerpt": "\n".join(traceback_lines[-40:]),
        }

def build_env(action: dict[str, Any], tier: str, run_name: str, save_dir: Path, args: Any, proxy_max_tokens_override: str | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    env = os.environ.copy()
    overrides = action.get("env_overrides", [])
    rejected = validate_env_overrides(
        overrides,
        fallback_architecture=str(action.get("family_name") or BASE_ENV_DEFAULTS["ARCHITECTURE"]),
    )
    if rejected:
        raise RuntimeError(
            f"Agent env_overrides contain disallowed keys: {', '.join(rejected)}. "
            "Register the architecture-specific keys in research.override_registry."
        )
    config = config_from_overrides(overrides)
    has_architecture_override = False
    if isinstance(overrides, dict):
        has_architecture_override = "ARCHITECTURE" in overrides
    else:
        has_architecture_override = any(item.get("key") == "ARCHITECTURE" for item in overrides)
    if action.get("family_name") and not has_architecture_override:
        config["ARCHITECTURE"] = str(action["family_name"]).lower()
    env.update(config)
    env["RUN_NAME"] = run_name
    env["SAVE_DIR"] = str(save_dir)
    env["WANDB_PROJECT"] = env.get("WANDB_PROJECT", "qwen3-0.6B-auto")
    proxy_tokens = proxy_max_tokens_override if proxy_max_tokens_override is not None else args.proxy_max_tokens
    env["MAX_TOKENS"] = args.full_max_tokens if tier == "full" else proxy_tokens

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
        "GATED_TEMPERATURE": ("gated_temperature", float),
        "GATED_INIT_LOGIT": ("gated_init_logit", float),
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


def read_step_records(checkpoint_dir: Path | None) -> list[dict[str, Any]]:
    if checkpoint_dir is None:
        return []
    metrics = checkpoint_dir / "metrics.jsonl"
    if not metrics.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(metrics) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "step":
                records.append(rec)
    return records


def summarize_curve_metrics(step_records: list[dict[str, Any]]) -> dict[str, float | bool | None]:
    if not step_records:
        return {
            "curve_start_fvu": None,
            "curve_mid_fvu": None,
            "curve_end_fvu": None,
            "curve_late_slope": None,
            "curve_still_improving": None,
        }
    size = len(step_records)
    first = step_records[max(0, size // 5 - 1)]
    mid = step_records[max(0, size // 2 - 1)]
    end = step_records[-1]
    tail_start = step_records[max(0, int(size * 0.8) - 1)]
    start_fvu = extract_step_fvu(first)
    mid_fvu = extract_step_fvu(mid)
    end_fvu = extract_step_fvu(end)
    tail_fvu = extract_step_fvu(tail_start)
    late_slope = None
    still_improving = None
    start_tokens = float(tail_start.get("total_tokens", 0) or 0)
    end_tokens = float(end.get("total_tokens", 0) or 0)
    if tail_fvu is not None and end_fvu is not None and end_tokens > start_tokens:
        late_slope = (end_fvu - tail_fvu) / (end_tokens - start_tokens)
        still_improving = end_fvu < tail_fvu - 1e-4
    return {
        "curve_start_fvu": start_fvu,
        "curve_mid_fvu": mid_fvu,
        "curve_end_fvu": end_fvu,
        "curve_late_slope": late_slope,
        "curve_still_improving": still_improving,
    }


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
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'; "
            "try:\n"
            "    import torch_npu; device = 'npu' if torch.npu.is_available() else device\n"
            "except ImportError: pass; "
            f"cfg = SparseCoderConfig(architecture='{arch}', k={k}, expansion_factor={ef}); "
            "sae = SparseCoder(1024, cfg, device=device, dtype=torch.float32); "
            "x = torch.randn(4, 1024, device=device); "
            "out = sae(x); out.fvu.backward(); print('sanity: OK')"
        ),
    ]
    try:
        subprocess.run(
            cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as exc:
        raise SanityCheckError(
            cmd=cmd,
            returncode=exc.returncode,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        ) from exc


def terminate_process_group(process: subprocess.Popen, timeout: int = DEFAULT_PROCESS_TERM_TIMEOUT_SEC) -> None:
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
    import shutil
    import socket
    import urllib.parse

    if not shutil.which("codex"):
        raise RuntimeError(
            "codex CLI not found on PATH. Install it or adjust PATH before running."
        )

    if agent_proxy:
        try:
            parsed = urllib.parse.urlparse(agent_proxy)
            host = parsed.hostname
            port = parsed.port
            if not host or not port:
                raise RuntimeError(
                    f"Agent proxy {agent_proxy} is not a valid proxy URL. "
                    "Expected something like http://127.0.0.1:23234."
                )
            with socket.create_connection((host, port), timeout=5):
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
        "-m",
        "research.controller",
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
    result = run(cmd, cwd=REPO_ROOT)
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
    return parsed


def _run_sanity_for_round(
    action: dict[str, Any],
    tier: str,
    round_id: int,
    round_ctx: dict[str, Any],
) -> None:
    """Run sanity check and log events. Raises on failure."""
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
    args: Any,
    round_id: int,
    memory: dict[str, Any],
    round_ctx: dict[str, Any],
    proxy_max_tokens_override: str | None = None,
    proxy_mode: str = "fast",
    evaluation_basis: str = "terminal_only",
) -> dict[str, str]:
    run_stamp = int(time.time())
    run_name = f"round{round_id:04d}_{tier}_{run_stamp}"
    save_dir = SAVE_ROOT / f"round_{round_id:04d}_{tier}"
    save_dir.mkdir(parents=True, exist_ok=True)
    env, config_json = build_env(action, tier, run_name, save_dir, args, proxy_max_tokens_override=proxy_max_tokens_override)
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
        proxy_mode=proxy_mode if tier == "proxy" else None,
        evaluation_basis=evaluation_basis if tier == "proxy" else None,
    )
    write_status(
        "training_started",
        round=round_id,
        tier=tier,
        run_name=run_name,
        log_path=str(log_path),
        config=config_json,
        baseline_tokens_per_sec=baseline_tps,
        proxy_mode=proxy_mode if tier == "proxy" else None,
        evaluation_basis=evaluation_basis if tier == "proxy" else None,
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
                    proxy_mode=proxy_mode if tier == "proxy" else None,
                    evaluation_basis=evaluation_basis if tier == "proxy" else None,
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
                    proxy_mode=proxy_mode if tier == "proxy" else None,
                    evaluation_basis=evaluation_basis if tier == "proxy" else None,
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
                    proxy_mode=proxy_mode if tier == "proxy" else None,
                    evaluation_basis=evaluation_basis if tier == "proxy" else None,
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
    step_records = read_step_records(checkpoint_dir)
    curve_metrics = summarize_curve_metrics(step_records)
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
    result["proxy_mode"] = proxy_mode if tier == "proxy" else ""
    result["evaluation_basis"] = evaluation_basis if tier == "proxy" else ""
    for key, value in curve_metrics.items():
        if value is None:
            result[key] = ""
        elif isinstance(value, bool):
            result[key] = "true" if value else "false"
        else:
            result[key] = f"{value:.6f}"
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
        proxy_mode=proxy_mode if tier == "proxy" else None,
        evaluation_basis=evaluation_basis if tier == "proxy" else None,
        **curve_metrics,
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
        proxy_mode=proxy_mode if tier == "proxy" else None,
        evaluation_basis=evaluation_basis if tier == "proxy" else None,
        **curve_metrics,
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
