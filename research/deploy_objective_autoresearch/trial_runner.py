from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SearchConfig
from .metrics_extractor import extract_trial_snapshot, read_latest_step_record
from .state_store import TrialRecord


@dataclass(slots=True)
class TrialRunResult:
    returncode: int
    target_tokens: int
    checkpoint_root: str | None
    metrics_path: str | None
    log_path: str
    tokens_seen: int
    error_type: str | None = None
    error_summary: str | None = None


def next_checkpoint_target(tokens_seen: int, checkpoint_interval_tokens: int) -> int:
    if checkpoint_interval_tokens <= 0:
        raise ValueError("checkpoint_interval_tokens must be positive")
    completed = tokens_seen // checkpoint_interval_tokens
    return (completed + 1) * checkpoint_interval_tokens


def _pick_master_port() -> int:
    # Avoid opening probe sockets in restricted environments; a deterministic
    # per-process offset is good enough because this framework runs one trial
    # at a time.
    return 29501 + (os.getpid() % 1000)


def build_trial_env(
    *,
    cfg: SearchConfig,
    params: dict[str, str | int | float],
    run_name: str,
    save_dir: Path,
    target_tokens: int,
    resume: bool,
    master_port: int | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "NPROC_PER_NODE": str(cfg.nproc_per_node),
            "MASTER_PORT": str(master_port or _pick_master_port()),
            "MODEL_PATH": cfg.model_path,
            "DATASET_PATH": cfg.dataset_path,
            "ELBOW_THRESHOLD_PATH": cfg.elbow_threshold_path,
            "WANDB_PROJECT": cfg.wandb_project,
            "SAVE_DIR": str(save_dir),
            "RUN_NAME": run_name,
            "MAX_TOKENS": str(target_tokens),
            "ARCHITECTURE": cfg.architecture,
            "HOOKPOINTS": cfg.hookpoints,
            "EXPANSION_FACTOR": str(cfg.expansion_factor),
            "OPTIMIZER": cfg.optimizer,
            "BATCH_SIZE": str(cfg.batch_size),
            "GRAD_ACC_STEPS": str(cfg.grad_acc_steps),
            "MICRO_ACC_STEPS": str(cfg.micro_acc_steps),
            "DEAD_FEATURE_THRESHOLD": str(cfg.dead_feature_threshold),
            "USE_HADAMARD": str(cfg.use_hadamard),
            "COMPILE_MODEL": str(cfg.compile_model),
            "PRINT_COST_BREAKDOWN": str(cfg.print_cost_breakdown),
            "RESUME": "1" if resume else "0",
        }
    )
    for key, value in params.items():
        env[str(key)] = str(value)
    return env


def latest_checkpoint_dir(save_dir: Path, run_name: str) -> Path | None:
    matches = sorted(save_dir.glob(f"{run_name}*"))
    return matches[-1] if matches else None


def extract_training_failure_details(log_path: Path) -> tuple[str | None, str | None]:
    if not log_path.exists():
        return None, None
    lines = [line.strip() for line in log_path.read_text(errors="replace").splitlines()]
    lines = [line for line in lines if line]
    for line in reversed(lines):
        if "Error:" in line or line.endswith("Error") or line.endswith("Exception"):
            error_type = line.split(":", 1)[0]
            return error_type, line
    if lines:
        return None, lines[-1]
    return None, None


def launch_trial_process(env: dict[str, str], log_path: Path) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as _:
        pass
    log_fh = log_path.open("ab")
    proc = subprocess.Popen(
        ["bash", "scripts/autoresearch_test.sh"],
        cwd=Path.cwd(),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    log_fh.close()
    return proc


def terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def wait_for_trial(
    *,
    proc: subprocess.Popen[bytes],
    cfg: SearchConfig,
    run_name: str,
    save_dir: Path,
    target_tokens: int,
    log_path: Path,
) -> TrialRunResult:
    last_metrics_update = time.time()
    checkpoint_root: Path | None = None
    metrics_path: Path | None = None
    reached_target_at: float | None = None

    while True:
        checkpoint_root = latest_checkpoint_dir(save_dir, run_name)
        if checkpoint_root is not None:
            candidate_metrics = checkpoint_root / "metrics.jsonl"
            if candidate_metrics.exists():
                metrics_path = candidate_metrics
                mtime = candidate_metrics.stat().st_mtime
                last_metrics_update = max(last_metrics_update, mtime)
                latest = read_latest_step_record(candidate_metrics)
                if latest is not None and int(latest.get("total_tokens") or 0) >= target_tokens:
                    if reached_target_at is None:
                        reached_target_at = time.time()

        returncode = proc.poll()
        if returncode is not None:
            latest_tokens = 0
            if metrics_path is not None:
                latest = read_latest_step_record(metrics_path)
                if latest is not None:
                    latest_tokens = int(latest.get("total_tokens") or 0)
            error_type, error_summary = (None, None)
            if returncode != 0:
                error_type, error_summary = extract_training_failure_details(log_path)
            return TrialRunResult(
                returncode=returncode,
                target_tokens=target_tokens,
                checkpoint_root=str(checkpoint_root) if checkpoint_root else None,
                metrics_path=str(metrics_path) if metrics_path else None,
                log_path=str(log_path),
                tokens_seen=latest_tokens,
                error_type=error_type,
                error_summary=error_summary,
            )

        now = time.time()
        if now - last_metrics_update > cfg.stall_timeout_sec:
            terminate_process_group(proc)
            return TrialRunResult(
                returncode=124,
                target_tokens=target_tokens,
                checkpoint_root=str(checkpoint_root) if checkpoint_root else None,
                metrics_path=str(metrics_path) if metrics_path else None,
                log_path=str(log_path),
                tokens_seen=0,
                error_type="stall_timeout",
                error_summary=(
                    f"No metrics update for {cfg.stall_timeout_sec}s while waiting for {target_tokens} tokens"
                ),
            )

        if reached_target_at is not None and now - reached_target_at > cfg.finish_grace_sec:
            terminate_process_group(proc)
            return TrialRunResult(
                returncode=125,
                target_tokens=target_tokens,
                checkpoint_root=str(checkpoint_root) if checkpoint_root else None,
                metrics_path=str(metrics_path) if metrics_path else None,
                log_path=str(log_path),
                tokens_seen=target_tokens,
                error_type="finish_grace_timeout",
                error_summary="Target tokens reached but process did not exit in time",
            )

        time.sleep(cfg.poll_interval_sec)


def run_trial_segment(cfg: SearchConfig, trial: TrialRecord) -> TrialRunResult:
    save_dir = Path(trial.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    target_tokens = next_checkpoint_target(
        trial.tokens_seen,
        cfg.checkpoint_interval_tokens,
    )
    env = build_trial_env(
        cfg=cfg,
        params=trial.params,
        run_name=trial.trial_id,
        save_dir=save_dir,
        target_tokens=target_tokens,
        resume=trial.tokens_seen > 0,
    )
    log_path = Path(trial.log_path)
    proc = launch_trial_process(env, log_path)
    return wait_for_trial(
        proc=proc,
        cfg=cfg,
        run_name=trial.trial_id,
        save_dir=save_dir,
        target_tokens=target_tokens,
        log_path=log_path,
    )


def update_trial_from_result(cfg: SearchConfig, trial: TrialRecord, result: TrialRunResult) -> TrialRecord:
    trial.target_tokens = result.target_tokens
    trial.returncode = result.returncode
    trial.checkpoint_root = result.checkpoint_root
    trial.metrics_path = result.metrics_path
    if result.error_summary:
        trial.invalid_reason = result.error_summary
    if result.returncode != 0 or result.metrics_path is None:
        trial.status = "stopped"
        trial.tokens_seen = max(trial.tokens_seen, result.tokens_seen)
        return trial

    snapshot = extract_trial_snapshot(
        log_path=Path(result.log_path),
        metrics_path=Path(result.metrics_path),
        hook_metric_prefix=cfg.hookpoints,
        checkpoint_interval_tokens=cfg.checkpoint_interval_tokens,
        window_start_tokens=trial.tokens_seen,
    )
    trial.tokens_seen = snapshot.tokens_seen
    trial.checkpoint_decisions = snapshot.checkpoint_count
    trial.total_cost_ratio = snapshot.total_cost_ratio
    trial.latest_exceed_alpha_0_50 = snapshot.latest_exceed_alpha_0_50
    trial.best_exceed_alpha_0_50 = snapshot.best_exceed_alpha_0_50
    trial.latest_fvu = snapshot.latest_fvu
    trial.best_fvu = snapshot.best_fvu
    trial.best_objective = snapshot.best_objective
    trial.delta_best_exceed = snapshot.delta_best_exceed
    trial.delta_best_fvu = snapshot.delta_best_fvu
    trial.delta_best_objective = snapshot.delta_best_objective
    trial.invalid_reason = snapshot.invalid_reason
    trial.status = "checkpoint_ready" if snapshot.invalid_reason is None else "stopped"
    return trial
