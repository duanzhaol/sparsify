"""
Minimal unattended nightly runner for SAE autoresearch.

This version stays intentionally simple:
- runs a fixed queue of parameter / architecture experiments
- launches the validated base-case shell script
- watches for progress via metrics.jsonl timestamps
- records results via controller.py
- promotes improved proxy runs to a full run automatically

It does not attempt autonomous code edits. That can be added later on top of
this stable execution layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = REPO_ROOT / "research"
SCRIPT_PATH = REPO_ROOT / "scripts" / "autoresearch_test.sh"
HISTORY_DIR = RESEARCH_DIR / "history"
LOG_DIR = HISTORY_DIR / "logs"
SAVE_ROOT = REPO_ROOT / "checkpoints" / "research_nightly"
CONTROLLER_PATH = RESEARCH_DIR / "controller.py"
RESULTS_PATH = HISTORY_DIR / "results.tsv"

DEFAULT_PROXY_MAX_TOKENS = "20000000"
DEFAULT_FULL_MAX_TOKENS = "200000000"
DEFAULT_PROXY_TIMEOUT_SEC = 30 * 60
DEFAULT_FULL_TIMEOUT_SEC = 2 * 60 * 60
DEFAULT_STALL_TIMEOUT_SEC = 15 * 60
DEFAULT_POLL_INTERVAL_SEC = 30


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    self_review: str
    env: dict[str, str]


def build_plan() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="baseline_topk_k128",
            description="baseline topk k128 ef8 signum",
            self_review="This improvement might be fake because proxy budget is short.",
            env={},
        ),
        ExperimentSpec(
            name="topk_k128_adam",
            description="topk k128 ef8 adam",
            self_review="This improvement might be fake because optimizer changes can need retuning.",
            env={"OPTIMIZER": "adam"},
        ),
        ExperimentSpec(
            name="topk_k128_lr4e4",
            description="topk k128 ef8 signum lr4e-4",
            self_review="This improvement might be fake because lower lr may just change early-run speed.",
            env={"LR": "4e-4"},
        ),
        ExperimentSpec(
            name="topk_k128_lr1p6e3",
            description="topk k128 ef8 signum lr1.6e-3",
            self_review="This improvement might be fake because higher lr can look good briefly before diverging.",
            env={"LR": "1.6e-3"},
        ),
        ExperimentSpec(
            name="topk_k64",
            description="topk k64 ef8 signum",
            self_review="This improvement might be fake because lower K can lag at short budgets.",
            env={"K": "64"},
        ),
        ExperimentSpec(
            name="gated_k128",
            description="gated k128 ef8 signum",
            self_review="This improvement might be fake because architecture warmup may differ from topk.",
            env={"ARCHITECTURE": "gated"},
        ),
        ExperimentSpec(
            name="gated_k64",
            description="gated k64 ef8 signum",
            self_review="This improvement might be fake because proxy budget may under-train gated routing at low K.",
            env={"ARCHITECTURE": "gated", "K": "64"},
        ),
        ExperimentSpec(
            name="jumprelu_k128",
            description="jumprelu k128 ef8 signum",
            self_review="This improvement might be fake because JumpReLU threshold dynamics may need longer training.",
            env={
                "ARCHITECTURE": "jumprelu",
                "JUMPRELU_INIT_THRESHOLD": "0.001",
                "JUMPRELU_BANDWIDTH": "0.001",
            },
        ),
        ExperimentSpec(
            name="group_topk_k128",
            description="group_topk k128 ef8 signum g16 a4",
            self_review="This improvement might be fake because routing overhead may change short-run behavior.",
            env={
                "ARCHITECTURE": "group_topk",
                "NUM_GROUPS": "16",
                "ACTIVE_GROUPS": "4",
            },
        ),
    ]


def ensure_setup() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["python", str(CONTROLLER_PATH), "init"],
        cwd=RESEARCH_DIR,
        check=True,
    )


def load_results_index() -> dict[str, dict[str, str]]:
    if not RESULTS_PATH.exists():
        return {}
    with open(RESULTS_PATH, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {
            row["description"]: row
            for row in reader
            if row.get("description")
        }


def serialize_config(spec: ExperimentSpec, tier: str) -> dict[str, str]:
    config = {
        "architecture": spec.env.get("ARCHITECTURE", "topk"),
        "expansion_factor": int(spec.env.get("EXPANSION_FACTOR", "8")),
        "k": int(spec.env.get("K", "128")),
        "optimizer": spec.env.get("OPTIMIZER", "signum"),
        "lr": spec.env.get("LR", "8e-4"),
        "hookpoints": spec.env.get("HOOKPOINTS", "layers.[3].self_attn.o_proj"),
        "batch_size": int(spec.env.get("BATCH_SIZE", "1")),
        "grad_acc_steps": int(spec.env.get("GRAD_ACC_STEPS", "8")),
        "use_hadamard": spec.env.get("USE_HADAMARD", "0") == "1",
        "tier": tier,
    }
    if "NUM_GROUPS" in spec.env:
        config["num_groups"] = int(spec.env["NUM_GROUPS"])
    if "ACTIVE_GROUPS" in spec.env:
        config["active_groups"] = int(spec.env["ACTIVE_GROUPS"])
    if "JUMPRELU_INIT_THRESHOLD" in spec.env:
        config["jumprelu_init_threshold"] = float(spec.env["JUMPRELU_INIT_THRESHOLD"])
    if "JUMPRELU_BANDWIDTH" in spec.env:
        config["jumprelu_bandwidth"] = float(spec.env["JUMPRELU_BANDWIDTH"])
    return config


def latest_checkpoint_dir(save_dir: Path, run_name: str) -> Path | None:
    matches = sorted(save_dir.glob(f"{run_name}*"))
    return matches[-1] if matches else None


def metrics_last_update(checkpoint_dir: Path | None) -> float | None:
    if checkpoint_dir is None:
        return None
    metrics = checkpoint_dir / "metrics.jsonl"
    if not metrics.exists():
        return None
    return metrics.stat().st_mtime


def run_one(spec: ExperimentSpec, tier: str, args: argparse.Namespace) -> str:
    run_stamp = int(time.time())
    run_name = f"{spec.name}_{tier}_{run_stamp}"
    save_dir = SAVE_ROOT / f"{spec.name}_{tier}"
    save_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(spec.env)
    env["RUN_NAME"] = run_name
    env["SAVE_DIR"] = str(save_dir)
    env["WANDB_PROJECT"] = env.get("WANDB_PROJECT", "qwen3-0.6B-auto")
    env["MAX_TOKENS"] = args.full_max_tokens if tier == "full" else args.proxy_max_tokens

    timeout_sec = args.full_timeout_sec if tier == "full" else args.proxy_timeout_sec
    log_path = LOG_DIR / f"{run_name}.log"
    config_path = LOG_DIR / f"{run_name}.config.json"
    config_path.write_text(json.dumps(serialize_config(spec, tier), indent=2))

    start = time.time()
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

            elapsed = time.time() - start
            if elapsed > timeout_sec:
                process.kill()
                break
            if time.time() - last_progress > args.stall_timeout_sec:
                process.kill()
                break

    process.wait()
    checkpoint_dir = latest_checkpoint_dir(save_dir, run_name)

    record_cmd = [
        "python",
        str(CONTROLLER_PATH),
        "record",
        "--log",
        str(log_path),
        "--tier",
        tier,
        "--description",
        spec.description if tier == "proxy" else f"{spec.description} full",
        "--self-review",
        spec.self_review,
        "--config-json",
        str(config_path),
    ]
    if checkpoint_dir is not None:
        record_cmd.extend(["--checkpoint-dir", str(checkpoint_dir)])

    record = subprocess.run(
        record_cmd,
        cwd=RESEARCH_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    print(record.stdout, end="")

    decision = "crash"
    for line in record.stdout.splitlines():
        if line.startswith("decision:"):
            decision = line.split(":", 1)[1].strip()
            break
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal unattended SAE runner")
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--proxy-max-tokens", default=DEFAULT_PROXY_MAX_TOKENS)
    parser.add_argument("--full-max-tokens", default=DEFAULT_FULL_MAX_TOKENS)
    parser.add_argument("--proxy-timeout-sec", type=int, default=DEFAULT_PROXY_TIMEOUT_SEC)
    parser.add_argument("--full-timeout-sec", type=int, default=DEFAULT_FULL_TIMEOUT_SEC)
    parser.add_argument("--stall-timeout-sec", type=int, default=DEFAULT_STALL_TIMEOUT_SEC)
    parser.add_argument("--poll-interval-sec", type=int, default=DEFAULT_POLL_INTERVAL_SEC)
    args = parser.parse_args()

    ensure_setup()
    results_index = load_results_index()
    plan = build_plan()
    ran = 0

    for spec in plan:
        if args.max_experiments is not None and ran >= args.max_experiments:
            break
        full_desc = f"{spec.description} full"
        prior_proxy = results_index.get(spec.description)
        prior_full = results_index.get(full_desc)

        if prior_proxy and prior_proxy.get("decision") == "promote" and not prior_full:
            run_one(spec, "full", args)
            results_index = load_results_index()
            continue

        if prior_proxy:
            continue

        decision = run_one(spec, "proxy", args)
        results_index = load_results_index()
        ran += 1
        if decision == "promote":
            run_one(spec, "full", args)
            results_index = load_results_index()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
