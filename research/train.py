"""
SAE training experiment runner for autoresearch.

This is the file the agent modifies. It defines the experiment configuration,
builds the training command, runs it, and prints results in a parseable format.

The agent changes CONFIG (and optionally edits sparsify source files) to explore
different SAE architectures and hyperparameters. The goal is to minimize FVU
at the smallest possible K.

Usage:
    python train.py                # run proxy experiment
    python train.py --full         # run full experiment (for promoted candidates)
    python train.py > run.log 2>&1 # redirect output (recommended in loop)
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Import fixed constants from prepare.py
from prepare import (
    MODEL_PATH,
    DATASET_PATH,
    ELBOW_THRESHOLD_PATH,
    PROXY_MAX_TOKENS,
    FULL_MAX_TOKENS,
    PROXY_TIMEOUT,
    FULL_TIMEOUT,
    PROXY_HOOKPOINT,
    get_final_fvu,
    parse_summary,
)

# ============================================================================
# EXPERIMENT CONFIGURATION — Agent modifies this section
# ============================================================================

CONFIG = {
    # --- SAE architecture ---
    "architecture": "topk",         # topk | gated | jumprelu | group_topk
    "expansion_factor": 8,          # latent dim = expansion_factor * d_in
    "k": 128,                       # number of active features

    # --- Group TopK (only when architecture=group_topk) ---
    # "num_groups": 16,
    # "active_groups": 4,

    # --- JumpReLU (only when architecture=jumprelu) ---
    # "jumprelu_init_threshold": 0.001,
    # "jumprelu_bandwidth": 0.001,

    # --- Optimizer ---
    "optimizer": "signum",          # signum | adam
    "lr": 8e-4,

    # --- Training ---
    "batch_size": 1,
    "grad_acc_steps": 8,
    "auxk_alpha": 0.03125,
    "dead_feature_threshold": 10_000_000,

    # --- Regularization ---
    # "ortho_lambda": 0.0,

    # --- Matryoshka multi-K training ---
    # "matryoshka_ks": [32, 64],
    # "matryoshka_weights": [1.0, 1.0],

    # --- Hadamard rotation ---
    "use_hadamard": False,

    # --- Hookpoints (proxy uses single layer for speed) ---
    "hookpoints": [PROXY_HOOKPOINT],
}

# ============================================================================
# Below here: experiment runner. Agent usually does not need to modify this.
# ============================================================================

SAVE_DIR = "checkpoints/research"
NPROC = 2  # number of GPUs


def build_command(config: dict, is_full: bool = False) -> list[str]:
    """Build the sparsify CLI command from config dict."""
    max_tokens = FULL_MAX_TOKENS if is_full else PROXY_MAX_TOKENS
    run_tag = f"{'full' if is_full else 'proxy'}_{config['architecture']}_k{config['k']}_ef{config['expansion_factor']}"
    run_name = f"{run_tag}_{int(time.time())}"

    cmd = [
        "torchrun",
        "--nproc_per_node", str(NPROC),
        "--master_port", "29502",
        "-m", "sparsify",
        MODEL_PATH,
        DATASET_PATH,
        "--split", "train",
        "--ctx_len", "2048",
        "--max_examples", "1000000",
        "--shuffle_seed", "1127",
        "--data_preprocessing_num_proc", "120",
        "--init_seeds", "1127",
        "--normalize_decoder",
        "--num_latents", "0",
        "--save_every", "1000",
        "--save_best",
        "--save_dir", SAVE_DIR,
        "--run_name", run_name,
        "--save_metrics_jsonl",
        "--nolog_to_wandb",
        "--wandb_log_frequency", "1",
        "--max_tokens", str(max_tokens),
        "--compile_model",
    ]

    # Elbow thresholds
    if os.path.exists(ELBOW_THRESHOLD_PATH):
        cmd.extend(["--elbow_threshold_path", ELBOW_THRESHOLD_PATH])

    # Map config dict to CLI args
    simple_args = {
        "architecture": "--architecture",
        "expansion_factor": "--expansion_factor",
        "k": "-k",
        "optimizer": "--optimizer",
        "lr": "--lr",
        "batch_size": "--batch_size",
        "grad_acc_steps": "--grad_acc_steps",
        "auxk_alpha": "--auxk_alpha",
        "dead_feature_threshold": "--dead_feature_threshold",
        "ortho_lambda": "--ortho_lambda",
        "jumprelu_init_threshold": "--jumprelu_init_threshold",
        "jumprelu_bandwidth": "--jumprelu_bandwidth",
        "num_groups": "--num_groups",
        "active_groups": "--active_groups",
    }

    for key, flag in simple_args.items():
        if key in config and config[key] is not None:
            cmd.extend([flag, str(config[key])])

    # Boolean flags
    if config.get("use_hadamard"):
        cmd.append("--use_hadamard")
    else:
        cmd.append("--nouse_hadamard")

    # List args
    if "hookpoints" in config:
        for hp in config["hookpoints"]:
            cmd.extend(["--hookpoints", hp])

    if "matryoshka_ks" in config:
        for mk in config["matryoshka_ks"]:
            cmd.extend(["--matryoshka_ks", str(mk)])

    if "matryoshka_weights" in config:
        for mw in config["matryoshka_weights"]:
            cmd.extend(["--matryoshka_weights", str(mw)])

    return cmd, run_name


def find_checkpoint_dir(run_name: str) -> str | None:
    """Find the actual checkpoint directory (includes timestamp suffix)."""
    base = Path(SAVE_DIR)
    if not base.exists():
        return None
    # Trainer appends dp/bs/ga/ef/k/timestamp to run_name
    matches = sorted(base.glob(f"{run_name}*"))
    if matches:
        return str(matches[-1])
    # Also check for exact match
    exact = base / run_name
    if exact.exists():
        return str(exact)
    return None


def run_experiment(is_full: bool = False):
    """Run the experiment and print results."""
    cmd, run_name = build_command(CONFIG, is_full)
    timeout = FULL_TIMEOUT if is_full else PROXY_TIMEOUT
    mode = "FULL" if is_full else "PROXY"

    print(f"=== {mode} EXPERIMENT ===")
    print(f"run_name: {run_name}")
    print(f"architecture: {CONFIG['architecture']}")
    print(f"k: {CONFIG['k']}")
    print(f"expansion_factor: {CONFIG['expansion_factor']}")
    print(f"max_tokens: {FULL_MAX_TOKENS if is_full else PROXY_MAX_TOKENS}")
    print(f"hookpoints: {CONFIG.get('hookpoints', ['default'])}")
    print(f"command: {' '.join(cmd)}")
    print("---")
    sys.stdout.flush()

    t0 = time.time()

    # Heartbeat: background thread polls metrics.jsonl and prints progress
    stop_heartbeat = threading.Event()

    def heartbeat_loop(run_name_prefix):
        """Print a progress line every 10s by reading metrics.jsonl."""
        last_step = -1
        while not stop_heartbeat.is_set():
            stop_heartbeat.wait(10)
            ckpt = find_checkpoint_dir(run_name_prefix)
            if not ckpt:
                elapsed = time.time() - t0
                print(f"[heartbeat] {elapsed:.0f}s elapsed, waiting for first step...", flush=True)
                continue
            metrics_path = Path(ckpt) / "metrics.jsonl"
            if not metrics_path.exists():
                continue
            # Read last step line
            last_line = None
            with open(metrics_path) as f:
                for line in f:
                    if '"type": "step"' in line:
                        last_line = line
            if last_line:
                try:
                    rec = json.loads(last_line)
                    step = rec.get("step", "?")
                    tokens = rec.get("total_tokens", "?")
                    # Find any FVU key
                    fvu_val = "?"
                    for k, v in rec.items():
                        if k.endswith("/fvu") and isinstance(v, (int, float)):
                            fvu_val = f"{v:.4f}"
                            break
                    if step != last_step:
                        last_step = step
                        elapsed = time.time() - t0
                        print(f"[heartbeat] step={step} tokens={tokens} fvu={fvu_val} elapsed={elapsed:.0f}s", flush=True)
                except (json.JSONDecodeError, KeyError):
                    pass

    hb_thread = threading.Thread(target=heartbeat_loop, args=(run_name,), daemon=True)
    hb_thread.start()

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=False,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        wall_time = time.time() - t0
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        wall_time = time.time() - t0
        returncode = -1
        print(f"\n[TIMEOUT] Experiment killed after {timeout}s", flush=True)
    finally:
        stop_heartbeat.set()
        hb_thread.join(timeout=5)

    # Find checkpoint and parse results
    ckpt_dir = find_checkpoint_dir(run_name)
    fvu = None
    dead_pct = None
    total_tokens = None

    if ckpt_dir:
        summary = parse_summary(ckpt_dir)
        if summary:
            total_tokens = summary.get("total_tokens", 0)
            # Get FVU for the hookpoint(s)
            fvu_dict = summary.get("best_fvu") or summary.get("final_fvu", {})
            if fvu_dict:
                vals = [v for v in fvu_dict.values() if isinstance(v, (int, float))]
                fvu = sum(vals) / len(vals) if vals else None

    # Try to get peak memory from nvidia-smi (rough)
    peak_mem_gb = 0.0
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if smi.returncode == 0:
            mems = [float(x.strip()) for x in smi.stdout.strip().split("\n") if x.strip()]
            peak_mem_gb = max(mems) / 1024 if mems else 0.0
    except Exception:
        pass

    # Print parseable summary
    print("\n---")
    print(f"status:           {'ok' if returncode == 0 else 'crash' if returncode > 0 else 'timeout'}")
    print(f"val_fvu:          {fvu:.6f}" if fvu is not None else "val_fvu:          nan")
    print(f"k:                {CONFIG['k']}")
    print(f"architecture:     {CONFIG['architecture']}")
    print(f"wall_time_sec:    {wall_time:.1f}")
    print(f"peak_memory_gb:   {peak_mem_gb:.1f}")
    print(f"total_tokens:     {total_tokens}" if total_tokens else "total_tokens:     0")
    print(f"checkpoint:       {ckpt_dir}" if ckpt_dir else "checkpoint:       none")
    print(f"expansion_factor: {CONFIG['expansion_factor']}")

    return returncode == 0


if __name__ == "__main__":
    is_full = "--full" in sys.argv
    success = run_experiment(is_full)
    sys.exit(0 if success else 1)
