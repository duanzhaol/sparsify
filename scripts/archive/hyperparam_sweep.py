#!/usr/bin/env python3
"""
Hyperparameter sweep script for SAE training.

Usage:
    python scripts/hyperparam_sweep.py
    python scripts/hyperparam_sweep.py --dry-run  # Preview commands without running
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path


# ============================================================================
# CONFIGURATION - Modify these values for your experiments
# ============================================================================

# Base configuration (shared across all runs)
BASE_CONFIG = {
    "model": "~/models/Qwen3-8B/",
    "dataset": "~/fineweb-edu/sample/10BT",
    "split": "train",
    "ctx_len": 2048,
    "hookpoints": "layers.0.self_attn.o_proj",
    "hook_mode": "input",

    # Training parameters
    "batch_size": 1,
    "grad_acc_steps": 8,
    "micro_acc_steps": 1,
    "max_tokens": 100_000_000,  # 100M tokens per run

    # SAE architecture (fixed)
    "activation": "topk",
    "normalize_decoder": True,
    "num_latents": 0,
    "multi_topk": False,
    "skip_connection": False,

    # Optimization
    "loss_fn": "fvu",
    "optimizer": "signum",
    "lr": 5e-3,
    "auxk_alpha": 0.03125,
    "dead_feature_threshold": 10000000,

    # Logging and saving
    "save_every": 100,
    "save_best": True,
    "save_dir": "checkpoints",
    "log_to_wandb": True,
    "wandb_log_frequency": 1,

    # Data preprocessing
    "data_preprocessing_num_proc": 8,
    "shuffle_seed": 42,
    "text_column": "text",
    "init_seeds": 0,

    # Additional settings
    "elbow_threshold_path": "~/sparsify/thresholds.json",
    "exceed_alphas": [0.05, 0.10, 0.20, 0.50, 1.0, 2.0],
}

# Hyperparameters to sweep
SWEEP_PARAMS = {
    "expansion_factor": [4, 8, 16],      # Sweep over different expansion factors
    "k": [16, 24, 32, 40, 48, 64],       # Sweep over different topk values
}

# Distributed training settings
NUM_GPUS = 8
MASTER_PORT = 29500  # Base port, will increment for each run to avoid conflicts

# ============================================================================
# Script Logic - Usually don't need to modify below
# ============================================================================


def generate_run_name(params):
    """Generate a descriptive run name from parameters."""
    expansion = params["expansion_factor"]
    k = params["k"]
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return f"sweep_ef{expansion}_k{k}_{timestamp}"


def build_command(params, run_name, gpu_count, port):
    """Build the training command."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={gpu_count}",
        f"--master_port={port}",
        "-m", "sparsify",
        params["model"],
        params["dataset"],
    ]

    # Add all parameters
    for key, value in params.items():
        if key in ["model", "dataset"]:
            continue  # Already added as positional args

        # Convert key to command line format
        flag = f"--{key}"

        # Handle different value types
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, list):
            cmd.append(flag)
            cmd.extend(map(str, value))
        else:
            cmd.extend([flag, str(value)])

    # Add run name
    cmd.extend(["--run_name", run_name])

    return cmd


def run_experiment(params, run_name, gpu_count, port, dry_run=False):
    """Run a single experiment."""
    cmd = build_command(params, run_name, gpu_count, port)

    print("\n" + "="*80)
    print(f"Experiment: {run_name}")
    print(f"Parameters: expansion_factor={params['expansion_factor']}, k={params['k']}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print()

    if dry_run:
        print("[DRY RUN] Would execute the above command\n")
        return True

    # Run the command
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ Experiment completed successfully in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Experiment failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for SAE training")
    parser.add_argument("--dry-run", action="store_true", help="Preview commands without executing")
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue to next experiment if one fails")
    parser.add_argument("--max-examples", type=int, help="Override max_examples for quick testing")
    parser.add_argument("--gpus", type=int, default=NUM_GPUS, help="Number of GPUs to use")
    args = parser.parse_args()

    # Generate all parameter combinations
    param_names = list(SWEEP_PARAMS.keys())
    param_values = list(SWEEP_PARAMS.values())
    combinations = list(product(*param_values))

    print(f"\n{'='*80}")
    print(f"Hyperparameter Sweep Configuration")
    print(f"{'='*80}")
    print(f"Total experiments: {len(combinations)}")
    print(f"Sweep parameters:")
    for name, values in SWEEP_PARAMS.items():
        print(f"  - {name}: {values}")
    print(f"GPUs per experiment: {args.gpus}")
    print(f"Tokens per experiment: {BASE_CONFIG.get('max_tokens', 'N/A'):,}")

    if args.max_examples:
        print(f"Override: max_examples={args.max_examples}")

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - Commands will not be executed")
    print(f"{'='*80}\n")

    # Confirm before starting
    if not args.dry_run:
        response = input("Start sweep? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Run all experiments
    results = []
    port = MASTER_PORT

    for i, combo in enumerate(combinations, 1):
        # Create parameter dict for this experiment
        params = BASE_CONFIG.copy()
        for name, value in zip(param_names, combo):
            params[name] = value

        # Override max_examples if specified (for quick testing)
        if args.max_examples:
            params["max_examples"] = args.max_examples

        # Generate run name
        run_name = generate_run_name(params)

        print(f"\n{'#'*80}")
        print(f"# Experiment {i}/{len(combinations)}")
        print(f"{'#'*80}")

        # Run experiment
        success = run_experiment(params, run_name, args.gpus, port, args.dry_run)
        results.append({
            "run_name": run_name,
            "expansion_factor": params["expansion_factor"],
            "k": params["k"],
            "success": success,
        })

        # Increment port to avoid conflicts
        port += 1

        # Stop on failure unless continue-on-error is set
        if not success and not args.continue_on_error and not args.dry_run:
            print("\n⚠️  Stopping sweep due to failure. Use --continue-on-error to continue.")
            break

        # Small delay between experiments
        if not args.dry_run and i < len(combinations):
            print("\nWaiting 5 seconds before next experiment...")
            time.sleep(5)

    # Summary
    print("\n" + "="*80)
    print("Sweep Summary")
    print("="*80)

    if args.dry_run:
        print(f"Generated {len(combinations)} experiment configurations")
    else:
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        print(f"Completed: {len(results)}/{len(combinations)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        print("\nResults:")
        for r in results:
            status = "✓" if r["success"] else "✗"
            print(f"  {status} {r['run_name']} (ef={r['expansion_factor']}, k={r['k']})")

        # Generate wandb comparison link if using wandb
        if BASE_CONFIG.get("log_to_wandb") and successful > 0:
            print("\n💡 Tip: Compare runs in WandB:")
            print("   1. Go to your WandB project")
            print("   2. Select all runs starting with 'sweep_'")
            print("   3. Click 'Compare' to see side-by-side metrics")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
