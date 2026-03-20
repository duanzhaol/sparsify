"""
One-time environment validation and data check for SAE autoresearch.

Usage:
    python prepare.py              # validate everything
    python prepare.py --smoke      # also run a minimal forward+backward smoke test

This script is READ-ONLY for the agent. Do not modify.
"""

import argparse
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Fixed paths (match train_qwen3_0p6b.sh defaults)
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.expanduser("~/models/Qwen3-0.6B")
DATASET_PATH = os.path.expanduser(
    "~/fineweb-edu/sample/10BT-tokenized-qwen3-2048"
)
ELBOW_THRESHOLD_PATH = os.path.expanduser(
    "~/sparsify/thresholds/Qwen3-0.6B/thresholds_q.json"
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def check_paths():
    """Verify model, dataset, and thresholds exist."""
    ok = True
    for label, path in [
        ("Model", MODEL_PATH),
        ("Dataset", DATASET_PATH),
        ("Elbow thresholds", ELBOW_THRESHOLD_PATH),
    ]:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}: {path}")
        if not exists:
            ok = False
    return ok


def check_gpu():
    """Verify CUDA GPUs are available."""
    if not torch.cuda.is_available():
        print("  [MISSING] No CUDA GPUs detected")
        return False
    n = torch.cuda.device_count()
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  [OK] GPU {i}: {name}, {mem:.1f}GB")
    return True


def check_sparsify():
    """Verify sparsify is importable."""
    try:
        import sparsify  # noqa: F401
        from sparsify.config import SparseCoderConfig, TrainConfig  # noqa: F401
        from sparsify.trainer import Trainer  # noqa: F401

        print("  [OK] sparsify importable")
        return True
    except ImportError as e:
        print(f"  [MISSING] sparsify import failed: {e}")
        return False


def smoke_test():
    """Run a minimal SAE forward+backward to verify everything works."""
    print("\nSmoke test: minimal SAE forward+backward...")
    t0 = time.time()

    from sparsify import SparseCoder
    from sparsify.config import SparseCoderConfig

    cfg = SparseCoderConfig(expansion_factor=8, k=32)
    sae = SparseCoder(1024, cfg, device="cuda", dtype=torch.float32)

    x = torch.randn(4, 1024, device="cuda")
    out = sae(x)
    out.fvu.backward()

    dt = time.time() - t0
    print(f"  [OK] forward+backward in {dt:.2f}s, FVU={out.fvu.item():.4f}")

    # Test save/load roundtrip
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        sae.save_to_disk(tmpdir)
        sae2 = SparseCoder.load_any(tmpdir, device="cuda")
        out2 = sae2(x)
        print(f"  [OK] save/load roundtrip, FVU={out2.fvu.item():.4f}")

    return True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate autoresearch environment")
    parser.add_argument(
        "--smoke", action="store_true", help="Run minimal forward/backward smoke test"
    )
    args = parser.parse_args()

    print("=== SAE Autoresearch Environment Check ===\n")

    print("Paths:")
    paths_ok = check_paths()

    print("\nGPU:")
    gpu_ok = check_gpu()

    print("\nSparsify:")
    sparsify_ok = check_sparsify()

    if not (paths_ok and gpu_ok and sparsify_ok):
        print("\n[FAIL] Environment check failed. Fix issues above.")
        sys.exit(1)

    if args.smoke:
        try:
            smoke_test()
        except Exception as e:
            print(f"\n[FAIL] Smoke test failed: {e}")
            sys.exit(1)

    print("\n[PASS] Environment ready for autoresearch.")
