"""Benchmark scatter_add_ alternatives on Ascend NPU.

Measures three timing modes to capture real-world performance:
  1. NPU Event timing  — device-side elapsed time (like CUDA events)
  2. Pipeline timing    — queue N ops back-to-back without sync, sync once
  3. Sync timing        — synchronize after each iteration (original method)

The key difference: CPU fallback causes pipeline stalls that only show up
in modes 1 and 2, but are masked by mode 3's per-iteration sync.
"""

import time
import torch
import torch_npu


# ── Approaches ─────────────────────────────────────────────────────

def baseline_scatter_matmul(indices, acts, W_T):
    """Current: scatter_add_(dim=1) + matmul. Falls back to AI_CPU."""
    N, k = indices.shape
    M = W_T.shape[0]
    S = torch.zeros(N, M, dtype=acts.dtype, device=acts.device)
    S.scatter_add_(1, indices.long(), acts)
    return S @ W_T


def approach_a_index_put(indices, acts, W_T):
    """index_put_ with accumulate=True + matmul."""
    N, k = indices.shape
    M = W_T.shape[0]
    rows = torch.arange(N, device=acts.device).unsqueeze(1).expand_as(indices)
    S = torch.zeros(N, M, dtype=acts.dtype, device=acts.device)
    S.index_put_((rows, indices.long()), acts, accumulate=True)
    return S @ W_T


def approach_b_gather_mul_sum(indices, acts, W_T):
    """Gather rows + elementwise mul + sum. No dense S matrix."""
    selected = W_T[indices]  # (N, k, d_in)
    return (acts.unsqueeze(-1) * selected).sum(dim=1)


def approach_c_gather_bmm(indices, acts, W_T):
    """Gather rows + batch matmul. Uses AI_CORE Cube."""
    selected = W_T[indices]  # (N, k, d_in)
    return torch.bmm(acts.unsqueeze(1), selected).squeeze(1)


APPROACHES = {
    "baseline (scatter+mm)": baseline_scatter_matmul,
    "A (index_put+mm)":      approach_a_index_put,
    "B (gather+mul+sum)":    approach_b_gather_mul_sum,
    "C (gather+bmm)":        approach_c_gather_bmm,
}


# ── Timing helpers ─────────────────────────────────────────────────

def time_with_events(fn, args, warmup=10, repeats=50):
    """Device-side timing using NPU events (like CUDA events)."""
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()

    start_evt = torch.npu.Event(enable_timing=True)
    end_evt = torch.npu.Event(enable_timing=True)

    times = []
    for _ in range(repeats):
        start_evt.record()
        fn(*args)
        end_evt.record()
        torch.npu.synchronize()
        times.append(start_evt.elapsed_time(end_evt))  # ms

    return sum(times) / len(times)


def time_pipeline(fn, args, warmup=10, repeats=100):
    """Pipeline timing: queue many ops, sync once at end."""
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    torch.npu.synchronize()
    total = (time.perf_counter() - start) * 1000  # ms

    return total / repeats


def time_sync(fn, args, warmup=10, repeats=50):
    """Sync timing: synchronize after each iteration (original method)."""
    for _ in range(warmup):
        fn(*args)
        torch.npu.synchronize()

    times = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        fn(*args)
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times)


# ── Workloads ──────────────────────────────────────────────────────

WORKLOADS = [
    ("small",  1024, 4096,  32,   64),
    ("medium", 4096, 8192,  32, 1024),
    ("large",  8192, 16384, 64, 1024),
]


def check_correctness(fn, args, ref_out, name):
    """Check output matches reference within tolerance."""
    out = fn(*args)
    max_diff = (out - ref_out).abs().max().item()
    ok = max_diff < 1e-3
    return ok, max_diff


def main():
    device = torch.npu.get_device_name(0)
    print(f"Device: {device}")
    print()

    timing_modes = [
        ("event",    time_with_events),
        ("pipeline", time_pipeline),
        ("sync",     time_sync),
    ]

    for wl_name, N, M, k, d_in in WORKLOADS:
        W_T = torch.randn(M, d_in, device="npu")
        indices = torch.randint(0, M, (N, k), device="npu")
        acts = torch.randn(N, k, device="npu")
        args = (indices, acts, W_T)

        # Reference output
        ref_out = baseline_scatter_matmul(*args)

        print(f"{'=' * 90}")
        print(f"  Workload: {wl_name}  (N={N}, M={M}, k={k}, d_in={d_in})")
        print(f"  S matrix: {N}x{M} = {N*M*4/1024/1024:.0f}MB  |  "
              f"Gather: {N}x{k}x{d_in} = {N*k*d_in*4/1024/1024:.0f}MB")
        print(f"{'=' * 90}")

        # Header
        print(f"  {'Approach':<24s}", end="")
        for mode_name, _ in timing_modes:
            print(f"  {mode_name:>10s}", end="")
        print(f"  {'correct':>8s}  {'max_diff':>10s}")
        print(f"  {'-' * 84}")

        for approach_name, fn in APPROACHES.items():
            ok, max_diff = check_correctness(fn, args, ref_out, approach_name)

            print(f"  {approach_name:<24s}", end="")
            for mode_name, timer_fn in timing_modes:
                t = timer_fn(fn, args)
                print(f"  {t:>8.3f}ms", end="")
            print(f"  {'PASS' if ok else 'FAIL':>8s}  {max_diff:>10.2e}")

        print()


if __name__ == "__main__":
    main()
