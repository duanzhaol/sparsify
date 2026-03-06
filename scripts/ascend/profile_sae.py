"""Profile SAE training operators on Ascend NPU.

Usage:
    python scripts/ascend/profile_sae.py

Produces:
    prof_output/ directory with:
    - operator_details.csv   (torch op name <-> NPU kernel mapping)
    - op_statistic.csv       (CANN kernel time aggregation)
    - op_summary*.csv        (per-invocation kernel details)
    - trace.json             (Chrome trace, open in chrome://tracing)
"""

import csv
import glob
import os
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import torch_npu.profiler as npu_profiler
from torch_npu.profiler import ProfilerActivity

from sparsify.config import SparseCoderConfig
from sparsify.sparse_coder import SparseCoder

# ---------------------------------------------------------------------------
# Config - adjust these to match your real training
# ---------------------------------------------------------------------------
D_IN = 512           # Qwen3-0.6B q_proj input dim
EXPANSION = 8
K = 128
BATCH = 2
SEQ_LEN = 2048       # tokens per sequence
N = BATCH * SEQ_LEN  # total tokens per step

WARMUP_STEPS = 5     # let CANN JIT-compile kernels
PROFILE_STEPS = 3    # steps to actually profile

DEVICE = "npu:0"
OUT_DIR = Path("./prof_output")


# ---------------------------------------------------------------------------
# Helpers to pretty-print the generated CSVs
# ---------------------------------------------------------------------------

def _find_files(root, pattern):
    return sorted(Path(p) for p in glob.glob(str(root / "**" / pattern), recursive=True))


def _print_csv_table(csv_path, sort_col=None, max_rows=40, title=None):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print(f"  (empty: {csv_path})")
        return

    if sort_col and sort_col in rows[0]:
        try:
            rows.sort(key=lambda r: float(r[sort_col]), reverse=True)
        except (ValueError, TypeError):
            pass

    headers = list(rows[0].keys())
    col_widths = {h: len(h) for h in headers}
    display_rows = rows[:max_rows]
    for row in display_rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row.get(h, ""))))

    MAX_COL = 48
    col_widths = {h: min(w, MAX_COL) for h, w in col_widths.items()}

    header_line = " | ".join(h.ljust(col_widths[h])[:col_widths[h]] for h in headers)
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)

    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}")
    print(f"  Source: {csv_path}")
    print(f"  {header_line}")
    print(f"  {sep_line}")
    for row in display_rows:
        line = " | ".join(
            str(row.get(h, "")).ljust(col_widths[h])[:col_widths[h]]
            for h in headers
        )
        print(f"  {line}")
    if len(rows) > max_rows:
        print(f"  ... ({len(rows) - max_rows} more rows omitted)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.npu.set_device(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = SparseCoderConfig(expansion_factor=EXPANSION, k=K)
    sae = SparseCoder(D_IN, cfg, device=DEVICE, dtype=torch.float32)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    print(f"Warming up {WARMUP_STEPS} steps ...")
    for _ in range(WARMUP_STEPS):
        x = torch.randn(N, D_IN, device=DEVICE)
        out = sae(x)
        loss = out.fvu
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.npu.synchronize()

    print(f"Profiling {PROFILE_STEPS} steps ...")

    experimental_config = npu_profiler._ExperimentalConfig(
        profiler_level=npu_profiler.ProfilerLevel.Level1,
        aic_metrics=npu_profiler.AiCMetrics.PipeUtilization,
        export_type=npu_profiler.ExportType.Text,
    )

    schedule = npu_profiler.schedule(
        wait=0, warmup=1, active=PROFILE_STEPS, repeat=1,
    )

    with npu_profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        schedule=schedule,
        on_trace_ready=npu_profiler.tensorboard_trace_handler(str(OUT_DIR)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        experimental_config=experimental_config,
    ) as prof:
        for step in range(1 + PROFILE_STEPS):
            x = torch.randn(N, D_IN, device=DEVICE)
            out = sae(x)
            loss = out.fvu
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.npu.synchronize()
            prof.step()

    # ---- Generate Chrome trace JSON --------------------------------------
    #
    # torch_npu.profiler does NOT auto-generate chrome trace.
    # We do it in two ways:
    #   1) prof.export_chrome_trace() - uses the profiler's internal path
    #   2) Fallback: offline analyse() which generates msprof_*.json
    #
    print("\nGenerating Chrome trace ...")
    prof_dirs = sorted(OUT_DIR.glob("*_ascend_pt"), key=lambda p: p.stat().st_mtime)
    if prof_dirs:
        latest_prof = prof_dirs[-1]
        trace_json = latest_prof / "trace.json"
        try:
            prof.export_chrome_trace(str(trace_json))
            print(f"  Chrome trace saved: {trace_json}")
            print(f"  -> Open chrome://tracing and load this file.")
        except Exception as e:
            print(f"  export_chrome_trace failed: {e}")
            print(f"  Falling back to offline analyse ...")
            try:
                from torch_npu.profiler import analyse
                analyse(str(latest_prof))
                msprof_jsons = _find_files(latest_prof, "msprof_*.json")
                if msprof_jsons:
                    print(f"  Chrome trace generated:")
                    for j in msprof_jsons:
                        print(f"    {j}")
                    print(f"  -> Open chrome://tracing and load any of the JSON files.")
                else:
                    print(f"  analyse() completed but no msprof_*.json found.")
            except Exception as e2:
                print(f"  Offline analyse also failed: {e2}")
    else:
        print(f"  No *_ascend_pt directory found in {OUT_DIR}")

    # ---- Print CSV summaries ---------------------------------------------
    print(f"\nScanning output: {OUT_DIR.resolve()}")

    for csv_path in _find_files(OUT_DIR, "operator_details.csv"):
        _print_csv_table(
            csv_path,
            sort_col="Device Self Duration(us)",
            title="Operator Details (PyTorch op -> NPU kernel time)",
        )

    for csv_path in _find_files(OUT_DIR, "op_statistic*.csv"):
        _print_csv_table(
            csv_path,
            sort_col="Total Time(us)",
            title="Op Statistic (CANN kernel aggregation)",
        )

    for csv_path in _find_files(OUT_DIR, "op_summary*.csv"):
        _print_csv_table(
            csv_path,
            sort_col="Task Duration(us)",
            title="Op Summary (per-invocation kernel details)",
        )

    # ---- List all generated files ----------------------------------------
    all_files = sorted(f for f in OUT_DIR.rglob("*") if f.is_file())
    if all_files:
        print(f"\nAll generated files ({len(all_files)}):")
        for f in all_files:
            size_kb = f.stat().st_size / 1024
            print(f"  {f}  ({size_kb:.1f} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
