# LUTurbo Compensation Hotdims Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runtime-oriented hot-dimension statistics package for LUTurbo online compensation, plus offline summary/plot tooling, while keeping the main inference path near-zero overhead when stats are disabled.

**Architecture:** Add a standalone `experiments/compensation_hotdims/` package with a no-op collector, a real runtime collector that consumes already-selected input dims from the runtime, and offline summarization/plot entry points. Keep the runtime API thin and self-contained so the real LUTurbo runtime can call it without pulling in training code, and cover the behavior with focused unit tests.

**Tech Stack:** Python 3.12, PyTorch tensors for runtime-facing APIs, JSON/JSONL outputs, NumPy for summary metrics, Matplotlib for plots, Pytest for tests.

---

## File Structure

- Create: `experiments/compensation_hotdims/__init__.py`
- Create: `experiments/compensation_hotdims/schema.py`
- Create: `experiments/compensation_hotdims/runtime_stats.py`
- Create: `experiments/compensation_hotdims/summarize.py`
- Create: `experiments/compensation_hotdims/analyze_runtime_dump.py`
- Create: `experiments/compensation_hotdims/plot.py`
- Create: `experiments/compensation_hotdims/README.md`
- Create: `tests/test_compensation_hotdims.py`

## Task 1: Runtime Collector Contract

**Files:**
- Create: `tests/test_compensation_hotdims.py`
- Create: `experiments/compensation_hotdims/__init__.py`
- Create: `experiments/compensation_hotdims/schema.py`
- Create: `experiments/compensation_hotdims/runtime_stats.py`

- [ ] **Step 1: Write the failing tests for the no-op path and the runtime-facing collector API**

```python
def test_noop_collector_ignores_records(tmp_path: Path):
    collector = build_hotdim_collector(
        enabled=False,
        output_dir=tmp_path / "comp_hotdims",
        layer_input_dims={"layers.0.self_attn.q_proj": 8},
    )
    collector.record_batch(
        "layers.0.self_attn.q_proj",
        selected_input_dims=torch.tensor([[1, 3], [2, 4]], dtype=torch.int64),
    )
    collector.finalize()
    assert not (tmp_path / "comp_hotdims").exists()


def test_runtime_collector_tracks_counts_hist_and_window_snapshot(tmp_path: Path):
    collector = build_hotdim_collector(
        enabled=True,
        output_dir=tmp_path / "comp_hotdims",
        layer_input_dims={"layers.0.self_attn.q_proj": 8},
        window_tokens=2,
        trace_sample_rate=0.0,
    )
    collector.record_batch(
        "layers.0.self_attn.q_proj",
        selected_input_dims=torch.tensor([[1, 3, 7], [1, 4, 7]], dtype=torch.int64),
        selected_counts=torch.tensor([3, 2], dtype=torch.int64),
    )
    collector.finalize()

    summary = json.loads((tmp_path / "comp_hotdims/layers.0.self_attn.q_proj/summary.json").read_text())
    assert summary["token_count"] == 2
    assert summary["total_selected_events"] == 5
    assert summary["selected_count_hist"][1] == 2
    assert summary["selected_count_hist"][7] == 1
    assert summary["selected_k_hist"] == {"2": 1, "3": 1}
```

- [ ] **Step 2: Run the tests to verify they fail for the missing package**

Run: `pytest tests/test_compensation_hotdims.py -q`
Expected: FAIL with import or attribute errors for `build_hotdim_collector`

- [ ] **Step 3: Implement the minimal collector and schema types**

```python
class NoOpCompensationHotdimCollector:
    def record_batch(self, *args, **kwargs) -> None:
        return None

    def finalize(self) -> None:
        return None


def build_hotdim_collector(*, enabled: bool, **kwargs):
    if not enabled:
        return NoOpCompensationHotdimCollector()
    return CompensationHotdimCollector(**kwargs)
```

- [ ] **Step 4: Run the tests to verify the collector API passes**

Run: `pytest tests/test_compensation_hotdims.py -q`
Expected: PASS for the no-op and count/histogram collector cases

- [ ] **Step 5: Commit the collector contract**

```bash
git add tests/test_compensation_hotdims.py experiments/compensation_hotdims/__init__.py experiments/compensation_hotdims/schema.py experiments/compensation_hotdims/runtime_stats.py
git commit -m "Add compensation hotdim runtime collector"
```

## Task 2: Summary Metrics and JSON/JSONL Outputs

**Files:**
- Modify: `tests/test_compensation_hotdims.py`
- Modify: `experiments/compensation_hotdims/runtime_stats.py`
- Modify: `experiments/compensation_hotdims/schema.py`

- [ ] **Step 1: Add failing tests for coverage, hotness ratio, score accumulation, and window summaries**

```python
def test_runtime_collector_writes_summary_with_concentration_metrics(tmp_path: Path):
    collector = build_hotdim_collector(
        enabled=True,
        output_dir=tmp_path / "comp_hotdims",
        layer_input_dims={"layers.1.self_attn.q_proj": 8},
        window_tokens=2,
        trace_sample_rate=0.0,
        include_scores=True,
    )
    collector.record_batch(
        "layers.1.self_attn.q_proj",
        selected_input_dims=torch.tensor([[2, 5], [2, 7]], dtype=torch.int64),
        selected_scores=torch.tensor([[0.9, 0.1], [0.7, 0.2]], dtype=torch.float32),
    )
    collector.finalize()

    summary = json.loads((tmp_path / "comp_hotdims/layers.1.self_attn.q_proj/summary.json").read_text())
    assert summary["coverage_by_rank"]["top_1"] == 0.5
    assert summary["top_dims_by_count"][0]["dim"] == 2
    assert summary["top_dims_by_score"][0]["dim"] == 2
    assert summary["top_dims_by_count"][0]["hotness_ratio"] > 1.0

    windows = [json.loads(line) for line in (tmp_path / "comp_hotdims/layers.1.self_attn.q_proj/windows.jsonl").read_text().splitlines()]
    assert len(windows) == 1
    assert windows[0]["coverage_by_rank"]["top_4"] == 1.0
```

- [ ] **Step 2: Run the tests to verify the new summary expectations fail**

Run: `pytest tests/test_compensation_hotdims.py -q`
Expected: FAIL on missing summary fields or incorrect metric calculations

- [ ] **Step 3: Implement summary metric helpers and flush logic**

```python
def compute_coverage_by_rank(sorted_counts: np.ndarray, total_events: int) -> dict[str, float]:
    return {
        "top_1": float(sorted_counts[:1].sum() / total_events) if total_events else 0.0,
        "top_4": float(sorted_counts[:4].sum() / total_events) if total_events else 0.0,
        "top_16": float(sorted_counts[:16].sum() / total_events) if total_events else 0.0,
        "top_64": float(sorted_counts[:64].sum() / total_events) if total_events else 0.0,
        "top_128": float(sorted_counts[:128].sum() / total_events) if total_events else 0.0,
    }
```

- [ ] **Step 4: Run the tests to verify the JSON/JSONL summaries pass**

Run: `pytest tests/test_compensation_hotdims.py -q`
Expected: PASS for count, score, and window-summary behavior

- [ ] **Step 5: Commit the summary metrics**

```bash
git add tests/test_compensation_hotdims.py experiments/compensation_hotdims/runtime_stats.py experiments/compensation_hotdims/schema.py
git commit -m "Add compensation hotdim summary metrics"
```

## Task 3: Offline Summarization CLI

**Files:**
- Modify: `tests/test_compensation_hotdims.py`
- Create: `experiments/compensation_hotdims/summarize.py`
- Create: `experiments/compensation_hotdims/analyze_runtime_dump.py`

- [ ] **Step 1: Add failing tests for offline TSV/JSON summary generation**

```python
def test_analyze_runtime_dump_writes_layer_and_global_rankings(tmp_path: Path):
    out_dir = tmp_path / "comp_hotdims"
    collector = build_hotdim_collector(
        enabled=True,
        output_dir=out_dir,
        layer_input_dims={
            "layers.0.self_attn.q_proj": 8,
            "layers.1.self_attn.q_proj": 8,
        },
        window_tokens=2,
        trace_sample_rate=0.0,
    )
    collector.record_batch("layers.0.self_attn.q_proj", torch.tensor([[1, 2], [1, 3]], dtype=torch.int64))
    collector.record_batch("layers.1.self_attn.q_proj", torch.tensor([[4, 5], [4, 6]], dtype=torch.int64))
    collector.finalize()

    exit_code = main(["--input_dir", str(out_dir), "--output_dir", str(tmp_path / "analysis")])
    assert exit_code == 0
    assert (tmp_path / "analysis/layer_summary.tsv").exists()
    assert (tmp_path / "analysis/global_layer_ranking.tsv").exists()
```

- [ ] **Step 2: Run the tests to verify the CLI is missing**

Run: `pytest tests/test_compensation_hotdims.py -q`
Expected: FAIL with missing module or missing `main`

- [ ] **Step 3: Implement the offline summarizer and CLI wrapper**

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)
    write_analysis_outputs(Path(args.input_dir), Path(args.output_dir))
    return 0
```

- [ ] **Step 4: Run the tests to verify offline summaries pass**

Run: `pytest tests/test_compensation_hotdims.py -q`
Expected: PASS for runtime dump analysis outputs

- [ ] **Step 5: Commit the offline summarizer**

```bash
git add tests/test_compensation_hotdims.py experiments/compensation_hotdims/summarize.py experiments/compensation_hotdims/analyze_runtime_dump.py
git commit -m "Add compensation hotdim offline analysis"
```

## Task 4: Plotting and Usage Documentation

**Files:**
- Modify: `tests/test_compensation_hotdims.py`
- Create: `experiments/compensation_hotdims/plot.py`
- Create: `experiments/compensation_hotdims/README.md`

- [ ] **Step 1: Add a failing smoke test for plot generation**

```python
def test_plotting_smoke_writes_pngs(tmp_path: Path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "layer_summary.tsv").write_text(
        "layer_name\\ttoken_count\\ttop_64_coverage\\tgini\\n"
        "layers.0.self_attn.q_proj\\t2\\t1.0\\t0.5\\n"
    )
    exit_code = plot_main(["--analysis_dir", str(analysis_dir), "--output_dir", str(tmp_path / "plots")])
    assert exit_code == 0
    assert (tmp_path / "plots/coverage_by_layer.png").exists()
```

- [ ] **Step 2: Run the tests to verify plotting is missing**

Run: `pytest tests/test_compensation_hotdims.py -q`
Expected: FAIL on missing plot CLI

- [ ] **Step 3: Implement minimal plotting and README usage docs**

```python
def plot_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)
    write_default_plots(Path(args.analysis_dir), Path(args.output_dir))
    return 0
```

- [ ] **Step 4: Run the tests to verify the package is green**

Run: `pytest tests/test_compensation_hotdims.py -q`
Expected: PASS for collector, analysis, and plotting smoke tests

- [ ] **Step 5: Commit the plotting/docs task**

```bash
git add tests/test_compensation_hotdims.py experiments/compensation_hotdims/plot.py experiments/compensation_hotdims/README.md
git commit -m "Add compensation hotdim plots and docs"
```

## Final Verification

**Files:**
- Verify: `tests/test_compensation_hotdims.py`
- Verify: `experiments/compensation_hotdims/*.py`

- [ ] **Step 1: Run the focused test suite**

Run: `pytest tests/test_compensation_hotdims.py -q`
Expected: PASS

- [ ] **Step 2: Byte-compile the new package**

Run: `python -m py_compile experiments/compensation_hotdims/*.py`
Expected: no output

- [ ] **Step 3: Inspect git status**

Run: `git status --short`
Expected: only the intended experiment package, tests, and plan/spec docs changed

