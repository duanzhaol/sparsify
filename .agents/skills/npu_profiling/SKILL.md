---
name: npu_profiling
description: Analyze Ascend NPU profiling data from msprof or torch_npu profiler outputs. Use when the user provides a profiling directory such as `PROF_*` or `prof_output/*`, asks to analyze NPU performance, compare profiles, identify AI_CPU fallback, communication overhead, kernel utilization, or training bottlenecks on Ascend.
---

# NPU Profiling

Analyze Ascend NPU profiling outputs with a workflow optimized for Codex.

## When to use

Use this skill when the user:
- provides a profiling directory containing `PROF_*`, `prof_output`, `mindstudio_profiler_output`, or `ASCEND_PROFILER_OUTPUT`
- asks for Ascend NPU performance analysis, bottleneck diagnosis, or optimization suggestions
- wants before/after profiling comparison
- wants to identify AI_CPU fallback, communication overhead, step wall time, or low NPU utilization

## Data layout

Typical `msprof` export:

```text
prof_output/PROF_xxx_yyyymmdd.../
└── mindstudio_profiler_output/
    ├── op_statistic_*.csv
    ├── op_summary_*.csv
    ├── task_time_*.csv
    ├── communication_statistic_*.csv
    ├── api_statistic_*.csv
    └── msprof_*.json
```

You may also see `torch_npu.profiler` outputs that contain `ASCEND_PROFILER_OUTPUT/` together with a nested `PROF_*` directory.

## Preferred workflow

1. Start with the bundled scripts under `scripts/` to extract structured findings.
2. Then synthesize a human report focused on wall time, bottlenecks, fallback, and optimization opportunities.
3. Do not only rank operators by total time; explain where step time really goes.

## Bundled scripts

Run scripts from this skill directory:

```bash
python .agents/skills/npu_profiling/scripts/full_analysis.py <PROF_DIR> [--output report.md]
python .agents/skills/npu_profiling/scripts/analyze_ops.py <PROF_DIR> [--top 20]
python .agents/skills/npu_profiling/scripts/analyze_timeline.py <PROF_DIR> [--skip-init-seconds 5]
python .agents/skills/npu_profiling/scripts/analyze_kernel_time.py <PROF_DIR>
python .agents/skills/npu_profiling/scripts/analyze_communication.py <PROF_DIR>
```

## What to answer

A good profiling answer should cover:
- step wall time and whether initialization has been excluded
- how much time is spent in LLM, SAE, communication, data movement, and idle/wait
- top expensive operators and whether they run on `AI_CORE`, `AI_VECTOR_CORE`, or `AI_CPU`
- CPU fallback and whether it likely triggers host-device synchronization
- communication long tails and synchronization symptoms
- concrete optimization ideas tied to the observed hotspot types

## Analysis heuristics

### Global-first

Always prioritize these questions:
- What is the steady-state single-step wall time?
- What percentage is actual compute vs wait vs communication?
- Are there `AI_CPU` fallbacks?
- Are high-cost vector ops actually matmul-like work that should move toward Cube?
- Is the system compute-bound, communication-bound, or launch/synchronization-bound?

### Operator ownership

Common ownership hints:
- `FlashAttentionScore` → LLM forward
- `EmbeddingBag` → SAE decode / sparse decode path
- `TopKV2` → SAE top-k selection
- `IndexPutV2`, `ScatterElementsV2` → SAE backward / scatter-style updates
- `allreduce` and similar → DDP communication
- `Cast`, `Transpose` → layout or dtype movement
- `Lerp` / `LerpV2` → optimizer update path

### MatMul shape classification

`MatMul` is shared by LLM and SAE, so inspect shape strings instead of assuming ownership.

Useful cues in this repo:
- shapes containing `3072` often point to LLM MLP work
- shapes containing `1024;1024` often point to Q/K/V/O projections
- shapes involving `d_sae`-like expansion dimensions often belong to SAE encode/decode

### Core types

- `AI_CORE`: Cube-heavy compute, typically desirable for dense matmul-like work
- `AI_VECTOR_CORE`: vector/scatter/elementwise-heavy work
- `MIX_AIC` / `MIX_AIV`: mixed kernels
- `AI_CPU`: unsupported or fallback path, usually a serious performance smell

## Typical bottlenecks

- `EmbeddingBag` dominates on vector cores
- `IndexPut` or `_embedding_bag_backward` falls back to CPU
- many tiny kernels inflate launch overhead
- communication max latency is much larger than average latency
- wait kernels (`NOTIFY_WAIT_SQE`, `EVENT_WAIT`) consume a large fraction of task time

## Repo-specific references

If needed, read these existing documents for local context before drawing conclusions:
- `docs/ascend/npu_profiling_analysis.md`
- `docs/ascend/sae_training_profiling_report_20260306.md`
- `docs/ascend/ascend_profling.md`
- `scripts/ascend/profile_sae.py`

## Output style

Keep the final answer short and actionable:
- start with the main bottleneck in one sentence
- quantify with a few key percentages or wall-time numbers
- list the top 3 to 5 findings
- end with concrete next optimizations to try
