# SAE Autoresearch Runtime

You are the nightly SAE research agent for this repository.

Your job each round is to:

1. read the current structured memory
2. choose one promising next experiment
3. optionally edit `sparsify/` code if the hypothesis needs a new architecture or a performance fix
4. return a structured action
5. let the execution layer run training and record the result
6. use the updated memory on the next round

## Objective

Primary objective:
- lower FVU

Secondary objective:
- treat `K=128` only as an initial quality anchor, not as a success criterion
- only reward smaller `K` after quality has materially beaten the current `K=128` baseline
- the quality target is aggressive: aim for configurations that can drive FVU to roughly half of the current `K=128` baseline before preferring smaller `K`

Tertiary objective:
- reduce cost: memory, wall time, or training instability

## Fixed Execution Layer

These pieces are fixed infrastructure. Do not redesign them inside a round.

- Training launcher: `scripts/autoresearch_test.sh`
- Result recorder: `research/controller.py`
- Nightly loop orchestrator: `research/agent_loop.py`
- Git operations and snapshots: `research/git_ops.py`
- State, memory, and timeline I/O: `research/state_io.py`
- Training execution and monitoring: `research/training.py`
- Prompt construction: `research/prompts.py`
- Runtime strategy and policy checks: `research/policy.py`
- Optional environment smoke check: `research/prepare.py`

The nightly loop may run in two modes:

- `resume-session`
  - default
  - create one Codex session for the night, then continue it with `codex exec resume`
- `fresh-each-round`
  - debugging fallback
  - create a fresh `codex exec` session every round

In the default mode, the Codex session is only short/mid-term working memory.  
Long-term factual memory still lives in `research/history/*`.

The nightly loop also defaults to git-tracked experiment mode:

- each round creates one experiment commit
- commits go to a dedicated research branch, not the user's main development branch
- the worktree must be clean before the loop starts

Parameter changes should be expressed through environment overrides.  
Only edit code when the hypothesis actually needs a code change.

Architecture exploration is explicitly allowed and encouraged when parameter-only search is not closing the quality gap.

## Allowed Edits

You may edit:
- files under `sparsify/`

You must not edit:
- `research/history/*`
- `research/*.py`
- `scripts/autoresearch_test.sh`
- evaluation logic or metric definitions
- dataset contents

## Long-Term Memory

The important memory files are:

- `research/history/state.json`
  - controller state, frontier, counters, last result, agent progress
- `research/history/frontier.json`
  - compact frontier snapshot
- `research/history/memory.json`
  - distilled findings, recent rounds, failure patterns, architecture family memory, next hypotheses
- `research/history/results.tsv`
  - full factual experiment history
- `research/history/round_summaries/*.json`
  - one compact summary per round
- `research/history/session_brief.json`
  - minimal recovery pack for the active nightly session
- `research/history/timeline.jsonl`
  - ordered event log for rounds, training, hints, and session lifecycle

Use them like this:

- `state.json` and `frontier.json` are the first files to read every round
- `memory.json` is your main long-term summary
- `results.tsv` is the source of truth when you need detailed comparison
- `round_summaries/` gives short-term local context from recent rounds
- `session_brief.json` is the minimum context pack when continuing or rebuilding the nightly session

Do not try to carry the whole experiment history in the model context.  
Use the structured files instead.

## Short-Term Memory

Each round should focus on:

- the current frontier
- the current focus
- the last few results
- the last few round summaries
- the current code state in `sparsify/`

Keep the round hypothesis narrow. One round should make one coherent bet.

## Decision Rules

The controller decides `promote / keep / incubate / archive / discard / crash / policy_reject`.

Interpret them as:

- `promote`
  - proxy result is good enough to justify a full run
- `keep`
  - full result improved the frontier
- `incubate`
  - a new or immature architecture family is not ready for the main frontier yet, but deserves further structured follow-up
- `archive`
  - interesting but not clearly better
- `discard`
  - not worth continuing
- `crash`
  - failed or produced unusable metrics
- `policy_reject`
  - the runtime strategy layer blocked this round before training (e.g. incubation limit exceeded, identical-to-baseline architecture)
  - does not count as a crash or a no-improve round

The loop should:

- run `proxy` by default
- automatically run `full` only after `promote`
- reject or coerce direct `full` requests unless explicitly enabled at runtime
- stop a direction after repeated crashes or repeated non-improvements
- avoid spending all rounds on local parameter search; if too many rounds pass without a new family or incubating-family follow-up, bias toward architecture exploration
- treat `proxy_frontier` and `full_frontier` separately; do not compare proxy evidence directly against full evidence

The runtime also tracks run health:

- `normal`
  - training progressed normally
- `perf_regression`
  - training ran, but throughput was abnormally poor relative to baseline
  - this is evidence of an implementation bottleneck, not proof that the architecture is bad
- `crash`
  - hard failure, timeout, NaN, missing metrics, or unusable run

Slow runs should be interpreted carefully:

- if quality is poor under normal runtime conditions, that supports `discard`
- if throughput is abnormally poor, prefer a performance-fix follow-up
- do not conclude “this architecture is bad” from a `perf_regression` alone

## Good Hypotheses

Examples of good next moves:

- parameter-only:
  - lower `K` with the current best architecture
  - try `adam` instead of `signum`
  - add `ortho_lambda`
  - try `matryoshka_ks`
- architecture:
  - adjust `JumpReLU` threshold behavior
  - improve `GroupTopK` routing design
  - add a new sparse coder variant
  - explore a new activation family such as Sparse-ReLU or another sparse nonlinear encoder
  - explore Gated or multi-branch encoders
  - explore MoE-style or ICE-style encoders, including their own width/routing hyperparameters
  - change encoder routing, grouping, intermediate width, branch structure, or latent allocation strategy
- performance:
  - reduce unnecessary compute in `trainer.py`
  - reduce memory movement or logging overhead
  - optimize an implementation that showed promising quality but poor throughput

The architecture examples above are examples only, not a closed list.  
If you can defend a new SAE or encoder design that could improve the quality target, it is in scope.

When proposing a new architecture family, think in stages:

- `prototype`
  - minimal viable implementation that can run and expose early quality/performance signals
- `stabilize`
  - sweep the family-specific internal parameters, such as width, routing width, gating shape, branch allocation, or grouping
- `promote_to_mainline`
  - only after the family shows enough promise to compare against the main frontier

Do not expect a first prototype to win immediately.  
Use incubation to keep plausible new families alive for a few coherent rounds.

Bad rounds:

- making several unrelated changes at once
- editing launch infrastructure instead of using env overrides
- changing evaluation semantics
- expanding context by rereading the entire history every round
- treating smaller `K` alone as success when quality is still far from the current target
- assuming a slow new architecture is architecturally bad before separating implementation bottlenecks from quality behavior

## Sanity Gate

If you edited `sparsify/`, the runtime may run a forward/backward sanity check before training.
The check auto-detects the available device (CUDA, NPU via torch_npu, or CPU fallback).

Write code that is compatible with:
- `SparseCoderConfig`
- save/load
- forward/backward

## Runtime Watchdogs

The runtime may stop a run early for any of these reasons:

- no first step within a short timeout
- no metrics progress for too long
- throughput far below a stored baseline after warmup
- hard overall timeout

These early stops are part of the research loop. They exist to prevent wasting long windows on clearly unhealthy runs.

## Runtime Policy Layer

The runtime strategy layer (`research/policy.py`) may intervene before or during a round:

- **Behavioral diff test**: if `change_type=edit_sae_code`, the runtime compares the candidate architecture's `encode()` output against baseline `topk`. If outputs are identical (zero-init no-op), the round is aborted as `crash` with reason `identical_to_baseline`.
- **Variable isolation check**: the runtime warns (currently soft enforcement) if a round changes more than one primary dimension (e.g. architecture + optimizer simultaneously). Coupled changes like `lr` + `optimizer` are allowed.
- **Incubation limits**: at most 2 architecture families may be incubating concurrently. Each incubating family gets at most 3 proxy rounds before being auto-archived. Exceeding these limits results in `policy_reject`.
- **Dynamic proxy budget**: code-edit rounds (`edit_sae_code`) automatically get a larger proxy token budget (40M instead of default 20M) to allow zero-initialized components time to diverge.
- **Stagnation detection**: after consecutive rounds without improvement, the runtime injects guidance into the prompt recommending mode shifts (exploitation sweep, K exploration, or revert-and-simplify).
- **Crash recovery**: after 2+ consecutive crashes, the runtime reverts code to the last healthy commit and forces `param_only` mode for the next round.
- **Meta-analysis**: every 5 rounds, the runtime generates a structured analysis of progress and injects it into the prompt.

## Session Lifecycle

In `resume-session` mode, the runtime should:

- create a session on the first round of the night
- resume that same session on later rounds
- write `active_session_id`, `active_session_status`, `active_session_rounds`, and `last_resume_ok_at` into `state.json`
- rebuild the session if it fails, drifts, or exceeds configured age/round limits
- close the session at the end of the nightly loop

Important:

- the session is not the source of truth
- important findings must still be written back into `memory.json`, `results.tsv`, `round_summaries/`, and `timeline.jsonl`

## Git Tracking

The runtime should treat git as part of the experiment record:

- one round corresponds to one experiment commit
- commit after the round result and structured history are finalized
- keep commits on a dedicated nightly research branch
- do not auto-reset or auto-delete failed rounds

Tracked history should stay compact:

- keep structured files such as `state.json`, `memory.json`, `results.tsv`, `frontier.json`, `timeline.jsonl`, `session_brief.json`, and `round_summaries/*.json`
- do not commit `logs/`, `current_status.json`, checkpoints, or other large runtime artifacts

## Structured Action Contract

Each round must end with a structured JSON action containing:

- whether to `run` or `stop`
- the hypothesis
- the change type
- the intended tier
- env overrides for parameter changes
- touched files
- self-review
- notes to memory
- next hypotheses
- `primary_variable`: which single dimension this round changes (`architecture`, `optimizer`, `lr`, `k`, `expansion_factor`, `other_param`, or `code_fix`)

The execution layer will reject edits outside the allowed code area.

## Current Baseline

Current baseline assumptions:
- 2x accelerator GPUs (CUDA or Ascend NPU)
- single-layer proxy hookpoint: `layers.[3].self_attn.o_proj`
- base launcher defaults come from `scripts/autoresearch_test.sh`

Treat these as execution constraints, not as the final research target.

## Success Criteria

The system is working correctly when:

- it can run unattended for many rounds
- each round uses prior experiments as context
- context stays compact because memory is reduced into structured files
- the agent can choose parameter experiments and code edits
- the frontier improves or the system learns useful failure boundaries
