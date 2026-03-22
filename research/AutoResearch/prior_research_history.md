# Prior Research History: SAE Architecture Search

## Document Status: Reference Only

**This document is a loose reference, not a source of truth.** The authoritative state is the live frontier maintained by the autoresearch loop (`state.json` frontier).

Reasons to treat this document with skepticism:
1. Many earlier conclusions turned out to be wrong (e.g., "EF<32 is always bad")
2. All FVU numbers were obtained under a different training token budget — absolute values are meaningless; only relative ordering within the same conditions has any signal
3. Any specific claim here should be verified by running your own experiments rather than taken at face value

Use this document for: starting priors, avoiding known dead ends, engineering checklists.
Do NOT use it for: definitive architecture rankings, fixed hyperparameter recipes, or skipping experiments you haven't run yourself.

---

## Overview

~240 experiments across ~30 architecture families. Key outcomes: 45 keep, 55 promote, 62 incubate, 48 discard, 27 crash.

The main finding is a **piecewise Pareto frontier** across three dimensions: **(K, EF, FVU)** — lower is better in all. No single architecture wins everywhere.

---

## Current Frontier

The frontier is tracked as a 3D Pareto surface over (K, EF, FVU). Each slot is keyed by `{K}_{EF}`.

| K | EF | Architecture | FVU | AUXK_ALPHA |
|---|---|---|---:|---|
| 1 | 12 | `lowrank_residual` | 0.1996 | 0.03125 |
| 2 | 12 | `lowrank_residual` | 0.1800 | 0.03125 |
| 4 | 12 | `lowrank_residual` | 0.1581 | 0.03125 |
| 6 | 12 | `lowrank_two_stage_residual` | 0.1488 | 0.03125 |
| 8 | 12 | `lowrank_two_stage_residual` | 0.1392 | 0.03125 |
| 16 | 12 | `routed_lowrank_two_stage_residual` | 0.1325 | 0.0 |
| 24 | 12 | `routed_lowrank_two_stage_residual` | 0.1225 | 0.0 |
| 32 | 12 | `lowrank_two_stage_residual` | 0.0963 | 0.0 |
| 40 | 12 | `lowrank_two_stage_residual` | 0.0903 | 0.0 |
| 48 | 32 | `lowrank_gated_residual` | 0.0874 | 0.0 |
| 56–128 | 32 | `lowrank_gated_residual` | 0.084–0.066 | 0.0 |

Key pattern: EF=12 families own K<=40; `lowrank_gated_residual` @ EF=32 owns K>=48.

---

## Frontier Architecture Families

### `lowrank_gated_residual` — high-K anchor
- Best at K>=48 with EF=32
- Best absolute FVU: 0.066 at K=128
- Not competitive at low K or low EF

### `lowrank_two_stage_residual` — low-K workhorse
- Best at K=6/8/32/40 with EF=12
- Disproved the early assumption that EF<32 is unviable for lowrank families

### `routed_lowrank_two_stage_residual` — mid-K transition
- Best at K=16/24 with EF=12
- Routing helps in the K=16–32 band but not at extremes

### `lowrank_residual` — ultra-low-K tail
- Best at K=1/2/4 with EF=12
- Also competitive at high K (0.068 at K=128 EF=32), but behind `lowrank_gated_residual`

---

## Robust Hyperparameters

- **Optimizer**: signum
- **LR**: 1.6e-3
- **High-K recipe**: `lowrank_gated_residual`, EF=32, AUXK_ALPHA=0.0
- **Low-K recipe**: EF=12 family (depends on K), AUXK_ALPHA unsettled

### AUXK_ALPHA caveat

The low-K frontier is split: K>=16 points used AUXK_ALPHA=0.0, K<=8 points used 0.03125 (due to recipe drift — agent omitted the override). This means cross-K comparisons in the low-K range are not perfectly controlled. A cleanup sweep under consistent AUXK_ALPHA would resolve this.

---

## Rejected Directions

**Do not retry** without a fundamentally new idea:

- **Support-selection gimmicks**: `batch_topk`, `adaptive_budget_topk`, `bucketed_topk`, `group_topk`, `factorized_topk` — all failed
- **Whitening/preconditioning**: `whitened_topk`, `whitened_lowrank_*`, `preconditioned_topk` — weak or unstable
- **Discrete/codebook**: `codebook_topk`, `product_codebook_topk`, `residual_vq`, `residual_vq_topk` — no compelling result
- **Other dead ends**: `routed`, `multi_branch_gated`, `residual_topk`, `lowrank_grouped_residual`

**Stepping stones** (contributed ideas but now outclassed):
- `bucketed_lowrank_residual` — first proof that EF=12 low-K lane works
- `gated`, `jumprelu`, `two_stage_residual` — outclassed by lowrank variants

---

## Engineering Lessons

1. **Most crashes were integration bugs, not scientific negatives**: missing config registration, missing factory dispatch, DDP unused-parameter errors, malformed env_overrides. Don't interpret these as evidence against the architecture.

2. **Architecture integration checklist is mandatory** for new families:
   - Config whitelist in `sparsify/config.py`
   - Factory dispatch in `sparsify/sparse_coder.py`
   - Behavioral diff test (encode output must differ from baseline topk)
   - Sanity backward pass

3. **Always pin all recipe fields explicitly**: ARCHITECTURE, K, EXPANSION_FACTOR, OPTIMIZER, LR, AUXK_ALPHA. Omitting any causes silent fallback to launcher defaults (this caused the AUXK_ALPHA drift).

4. **Controller/prompt structure matters more than backend choice**: same-(K,EF) comparison emphasis, in-round repair loops, and operator hints drove the biggest productivity gains.

---

## Recommended Starting Priors

When starting a new exploration:

1. **Use EF as a frontier dimension** alongside K and FVU — not a fixed constant
2. For high-K quality: start from `lowrank_gated_residual`, EF=32, signum, LR=1.6e-3, AUXK_ALPHA=0.0
3. For low-K tradeoff: start from EF=12 families (`lowrank_two_stage_residual`, `lowrank_residual`), signum, LR=1.6e-3
4. Explicitly decide AUXK_ALPHA per regime — do not let it drift
5. Change one variable per round (architecture+EF coupling is acceptable; K+EF is not)
