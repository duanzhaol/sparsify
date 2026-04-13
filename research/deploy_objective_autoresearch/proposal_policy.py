from __future__ import annotations

from typing import Iterable

from .config import DynamicBounds, SearchConfig


def _round_to_step(value: int, step: int, minimum: int, maximum: int) -> int:
    rounded = int(round(value / step) * step)
    rounded = max(minimum, rounded)
    rounded = min(maximum, rounded)
    return rounded


def normalize_candidate_params(
    raw: dict[str, int | float],
    cfg: SearchConfig | None = None,
) -> dict[str, int | float]:
    bounds = cfg.bounds if cfg is not None else DynamicBounds()
    k = _round_to_step(int(raw.get("K", 64)), bounds.k_step, bounds.k_min, bounds.k_max)
    num_experts = _round_to_step(
        int(raw.get("NUM_EXPERTS", 256)),
        bounds.num_experts_step,
        bounds.num_experts_min,
        bounds.num_experts_max,
    )
    latents = _round_to_step(
        int(raw.get("LATENTS_PER_EXPERT", 64)),
        bounds.latents_per_expert_step,
        bounds.latents_per_expert_min,
        bounds.latents_per_expert_max,
    )
    active_experts = int(raw.get("ACTIVE_EXPERTS", 2))
    active_experts = max(bounds.active_experts_min, active_experts)
    active_experts = min(bounds.active_experts_max, active_experts)
    lr = float(raw.get("LR", 8e-4))
    lr = min(bounds.lr_max, max(bounds.lr_min, lr))
    auxk_alpha = float(raw.get("AUXK_ALPHA", 0.03125))
    auxk_alpha = min(bounds.auxk_alpha_max, max(bounds.auxk_alpha_min, auxk_alpha))
    return {
        "K": k,
        "NUM_EXPERTS": num_experts,
        "LATENTS_PER_EXPERT": latents,
        "ACTIVE_EXPERTS": active_experts,
        "LR": lr,
        "AUXK_ALPHA": auxk_alpha,
    }


def params_signature(params: dict[str, int | float]) -> str:
    return "|".join(f"{key}={params[key]}" for key in sorted(params))


def _candidate_pool(anchor: dict[str, int | float], cheaper_first: bool) -> list[dict[str, int | float]]:
    k = int(anchor["K"])
    experts = int(anchor["NUM_EXPERTS"])
    latents = int(anchor["LATENTS_PER_EXPERT"])
    active = int(anchor["ACTIVE_EXPERTS"])
    lr = float(anchor["LR"])
    auxk = float(anchor["AUXK_ALPHA"])

    cheaper = [
        {**anchor, "K": k - 16},
        {**anchor, "K": k - 8},
        {**anchor, "LATENTS_PER_EXPERT": latents - 16},
        {**anchor, "LATENTS_PER_EXPERT": latents - 8},
        {**anchor, "NUM_EXPERTS": experts - 64},
        {**anchor, "NUM_EXPERTS": experts - 32},
        {**anchor, "NUM_EXPERTS": experts + 32, "LATENTS_PER_EXPERT": latents - 16},
        {**anchor, "NUM_EXPERTS": experts + 64, "LATENTS_PER_EXPERT": latents - 16},
    ]
    quality = [
        {**anchor, "K": k + 8},
        {**anchor, "K": k + 16},
        {**anchor, "LATENTS_PER_EXPERT": latents + 8},
        {**anchor, "LATENTS_PER_EXPERT": latents + 16},
        {**anchor, "NUM_EXPERTS": experts + 32},
        {**anchor, "NUM_EXPERTS": experts + 64},
        {**anchor, "NUM_EXPERTS": experts - 32, "LATENTS_PER_EXPERT": latents + 16},
        {**anchor, "NUM_EXPERTS": experts - 64, "LATENTS_PER_EXPERT": latents + 16},
    ]
    train = [
        {**anchor, "LR": lr * 0.75},
        {**anchor, "LR": lr * 1.25},
        {**anchor, "AUXK_ALPHA": auxk * 0.5},
        {**anchor, "AUXK_ALPHA": auxk * 1.5},
        {**anchor, "ACTIVE_EXPERTS": max(1, active - 1)},
        {**anchor, "ACTIVE_EXPERTS": min(4, active + 1)},
    ]
    jump = [
        {**anchor, "K": 48, "NUM_EXPERTS": max(128, experts - 96), "LATENTS_PER_EXPERT": max(32, latents - 24)},
        {**anchor, "K": 96, "NUM_EXPERTS": min(768, experts + 96), "LATENTS_PER_EXPERT": min(160, latents + 24)},
    ]
    if cheaper_first:
        return [*cheaper, *quality, *train, *jump]
    return [*quality, *cheaper, *train, *jump]


def propose_next_params(
    cfg: SearchConfig,
    *,
    attempted_signatures: Iterable[str],
    incumbent_trial: object | None = None,
    current_trial: object | None = None,
) -> dict[str, int | float]:
    attempted = set(attempted_signatures)
    anchor = cfg.baseline_params()
    if not attempted and incumbent_trial is None and current_trial is None:
        return normalize_candidate_params(anchor, cfg)
    cheaper_first = False

    for trial in (current_trial, incumbent_trial):
        if trial is None:
            continue
        params = getattr(trial, "params", None)
        if params:
            anchor = dict(params)
            break

    incumbent_objective = getattr(incumbent_trial, "best_objective", None)
    current_objective = getattr(current_trial, "best_objective", None)
    current_cost = getattr(current_trial, "total_cost_ratio", None)

    if current_trial is not None and current_cost is not None:
        cheaper_first = current_cost >= 0.08
    if (
        incumbent_objective is not None
        and current_objective is not None
        and current_objective > incumbent_objective + 0.02
    ):
        cheaper_first = True

    for candidate in _candidate_pool(anchor, cheaper_first=cheaper_first):
        normalized = normalize_candidate_params(candidate, cfg)
        if params_signature(normalized) not in attempted:
            return normalized

    baseline = normalize_candidate_params(anchor, cfg)
    attempts = len(attempted)
    exploratory = normalize_candidate_params(
        {
            **baseline,
            "K": baseline["K"] + ((attempts % 5) - 2) * 8,
            "NUM_EXPERTS": baseline["NUM_EXPERTS"] + ((attempts % 7) - 3) * 32,
            "LATENTS_PER_EXPERT": baseline["LATENTS_PER_EXPERT"] + ((attempts % 3) - 1) * 16,
            "LR": float(baseline["LR"]) * (1.0 + ((attempts % 4) - 1.5) * 0.1),
        },
        cfg,
    )
    return exploratory
