"""Override validation and baseline resolution for the autoresearch loop."""

from __future__ import annotations

from typing import Any

from .types import BASE_ENV_DEFAULTS


COMMON_OVERRIDE_KEYS = set(BASE_ENV_DEFAULTS.keys()) | {
    "NUM_GROUPS",
    "ACTIVE_GROUPS",
    "ORTHO_LAMBDA",
    "RESIDUAL_FROM",
    "MATRYOSHKA_KS",
    "MATRYOSHKA_WEIGHTS",
}

_LOWRANK_KEYS = {"TRUNK_RANK"}
_TWOSTAGE_KEYS = {"TRUNK_RANK", "STAGE1_RATIO"}
_SOFT_CODEBOOK_TWOSTAGE_KEYS = {"TRUNK_RANK", "NUM_CODES", "STAGE1_RATIO"}
_EXPERT_KEYS = {"NUM_EXPERTS", "ACTIVE_EXPERTS", "LATENTS_PER_EXPERT"}
_FACTORIZED_EXPERT_KEYS = {
    "FACTORIZED_HIDDEN_DIM",
    "NUM_EXPERTS",
    "ACTIVE_EXPERTS",
    "LATENTS_PER_EXPERT",
}
_FACTORIZED_EXPERT_RESIDUAL_KEYS = _FACTORIZED_EXPERT_KEYS | {"STAGE1_RATIO"}
_SHARED_LOWRANK_ROUTED_EXPERT_KEYS = _FACTORIZED_EXPERT_KEYS
_SHARED_LOWRANK_ROUTED_EXPERT_RESIDUAL_KEYS = (
    _FACTORIZED_EXPERT_KEYS | {"STAGE1_RATIO"}
)
_LOWRANK_EXPERT_KEYS = {"TRUNK_RANK", "NUM_EXPERTS", "ACTIVE_EXPERTS", "LATENTS_PER_EXPERT"}
_LOWRANK_EXPERT_RESIDUAL_KEYS = {
    "TRUNK_RANK",
    "STAGE1_RATIO",
    "NUM_EXPERTS",
    "ACTIVE_EXPERTS",
    "LATENTS_PER_EXPERT",
}

ARCHITECTURE_OVERRIDE_KEYS: dict[str, set[str]] = {
    "topk": set(),
    "jumprelu": {
        "JUMPRELU_INIT_THRESHOLD",
        "JUMPRELU_BANDWIDTH",
    },
    "gated": {
        "GATED_TEMPERATURE",
        "GATED_INIT_LOGIT",
    },
    "expert_topk": _EXPERT_KEYS,
    "expert_jumprelu": _EXPERT_KEYS | {"JUMPRELU_INIT_THRESHOLD", "JUMPRELU_BANDWIDTH"},
    "product_key_expert_jumprelu": _EXPERT_KEYS | {"JUMPRELU_INIT_THRESHOLD", "JUMPRELU_BANDWIDTH"},
    "adaptive_active_product_key_expert_jumprelu": _EXPERT_KEYS | {"JUMPRELU_INIT_THRESHOLD", "JUMPRELU_BANDWIDTH"},
    "product_key_factorized_expert_topk": _FACTORIZED_EXPERT_KEYS,
    "shared_product_key_expert_jumprelu": _EXPERT_KEYS | {"STAGE1_RATIO", "JUMPRELU_INIT_THRESHOLD", "JUMPRELU_BANDWIDTH"},
    "shared_product_key_factorized_expert_topk": _FACTORIZED_EXPERT_KEYS,
    "hashed_expert_jumprelu": _EXPERT_KEYS | {"JUMPRELU_INIT_THRESHOLD", "JUMPRELU_BANDWIDTH"},
    "shared_routed_expert_topk": _EXPERT_KEYS,
    "shared_two_stage_residual_expert": _EXPERT_KEYS | {"STAGE1_RATIO"},
    "factorized_topk": {"FACTORIZED_HIDDEN_DIM"},
    "factorized_expert_topk": _FACTORIZED_EXPERT_KEYS,
    "shared_routed_factorized_expert_residual": _FACTORIZED_EXPERT_RESIDUAL_KEYS,
    "shared_lowrank_routed_expert_topk": _SHARED_LOWRANK_ROUTED_EXPERT_KEYS,
    "shared_lowrank_routed_expert_residual": _SHARED_LOWRANK_ROUTED_EXPERT_RESIDUAL_KEYS,
    "lowrank_residual": _LOWRANK_KEYS,
    "lowrank_expert_topk": _LOWRANK_EXPERT_KEYS,
    "lowrank_expert_jumprelu": _LOWRANK_EXPERT_KEYS | {"JUMPRELU_INIT_THRESHOLD", "JUMPRELU_BANDWIDTH"},
    "lowrank_expert_residual": _LOWRANK_EXPERT_RESIDUAL_KEYS,
    "two_stage_residual_expert": {
        "STAGE1_RATIO",
        "NUM_EXPERTS",
        "ACTIVE_EXPERTS",
        "LATENTS_PER_EXPERT",
    },
    "lowrank_gated_residual": _LOWRANK_KEYS,
    "lowrank_jumprelu_residual": _LOWRANK_KEYS,
    "lowrank_grouped_residual": _LOWRANK_KEYS,
    "lowrank_adaptive_budget_residual": _LOWRANK_KEYS,
    "bucketed_lowrank_residual": _LOWRANK_KEYS,
    "whitened_lowrank_residual": _LOWRANK_KEYS,
    "whitened_lowrank_gated_residual": _LOWRANK_KEYS,
    "lowrank_multi_branch_residual": _LOWRANK_KEYS,
    "lowrank_factorized_residual": {"TRUNK_RANK", "FACTORIZED_HIDDEN_DIM"},
    "lowrank_two_stage_residual": _TWOSTAGE_KEYS,
    "routed_lowrank_two_stage_residual": _TWOSTAGE_KEYS,
    "lowrank_residual_vq": {"TRUNK_RANK", "NUM_CODES"},
    "lowrank_soft_codebook_residual": {"TRUNK_RANK", "NUM_CODES"},
    "lowrank_grouped_soft_codebook_residual": {"TRUNK_RANK", "NUM_CODES"},
    "lowrank_gated_soft_codebook_residual": {"TRUNK_RANK", "NUM_CODES"},
    "lowrank_two_stage_soft_codebook_residual": _SOFT_CODEBOOK_TWOSTAGE_KEYS,
    "bucketed_lowrank_two_stage_soft_codebook_residual": _SOFT_CODEBOOK_TWOSTAGE_KEYS,
    "whitened_lowrank_two_stage_soft_codebook_residual": _SOFT_CODEBOOK_TWOSTAGE_KEYS,
    "lowrank_asymmetric_two_stage_soft_codebook_residual": _SOFT_CODEBOOK_TWOSTAGE_KEYS,
    "routed_lowrank_two_stage_soft_codebook_residual": _SOFT_CODEBOOK_TWOSTAGE_KEYS,
    "routed_lowrank_asymmetric_two_stage_soft_codebook_residual": _SOFT_CODEBOOK_TWOSTAGE_KEYS,
}

STRUCTURAL_OVERRIDE_KEYS = {
    "ARCHITECTURE",
    "K",
    "EXPANSION_FACTOR",
    "NUM_GROUPS",
    "ACTIVE_GROUPS",
    "MATRYOSHKA_KS",
    "RESIDUAL_FROM",
    "TRUNK_RANK",
    "NUM_CODES",
    "STAGE1_RATIO",
    "FACTORIZED_HIDDEN_DIM",
    "NUM_EXPERTS",
    "ACTIVE_EXPERTS",
    "LATENTS_PER_EXPERT",
}

DYNAMIC_OVERRIDE_KEYS = {
    "LR",
    "AUXK_ALPHA",
    "JUMPRELU_INIT_THRESHOLD",
    "JUMPRELU_BANDWIDTH",
    "GATED_TEMPERATURE",
    "GATED_INIT_LOGIT",
    "ORTHO_LAMBDA",
    "MATRYOSHKA_WEIGHTS",
}

CONFIG_JSON_TO_ENV = {
    "architecture": "ARCHITECTURE",
    "expansion_factor": "EXPANSION_FACTOR",
    "k": "K",
    "optimizer": "OPTIMIZER",
    "lr": "LR",
    "hookpoints": "HOOKPOINTS",
    "batch_size": "BATCH_SIZE",
    "grad_acc_steps": "GRAD_ACC_STEPS",
    "micro_acc_steps": "MICRO_ACC_STEPS",
    "auxk_alpha": "AUXK_ALPHA",
    "dead_feature_threshold": "DEAD_FEATURE_THRESHOLD",
    "use_hadamard": "USE_HADAMARD",
    "jumprelu_init_threshold": "JUMPRELU_INIT_THRESHOLD",
    "jumprelu_bandwidth": "JUMPRELU_BANDWIDTH",
    "gated_temperature": "GATED_TEMPERATURE",
    "gated_init_logit": "GATED_INIT_LOGIT",
    "num_groups": "NUM_GROUPS",
    "active_groups": "ACTIVE_GROUPS",
    "ortho_lambda": "ORTHO_LAMBDA",
    "residual_from": "RESIDUAL_FROM",
    "matryoshka_ks": "MATRYOSHKA_KS",
    "matryoshka_weights": "MATRYOSHKA_WEIGHTS",
    "trunk_rank": "TRUNK_RANK",
    "num_codes": "NUM_CODES",
    "stage1_ratio": "STAGE1_RATIO",
    "factorized_hidden_dim": "FACTORIZED_HIDDEN_DIM",
    "num_experts": "NUM_EXPERTS",
    "active_experts": "ACTIVE_EXPERTS",
    "latents_per_expert": "LATENTS_PER_EXPERT",
}


def _iter_overrides(overrides: list | dict) -> list[tuple[str, str]]:
    if isinstance(overrides, dict):
        return [(str(key), str(value)) for key, value in overrides.items()]
    pairs: list[tuple[str, str]] = []
    for item in overrides:
        key = item.get("key")
        if not key:
            continue
        pairs.append((str(key), str(item.get("value", ""))))
    return pairs


def resolve_architecture_from_overrides(overrides: list | dict, fallback: str = "topk") -> str:
    for key, value in _iter_overrides(overrides):
        if key == "ARCHITECTURE" and value:
            return value.lower()
    return fallback.lower()


def allowed_override_keys_for_architecture(architecture: str) -> set[str]:
    return COMMON_OVERRIDE_KEYS | ARCHITECTURE_OVERRIDE_KEYS.get(architecture.lower(), set())


def validate_env_overrides(overrides: list | dict, fallback_architecture: str = "topk") -> list[str]:
    architecture = resolve_architecture_from_overrides(overrides, fallback=fallback_architecture)
    allowed = allowed_override_keys_for_architecture(architecture)
    rejected: list[str] = []
    for key, _ in _iter_overrides(overrides):
        if key not in allowed:
            rejected.append(key)
    return rejected


def config_from_overrides(overrides: list | dict, base_config: dict[str, str] | None = None) -> dict[str, str]:
    config = dict(base_config or BASE_ENV_DEFAULTS)
    for key, value in _iter_overrides(overrides):
        config[key] = value
    return config


def env_config_from_runtime_config(config_json: dict[str, Any]) -> dict[str, str]:
    config = dict(BASE_ENV_DEFAULTS)
    for key, env_key in CONFIG_JSON_TO_ENV.items():
        if key not in config_json:
            continue
        value = config_json[key]
        if value is None:
            continue
        if env_key == "USE_HADAMARD":
            config[env_key] = "1" if bool(value) else "0"
        elif isinstance(value, list):
            config[env_key] = ",".join(str(v) for v in value)
        else:
            config[env_key] = str(value)
    return config


def changed_override_keys(overrides: list | dict, baseline_config: dict[str, str]) -> set[str]:
    changed: set[str] = set()
    for key, value in _iter_overrides(overrides):
        if value != str(baseline_config.get(key, "")):
            changed.add(key)
    return changed


def determine_parameter_behavior(change_type: str, changed_keys: set[str]) -> str:
    if change_type == "edit_perf_code":
        return "implementation"
    if changed_keys & STRUCTURAL_OVERRIDE_KEYS:
        return "structural"
    if changed_keys & DYNAMIC_OVERRIDE_KEYS:
        return "dynamic"
    return "default"
