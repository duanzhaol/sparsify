"""Override validation and baseline resolution for the autoresearch loop."""

from __future__ import annotations

from typing import Any

from research.state_io import BASE_ENV_DEFAULTS


COMMON_OVERRIDE_KEYS = set(BASE_ENV_DEFAULTS.keys()) | {
    "NUM_GROUPS",
    "ACTIVE_GROUPS",
    "ORTHO_LAMBDA",
    "RESIDUAL_FROM",
    "MATRYOSHKA_KS",
    "MATRYOSHKA_WEIGHTS",
}

ARCHITECTURE_OVERRIDE_KEYS = {
    "topk": set(),
    "jumprelu": {
        "JUMPRELU_INIT_THRESHOLD",
        "JUMPRELU_BANDWIDTH",
    },
    "gated": {
        "GATED_TEMPERATURE",
        "GATED_INIT_LOGIT",
    },
}

STRUCTURAL_OVERRIDE_KEYS = {
    "ARCHITECTURE",
    "K",
    "EXPANSION_FACTOR",
    "NUM_GROUPS",
    "ACTIVE_GROUPS",
    "MATRYOSHKA_KS",
    "RESIDUAL_FROM",
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


def resolve_baseline_config(
    state: dict[str, Any],
    action: dict[str, Any],
    tier: str | None = None,
) -> dict[str, str]:
    overrides = action.get("env_overrides", [])
    requested = config_from_overrides(overrides)
    architecture = str(action.get("family_name") or requested.get("ARCHITECTURE", BASE_ENV_DEFAULTS["ARCHITECTURE"])).lower()
    requested_k = str(requested.get("K", BASE_ENV_DEFAULTS["K"]))
    frontier_names = []
    if tier == "full":
        frontier_names.extend(["full_frontier", "proxy_frontier"])
    elif tier == "proxy":
        frontier_names.extend(["proxy_frontier", "full_frontier"])
    else:
        frontier_names.extend(["full_frontier", "proxy_frontier"])

    for frontier_name in frontier_names:
        frontier = state.get(frontier_name, {})
        point = frontier.get(requested_k)
        if not isinstance(point, dict):
            continue
        point_arch = str(point.get("architecture") or point.get("config", {}).get("architecture") or "").lower()
        if point_arch != architecture:
            continue
        config_json = point.get("config")
        if isinstance(config_json, dict):
            return env_config_from_runtime_config(config_json)

    for frontier_name in frontier_names:
        frontier = state.get(frontier_name, {})
        for point in frontier.values():
            if not isinstance(point, dict):
                continue
            point_arch = str(point.get("architecture") or point.get("config", {}).get("architecture") or "").lower()
            if point_arch != architecture:
                continue
            config_json = point.get("config")
            if isinstance(config_json, dict):
                return env_config_from_runtime_config(config_json)

    return dict(BASE_ENV_DEFAULTS)


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
