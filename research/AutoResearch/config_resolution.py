"""Shared action/config resolution helpers for AutoResearch.

This module keeps the agent-visible reference config, policy validation,
runtime execution, and persisted round summaries aligned to the same
env-style configuration semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .compatibility import is_compatible_label
from .override_registry import (
    allowed_override_keys_for_architecture,
    config_from_overrides,
    env_config_from_runtime_config,
)
from .target_profile import budget_accesses, default_target_profile, profile_matches
from .types import Action, BASE_ENV_DEFAULTS


@dataclass
class ResolvedActionConfig:
    reference_env_config: dict[str, str]
    candidate_env_config: dict[str, str]
    changed_keys: list[str]
    reference_source: str


def stable_env_items(config: dict[str, Any]) -> list[tuple[str, str]]:
    keys = list(BASE_ENV_DEFAULTS.keys())
    extra = sorted(k for k in config.keys() if k not in BASE_ENV_DEFAULTS)
    items: list[tuple[str, str]] = []
    for key in keys + extra:
        value = config.get(key)
        if value in (None, ""):
            continue
        items.append((key, str(value)))
    return items


def render_env_config(config: dict[str, Any]) -> str:
    return "\n".join(f"  {key}={value}" for key, value in stable_env_items(config))


def changed_keys_from_candidate(
    reference_env_config: dict[str, str],
    candidate_env_config: dict[str, str],
) -> list[str]:
    return [
        key
        for key, ref_val in stable_env_items(reference_env_config)
        if str(candidate_env_config.get(key, "")) != str(ref_val)
    ] + [
        key
        for key in sorted(candidate_env_config.keys())
        if key not in reference_env_config and str(candidate_env_config.get(key, "")) != ""
    ]


def summary_invalid_reason(summary: dict[str, Any]) -> str | None:
    if not summary:
        return None
    result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
    invalid = result.get("invalid_reason") or summary.get("invalid_reason")
    if invalid:
        return str(invalid)
    if str(result.get("error_type") or "") == "config_mismatch":
        return str(result.get("error_summary") or "config_mismatch")
    termination_reason = str(result.get("termination_reason") or "")
    if "INVALID" in termination_reason:
        return termination_reason
    description = str(result.get("description") or summary.get("description") or "")
    if "[INVALID" in description:
        return description
    return None


def summary_is_usable_reference(summary: dict[str, Any]) -> bool:
    if not summary or not isinstance(summary, dict):
        return False
    result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
    if str(result.get("decision") or "") in {"policy_reject", "crash"}:
        return False
    return summary_invalid_reason(summary) is None


def summary_config_source(summary: dict[str, Any]) -> str:
    if isinstance(summary.get("resolved_candidate_env_config"), dict) and summary.get("resolved_candidate_env_config"):
        return "resolved_candidate_env_config"
    if isinstance(summary.get("runtime_env_config"), dict) and summary.get("runtime_env_config"):
        return "runtime_env_config"
    if isinstance(summary.get("runtime_config_json"), dict) and summary.get("runtime_config_json"):
        return "runtime_config_json"
    result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
    checkpoint = result.get("checkpoint")
    if checkpoint and (Path(str(checkpoint)) / "config.json").exists():
        return "checkpoint_config"
    return "legacy_env_overrides"


def _config_from_checkpoint_summary(summary: dict[str, Any]) -> dict[str, str] | None:
    result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
    checkpoint = result.get("checkpoint")
    if not checkpoint:
        return None
    config_path = Path(str(checkpoint)) / "config.json"
    if not config_path.exists():
        return None
    try:
        raw = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return None

    config = dict(BASE_ENV_DEFAULTS)
    for src_key, env_key in {
        "optimizer": "OPTIMIZER",
        "lr": "LR",
        "batch_size": "BATCH_SIZE",
        "grad_acc_steps": "GRAD_ACC_STEPS",
        "micro_acc_steps": "MICRO_ACC_STEPS",
        "auxk_alpha": "AUXK_ALPHA",
        "dead_feature_threshold": "DEAD_FEATURE_THRESHOLD",
    }.items():
        value = raw.get(src_key)
        if value is not None:
            config[env_key] = str(value)

    if raw.get("hookpoints") is not None:
        hookpoints = raw["hookpoints"]
        if isinstance(hookpoints, list):
            config["HOOKPOINTS"] = ",".join(_normalize_hookpoint(str(v)) for v in hookpoints)
        else:
            config["HOOKPOINTS"] = _normalize_hookpoint(str(hookpoints))
    if raw.get("use_hadamard") is not None:
        config["USE_HADAMARD"] = "1" if bool(raw.get("use_hadamard")) else "0"
    if raw.get("compile_model") is not None:
        config["COMPILE_MODEL"] = "1" if bool(raw.get("compile_model")) else "0"

    sae = raw.get("sae", {}) if isinstance(raw.get("sae"), dict) else {}
    if sae.get("architecture") is not None:
        config["ARCHITECTURE"] = str(sae.get("architecture")).lower()
    allowed_keys = allowed_override_keys_for_architecture(config["ARCHITECTURE"])
    for src_key, env_key in {
        "architecture": "ARCHITECTURE",
        "expansion_factor": "EXPANSION_FACTOR",
        "k": "K",
        "trunk_rank": "TRUNK_RANK",
        "num_codes": "NUM_CODES",
        "stage1_ratio": "STAGE1_RATIO",
        "factorized_hidden_dim": "FACTORIZED_HIDDEN_DIM",
        "num_experts": "NUM_EXPERTS",
        "jumprelu_init_threshold": "JUMPRELU_INIT_THRESHOLD",
        "jumprelu_bandwidth": "JUMPRELU_BANDWIDTH",
        "gated_temperature": "GATED_TEMPERATURE",
        "gated_init_logit": "GATED_INIT_LOGIT",
        "ortho_lambda": "ORTHO_LAMBDA",
        "residual_from": "RESIDUAL_FROM",
    }.items():
        value = sae.get(src_key)
        if value is not None:
            if env_key not in BASE_ENV_DEFAULTS and env_key not in allowed_keys:
                continue
            config[env_key] = str(value)
    return config


def _normalize_hookpoint(raw: str) -> str:
    return re.sub(r"layers\.(\d+)\.", r"layers.[\1].", str(raw))


def structured_config_from_round_summary(summary: dict[str, Any]) -> dict[str, str] | None:
    """Return only trustworthy persisted configs from a round summary.

    This intentionally excludes the legacy action/env-overrides fallback,
    which may inherit today's BASE_ENV_DEFAULTS and therefore cannot be used
    for target-profile attribution.
    """
    if not summary:
        return None

    for field in ("resolved_candidate_env_config", "runtime_env_config"):
        cfg = summary.get(field)
        if isinstance(cfg, dict) and cfg:
            return {str(k): str(v) for k, v in cfg.items() if v is not None}

    runtime_cfg = summary.get("runtime_config_json")
    if isinstance(runtime_cfg, dict) and runtime_cfg:
        return env_config_from_runtime_config(runtime_cfg)

    return _config_from_checkpoint_summary(summary)


def config_from_round_summary(summary: dict[str, Any]) -> dict[str, str] | None:
    if not summary:
        return None

    checkpoint_cfg = structured_config_from_round_summary(summary)
    if checkpoint_cfg is not None:
        return checkpoint_cfg

    action = summary.get("action", {})
    family_name = str(
        summary.get("family_name")
        or action.get("family_name")
        or ""
    ).lower()
    if not family_name:
        return None

    config = dict(BASE_ENV_DEFAULTS)
    for item in action.get("env_overrides", []) or []:
        key = item.get("key")
        value = item.get("value")
        if key is not None and value is not None:
            config[str(key)] = str(value)
    if family_name:
        config["ARCHITECTURE"] = family_name
    return config


def frontier_entry_to_env_config(entry: dict[str, Any]) -> dict[str, str]:
    config = dict(BASE_ENV_DEFAULTS)
    raw = entry.get("config", {})

    mapping = {
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
        "num_groups": "NUM_GROUPS",
        "active_groups": "ACTIVE_GROUPS",
        "jumprelu_init_threshold": "JUMPRELU_INIT_THRESHOLD",
        "jumprelu_bandwidth": "JUMPRELU_BANDWIDTH",
        "gated_temperature": "GATED_TEMPERATURE",
        "gated_init_logit": "GATED_INIT_LOGIT",
        "ortho_lambda": "ORTHO_LAMBDA",
        "residual_from": "RESIDUAL_FROM",
        "matryoshka_ks": "MATRYOSHKA_KS",
        "matryoshka_weights": "MATRYOSHKA_WEIGHTS",
        "trunk_rank": "TRUNK_RANK",
        "num_codes": "NUM_CODES",
        "stage1_ratio": "STAGE1_RATIO",
        "factorized_hidden_dim": "FACTORIZED_HIDDEN_DIM",
        "num_experts": "NUM_EXPERTS",
        "family_name": "FAMILY_NAME",
        "family_stage": "FAMILY_STAGE",
    }

    for cfg_key, env_key in mapping.items():
        value = raw.get(cfg_key)
        if value is None:
            continue
        if env_key == "USE_HADAMARD":
            config[env_key] = "1" if bool(value) else "0"
        elif isinstance(value, list):
            config[env_key] = ",".join(str(v) for v in value)
        else:
            config[env_key] = str(value)

    if entry.get("architecture") is not None:
        config["ARCHITECTURE"] = str(entry["architecture"]).lower()
    if entry.get("k") is not None:
        config["K"] = str(entry["k"])
    if entry.get("ef") is not None:
        config["EXPANSION_FACTOR"] = str(entry["ef"])

    config.pop("FAMILY_NAME", None)
    config.pop("FAMILY_STAGE", None)
    return config


def best_frontier_entry(
    frontier: dict[str, Any],
    family_name: str | None = None,
    registry: dict[str, str] | None = None,
    prefer_feasible: bool = False,
) -> dict[str, Any] | None:
    current_profile = default_target_profile()

    def _pick_best(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
        best, best_fvu = None, float("inf")
        for entry in entries:
            try:
                fvu = float(entry.get("fvu", float("inf")))
            except (TypeError, ValueError):
                continue
            if fvu < best_fvu:
                best_fvu = fvu
                best = entry
        return best

    target_family = (family_name or "").lower()
    feasible: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []

    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        entry_family = str(
            entry.get("config", {}).get("family_name")
            or entry.get("architecture")
            or ""
        ).lower()
        if target_family and entry_family != target_family:
            continue
        if registry is not None and not is_compatible_label(registry.get(entry_family)):
            continue
        entry_cfg = dict(entry.get("config", {}) or {})
        if entry.get("target_profile") is not None and "target_profile" not in entry_cfg:
            entry_cfg["target_profile"] = entry["target_profile"]
        if not profile_matches(entry_cfg, current_profile):
            continue
        all_candidates.append(entry)
        if prefer_feasible:
            total_cost = entry.get("total_cost")
            if total_cost is None:
                sel = entry.get("selection_cost")
                deploy = entry.get("deployment_accesses", 0) or 0
                total_cost = float(sel) + float(deploy) if sel is not None else None
            if total_cost is not None and float(total_cost) <= budget_accesses(entry_cfg):
                feasible.append(entry)

    if prefer_feasible and feasible:
        return _pick_best(feasible)
    return _pick_best(all_candidates)


def latest_active_family_name(
    families: dict[str, Any],
    registry: dict[str, str] | None = None,
) -> str | None:
    best_name: str | None = None
    best_round = -1
    for name, family in families.items():
        if registry is not None and not is_compatible_label(registry.get(str(name).lower())):
            continue
        if family.get("status") != "active":
            continue
        last_round = int(family.get("last_round") or -1)
        if last_round > best_round:
            best_round = last_round
            best_name = name
    return best_name


def resolve_mainline_snapshot(state: Any) -> dict[str, Any]:
    registry = state.load_compatibility_registry()
    best_entry = best_frontier_entry(
        state.frontier,
        registry=registry,
        prefer_feasible=True,
    )
    if best_entry is not None:
        config = frontier_entry_to_env_config(best_entry)
        family_name = str(
            best_entry.get("config", {}).get("family_name")
            or best_entry.get("architecture")
            or config.get("ARCHITECTURE", "topk")
        ).lower()
        return {
            "family_name": family_name,
            "config": config,
            "source": "frontier_best",
        }

    config = dict(BASE_ENV_DEFAULTS)
    return {
        "family_name": config["ARCHITECTURE"],
        "config": config,
        "source": "target_profile_baseline",
    }


def latest_successful_family_config(
    state: Any,
    family_name: str,
) -> dict[str, str] | None:
    target = (family_name or "").lower()
    current_profile = default_target_profile()
    if not target:
        return None
    for summary in reversed(state.recent_round_summaries(limit=50)):
        if not isinstance(summary, dict):
            continue
        summary_family = str(
            summary.get("family_name")
            or summary.get("action", {}).get("family_name")
            or summary.get("result", {}).get("architecture")
            or ""
        ).lower()
        if summary_family != target:
            continue
        if not summary_is_usable_reference(summary):
            continue
        cfg = structured_config_from_round_summary(summary)
        if cfg is not None and profile_matches(cfg, current_profile):
            return cfg
    return None


def resolve_reference_env_config(
    action: Action,
    state: Any,
) -> tuple[dict[str, str], str]:
    current_profile = default_target_profile()
    if action.reference_round is not None:
        explicit = structured_config_from_round_summary(state.load_round_summary(action.reference_round))
        if explicit is not None and profile_matches(explicit, current_profile):
            return explicit, f"reference_round:r{action.reference_round}"

    mainline = resolve_mainline_snapshot(state)
    action_family = (action.family_name or "").lower()
    mainline_family = mainline["family_name"]

    if action_family == mainline_family or not action_family:
        return dict(mainline["config"]), f"mainline:{mainline['source']}"

    recent_family_cfg = latest_successful_family_config(state, action_family)
    if recent_family_cfg is not None:
        return recent_family_cfg, f"family_recent:{action_family}"

    registry = state.load_compatibility_registry()
    family_best = best_frontier_entry(
        state.frontier,
        family_name=action_family,
        registry=registry,
        prefer_feasible=True,
    )
    if family_best is not None:
        return frontier_entry_to_env_config(family_best), f"family_frontier:{action_family}"

    return dict(mainline["config"]), f"mainline:{mainline['source']}"


def resolve_action_configs(
    action: Action,
    state: Any,
) -> ResolvedActionConfig:
    reference_env_config, source = resolve_reference_env_config(action, state)
    candidate_env_config = config_from_overrides(
        action.env_overrides,
        base_config=reference_env_config,
    )
    changed_keys = changed_keys_from_candidate(reference_env_config, candidate_env_config)
    return ResolvedActionConfig(
        reference_env_config=reference_env_config,
        candidate_env_config=candidate_env_config,
        changed_keys=changed_keys,
        reference_source=source,
    )
