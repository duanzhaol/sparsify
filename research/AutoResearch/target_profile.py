"""Shared target-profile metadata for AutoResearch.

This keeps the training hookpoint and the deployment cost proxy aligned across
controller/policy/prompt/runtime state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


Q_PROJ_FUSED_QKV_PROFILE_ID = "qwen3_0p6b_layer3_q_proj_fused_qkv_proxy_v1"
K_PROJ_ACTUAL_PROFILE_ID = "qwen3_0p6b_k_proj_actual_v1"
V_PROJ_ACTUAL_PROFILE_ID = "qwen3_0p6b_v_proj_actual_v1"
O_PROJ_ACTUAL_PROFILE_ID = "qwen3_0p6b_o_proj_actual_v1"
UP_PROJ_ACTUAL_PROFILE_ID = "qwen3_0p6b_up_proj_actual_v1"


@dataclass(frozen=True)
class TargetProfile:
    profile_id: str
    training_hookpoint: str
    d_in: int
    n_output: int
    elbow_threshold_path: str
    cost_model_label: str

    @property
    def original_matmul_accesses(self) -> int:
        return self.d_in * self.n_output

    def budget_accesses(self, multiplier: float = 1.5) -> float:
        return float(multiplier * self.original_matmul_accesses)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["original_matmul_accesses"] = self.original_matmul_accesses
        return data


DEFAULT_TARGET_PROFILE = TargetProfile(
    profile_id=Q_PROJ_FUSED_QKV_PROFILE_ID,
    training_hookpoint="layers.[3].self_attn.q_proj",
    d_in=1024,
    n_output=4096,
    elbow_threshold_path="thresholds/Qwen3-0.6B/thresholds_q.json",
    cost_model_label=(
        "train on q_proj activations; cost proxy uses fused QKV deployment "
        "matrix 1024x4096 and counts deploy libraries as d_in+n_output"
    ),
)

_K_PROJ_PROFILE = TargetProfile(
    profile_id=K_PROJ_ACTUAL_PROFILE_ID,
    training_hookpoint="layers.[3].self_attn.k_proj",
    d_in=1024,
    n_output=1024,
    elbow_threshold_path="thresholds/Qwen3-0.6B/thresholds_q.json",
    cost_model_label="train on k_proj activations; cost uses actual k_proj matrix 1024x1024",
)

_V_PROJ_PROFILE = TargetProfile(
    profile_id=V_PROJ_ACTUAL_PROFILE_ID,
    training_hookpoint="layers.[3].self_attn.v_proj",
    d_in=1024,
    n_output=1024,
    elbow_threshold_path="thresholds/Qwen3-0.6B/thresholds_q.json",
    cost_model_label="train on v_proj activations; cost uses actual v_proj matrix 1024x1024",
)

_O_PROJ_PROFILE = TargetProfile(
    profile_id=O_PROJ_ACTUAL_PROFILE_ID,
    training_hookpoint="layers.[3].self_attn.o_proj",
    d_in=2048,
    n_output=1024,
    elbow_threshold_path="thresholds/Qwen3-0.6B/thresholds_o.json",
    cost_model_label="train on o_proj activations; cost uses actual o_proj matrix 2048x1024",
)

_UP_PROJ_PROFILE = TargetProfile(
    profile_id=UP_PROJ_ACTUAL_PROFILE_ID,
    training_hookpoint="layers.[3].mlp.up_proj",
    d_in=1024,
    n_output=6144,
    elbow_threshold_path="thresholds/Qwen3-0.6B/thresholds_up.json",
    cost_model_label=(
        "train on up_proj activations; cost proxy uses fused gate/up deployment "
        "matrix 1024x6144 and counts deploy libraries as d_in+n_output"
    ),
)


def default_target_profile() -> TargetProfile:
    return DEFAULT_TARGET_PROFILE


def resolve_target_profile(config: dict[str, Any] | None = None) -> TargetProfile:
    cfg = config or {}
    explicit = _profile_from_explicit_config(cfg)
    if explicit is not None:
        return explicit

    hookpoint = _first_hookpoint(
        cfg.get("hookpoints")
        or cfg.get("HOOKPOINTS")
        or DEFAULT_TARGET_PROFILE.training_hookpoint
    )

    if hookpoint.endswith(".self_attn.q_proj"):
        return TargetProfile(
            profile_id=DEFAULT_TARGET_PROFILE.profile_id,
            training_hookpoint=hookpoint,
            d_in=DEFAULT_TARGET_PROFILE.d_in,
            n_output=DEFAULT_TARGET_PROFILE.n_output,
            elbow_threshold_path=DEFAULT_TARGET_PROFILE.elbow_threshold_path,
            cost_model_label=DEFAULT_TARGET_PROFILE.cost_model_label,
        )
    if hookpoint.endswith(".self_attn.k_proj"):
        return TargetProfile(
            profile_id=_K_PROJ_PROFILE.profile_id,
            training_hookpoint=hookpoint,
            d_in=_K_PROJ_PROFILE.d_in,
            n_output=_K_PROJ_PROFILE.n_output,
            elbow_threshold_path=_K_PROJ_PROFILE.elbow_threshold_path,
            cost_model_label=_K_PROJ_PROFILE.cost_model_label,
        )
    if hookpoint.endswith(".self_attn.v_proj"):
        return TargetProfile(
            profile_id=_V_PROJ_PROFILE.profile_id,
            training_hookpoint=hookpoint,
            d_in=_V_PROJ_PROFILE.d_in,
            n_output=_V_PROJ_PROFILE.n_output,
            elbow_threshold_path=_V_PROJ_PROFILE.elbow_threshold_path,
            cost_model_label=_V_PROJ_PROFILE.cost_model_label,
        )
    if hookpoint.endswith(".self_attn.o_proj"):
        return TargetProfile(
            profile_id=_O_PROJ_PROFILE.profile_id,
            training_hookpoint=hookpoint,
            d_in=_O_PROJ_PROFILE.d_in,
            n_output=_O_PROJ_PROFILE.n_output,
            elbow_threshold_path=_O_PROJ_PROFILE.elbow_threshold_path,
            cost_model_label=_O_PROJ_PROFILE.cost_model_label,
        )
    if hookpoint.endswith(".mlp.up_proj"):
        return TargetProfile(
            profile_id=_UP_PROJ_PROFILE.profile_id,
            training_hookpoint=hookpoint,
            d_in=_UP_PROJ_PROFILE.d_in,
            n_output=_UP_PROJ_PROFILE.n_output,
            elbow_threshold_path=_UP_PROJ_PROFILE.elbow_threshold_path,
            cost_model_label=_UP_PROJ_PROFILE.cost_model_label,
        )
    return DEFAULT_TARGET_PROFILE


def profile_matches(config: dict[str, Any] | None, target: TargetProfile | None = None) -> bool:
    return resolve_target_profile(config).profile_id == (target or DEFAULT_TARGET_PROFILE).profile_id


def original_matmul_accesses(config: dict[str, Any] | None = None) -> int:
    return resolve_target_profile(config).original_matmul_accesses


def budget_accesses(
    config: dict[str, Any] | None = None,
    multiplier: float = 1.5,
) -> float:
    return resolve_target_profile(config).budget_accesses(multiplier)


def _first_hookpoint(raw: Any) -> str:
    if isinstance(raw, (list, tuple)):
        return str(raw[0]) if raw else ""
    text = str(raw or "")
    if "," in text:
        return text.split(",", 1)[0].strip()
    return text.strip()


def _profile_from_explicit_config(config: dict[str, Any]) -> TargetProfile | None:
    raw = config.get("target_profile")
    if not isinstance(raw, dict):
        raw = config.get("TARGET_PROFILE")
    if not isinstance(raw, dict):
        return None

    profile_id = raw.get("profile_id")
    hookpoint = raw.get("training_hookpoint")
    d_in = raw.get("d_in")
    n_output = raw.get("n_output")
    elbow = raw.get("elbow_threshold_path")
    label = raw.get("cost_model_label")
    if not all(v is not None for v in (profile_id, hookpoint, d_in, n_output, elbow, label)):
        return None

    try:
        return TargetProfile(
            profile_id=str(profile_id),
            training_hookpoint=str(hookpoint),
            d_in=int(d_in),
            n_output=int(n_output),
            elbow_threshold_path=str(elbow),
            cost_model_label=str(label),
        )
    except (TypeError, ValueError):
        return None
