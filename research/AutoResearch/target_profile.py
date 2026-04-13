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

QWEN3_4B_Q_PROJ_FUSED_QKV_PROFILE_ID = "qwen3_4b_q_proj_fused_qkv_proxy_v1"
QWEN3_4B_K_PROJ_ACTUAL_PROFILE_ID = "qwen3_4b_k_proj_actual_v1"
QWEN3_4B_V_PROJ_ACTUAL_PROFILE_ID = "qwen3_4b_v_proj_actual_v1"
QWEN3_4B_O_PROJ_ACTUAL_PROFILE_ID = "qwen3_4b_o_proj_actual_v1"
QWEN3_4B_UP_PROJ_ACTUAL_PROFILE_ID = "qwen3_4b_up_proj_actual_v1"


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

_QWEN3_4B_Q_PROJ_PROFILE = TargetProfile(
    profile_id=QWEN3_4B_Q_PROJ_FUSED_QKV_PROFILE_ID,
    training_hookpoint="layers.[17].self_attn.q_proj",
    d_in=2560,
    n_output=6144,
    elbow_threshold_path="thresholds/Qwen3-4B/thresholds_q.json",
    cost_model_label=(
        "train on q_proj activations; cost proxy uses fused QKV deployment "
        "matrix 2560x6144 and counts deploy libraries as d_in+n_output"
    ),
)

_QWEN3_4B_K_PROJ_PROFILE = TargetProfile(
    profile_id=QWEN3_4B_K_PROJ_ACTUAL_PROFILE_ID,
    training_hookpoint="layers.[17].self_attn.k_proj",
    d_in=2560,
    n_output=1024,
    elbow_threshold_path="thresholds/Qwen3-4B/thresholds_q.json",
    cost_model_label="train on k_proj activations; cost uses actual k_proj matrix 2560x1024",
)

_QWEN3_4B_V_PROJ_PROFILE = TargetProfile(
    profile_id=QWEN3_4B_V_PROJ_ACTUAL_PROFILE_ID,
    training_hookpoint="layers.[17].self_attn.v_proj",
    d_in=2560,
    n_output=1024,
    elbow_threshold_path="thresholds/Qwen3-4B/thresholds_q.json",
    cost_model_label="train on v_proj activations; cost uses actual v_proj matrix 2560x1024",
)

_QWEN3_4B_O_PROJ_PROFILE = TargetProfile(
    profile_id=QWEN3_4B_O_PROJ_ACTUAL_PROFILE_ID,
    training_hookpoint="layers.[17].self_attn.o_proj",
    d_in=4096,
    n_output=2560,
    elbow_threshold_path="thresholds/Qwen3-4B/thresholds_o.json",
    cost_model_label="train on o_proj activations; cost uses actual o_proj matrix 4096x2560",
)

_QWEN3_4B_UP_PROJ_PROFILE = TargetProfile(
    profile_id=QWEN3_4B_UP_PROJ_ACTUAL_PROFILE_ID,
    training_hookpoint="layers.[17].mlp.up_proj",
    d_in=2560,
    n_output=19456,
    elbow_threshold_path="thresholds/Qwen3-4B/thresholds_up.json",
    cost_model_label=(
        "train on up_proj activations; cost proxy uses fused gate/up deployment "
        "matrix 2560x19456 and counts deploy libraries as d_in+n_output"
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
    default_elbow = _default_elbow_threshold_path(cfg)

    if _is_qwen3_4b_config(cfg):
        if hookpoint.endswith(".self_attn.q_proj"):
            return _copy_profile(
                _QWEN3_4B_Q_PROJ_PROFILE,
                hookpoint,
                default_elbow,
            )
        if hookpoint.endswith(".self_attn.k_proj"):
            return _copy_profile(
                _QWEN3_4B_K_PROJ_PROFILE,
                hookpoint,
                default_elbow,
            )
        if hookpoint.endswith(".self_attn.v_proj"):
            return _copy_profile(
                _QWEN3_4B_V_PROJ_PROFILE,
                hookpoint,
                default_elbow,
            )
        if hookpoint.endswith(".self_attn.o_proj"):
            return _copy_profile(
                _QWEN3_4B_O_PROJ_PROFILE,
                hookpoint,
                default_elbow,
            )
        if hookpoint.endswith(".mlp.up_proj"):
            return _copy_profile(
                _QWEN3_4B_UP_PROJ_PROFILE,
                hookpoint,
                default_elbow,
            )

    if hookpoint.endswith(".self_attn.q_proj"):
        return _copy_profile(DEFAULT_TARGET_PROFILE, hookpoint, default_elbow)
    if hookpoint.endswith(".self_attn.k_proj"):
        return _copy_profile(_K_PROJ_PROFILE, hookpoint, default_elbow)
    if hookpoint.endswith(".self_attn.v_proj"):
        return _copy_profile(_V_PROJ_PROFILE, hookpoint, default_elbow)
    if hookpoint.endswith(".self_attn.o_proj"):
        return _copy_profile(_O_PROJ_PROFILE, hookpoint, default_elbow)
    if hookpoint.endswith(".mlp.up_proj"):
        return _copy_profile(_UP_PROJ_PROFILE, hookpoint, default_elbow)
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
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "[":
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            return text[:idx].strip()
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


def _copy_profile(
    profile: TargetProfile,
    hookpoint: str,
    elbow_threshold_path: str | None = None,
) -> TargetProfile:
    return TargetProfile(
        profile_id=profile.profile_id,
        training_hookpoint=hookpoint,
        d_in=profile.d_in,
        n_output=profile.n_output,
        elbow_threshold_path=elbow_threshold_path or profile.elbow_threshold_path,
        cost_model_label=profile.cost_model_label,
    )


def _default_elbow_threshold_path(config: dict[str, Any]) -> str | None:
    value = config.get("elbow_threshold_path")
    if value in (None, ""):
        value = config.get("ELBOW_THRESHOLD_PATH")
    return str(value) if value not in (None, "") else None


def _is_qwen3_4b_config(config: dict[str, Any]) -> bool:
    model_path = config.get("model_path")
    if model_path in (None, ""):
        model_path = config.get("MODEL_PATH")
    if model_path in (None, ""):
        return False
    return "qwen3-4b" in str(model_path).lower()
