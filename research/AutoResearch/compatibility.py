"""Compatibility parsing and filtering for LUTurbo-oriented AutoResearch."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from .git_ops import REPO_ROOT
from .target_profile import default_target_profile

DIRECT_COMPATIBLE = "direct_compatible"
EXTENDED_COMPATIBLE = "extended_compatible"
INCOMPATIBLE = "incompatible"
UNKNOWN_COMPATIBILITY = "unknown"
COST_METRIC_VERSION = "objective_v1"

_LABEL_MAP = {
    "直接兼容": DIRECT_COMPATIBLE,
    "扩展兼容": EXTENDED_COMPATIBLE,
    "不兼容": INCOMPATIBLE,
    DIRECT_COMPATIBLE: DIRECT_COMPATIBLE,
    EXTENDED_COMPATIBLE: EXTENDED_COMPATIBLE,
    INCOMPATIBLE: INCOMPATIBLE,
}


def cost_entry_is_current(entry: dict[str, Any]) -> bool:
    """Return True when a stored frontier entry uses the current cost schema."""
    return str(entry.get("metric_version") or "") == COST_METRIC_VERSION


def extract_cost_extra_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract architecture-specific cost knobs from a config-like mapping."""
    extra: dict[str, Any] = {}
    for env_key, cfg_key in [
        ("trunk_rank", "trunk_rank"),
        ("num_codes", "num_codes"),
        ("stage1_ratio", "stage1_ratio"),
        ("factorized_hidden_dim", "factorized_hidden_dim"),
        ("num_experts", "num_experts"),
        ("active_experts", "active_experts"),
        ("latents_per_expert", "latents_per_expert"),
        ("TRUNK_RANK", "trunk_rank"),
        ("NUM_CODES", "num_codes"),
        ("STAGE1_RATIO", "stage1_ratio"),
        ("FACTORIZED_HIDDEN_DIM", "factorized_hidden_dim"),
        ("NUM_EXPERTS", "num_experts"),
        ("ACTIVE_EXPERTS", "active_experts"),
        ("LATENTS_PER_EXPERT", "latents_per_expert"),
    ]:
        val = config.get(env_key)
        if val is None or val == "":
            continue
        try:
            extra[cfg_key] = float(val) if "." in str(val) else int(val)
        except (ValueError, TypeError):
            continue
    return extra


def normalize_compatibility_label(label: str | None) -> str:
    if not label:
        return UNKNOWN_COMPATIBILITY
    cleaned = str(label).strip().strip("`")
    return _LABEL_MAP.get(cleaned, UNKNOWN_COMPATIBILITY)


def is_compatible_label(label: str | None) -> bool:
    normalized = normalize_compatibility_label(label)
    return normalized in {DIRECT_COMPATIBLE, EXTENDED_COMPATIBLE}


def parse_compatibility_registry(prior_text: str) -> dict[str, str]:
    """Parse family compatibility labels from markdown tables in prior history.

    Expected row shape:
    | `family` | ... | 直接兼容 | ... |
    """
    registry: dict[str, str] = {}
    if not prior_text:
        return registry

    for raw_line in prior_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        if "---" in line:
            continue

        cols = [part.strip() for part in line.split("|")[1:-1]]
        if len(cols) < 5:
            continue

        family = _normalize_family_name(cols[0])
        if not family:
            continue

        compat = normalize_compatibility_label(cols[4])
        if compat == UNKNOWN_COMPATIBILITY:
            continue

        registry[family] = compat

    return registry


def summarize_registry_counts(registry: dict[str, str]) -> str:
    direct = sum(1 for v in registry.values() if v == DIRECT_COMPATIBLE)
    extended = sum(1 for v in registry.values() if v == EXTENDED_COMPATIBLE)
    incompatible = sum(1 for v in registry.values() if v == INCOMPATIBLE)
    return (
        f"兼容性注册表：直接兼容 {direct} 个，"
        f"扩展兼容 {extended} 个，不兼容 {incompatible} 个。"
    )


def compatibility_hard_rules() -> str:
    """Short rules that must appear in every prompt."""
    profile = default_target_profile()
    return "\n".join([
        "LUTurbo 兼容性硬约束：",
        "1. 提案前必须先写出最终重构公式，并说明它如何导出到 LUTurbo/Lottable 推理链路。",
        "2. 每个 family 必须先判断兼容性：直接兼容 / 扩展兼容 / 不兼容。",
        "3. 明确标记为不兼容的 family，不得继续占据当前 objective leaderboard，也不得继续作为主线推进。",
        "4. 当前 leaderboard 只代表兼容 family 的 objective 排名；不兼容 family 只能作为历史参考，不能驱动后续决策。",
        f"当前 target：训练 hookpoint 为 {profile.training_hookpoint}。",
        f"成本 proxy：{profile.cost_model_label}。",
        "5. 成本硬约束：total_cost (encoder + deployment) 不得超过 1.5×h×n（原始 matmul 的 1.5 倍）。",
        "   超过此阈值的配置将被 policy 拦截。",
        "   降低 encoder 成本：减小 EXPANSION_FACTOR / TRUNK_RANK / NUM_CODES、使用低秩 scorer、探索非全字典选择机制。",
        "   降低部署成本：减小 K / TRUNK_RANK / NUM_CODES。",
        "   可通过 TRUNK_RANK / NUM_CODES / STAGE1_RATIO / FACTORIZED_HIDDEN_DIM 等 env_overrides 调节。",
        "6. 当前单目标为 objective_score = total_cost_ratio + exceed_alpha_0.50；FVU 只作诊断与 tie-break，不再是主优化轴。",
    ])


def compute_selection_cost(
    architecture: str,
    k: int = 128,
    ef: int = 12,
    d_in: int | None = None,
    n_output: int | None = None,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute encoder-side selection cost by instantiating the model.

    Uses subprocess to avoid import-time GPU issues, similar to
    ``policy.behavioral_diff_test()``.
    """
    if d_in is None or n_output is None:
        profile = default_target_profile()
        if d_in is None:
            d_in = profile.d_in
        if n_output is None:
            n_output = profile.n_output

    cfg_parts = [
        f"architecture='{architecture}'",
        f"k={k}",
        f"expansion_factor={ef}",
    ]
    if extra_config:
        for key, value in extra_config.items():
            if value is not None:
                cfg_parts.append(f"{key}={value!r}")

    code = f"""\
import sys, json; sys.path.insert(0, '.')
from sparsify.sparse_coder import _get_sae_class
from sparsify.config import SparseCoderConfig
cfg = SparseCoderConfig({', '.join(cfg_parts)})
cls = _get_sae_class('{architecture}')
m = cls({d_in}, cfg, device='cpu')
print("COST:" + json.dumps(m.selection_cost_estimate({int(n_output)})))
"""
    try:
        result = subprocess.run(
            ["python", "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in (result.stdout + result.stderr).splitlines():
            if line.startswith("COST:"):
                return json.loads(line[5:])
        return {"error": "parse_failed", "stderr": result.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as exc:
        return {"error": str(exc)}


_COST_TABLE_ARCHITECTURES = [
    "topk", "gated", "factorized_topk",
    "lowrank_residual", "lowrank_soft_codebook_residual",
    "lowrank_two_stage_soft_codebook_residual",
]
_COST_TABLE_EF_VALUES = [1, 2, 3, 4, 6, 8, 12]


def compute_selection_cost_table(
    architectures: list[str] | None = None,
    ef_values: list[int] | None = None,
    k: int = 32,
    d_in: int | None = None,
    n_output: int | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Batch-compute selection costs for (architecture, EF) combinations.

    Uses a single subprocess to avoid N separate invocations.
    Returns {(arch, ef): cost_dict} where cost_dict has keys:
    ratio, feasible, total_accesses, etc.
    """
    profile = default_target_profile()
    d_in = profile.d_in if d_in is None else d_in
    n_output = profile.n_output if n_output is None else n_output
    archs = architectures or _COST_TABLE_ARCHITECTURES
    efs = ef_values or _COST_TABLE_EF_VALUES

    combos = [(arch, ef) for arch in archs for ef in efs]
    if not combos:
        return {}

    # Build a single subprocess that computes all costs
    lines = [
        "import sys, json; sys.path.insert(0, '.')",
        "from sparsify.sparse_coder import _get_sae_class",
        "from sparsify.config import SparseCoderConfig",
    ]
    for arch, ef in combos:
        lines.append(f"try:")
        lines.append(f"    _cfg = SparseCoderConfig(architecture='{arch}', k={k}, expansion_factor={ef})")
        lines.append(f"    _cls = _get_sae_class('{arch}')")
        lines.append(f"    _m = _cls({d_in}, _cfg, device='cpu')")
        lines.append(f"    print('COST:{arch}|{ef}|' + json.dumps(_m.selection_cost_estimate({int(n_output)})))")
        lines.append(f"except Exception as _e:")
        lines.append(f"    print('COST:{arch}|{ef}|' + json.dumps({{'error': str(_e)}}))")

    code = "\n".join(lines)
    results: dict[tuple[str, int], dict[str, Any]] = {}
    try:
        proc = subprocess.run(
            ["python", "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        for line in (proc.stdout + proc.stderr).splitlines():
            if not line.startswith("COST:"):
                continue
            parts = line[5:].split("|", 2)
            if len(parts) == 3:
                arch_name, ef_str, payload = parts
                try:
                    results[(arch_name, int(ef_str))] = json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    pass
    except (subprocess.TimeoutExpired, Exception):
        pass

    return results


def extract_full_prior_document(prior_text: str) -> str:
    """Return the full prior document for fresh-session prompts."""
    return prior_text.strip()


def _normalize_family_name(raw: str) -> str:
    text = raw.strip().strip("`")
    if not text:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_]+", text):
        return ""
    return text.lower()
