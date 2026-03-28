"""Compatibility parsing and filtering for LUTurbo-oriented AutoResearch."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any, Iterable

from .git_ops import REPO_ROOT
from .target_profile import budget_accesses, default_target_profile, profile_matches, resolve_target_profile

DIRECT_COMPATIBLE = "direct_compatible"
EXTENDED_COMPATIBLE = "extended_compatible"
INCOMPATIBLE = "incompatible"
UNKNOWN_COMPATIBILITY = "unknown"

_LABEL_MAP = {
    "直接兼容": DIRECT_COMPATIBLE,
    "扩展兼容": EXTENDED_COMPATIBLE,
    "不兼容": INCOMPATIBLE,
    DIRECT_COMPATIBLE: DIRECT_COMPATIBLE,
    EXTENDED_COMPATIBLE: EXTENDED_COMPATIBLE,
    INCOMPATIBLE: INCOMPATIBLE,
}


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
        "3. 明确标记为不兼容的 family，不得继续占据 frontier，也不得继续作为主线推进。",
        "4. 当前 frontier 只代表兼容 family 的最优点；不兼容 family 只能作为历史参考，不能驱动后续决策。",
        f"当前 target：训练 hookpoint 为 {profile.training_hookpoint}。",
        f"成本 proxy：{profile.cost_model_label}。",
        "5. 成本硬约束：total_cost (encoder + deployment) 不得超过 1.5×h×n（原始 matmul 的 1.5 倍）。",
        "   超过此阈值的配置将被 policy 拦截。",
        "   降低 encoder 成本：减小 EXPANSION_FACTOR / TRUNK_RANK / NUM_CODES、使用低秩 scorer、探索非全字典选择机制。",
        "   降低部署成本：减小 K / TRUNK_RANK / NUM_CODES。",
        "   可通过 TRUNK_RANK / NUM_CODES / STAGE1_RATIO / FACTORIZED_HIDDEN_DIM 等 env_overrides 调节。",
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
_COST_TABLE_EF_VALUES = [2, 3, 4, 6, 8, 12]


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


def frontier_has_feasible_entry(
    frontier: dict[str, Any],
    registry: dict[str, str],
    d_in: int | None = None,
    n_output: int | None = None,
) -> bool:
    """Check if frontier has at least one feasible entry within total cost budget.

    Feasible means: total_cost / original_matmul_accesses <= 1.5
    where total_cost = encoder selection cost + deployment lookup cost.
    """
    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        cfg = dict(entry.get("config", {}) or {})
        if entry.get("target_profile") is not None and "target_profile" not in cfg:
            cfg["target_profile"] = entry["target_profile"]
        if d_in is None and not profile_matches(cfg, default_target_profile()):
            continue
        family_name = str(
            cfg.get("family_name") or entry.get("architecture") or ""
        ).lower()
        if not is_compatible_label(registry.get(family_name)):
            continue
        if d_in is not None:
            entry_budget = 1.5 * d_in * (n_output if n_output is not None else 4 * d_in)
            cost_d_in = d_in
            cost_n_output = n_output if n_output is not None else 4 * d_in
        else:
            entry_budget = budget_accesses(cfg)
            profile = resolve_target_profile(cfg)
            cost_d_in = profile.d_in
            cost_n_output = profile.n_output

        # Prefer stored total_cost; fallback to selection_cost + deployment
        tc = entry.get("total_cost")
        if tc is not None:
            if float(tc) <= entry_budget:
                return True
            continue
        sel_cost = entry.get("selection_cost")
        deploy = entry.get("deployment_accesses", 0) or 0
        if sel_cost is not None:
            if float(sel_cost) + float(deploy) <= entry_budget:
                return True
            continue
        # No stored cost — compute from entry fields
        cost = compute_selection_cost(
            str(entry.get("architecture") or cfg.get("architecture") or "topk"),
            k=int(entry.get("k") or cfg.get("k") or 128),
            ef=int(entry.get("ef") or cfg.get("expansion_factor") or 12),
            d_in=cost_d_in,
            n_output=cost_n_output,
        )
        if "error" not in cost and cost.get("combined_feasible", False):
            return True

    return False


def format_cost_summary(cost: dict[str, Any]) -> str:
    """Format a cost dict as a compact one-liner."""
    if "error" in cost:
        return f"cost=error({cost['error']})"
    ratio = cost.get("ratio", "?")
    budget = cost.get("budget_ratio", "?")
    feasible = "feasible" if cost.get("feasible") else "infeasible"
    return f"cost={ratio}x baseline (budget={budget}x, {feasible})"


def extract_full_prior_document(prior_text: str) -> str:
    """Return the full prior document for fresh-session prompts."""
    return prior_text.strip()


def extract_compatibility_digest(prior_text: str) -> str:
    """Extract the compact compatibility-focused sections for proposal prompts."""
    if not prior_text:
        return ""

    keep_headers = (
        "## 2. LUTurbo/Lottable 兼容性约束",
        "## 10.4 把“兼容性”写进 prompt，而不是靠事后人工筛选",
    )
    lines = prior_text.splitlines()
    extracted: list[str] = []
    capture = False
    for line in lines:
        if any(line.startswith(header) for header in keep_headers):
            capture = True
        elif capture and line.startswith("## "):
            capture = False

        if capture:
            extracted.append(line)

    return "\n".join(extracted).strip()


def filter_compatible_family_names(
    family_names: Iterable[str],
    registry: dict[str, str],
) -> list[str]:
    return [
        str(name).lower()
        for name in family_names
        if is_compatible_label(registry.get(str(name).lower()))
    ]


def _normalize_family_name(raw: str) -> str:
    text = raw.strip().strip("`")
    if not text:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_]+", text):
        return ""
    return text.lower()
