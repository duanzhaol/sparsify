"""Centralized prompt construction for the autoresearch framework.

All prompt assembly lives here. Each prompt section is a standalone function
returning a plain string (empty if nothing to contribute). Three compose
functions assemble sections in the correct 4-layer order.

Layer 1: Hard Constraints (role, rules, single-variable)
Layer 2: Current Decision State (policy, frontier, recent rounds, hints)
Layer 3: Rolling Memory (filtered families, failures, hypotheses)
Layer 4: Reference Digests (operator guide summary, prior research summary)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .compatibility import (
    compatibility_hard_rules,
    compute_selection_cost,
    compute_selection_cost_table,
    extract_compatibility_digest,
    extract_full_prior_document,
    format_cost_summary,
    is_compatible_label,
    summarize_registry_counts,
)

# ---------------------------------------------------------------------------
# Constants — Hard constraints baked into prompt template
# ---------------------------------------------------------------------------

ROLE_AND_OBJECTIVE = """\
你是本仓库中的 LUTurbo 自动研究 Agent。

系统使命：
- 为 LUTurbo 搜索可部署的基向量分解模块
- 真正目标是降低 LUTurbo 在 CPU 上的总在线成本与时延，而不只是把 sparsify-ascend 内部的 SAE 指标做高
- 优先考虑那些有希望同时降低“选择 + 系数计算 + 查表 + 在线补偿”总成本的方案"""

MODULE_CONTRACT = """\
模块契约：
- 学到的表示必须能导出到 LUTurbo 的“查表 + 加权求和 + 选择性补偿”推理流水线
- 最终重构必须能表示成一个或多个静态向量库上的有限加权和
- SAE 只是当前基线实现，不是必须坚持的唯一架构形式"""

PROXY_OBJECTIVE = """\
训练侧代理目标：
- 维护并改善 (total_cost, FVU) 的 2D Pareto frontier
- total_cost = encoder 选择成本 + 部署查表成本，两者加和后与 FVU 做 Pareto 权衡
- 用尽可能低的 total_cost 达到尽可能低的 FVU
- 当前阶段重点探索区域：total_cost < 0.5x；首要任务是在这个区域内尽量降低 FVU
- 0.5x-1.0x 区间可作为质量锚点或过渡对照，但不应继续作为默认主战场
- >1.0x 区间只保留少量高质量参考点，不应继续主导搜索注意力
- 这个 frontier 只是 LUTurbo 可用性的代理指标，不是最终系统指标
- K, EF, TRUNK_RANK, NUM_CODES 等参数均可自由调整，目标是 Pareto front 上的最优权衡
- encoder 选择成本由 EF 主导，降低主要靠降低 EXPANSION_FACTOR
- 部署查表成本由 K、trunk_rank、NUM_CODES 主导，K 增大会增加 K×n 查表访存
- 不同架构在相同 EF 下的 encoder 成本差异很大（见成本速查表）
- 可优先尝试天然低成本方向：极低 EF / 极低 K、factorized scorer、更少静态库、更短选择链路、轻量多专家/多子库方案
- 对 MoE-like 方向，只有在 router 足够轻、expert 更小、且最终仍能导出为静态子库有限加权和时才值得优先尝试"""

EXECUTION_LAYER = """\
执行层是固定的：
- 训练：scripts/autoresearch_test.sh
- 结果：research/controller.py
- 记忆：research/history/
- 当前执行沙盒：sparsify-ascend 是面向 LUTurbo 搜索的训练与评估环境"""

EDIT_RULES = """\
规则：
- 只能编辑 sparsify/ 下的文件
- 纯参数实验必须使用 env_overrides
- 每一轮只允许一个主假设
- 必须设置 primary_variable，明确本轮主变化维度
- 不要返回 command="stop"
- 最终必须返回一个符合 action schema 的 JSON 对象"""

SINGLE_VARIABLE_PRINCIPLE = "单变量原则：每一轮只改变一个主维度。"

HARD_CONSTRAINT_REMINDER = (
    "提醒：每轮只改一个主变量；只能编辑 sparsify/；最终返回 JSON。"
    "Frontier 基于 (total_cost, FVU) 2D Pareto front，total_cost = encoder + deployment。"
)

ARCHITECTURE_INTEGRATION_SKILL_PATH = Path(
    "/root/.codex/skills/sae-architecture-integration/SKILL.md"
)

# Hint prefixes to filter from tactical hints (legacy constraints no longer active)
_HARD_CONSTRAINT_HINT_PREFIXES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Layer 1: Hard Constraints
# ---------------------------------------------------------------------------


def section_hard_constraints() -> str:
    """All hard constraint text, always present at prompt top."""
    return "\n\n".join([
        ROLE_AND_OBJECTIVE,
        MODULE_CONTRACT,
        PROXY_OBJECTIVE,
        EXECUTION_LAYER,
        EDIT_RULES,
        SINGLE_VARIABLE_PRINCIPLE,
        compatibility_hard_rules(),
    ])


# ---------------------------------------------------------------------------
# Layer 2: Current Decision State
# ---------------------------------------------------------------------------


def section_policy_guidance(policy_guidance: str) -> str:
    if not policy_guidance:
        return ""
    return f"策略引导：\n{policy_guidance}"


def section_agent_state(
    round_index: int,
    consecutive_crashes: int,
    consecutive_no_improve: int,
    rounds_since_new_family: int,
) -> str:
    return (
        f"Agent 状态：round={round_index}, "
        f"consecutive_crashes={consecutive_crashes}, "
        f"consecutive_no_improve={consecutive_no_improve}, "
        f"rounds_since_new_family={rounds_since_new_family}"
    )


def section_frontier(
    frontier: dict[str, Any],
    registry: dict[str, str],
    limit: int = 10,
    cost_cache: dict[str, dict] | None = None,
) -> str:
    """按 total_cost 排序的 2D Pareto frontier (total_cost, FVU)。"""
    if cost_cache is None:
        cost_cache = {}
    entries: list[tuple[float, str]] = []
    low_cost_count = 0
    low_cost_best_fvu = float("inf")

    for _key, entry in frontier.items():
        if not isinstance(entry, dict):
            continue
        fvu = entry.get("fvu", "?")
        arch = entry.get("architecture", "?")
        cfg = entry.get("config", {})
        family_name = str(cfg.get("family_name") or arch).lower()
        k = entry.get("k", "?")
        ef = entry.get("ef", "?")
        lr = cfg.get("lr", "?")
        opt = cfg.get("optimizer", "?")

        if not is_compatible_label(registry.get(family_name)):
            continue

        # Get cost components: prefer stored, fallback to compute
        sel_ratio = None
        deploy_ratio = entry.get("deployment_ratio")
        combined_ratio = None
        total_cost_val = entry.get("total_cost")

        extra_config = _extract_cost_params(cfg)
        extra_key = "|".join(f"{ck}={cv}" for ck, cv in sorted(extra_config.items())) if extra_config else ""
        cost_key = f"{arch}|{k}|{ef}|{extra_key}"

        def _ensure_cost_cache() -> dict:
            if cost_key not in cost_cache:
                cost_cache[cost_key] = compute_selection_cost(
                    str(arch), k=int(k) if k != "?" else 128, ef=int(ef) if ef != "?" else 12,
                    extra_config=extra_config or None,
                )
            return cost_cache[cost_key]

        if total_cost_val is not None:
            # Have stored total_cost
            d_in = 1024
            original = d_in * 4 * d_in
            sel_cost = entry.get("selection_cost")
            sel_ratio = round(sel_cost / original, 2) if sel_cost and original > 0 else "?"
            combined_ratio = round(float(total_cost_val) / original, 2) if original > 0 else "?"
            if deploy_ratio is None:
                cost = _ensure_cost_cache()
                deploy_ratio = cost.get("deployment_ratio", "?") if "error" not in cost else "?"
        else:
            # No stored total_cost — compute
            cost = _ensure_cost_cache()
            if "error" not in cost:
                sel_ratio = cost.get("ratio", "?")
                deploy_ratio = cost.get("deployment_ratio", "?")
                combined_ratio = cost.get("combined_ratio", "?")
                total_cost_val = cost.get("combined_accesses", 0)
            else:
                sel_ratio = "?"
                deploy_ratio = "?"
                combined_ratio = "?"
                total_cost_val = 0

        combined_feasible = isinstance(combined_ratio, (int, float)) and combined_ratio <= 1.5
        if isinstance(combined_ratio, (int, float)) and combined_ratio < 0.5:
            low_cost_count += 1
            try:
                low_cost_best_fvu = min(low_cost_best_fvu, float(fvu))
            except (TypeError, ValueError):
                pass
        tag = " [FEASIBLE]" if combined_feasible else " [OVER BUDGET]"
        entries.append((
            float(total_cost_val),
            f"  total={combined_ratio}x (sel={sel_ratio}x + deploy={deploy_ratio}x)  fvu={fvu:<12}  arch={arch}  K={k} EF={ef}  lr={lr} opt={opt}{tag}",
            combined_feasible,
        ))

    entries.sort()

    parts = ["训练代理 frontier（2D Pareto: total_cost vs FVU）："]
    if entries:
        feasible_count = sum(1 for _, _, f in entries[:limit] if f)
        for _, line, _ in entries[:limit]:
            parts.append(line)
        if feasible_count == 0:
            parts.append("  !! 当前 frontier 所有点均超出总成本预算（>1.5x）。首要任务：找到成本可行的配置。")
        if low_cost_count == 0:
            parts.append("  !! <0.5x 低开销区域暂无 frontier 点，建议定向探索简单/更小/更稀疏的结构。")
        elif low_cost_count <= 2:
            if low_cost_best_fvu < float('inf'):
                parts.append(f"  !! <0.5x 区域仅有 {low_cost_count} 个点，当前最佳 FVU={low_cost_best_fvu:.4f}，仍明显欠覆盖。")
            else:
                parts.append(f"  !! <0.5x 区域仅有 {low_cost_count} 个点，仍可进一步探索。")
    else:
        parts.append("  （暂无条目）")

    return "\n".join(parts)


def section_selection_cost_status(
    frontier: dict[str, Any],
    registry: dict[str, str],
) -> str:
    """Show cost status for best <0.5x point, then feasible/global references."""
    best_global: dict[str, Any] | None = None
    best_global_fvu = float("inf")
    best_feasible: dict[str, Any] | None = None
    best_feasible_fvu = float("inf")
    best_low_cost: dict[str, Any] | None = None
    best_low_cost_fvu = float("inf")

    d_in = 1024
    budget = 1.5 * d_in * 4 * d_in
    low_cost_budget = 0.5 * d_in * 4 * d_in

    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        cfg = entry.get("config", {})
        family_name = str(cfg.get("family_name") or entry.get("architecture", "")).lower()
        if not is_compatible_label(registry.get(family_name)):
            continue
        fvu = float(entry.get("fvu", float("inf")))

        if fvu < best_global_fvu:
            best_global_fvu = fvu
            best_global = entry

        # Check feasibility
        tc = entry.get("total_cost")
        if tc is None:
            sel = entry.get("selection_cost")
            deploy = entry.get("deployment_accesses", 0) or 0
            tc = (float(sel) + float(deploy)) if sel is not None else None
        if tc is not None and float(tc) <= budget and fvu < best_feasible_fvu:
            best_feasible_fvu = fvu
            best_feasible = entry
        if tc is not None and float(tc) < low_cost_budget and fvu < best_low_cost_fvu:
            best_low_cost_fvu = fvu
            best_low_cost = entry

    if best_global is None:
        return ""

    parts = ["成本状态："]

    def _cost_for_entry(entry: dict) -> dict | None:
        arch = entry.get("architecture", "?")
        k = int(entry.get("k", 128))
        ef = int(entry.get("ef", 12))
        cfg = entry.get("config", {})
        extra_config = _extract_cost_params(cfg)
        cost = compute_selection_cost(arch, k=k, ef=ef, extra_config=extra_config or None)
        return cost if "error" not in cost else None

    def _format_entry(entry: dict, cost: dict, label: str) -> None:
        arch = entry.get("architecture", "?")
        k = int(entry.get("k", 128))
        ef = int(entry.get("ef", 12))
        combined_ratio = cost.get("combined_ratio", "?")
        combined_budget = cost.get("combined_budget_ratio", "?")
        feasible_tag = " [FEASIBLE]" if cost.get("combined_feasible", False) else " [OVER BUDGET]"

        parts.append(f"  {label} {arch} (K={k}, EF={ef}){feasible_tag}:")
        parts.append(
            f"    总成本: {combined_ratio}x | "
            f"预算比率: {combined_budget}x | "
            f"encoder: {cost['ratio']}x | 部署: {cost.get('deployment_ratio', '?')}x | "
            f"FVU: {entry.get('fvu', '?')}"
        )

    # Re-check feasibility with fresh computation (stored values may be stale)
    true_feasible: dict[str, Any] | None = None
    true_feasible_fvu = float("inf")
    true_feasible_cost: dict | None = None
    global_cost: dict | None = None

    # Compute cost for global best
    if best_global is not None:
        global_cost = _cost_for_entry(best_global)

    # Find true best feasible using fresh combined_feasible
    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        cfg = entry.get("config", {})
        family_name = str(cfg.get("family_name") or entry.get("architecture", "")).lower()
        if not is_compatible_label(registry.get(family_name)):
            continue
        fvu = float(entry.get("fvu", float("inf")))
        if fvu >= true_feasible_fvu:
            continue
        # Reuse global_cost if same entry
        if entry is best_global and global_cost is not None:
            cost = global_cost
        else:
            cost = _cost_for_entry(entry)
        if cost is not None and cost.get("combined_feasible", False):
            true_feasible = entry
            true_feasible_fvu = fvu
            true_feasible_cost = cost

    # Prefer the best true <0.5x point if one exists
    true_low_cost: dict[str, Any] | None = None
    true_low_cost_cost: dict | None = None
    true_low_cost_fvu = float("inf")
    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        cfg = entry.get("config", {})
        family_name = str(cfg.get("family_name") or entry.get("architecture", "")).lower()
        if not is_compatible_label(registry.get(family_name)):
            continue
        try:
            fvu = float(entry.get("fvu", float("inf")))
        except (TypeError, ValueError):
            continue
        if fvu >= true_low_cost_fvu:
            continue
        cost = global_cost if entry is best_global and global_cost is not None else _cost_for_entry(entry)
        if cost is not None and float(cost.get("combined_accesses", float("inf"))) < low_cost_budget:
            true_low_cost = entry
            true_low_cost_fvu = fvu
            true_low_cost_cost = cost

    shown_feasible = False
    if true_low_cost is not None and true_low_cost_cost is not None:
        _format_entry(true_low_cost, true_low_cost_cost, "当前最佳 <0.5x 点 →")
        # Show feasible separately only if it's a different entry
        if true_feasible is not None and true_feasible_cost is not None and true_feasible is not true_low_cost:
            _format_entry(true_feasible, true_feasible_cost, "最优可行点 →")
            shown_feasible = True
    elif true_feasible is not None and true_feasible_cost is not None:
        _format_entry(true_feasible, true_feasible_cost, "最优可行点 →")
        shown_feasible = True

    # Show global best only if different from already shown points
    if best_global is not None and global_cost is not None and best_global is not true_low_cost and best_global is not true_feasible:
        _format_entry(best_global, global_cost, "全局最低 FVU →")

    if true_feasible is None:
        parts.append("  !! 当前无可行点（所有 frontier 点的 total_cost > 1.5x）")

    parts.append(
        "  降低 encoder 成本: 降 EF / TRUNK_RANK / NUM_CODES / STAGE1_RATIO / FACTORIZED_HIDDEN_DIM"
    )
    parts.append(
        "  降低部署成本: 降 K / TRUNK_RANK / NUM_CODES"
    )

    return "\n".join(parts)


def section_cost_feasibility_table(registry: dict[str, str]) -> str:
    """成本可行性速查表：让 agent 知道各 (架构, EF) 组合是否在预算内。"""
    table = compute_selection_cost_table()
    if not table:
        return ""

    from .compatibility import _COST_TABLE_ARCHITECTURES, _COST_TABLE_EF_VALUES

    # Filter to compatible architectures only
    archs = [a for a in _COST_TABLE_ARCHITECTURES if is_compatible_label(registry.get(a))]
    if not archs:
        archs = _COST_TABLE_ARCHITECTURES  # fallback: show all if registry empty
    efs = _COST_TABLE_EF_VALUES

    # Build header
    max_name = max(len(a) for a in archs)
    header = f"{'架构':<{max_name}}  " + "  ".join(f"EF={ef:<3d}" for ef in efs)

    rows: list[str] = []
    for arch in archs:
        cells: list[str] = []
        for ef in efs:
            cost = table.get((arch, ef))
            if cost and "error" not in cost:
                ratio = cost.get("ratio", 0)
                feasible = cost.get("feasible", False)
                mark = "+" if feasible else "-"
                cells.append(f"{ratio}x{mark}")
            else:
                cells.append("?    ")
        row = f"{arch:<{max_name}}  " + "  ".join(f"{c:<7s}" for c in cells)
        rows.append(row)

    parts = [
        "Encoder 侧成本速查表（非最终 budget，ratio 指 encoder-only）：",
        f"  {header}",
    ]
    for row in rows:
        parts.append(f"  {row}")
    parts.append("  注意：以上仅为 encoder 选择成本（不依赖 K）。最终 budget 以 total_cost = selection + deployment 为准。")
    parts.append("  部署查表成本 K×n 额外贡献约：K=32 +3%，K=64 +6%，K=128 +12%。")

    return "\n".join(parts)


def section_recent_rounds(
    round_summaries: list[dict[str, Any]],
    registry: dict[str, str],
) -> str:
    """最近几轮兼容架构摘要，每轮一行。"""
    if not round_summaries:
        return ""
    compatible_lines: list[str] = []
    for summary in round_summaries:
        if not isinstance(summary, dict):
            continue
        family_name = str(
            summary.get("family_name")
            or summary.get("action", {}).get("family_name")
            or summary.get("result", {}).get("architecture")
            or ""
        ).lower()
        if family_name and not is_compatible_label(registry.get(family_name)):
            continue
        compatible_lines.append(f"  {_trim_round_summary(summary)}")
    if not compatible_lines:
        return "最近几轮（兼容架构）：\n  （最近没有兼容 family 的新结果）"
    return f"最近几轮（兼容架构，{len(compatible_lines)}）：\n" + "\n".join(compatible_lines)


def section_tactical_hints(hints: list[dict[str, Any]]) -> str:
    """只保留战术性 hint，过滤掉已经固化到模板里的 K/EF 硬约束。"""
    tactical = []
    for hint in hints:
        text = str(hint.get("text") or hint.get("message") or "").strip()
        if not text:
            continue
        if any(text.startswith(p) for p in _HARD_CONSTRAINT_HINT_PREFIXES):
            continue
        tactical.append(text)

    if not tactical:
        return ""
    lines = ["操作提示："]
    for i, t in enumerate(tactical, 1):
        lines.append(f"  {i}. {t}")
    return "\n".join(lines)


def section_compatibility_status(registry: dict[str, str]) -> str:
    if not registry:
        return ""
    return summarize_registry_counts(registry)


# ---------------------------------------------------------------------------
# Layer 3: Rolling Memory
# ---------------------------------------------------------------------------


def section_memory_brief(
    memory: dict[str, Any],
    frontier: dict[str, Any],
    registry: dict[str, str],
    recent_round_limit: int = 10,
) -> str:
    """精简记忆：只保留 frontier 家族、最近测试家族与近期失败。"""
    families = memory.get("architecture_families", {})

    # Which families to show
    frontier_families: set[str] = set()
    for entry in frontier.values():
        if isinstance(entry, dict):
            cfg = entry.get("config", {})
            fn = cfg.get("family_name", entry.get("architecture", ""))
            if fn and is_compatible_label(registry.get(fn.lower())):
                frontier_families.add(fn.lower())

    recent_families: set[str] = set()
    for rr in memory.get("recent_rounds", [])[-recent_round_limit:]:
        fn = rr.get("family_name", "")
        if fn and is_compatible_label(registry.get(fn.lower())):
            recent_families.add(fn.lower())

    show = frontier_families | recent_families

    parts: list[str] = ["架构家族（frontier 持有者 + 最近测试）："]
    for name in sorted(show):
        fam = families.get(name)
        if fam is None:
            continue
        status = fam.get("status", "?")
        best = fam.get("best_fvu")
        lr = fam.get("last_round", "?")
        history = fam.get("tested_configs", [])[-3:]
        hist_str = "; ".join(
            f"r{tc.get('round','?')} k{tc.get('k','?')} {tc.get('decision','?')}"
            for tc in history
        )
        best_str = f" best_fvu={best}" if best is not None else ""
        parts.append(f"  {name} [{status}]{best_str} last_r={lr}: {hist_str}")

    # Recent training failures
    for label, key, n in [
        ("Recent training failures", "recent_training_failures", 4),
        ("Recent sanity failures", "recent_sanity_failures", 3),
    ]:
        fails = memory.get(key, [])[-n:]
        if fails:
            parts.append("")
            label_cn = "近期训练失败" if key == "recent_training_failures" else "近期 sanity 失败"
            parts.append(f"{label_cn}：")
            for f in fails:
                if isinstance(f, dict):
                    parts.append(
                        f"  r{f.get('round','?')} {f.get('family_name','?')} "
                        f"{f.get('error_type','')}: {_truncate(str(f.get('error_summary','')), 80)}"
                    )

    filtered_count = sum(
        1
        for name in memory.get("architecture_families", {})
        if not is_compatible_label(registry.get(str(name).lower()))
    )
    if filtered_count:
        parts.append("")
        parts.append(f"已从运行时决策上下文中过滤 {filtered_count} 个不兼容 family。")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Layer 4: Reference Digests
# ---------------------------------------------------------------------------


def section_operator_guide_digest(guide_text: str, max_chars: int = 3000) -> str:
    """从 operator_guide.md 中提取 Runtime Priorities。"""
    if not guide_text:
        return ""

    lines = guide_text.split("\n")
    in_section = False
    extracted: list[str] = []

    for line in lines:
        if line.strip().startswith("## Runtime Priorities"):
            in_section = True
            extracted.append(line)
            continue
        if in_section:
            if line.startswith("## ") and "Runtime" not in line:
                break
            extracted.append(line)

    if extracted:
        text = "\n".join(extracted)
        if len(text) > max_chars:
            text = text[:max_chars - 20] + "\n[truncated]"
        return f"Operator guide 关键优先级：\n{text}"

    truncated = guide_text[:max_chars - 20] + "\n[truncated]"
    return f"Operator guide 摘要：\n{truncated}"


def section_prior_research_digest(prior_text: str) -> str:
    """兼容性摘要：不再只截取 ##1/##2。"""
    if not prior_text:
        return ""
    digest = extract_compatibility_digest(prior_text) or prior_text
    return f"历史研究关键结论：\n{digest}"


def section_prior_research_full(prior_text: str) -> str:
    """Fresh session must receive the full prior document."""
    if not prior_text:
        return ""
    return f"完整历史研究文档：\n{extract_full_prior_document(prior_text)}"


def section_architecture_checklist(
    checklist_text: str,
    memory: dict[str, Any],
    raw_summaries: list[dict[str, Any]],
) -> str:
    """按条件注入架构集成 checklist。

    这里使用 state.recent_round_summaries() 的原始 dict，
    而不是 recent_round_summaries_trimmed() 的字符串摘要。
    """
    if not checklist_text:
        return ""
    if not _should_include_checklist(memory, raw_summaries):
        return ""
    return f"架构集成检查清单：\n{checklist_text}"


# ---------------------------------------------------------------------------
# Compose functions
# ---------------------------------------------------------------------------


def compose_proposal(state: Any, policy_guidance: str) -> str:
    """四层结构的完整 proposal prompt。"""
    sections: list[str] = []

    registry = state.load_compatibility_registry()

    # Layer 1
    sections.append(section_hard_constraints())

    # Layer 2
    sections.append(section_policy_guidance(policy_guidance))
    sections.append(section_compatibility_status(registry))
    sections.append(section_agent_state(
        state.round_index, state.consecutive_crashes,
        state.consecutive_no_improve, state.rounds_since_new_family,
    ))
    sections.append(section_frontier(state.frontier, registry))
    sections.append(section_selection_cost_status(state.frontier, registry))
    sections.append(section_cost_feasibility_table(registry))
    sections.append(section_recent_rounds(
        state.recent_round_summaries(limit=5),
        registry,
    ))
    sections.append(section_tactical_hints(state.get_pending_hints()[:8]))

    # Layer 3
    sections.append(section_memory_brief(state.memory, state.frontier, registry))

    # Layer 4
    sections.append(section_operator_guide_digest(
        state.load_operator_guide_excerpt(),
    ))
    sections.append(section_prior_research_digest(
        state.load_prior_research(),
    ))
    sections.append(section_prior_research_full(
        state.load_prior_research(),
    ))

    checklist = _load_architecture_checklist()
    if checklist:
        raw = state.recent_round_summaries(limit=2)
        sections.append(section_architecture_checklist(checklist, state.memory, raw))

    return "\n\n".join(s for s in sections if s)


def compose_resume(state: Any, round_id: int, policy_guidance: str) -> str:
    """用于 session resume 的轻量增量 prompt。"""
    sections: list[str] = []

    registry = state.load_compatibility_registry()

    sections.append(f"继续 LUTurbo 研究会话。当前轮次：Round {round_id}。")
    sections.append("返回一个符合 action schema 的 JSON 对象，不要使用 markdown 代码块。")
    sections.append(HARD_CONSTRAINT_REMINDER)
    sections.append(compatibility_hard_rules())

    sections.append(section_policy_guidance(policy_guidance))
    sections.append(section_compatibility_status(registry))
    sections.append(section_agent_state(
        state.round_index, state.consecutive_crashes,
        state.consecutive_no_improve, state.rounds_since_new_family,
    ))
    sections.append(section_frontier(state.frontier, registry, limit=8))
    sections.append(section_selection_cost_status(state.frontier, registry))
    sections.append(section_cost_feasibility_table(registry))
    sections.append(section_recent_rounds(
        state.recent_round_summaries(limit=4),
        registry,
    ))
    sections.append(section_tactical_hints(state.get_pending_hints()[:4]))
    sections.append(section_prior_research_digest(
        state.load_prior_research(),
    ))

    # Open hypotheses
    hypotheses = state.memory.get("next_hypotheses", [])[:5]
    if hypotheses:
        lines = ["当前开放假设："]
        for h in hypotheses:
            lines.append(f"  - {_truncate(h, 100)}")
        sections.append("\n".join(lines))

    checklist = _load_architecture_checklist()
    if checklist:
        raw = state.recent_round_summaries(limit=2)
        sections.append(section_architecture_checklist(checklist, state.memory, raw))

    return "\n\n".join(s for s in sections if s)


def compose_repair(
    round_id: int,
    base_action: Any,  # Action
    failure_kind: str,
    failure_payload: dict[str, Any],
    repair_attempt: int,
    max_repair_attempts: int,
) -> str:
    """Repair prompt：修复工程性阻塞，但不改变实验设计。"""
    sections: list[str] = []

    sections.append(f"Round {round_id} 进入 repair 模式。")
    sections.append(
        "上一轮代码修改遇到了工程性阻塞。\n"
        "不要重新设计实验，不要修改 family_name 或 env_overrides。\n"
        "你的任务是补丁修复实现，让原始实验能够跑通。\n"
        "只能在 sparsify/ 内修改。\n"
        f"当前是第 {repair_attempt} / {max_repair_attempts} 次 repair 尝试。\n"
        "最终返回一个符合 action schema 的 JSON 对象。"
    )
    sections.append(HARD_CONSTRAINT_REMINDER)

    payload = {
        "round": round_id,
        "repair_attempt": repair_attempt,
        "max_repair_attempts": max_repair_attempts,
        "failure_kind": failure_kind,
        "base_action": {
            "family_name": base_action.family_name,
            "family_stage": base_action.family_stage,
            "change_type": base_action.change_type,
            "env_overrides": base_action.env_overrides,
            "summary": base_action.summary,
            "hypothesis": base_action.hypothesis,
        },
        "failure_payload": failure_payload,
    }
    sections.append(f"Repair 上下文：\n{json.dumps(payload, indent=2, ensure_ascii=False)}")

    checklist = _load_architecture_checklist()
    if checklist:
        sections.append(f"架构集成检查清单：\n{checklist}")

    return "\n\n".join(s for s in sections if s)


# ---------------------------------------------------------------------------
# Helpers (moved from agent.py)
# ---------------------------------------------------------------------------


def _load_architecture_checklist() -> str:
    """从 SKILL.md 加载架构集成检查清单。"""
    if not ARCHITECTURE_INTEGRATION_SKILL_PATH.exists():
        return ""
    text = ARCHITECTURE_INTEGRATION_SKILL_PATH.read_text().strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2].strip()
    return text


def _should_include_checklist(
    memory: dict[str, Any],
    recent_summaries: list[dict[str, Any]],
) -> bool:
    """判断是否需要注入架构 checklist。

    输入必须是 list[dict] 的原始 round summaries，而不是 list[str]。
    """
    for s in recent_summaries[-2:]:
        if not isinstance(s, dict):
            continue
        action = s.get("action", {})
        result = s.get("result", {})
        if str(action.get("change_type", "")) == "edit_sae_code":
            return True
        if s.get("family_stage") == "prototype" and result.get("decision") in ("crash", "policy_reject"):
            return True
    for entry in (
        list(memory.get("recent_training_failures", []))[-4:]
        + list(memory.get("recent_sanity_failures", []))[-2:]
    ):
        if isinstance(entry, dict):
            summary = str(entry.get("error_summary") or "").lower()
            if any(kw in summary for kw in ("unknown architecture", "dispatch", "registration")):
                return True
    return False


def summarize_results(rows: list[dict[str, str]]) -> list[str]:
    """Compress result rows into one-liner strings.

    Kept as public utility. No longer used in prompt assembly
    (section_recent_rounds uses the more informative round summaries).
    """
    lines: list[str] = []
    for row in rows:
        eid = row.get("experiment_id", "?")
        dec = row.get("decision", "?")
        fvu = row.get("val_fvu", "")
        k = row.get("k", "?")
        arch = row.get("architecture", "?")
        desc = row.get("description", "")
        fvu_part = f" fvu={fvu}" if fvu else ""
        desc_part = f" | {desc[:60]}" if desc else ""
        lines.append(f"{eid} {arch} k{k} {dec}{fvu_part}{desc_part}")
    return lines


def _trim_round_summary(summary: dict[str, Any]) -> str:
    action = summary.get("action", {})
    result = summary.get("result", {})
    round_id = summary.get("round", "?")
    family_name = summary.get("family_name", "?")
    change_type = action.get("change_type", "?")
    decision = result.get("decision", "?")
    fvu = result.get("val_fvu")
    duration = summary.get("duration_sec")
    hypothesis = _truncate(str(action.get("hypothesis") or ""), 80)
    fvu_part = f" fvu={fvu}" if fvu not in (None, "") else ""
    dur_part = f" {duration}s" if duration not in (None, "") else ""
    hyp_part = f" | {hypothesis}" if hypothesis else ""
    return f"r{round_id} {family_name} {change_type} -> {decision}{fvu_part}{dur_part}{hyp_part}"


def _extract_cost_params(config: dict[str, Any]) -> dict[str, Any]:
    """Extract architecture-specific params for cost estimation from config."""
    extra: dict[str, Any] = {}
    for key, cfg_key in [
        ("trunk_rank", "trunk_rank"),
        ("num_codes", "num_codes"),
        ("stage1_ratio", "stage1_ratio"),
        ("factorized_hidden_dim", "factorized_hidden_dim"),
        ("TRUNK_RANK", "trunk_rank"),
        ("NUM_CODES", "num_codes"),
        ("STAGE1_RATIO", "stage1_ratio"),
        ("FACTORIZED_HIDDEN_DIM", "factorized_hidden_dim"),
    ]:
        val = config.get(key)
        if val is not None and val != "":
            try:
                extra[cfg_key] = float(val) if "." in str(val) else int(val)
            except (ValueError, TypeError):
                pass
    return extra


def _truncate(s: str, limit: int) -> str:
    if not isinstance(s, str):
        return str(s)[:limit]
    return s if len(s) <= limit else s[:limit - 3] + "..."

