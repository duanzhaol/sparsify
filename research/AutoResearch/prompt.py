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
from .config_resolution import (
    config_from_round_summary,
    render_env_config,
    structured_config_from_round_summary,
    summary_config_source,
    summary_is_usable_reference,
)
from .target_profile import default_target_profile, profile_matches, resolve_target_profile

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
- 当前 run 训练 hookpoint 固定为 `layers.[3].self_attn.q_proj`
- 当前成本 proxy 按 fused QKV 部署矩阵 `1024 x 4096` 统计，而不是按 q_proj 单矩阵 `1024 x 2048` 统计
- 用尽可能低的 total_cost 达到尽可能低的 FVU
- 当前 target 上，plain `EF=1` selector family 已经提供了一批成本锚点；它们现在主要用于对照和校准，而不是默认继续作为唯一主线
- 当前阶段的主要任务不是继续细扫旧 family 的 K/LR 邻域，而是验证新的结构族是否能形成新的 Pareto 段
- 允许在 `0.3x-0.8x total_cost` 带内建立结构性 anchor；不要求所有轮次都继续压最低成本，只要求问题清晰、结构信息增量高
- 这个 frontier 只是 LUTurbo 可用性的代理指标，不是最终系统指标
- 旧位置、旧轮次、旧 family 排名都只算弱先验；在新 target 上必须重新验证
- K, EF, TRUNK_RANK, NUM_CODES 等参数均可自由调整，目标是 Pareto front 上的最优权衡
- encoder 选择成本由 EF 主导，降低主要靠降低 EXPANSION_FACTOR
- 部署查表成本由 K、trunk_rank、NUM_CODES 主导，K 增大会增加 K×n 查表访存
- 不同架构在相同 EF 下的 encoder 成本差异很大（见成本速查表）
- 接下来大部分探索预算应投向结构新意更高的方向，而不是继续在 `topk / jumprelu / factorized_topk` 上做重复的局部扫点
- 当前优先探索的方向是：轻量 expert 子库 / MoE-like 路由、`lowrank + expert`、`lowrank + expert + residual`、以及 `two-stage residual expert`
- 如果已实现 family 不覆盖这些方向，应优先做最小可运行原型，而不是退回旧 family 的局部微调
- 上述方向只是候选搜索空间，不是默认结论；不要因为旧经验预先排斥任何兼容 family
- 对 MoE-like 方向，只有在 router 足够轻、expert 更小、且最终仍能导出为静态子库有限加权和时才值得尝试"""

EXECUTION_LAYER = """\
执行层是固定的：
- 训练：scripts/autoresearch_test.sh
- 结果：research/controller.py
- 记忆：research/history/
- 当前执行沙盒：sparsify-ascend 是面向 LUTurbo 搜索的训练与评估环境
- 如果新增可调参数，不仅要在 sparsify/ 中实现，还必须同步接通 research/AutoResearch/ 下的 override/config-resolution/runner 持久化链路、scripts/autoresearch_test.sh 参数透传、以及必要的 resume/validation 路径；否则实验会静默回退到默认值
- 新增 tunable 参数时，允许编辑的最小必要范围是：sparsify/、research/AutoResearch/、scripts/autoresearch_test.sh；不要改其他路径"""

EDIT_RULES = """\
规则：
- 默认只编辑 sparsify/；只有在新增 tunable 参数或修复参数接线时，才允许同时修改 research/AutoResearch/ 与 scripts/autoresearch_test.sh 的必要文件
- 纯参数实验必须使用 env_overrides
- 每一轮只允许一个主假设
- 不要声明 primary_variable；系统会根据 reference_round 的完整配置自动判断本轮到底改了哪个参数
- 如果 change_type=param_only，默认必须显式给出 reference_round；只有当前 target profile 还没有任何可用 reference_round 时，才允许显式返回 `reference_round=null`，相对 `target_profile_baseline` 冷启动，并且仍然只能改 1 个 env 参数
- 新增 tunable 参数后的第一轮，必须先验证该参数真的进入了训练配置：至少检查 round*.config.json 和 checkpoint config.json 中该字段存在且取值正确
- 不要返回 command="stop"
- 最终必须返回一个符合 action schema 的 JSON 对象"""

SINGLE_VARIABLE_PRINCIPLE = (
    "单变量原则：每一轮只改变一个 env 参数。"
    "有 reference_round 时，env_overrides 是对该轮完整配置的 patch。"
    "若当前 target 仍是冷启动且暂无 reference_round，则允许显式返回 reference_round=null，相对 target_profile_baseline 只改 1 个 env 参数。"
)

HARD_CONSTRAINT_REMINDER = (
    "提醒：每轮只改一个 env 参数；不要声明 primary_variable；param_only 默认应显式给出 reference_round。"
    "只有当前 target 冷启动且暂无 reference_round 时，才允许显式返回 reference_round=null，相对 target_profile_baseline 只改 1 个 env 参数。"
    "默认只改 sparsify/；只有在新增 tunable 参数或修复参数接线时，才允许改 research/AutoResearch/ 与 scripts/autoresearch_test.sh 的必要文件。"
    "Frontier 基于 (total_cost, FVU) 2D Pareto front，total_cost = encoder + deployment；最终返回 JSON。"
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


def section_target_profile() -> str:
    profile = default_target_profile()
    return "\n".join([
        "当前 target profile：",
        f"  training_hookpoint={profile.training_hookpoint}",
        f"  cost_proxy={profile.cost_model_label}",
        f"  original_matmul={profile.d_in}x{profile.n_output} ({profile.original_matmul_accesses} accesses)",
    ])


def section_high_priority_directions() -> str:
    return "\n".join([
        "当前研究阶段：结构扩展优先。",
        "  plain `EF=1` selector family 现在主要作为 cost anchor / matched baseline；除 sanity 或严格对照外，不应继续占用大部分搜索预算。",
        "  当前要回答的是“新结构是否值得继续”，而不是“旧结构再调一个点会怎样”。",
        "",
        "当前高优先级未充分验证方向：",
        "  1. `expert_topk` / 轻量 MoE-like 子库路由。",
        "     目标是在保持很小 ACTIVE_EXPERTS 的同时扩大总容量；只有在 router 足够轻、总激活路径不膨胀、且最终仍能导出为静态子库有限加权和时才值得优先尝试。",
        "  2. `lowrank + expert`。",
        "     先让低秩 trunk 吃掉平滑主干，再让 expert 子库处理剩余稀疏结构。",
        "  3. `lowrank + expert + residual`。",
        "     把 trunk、子库路由、残差稀疏修正拆开，验证三段式结构是否能形成新的 Pareto 段。",
        "  4. `two-stage residual expert`。",
        "     先 coarse library，再用局部 expert residual 做第二段修正。",
        "  5. 如果当前已实现 family 不覆盖这些结构，应优先做最小可运行原型，而不是回到 `topk / jumprelu / factorized_topk` 的局部扫点。",
    ])


def _entry_config_with_target(entry: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(entry.get("config", {}) or {})
    if entry.get("target_profile") is not None and "target_profile" not in cfg:
        cfg["target_profile"] = entry["target_profile"]
    return cfg


def _summary_matches_current_target(summary: dict[str, Any]) -> bool:
    cfg = structured_config_from_round_summary(summary)
    return cfg is not None and profile_matches(cfg, default_target_profile())


def _current_target_summaries(round_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        summary for summary in round_summaries
        if isinstance(summary, dict) and _summary_matches_current_target(summary)
    ]


def _summary_round_ids(round_summaries: list[dict[str, Any]]) -> set[int]:
    round_ids: set[int] = set()
    for summary in round_summaries:
        try:
            round_ids.add(int(summary.get("round")))
        except (TypeError, ValueError, AttributeError):
            continue
    return round_ids


def _recent_summaries_with_latest(state: Any, limit: int = 20) -> list[dict[str, Any]]:
    """Return recent round summaries, forcing inclusion of the latest completed round.

    In resume-session mode the next worker should always see the immediately
    preceding round. Relying on a plain tail of round_summaries can miss that
    last round if prompt assembly observes a stale recent-summaries window.
    """
    recent_summaries = list(state.recent_round_summaries(limit=limit) or [])

    latest_round_id = int(getattr(state, "round_index", 0) or 0)
    if latest_round_id <= 0:
        return recent_summaries

    latest_summary = state.load_round_summary(latest_round_id)
    if not isinstance(latest_summary, dict) or not latest_summary:
        return recent_summaries

    latest_key = latest_round_id
    deduped: list[dict[str, Any]] = []
    seen_latest = False
    for summary in recent_summaries:
        try:
            round_id = int(summary.get("round"))
        except (TypeError, ValueError, AttributeError):
            deduped.append(summary)
            continue
        if round_id == latest_key:
            if not seen_latest:
                deduped.append(latest_summary)
                seen_latest = True
            continue
        deduped.append(summary)

    if not seen_latest:
        deduped.append(latest_summary)

    return deduped[-limit:]


def section_frontier(
    frontier: dict[str, Any],
    registry: dict[str, str],
    limit: int = 10,
    cost_cache: dict[str, dict] | None = None,
) -> str:
    """按 total_cost 排序的 2D Pareto frontier (total_cost, FVU)。"""
    if cost_cache is None:
        cost_cache = {}
    current_target = default_target_profile()
    entries: list[tuple[float, str]] = []
    saw_any_entry = False
    low_cost_count = 0
    low_cost_best_fvu = float("inf")

    for _key, entry in frontier.items():
        if not isinstance(entry, dict):
            continue
        saw_any_entry = True
        fvu = entry.get("fvu", "?")
        arch = entry.get("architecture", "?")
        cfg = _entry_config_with_target(entry)
        if not profile_matches(cfg, current_target):
            continue
        target_profile = resolve_target_profile(cfg)
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
        cost_key = (
            f"{target_profile.profile_id}|{target_profile.d_in}|{target_profile.n_output}|"
            f"{arch}|{k}|{ef}|{extra_key}"
        )

        def _ensure_cost_cache() -> dict:
            if cost_key not in cost_cache:
                cost_cache[cost_key] = compute_selection_cost(
                    str(arch),
                    k=int(k) if k != "?" else 128,
                    ef=int(ef) if ef != "?" else 12,
                    d_in=target_profile.d_in,
                    n_output=target_profile.n_output,
                    extra_config=extra_config or None,
                )
            return cost_cache[cost_key]

        if total_cost_val is not None:
            # Have stored total_cost
            original = target_profile.original_matmul_accesses
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
            parts.append("  !! <0.5x 低开销区域暂无 frontier 点，建议补充不同 family / K / EF / 结构参数的低成本样本。")
        elif low_cost_count <= 2:
            if low_cost_best_fvu < float('inf'):
                parts.append(f"  !! <0.5x 区域仅有 {low_cost_count} 个点，当前最佳 FVU={low_cost_best_fvu:.4f}，仍明显欠覆盖。")
            else:
                parts.append(f"  !! <0.5x 区域仅有 {low_cost_count} 个点，仍可进一步探索。")
    else:
        if saw_any_entry:
            parts.append("  （当前 target profile 下暂无 frontier 点；旧 target 历史已隔离）")
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

    current_target = default_target_profile()
    budget = 1.5 * current_target.original_matmul_accesses
    low_cost_budget = 0.5 * current_target.original_matmul_accesses

    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        cfg = _entry_config_with_target(entry)
        if not profile_matches(cfg, current_target):
            continue
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
        cfg = _entry_config_with_target(entry)
        target_profile = resolve_target_profile(cfg)
        extra_config = _extract_cost_params(cfg)
        cost = compute_selection_cost(
            arch,
            k=k,
            ef=ef,
            d_in=target_profile.d_in,
            n_output=target_profile.n_output,
            extra_config=extra_config or None,
        )
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
        cfg = _entry_config_with_target(entry)
        if not profile_matches(cfg, current_target):
            continue
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
        cfg = _entry_config_with_target(entry)
        if not profile_matches(cfg, current_target):
            continue
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
    current_target = default_target_profile()
    table = compute_selection_cost_table(
        d_in=current_target.d_in,
        n_output=current_target.n_output,
    )
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
        (
            f"当前成本 profile：hookpoint={current_target.training_hookpoint} | "
            f"proxy={current_target.d_in}x{current_target.n_output}"
        ),
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
        return "最近几轮（兼容架构）：\n  （当前 target 还没有历史结果）"
    saw_other_target = False
    compatible_lines: list[str] = []
    for summary in round_summaries:
        if not isinstance(summary, dict):
            continue
        if not summary_is_usable_reference(summary):
            continue
        if not _summary_matches_current_target(summary):
            saw_other_target = True
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
        if saw_other_target:
            return "最近几轮（兼容架构）：\n  （当前 target 下尚无新结果；旧 target 历史已隔离）"
        return "最近几轮（兼容架构）：\n  （最近没有兼容 family 的新结果）"
    return f"最近几轮（兼容架构，{len(compatible_lines)}）：\n" + "\n".join(compatible_lines)


def section_reference_configs(
    round_summaries: list[dict[str, Any]],
    registry: dict[str, str],
    limit: int = 2,
) -> str:
    """Show recent runnable full env configs that can serve as reference_round anchors."""
    if not round_summaries:
        return (
            "最近可用 reference_round 的完整配置：\n"
            "  （当前 target profile 尚无历史结果）\n"
            "  冷启动阶段请显式返回 reference_round=null，并相对 target_profile_baseline 只改 1 个 env 参数。"
        )

    usable: list[dict[str, Any]] = []
    saw_other_target = False
    for summary in round_summaries:
        if not isinstance(summary, dict):
            continue
        if not summary_is_usable_reference(summary):
            continue
        if not _summary_matches_current_target(summary):
            saw_other_target = True
            continue
        usable.append(summary)

    def _priority(summary: dict[str, Any]) -> tuple[int, int]:
        decision = str(summary.get("result", {}).get("decision") or "")
        round_id = int(summary.get("round") or 0)
        decision_rank = 0 if decision in {"keep", "archive"} else 1
        return (decision_rank, -round_id)

    def _family_name(summary: dict[str, Any]) -> str:
        result = summary.get("result", {})
        return str(
            summary.get("family_name")
            or summary.get("action", {}).get("family_name")
            or result.get("architecture")
            or ""
        ).lower()

    def _format_block(summary: dict[str, Any]) -> str | None:
        result = summary.get("result", {})
        decision = str(result.get("decision") or "")
        family_name = _family_name(summary)
        if family_name and not is_compatible_label(registry.get(family_name)):
            return None
        cfg = config_from_round_summary(summary)
        if not cfg:
            return None
        round_id = summary.get("round", "?")
        source = summary_config_source(summary)
        return (
            f"  r{round_id} {family_name or cfg.get('ARCHITECTURE', '?')} "
            f"(decision={decision}, source={source})\n{render_env_config(cfg)}"
        )

    ordered = sorted(usable, key=_priority)
    chosen: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    deferred: list[dict[str, Any]] = []
    for summary in ordered:
        family_name = _family_name(summary)
        if family_name and family_name in seen_families:
            deferred.append(summary)
            continue
        chosen.append(summary)
        if family_name:
            seen_families.add(family_name)

    blocks: list[str] = []
    for summary in [*chosen, *deferred]:
        if len(blocks) >= limit:
            break
        block = _format_block(summary)
        if block:
            blocks.append(block)
    if not blocks:
        if saw_other_target:
            return (
                "最近可用 reference_round 的完整配置：\n"
                "  （当前 target profile 尚无可用 reference_round；旧 target 结果已隔离）\n"
                "  冷启动阶段请显式返回 reference_round=null，并相对 target_profile_baseline 只改 1 个 env 参数。"
            )
        return ""
    return "最近可用 reference_round 的完整配置：\n" + "\n\n".join(blocks)


def section_tactical_hints(hints: list[dict[str, Any]]) -> str:
    """只保留战术性 hint，过滤掉已经固化到模板里的 K/EF 硬约束。"""
    tactical = []
    current_profile = default_target_profile()
    for hint in hints:
        text = str(hint.get("text") or hint.get("message") or "").strip()
        if not text:
            continue
        target_id = hint.get("target_profile_id")
        if target_id not in (None, "", current_profile.profile_id):
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
    round_summaries: list[dict[str, Any]],
    recent_round_limit: int = 10,
) -> str:
    """精简记忆：只保留 frontier 家族、最近测试家族与近期失败。"""
    families = memory.get("architecture_families", {})
    current_target = default_target_profile()
    target_summaries = _current_target_summaries(round_summaries)
    target_round_ids = _summary_round_ids(target_summaries)
    saw_other_target = bool(round_summaries) and not target_summaries

    # Which families to show
    frontier_families: set[str] = set()
    for entry in frontier.values():
        if isinstance(entry, dict):
            cfg = _entry_config_with_target(entry)
            if not profile_matches(cfg, current_target):
                continue
            fn = cfg.get("family_name", entry.get("architecture", ""))
            if fn and is_compatible_label(registry.get(fn.lower())):
                frontier_families.add(fn.lower())

    recent_families: set[str] = set()
    for summary in target_summaries[-recent_round_limit:]:
        fn = str(
            summary.get("family_name")
            or summary.get("action", {}).get("family_name")
            or summary.get("result", {}).get("architecture")
            or ""
        ).lower()
        if fn and is_compatible_label(registry.get(fn.lower())):
            recent_families.add(fn)

    show = frontier_families | recent_families

    parts: list[str] = ["架构家族（frontier 持有者 + 最近测试）："]
    if not show:
        if saw_other_target:
            parts.append("  （当前 target profile 尚无局部记忆；旧 target 历史已隔离）")
        else:
            parts.append("  （暂无局部记忆）")
    for name in sorted(show):
        fam = families.get(name)
        if fam is None:
            continue
        target_history = [
            tc
            for tc in fam.get("tested_configs", [])
            if isinstance(tc, dict)
            and target_round_ids
            and _safe_round_id(tc.get("round")) in target_round_ids
        ]
        best = _best_fvu_from_history(target_history)
        lr = target_history[-1].get("round", "?") if target_history else "?"
        history = target_history[-3:]
        hist_str = "; ".join(
            f"r{tc.get('round','?')} k{tc.get('k','?')} {tc.get('decision','?')}"
            for tc in history
        )
        best_str = f" best_fvu={best}" if best is not None else ""
        hist_part = f": {hist_str}" if hist_str else ""
        parts.append(f"  {name}{best_str} last_r={lr}{hist_part}")

    # Recent training failures
    for label, key, n in [
        ("Recent training failures", "recent_training_failures", 4),
        ("Recent sanity failures", "recent_sanity_failures", 3),
    ]:
        fails = [
            entry for entry in memory.get(key, [])
            if isinstance(entry, dict)
            and target_round_ids
            and _safe_round_id(entry.get("round")) in target_round_ids
        ][-n:]
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
    """Inject a concise background brief as weak prior, not hard conclusion."""
    if not prior_text:
        return ""
    digest = prior_text.strip()
    if len(digest) > 3500:
        digest = digest[:3480] + "\n[truncated]"
    return f"当前背景文档（弱先验，不是结论）：\n{digest}"


def section_prior_research_full(prior_text: str) -> str:
    """Fresh session must receive the full prior document."""
    if not prior_text:
        return ""
    return f"完整背景文档（弱先验，不是结论）：\n{extract_full_prior_document(prior_text)}"


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
    recent_summaries = _recent_summaries_with_latest(state, limit=20)

    # Layer 1
    sections.append(section_hard_constraints())

    # Layer 2
    sections.append(section_policy_guidance(policy_guidance))
    sections.append(section_compatibility_status(registry))
    sections.append(section_agent_state(
        state.round_index, state.consecutive_crashes,
        state.consecutive_no_improve, state.rounds_since_new_family,
    ))
    sections.append(section_target_profile())
    sections.append(section_high_priority_directions())
    sections.append(section_frontier(state.frontier, registry))
    sections.append(section_selection_cost_status(state.frontier, registry))
    sections.append(section_cost_feasibility_table(registry))
    sections.append(section_recent_rounds(
        recent_summaries[-5:],
        registry,
    ))
    sections.append(section_reference_configs(
        recent_summaries,
        registry,
        limit=4,
    ))
    sections.append(section_tactical_hints(state.get_pending_hints()))

    # Layer 3
    sections.append(section_memory_brief(state.memory, state.frontier, registry, recent_summaries))

    # Layer 4
    sections.append(section_operator_guide_digest(
        state.load_operator_guide_excerpt(),
    ))
    sections.append(section_prior_research_full(
        state.load_prior_research(),
    ))

    checklist = _load_architecture_checklist()
    if checklist:
        raw = recent_summaries[-2:]
        sections.append(section_architecture_checklist(checklist, state.memory, raw))

    return "\n\n".join(s for s in sections if s)


def compose_resume(state: Any, round_id: int, policy_guidance: str) -> str:
    """用于 session resume 的轻量增量 prompt。"""
    sections: list[str] = []

    registry = state.load_compatibility_registry()
    recent_summaries = _recent_summaries_with_latest(state, limit=20)
    target_recent_summaries = _current_target_summaries(recent_summaries)

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
    sections.append(section_target_profile())
    sections.append(section_high_priority_directions())
    sections.append(section_frontier(state.frontier, registry, limit=8))
    sections.append(section_selection_cost_status(state.frontier, registry))
    sections.append(section_cost_feasibility_table(registry))
    sections.append(section_recent_rounds(
        recent_summaries[-4:],
        registry,
    ))
    sections.append(section_reference_configs(
        recent_summaries,
        registry,
        limit=4,
    ))
    sections.append(section_tactical_hints(state.get_pending_hints()))
    sections.append(section_prior_research_digest(
        state.load_prior_research(),
    ))

    checklist = _load_architecture_checklist()
    if checklist:
        raw = recent_summaries[-2:]
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
        "默认只改最小必要文件；若阻塞原因是 tunable 参数未接通，可同步修复 research/AutoResearch/ 与 scripts/autoresearch_test.sh 中的必要 wiring。\n"
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
        ("num_experts", "num_experts"),
        ("TRUNK_RANK", "trunk_rank"),
        ("NUM_CODES", "num_codes"),
        ("STAGE1_RATIO", "stage1_ratio"),
        ("FACTORIZED_HIDDEN_DIM", "factorized_hidden_dim"),
        ("NUM_EXPERTS", "num_experts"),
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


def _safe_round_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _best_fvu_from_history(history: list[dict[str, Any]]) -> float | None:
    best: float | None = None
    for item in history:
        try:
            fvu = float(item.get("val_fvu"))
        except (TypeError, ValueError, AttributeError):
            continue
        if best is None or fvu < best:
            best = fvu
    return best
