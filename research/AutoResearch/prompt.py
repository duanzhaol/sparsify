"""Centralized prompt construction for the autoresearch framework.

All prompt assembly lives here. Each prompt section is a standalone function
returning a plain string (empty if nothing to contribute). Three compose
functions assemble sections in the correct 4-layer order.

Layer 1: Hard Constraints (role, rules, K/EF/single-variable)
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
    extract_compatibility_digest,
    extract_full_prior_document,
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
- 维护并改善 FVU 与 K 的 Pareto frontier
- 这个 frontier 只是 LUTurbo 可用性的代理指标，不是最终系统指标
- 更小的 K 只有在兼容性和潜在补偿行为仍然可接受时才真正有价值"""

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

K_WHITELIST = "K 约束：K 只能取 {4, 8, 16, 24, 32, 64, 96, 128} 中的值，禁止使用其他 K。"

EF_CONSTRAINT = "EF 约束：所有实验的 EXPANSION_FACTOR 固定为 12。每次 env_overrides 都必须显式设置 EXPANSION_FACTOR=12。"

SINGLE_VARIABLE_PRINCIPLE = "单变量原则：每一轮只改变一个主维度。"

HARD_CONSTRAINT_REMINDER = (
    "提醒：K 只能取 {4,8,16,24,32,64,96,128}；EF 永远为 12；"
    "每轮只改一个主变量；只能编辑 sparsify/；最终返回 JSON。"
)

ARCHITECTURE_INTEGRATION_SKILL_PATH = Path(
    "/root/.codex/skills/sae-architecture-integration/SKILL.md"
)

# Hint prefixes that are now in constants — filtered from tactical hints
_HARD_CONSTRAINT_HINT_PREFIXES = (
    "K values are restricted",
    "EXPANSION_FACTOR is now fixed",
)


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
        K_WHITELIST,
        EF_CONSTRAINT,
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
) -> str:
    """按 K 排序的训练代理 frontier，并区分 EF=12 主 regime 与其他历史参考。"""
    main: list[tuple[int, str]] = []

    for _key, entry in frontier.items():
        if not isinstance(entry, dict):
            continue
        k = int(entry.get("k", 0))
        ef = int(entry.get("ef", 0))
        fvu = entry.get("fvu", "?")
        arch = entry.get("architecture", "?")
        cfg = entry.get("config", {})
        family_name = str(cfg.get("family_name") or arch).lower()
        lr = cfg.get("lr", "?")
        opt = cfg.get("optimizer", "?")

        if ef == 12 and is_compatible_label(registry.get(family_name)):
            main.append((k, f"  k={k:>3d}  fvu={fvu:<12}  arch={arch}  lr={lr} opt={opt}"))

    main.sort()

    parts = ["训练代理 frontier（EF=12，当前主 regime）："]
    if main:
        for _, line in main[:limit]:
            parts.append(line)
    else:
        parts.append("  （暂无条目）")

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
        compatible_lines.append(f"  {_normalize_ef_text(_trim_round_summary(summary))}")
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
        text = _normalize_ef_text(text)
        if len(text) > max_chars:
            text = text[:max_chars - 20] + "\n[truncated]"
        return f"Operator guide 关键优先级：\n{text}"

    truncated = _normalize_ef_text(guide_text[:max_chars - 20]) + "\n[truncated]"
    return f"Operator guide 摘要：\n{truncated}"


def section_prior_research_digest(prior_text: str) -> str:
    """兼容性摘要：不再只截取 ##1/##2。"""
    if not prior_text:
        return ""
    digest = extract_compatibility_digest(prior_text) or prior_text
    return f"历史研究关键结论：\n{_normalize_ef_text(digest)}"


def section_prior_research_full(prior_text: str) -> str:
    """Fresh session must receive the full prior document."""
    if not prior_text:
        return ""
    return f"完整历史研究文档：\n{_normalize_ef_text(extract_full_prior_document(prior_text))}"


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
            lines.append(f"  - {_truncate(_normalize_ef_text(h), 100)}")
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


def _truncate(s: str, limit: int) -> str:
    if not isinstance(s, str):
        return str(s)[:limit]
    return s if len(s) <= limit else s[:limit - 3] + "..."


def _normalize_ef_text(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    replacements = [
        ("EF=16", "EF=12"),
        ("EF = 16", "EF = 12"),
        ("EF 16", "EF 12"),
        ("EXPANSION_FACTOR=16", "EXPANSION_FACTOR=12"),
        ("EXPANSION_FACTOR = 16", "EXPANSION_FACTOR = 12"),
        ("固定到 EF=16", "固定到 EF=12"),
        ("current `K=32, EF=16` backbone", "current `K=32, EF=12` backbone"),
    ]
    out = s
    for old, new in replacements:
        out = out.replace(old, new)
    return out
