"""Centralized prompt construction for the autoresearch framework.

All prompt assembly lives here. Each prompt section is a standalone function
returning a plain string (empty if nothing to contribute). Three compose
functions assemble sections in the correct 4-layer order.

Layer 1: Hard Constraints (role, rules, single-variable)
Layer 2: Current Decision State (policy, leaderboard, recent rounds, hints)
Layer 3: Rolling Memory (filtered families, failures, hypotheses)
Layer 4: Reference Digests (operator guide summary, prior research summary)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .compatibility import (
    compatibility_hard_rules,
    compute_selection_cost_table,
    extract_full_prior_document,
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
from .objective import EXCEED_FIELD, entry_objective_metrics, objective_rank_tuple
from .target_profile import default_target_profile, profile_matches

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
- 当前主目标是最小化单目标 `objective_score = total_cost_ratio + exceed_alpha_0.50`
- 其中 `total_cost_ratio = (encoder 选择成本 + 部署查表成本) / original_matmul_accesses`
- `exceed_alpha_0.50` 表示在 alpha=0.5 阈值下，超过误差阈值、需要在线补偿的比例
- 当前阶段目标不是只比 incumbent 小幅更好，而是要把 `objective_score` 继续压向 `0.30` 以下
- 当前 run 训练 hookpoint 固定为 `layers.[3].self_attn.q_proj`
- 当前成本 proxy 按 fused QKV 部署矩阵 `1024 x 4096` 统计，而不是按 q_proj 单矩阵 `1024 x 2048` 统计
- FVU 继续保留，但只作为诊断与 tie-break，不再是主优化轴
- 如果一个改动让 cost 略升但 exceed 明显下降，只要 objective_score 更低且仍满足预算，就算正收益
- 如果一个改动只让 FVU 更低，但 objective_score 没变好，则不应当作主要成功
- 旧位置、旧轮次、旧 family 排名都只算弱先验；在新 target 上必须重新验证
- K, EF, TRUNK_RANK, NUM_CODES 等参数均可自由调整，目标是更低的 objective_score
- encoder 选择成本由 EF 主导，降低主要靠降低 EXPANSION_FACTOR
- 部署成本由 K、trunk_rank、NUM_CODES 主导；每个静态库条目同时计入输入侧原子/value 访问与输出侧查表结果，因此 sparse 路径近似按 `K × (d_in + n_output)` 增长
- 不同架构在相同 EF 下的 encoder 成本差异很大（见成本速查表）
- 在比较两个候选时，优先问清楚：它是在降 cost，还是在降 exceed，还是两者都没有实质改善
- 局部参数微调只是辅助校准手段，不应连续多轮主导预算；若最近几轮都只是同一 family 上的小参数插值且 objective 没有扩展，应优先换结构槽位
- 对 MoE-like 方向，只有在 router 足够轻、expert 更小、且最终仍能导出为静态子库有限加权和时才值得尝试"""

EXECUTION_LAYER = """\
执行层是固定的：
- 训练：scripts/autoresearch_test.sh
- 结果：research/AutoResearch/controller.py
- 记忆：research/history/
- 当前执行沙盒：sparsify-ascend 是面向 LUTurbo 搜索的训练与评估环境
- 如果新增可调参数，不仅要在 sparsify/ 中实现，还必须同步接通 research/AutoResearch/ 下的 override/config-resolution/runner 持久化链路、scripts/autoresearch_test.sh 参数透传、以及必要的 resume/validation 路径；否则实验会静默回退到默认值
- `edit_sae_code` / `edit_perf_code` 返回的 JSON 必须描述“已经实际落盘的代码修改”，不是只表达计划；在返回 JSON 前，应先真实编辑工作区文件
- 新增 tunable 参数时，允许编辑的最小必要范围是：sparsify/、research/AutoResearch/、scripts/autoresearch_test.sh；不要改其他路径"""

EDIT_RULES = """\
规则：
- 默认只编辑 sparsify/；只有在新增 tunable 参数或修复参数接线时，才允许同时修改 research/AutoResearch/ 与 scripts/autoresearch_test.sh 的必要文件
- 纯参数实验必须使用 env_overrides
- 每一轮只允许一个主假设
- 不要声明 primary_variable；系统会根据 reference_round 的完整配置自动判断本轮到底改了哪个参数
- 如果 change_type=param_only，默认必须显式给出 reference_round；只有当前 target profile 还没有任何可用 reference_round 时，才允许显式返回 `reference_round=null`，相对 `target_profile_baseline` 冷启动，并且仍然只能改 1 个 env 参数
- 如果 change_type 是 `edit_sae_code` 或 `edit_perf_code`，返回前必须确认工作区中真的已有对应代码改动；若实际没有落盘改动，不要返回这两类 change_type
- `touched_files` 必须只列出本轮真实改动过并已保存到工作区的文件；不要把“计划改但尚未修改”的文件写进去
- 如果新增了 env key / tunable 参数，返回前至少完成本地接线自检：新 key 已进入 `override_registry` allowlist、`config_resolution` / `runner` / 必要的持久化路径已识别、`scripts/autoresearch_test.sh` 已透传；至少做一次本地校验，确认该 key 不会在训练启动前被判为 `Disallowed env override keys`
- 新增 tunable 参数后的第一轮，必须先验证该参数真的进入了训练配置：至少检查 round*.config.json 和 checkpoint config.json 中该字段存在且取值正确
- 不要返回 command="stop"
- 最终必须返回一个符合 action schema 的 JSON 对象
- 无论 fresh session 还是 resume session，都要显式返回全部 schema 字段；即使某项为空，也要返回 `[]` / `null` / `false`，不要省略字段"""

SINGLE_VARIABLE_PRINCIPLE = (
    "单变量原则：每一轮只改变一个 env 参数。"
    "有 reference_round 时，env_overrides 是对该轮完整配置的 patch。"
    "若当前 target 仍是冷启动且暂无 reference_round，则允许显式返回 reference_round=null，相对 target_profile_baseline 只改 1 个 env 参数。"
)

HARD_CONSTRAINT_REMINDER = (
    "提醒：每轮只改一个 env 参数；不要声明 primary_variable；param_only 默认应显式给出 reference_round。"
    "只有当前 target 冷启动且暂无 reference_round 时，才允许显式返回 reference_round=null，相对 target_profile_baseline 只改 1 个 env 参数。"
    "默认只改 sparsify/；只有在新增 tunable 参数或修复参数接线时，才允许改 research/AutoResearch/ 与 scripts/autoresearch_test.sh 的必要文件。"
    "如果返回 `edit_sae_code` / `edit_perf_code`，JSON 必须对应已经实际落盘的代码修改；不要只写计划。"
    "如果新增 env key，先本地确认 allowlist/wiring 已接通，再返回该 key。"
    "当前结果判定与 leaderboard 更新逻辑在 research/AutoResearch/controller.py，不在旧的 research/controller.py。"
    "当前单目标是 objective_score = total_cost_ratio + exceed_alpha_0.50；FVU 只作诊断。"
    "必须显式返回全部 schema 字段，即使为空也不要省略。最终返回 JSON。"
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


def _best_compatible_feasible_frontier_entry(
    frontier: dict[str, Any],
    registry: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    current_target = default_target_profile()
    best_entry: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_rank: tuple[float, float, float, float] | None = None

    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        cfg = _entry_config_with_target(entry)
        if not profile_matches(cfg, current_target):
            continue
        family_name = str(cfg.get("family_name") or entry.get("architecture", "")).lower()
        if family_name and not is_compatible_label(registry.get(family_name)):
            continue
        metrics = entry_objective_metrics(entry, cost_cache={})
        if not metrics["feasible"]:
            continue
        rank = objective_rank_tuple(
            metrics["objective_score"],
            metrics["total_cost_ratio"],
            metrics[EXCEED_FIELD],
            metrics["fvu"],
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_entry = entry
            best_metrics = metrics

    if best_entry is None or best_metrics is None:
        return None
    return best_entry, best_metrics


def section_high_priority_directions(
    frontier: dict[str, Any],
    registry: dict[str, str],
    target_objective: float = 0.30,
) -> str:
    incumbent = _best_compatible_feasible_frontier_entry(frontier, registry)
    incumbent_note = (
        "  incumbent 只应视作 objective anchor / matched baseline，不应自动变成默认 continuation plan。"
    )
    stage_gap_note = (
        "  当前阶段需要的是明显拉低 objective 的结构或容量突破，而不是继续围绕 incumbent 做保守小步扫描。"
    )
    mainline_note = (
        "  当前优先方向应同时包括：一是继续压榨 incumbent 主线附近真正高价值的 knee / 容量点；二是只在能明确解释为何会改善 small-K tradeoff 时才做新结构 probe。"
    )
    cost_band_note = (
        "  新结构不应默认占用明显更高 cost；若想进入更高 cost 带，必须能非常明确地换回更低 exceed。"
    )

    if incumbent is not None:
        entry, metrics = incumbent
        arch = entry.get("architecture", "?")
        k = entry.get("k", "?")
        obj = metrics["objective_score"]
        cost = metrics["total_cost_ratio"]
        exceed = metrics[EXCEED_FIELD]
        gap = float(obj) - target_objective if obj is not None else None
        incumbent_note = (
            f"  当前已验证 incumbent 是 `{arch}` (K={k})，"
            f"objective≈{float(obj):.6f} = cost {float(cost):.2f}x + exceed {float(exceed):.6f}。"
        )
        if arch == "expert_topk":
            mainline_note = (
                "  当前最强已验证主线是更大 expert 池配合更小 K 的 `expert_topk`；"
                "后续一方面应继续寻找这条线的 deployment knee / 容量配置，另一方面只在新结构能解释清楚 small-K exceed 为何会更慢退化时才值得挑战它。"
            )
        if cost is not None:
            low = max(0.0, float(cost) - 0.02)
            high = float(cost) + 0.04
            cost_band_note = (
                f"  当前可优先关注 `total_cost_ratio≈{low:.2f}~{high:.2f}` 这一带；"
                "若新结构想占用更高 cost，必须能非常明确地换回更低 exceed。"
            )
        if gap is not None and gap > 0.04:
            stage_gap_note = (
                f"  当前距离 `objective<{target_objective:.2f}` 仍差约 {gap:.3f}；"
                "这已经不是抠几个千分点能解决的问题，需要显著改变当前 cost/exceed tradeoff 的结构或容量突破。"
            )

    return "\n".join([
        "当前研究阶段：单目标 `objective_score = total_cost_ratio + exceed_alpha_0.50` 优先。",
        "  当前要回答的是“谁能把 objective 真正降下来”，而不是“谁的 FVU 最低”或“谁只把 cost 压得更左”。",
        stage_gap_note,
        incumbent_note,
        "  如果最近 2 轮没有把 objective 明显拉低，下一轮默认优先换结构槽位，而不是继续做同 family 的局部插值。",
        "  对 0.005~0.015 级别的小改进不要过度满足；除非它明确打开了一条新结构线，否则不值得连续多轮消耗预算。",
        "",
        "当前高优先级方向：",
        "  1. 新的 matched-objective architecture probe；优先回答新的结构槽位值不值得继续，而不是继续沿同一 family 做线性插值。",
        f"  2. {cost_band_note.strip()}",
        "  3. `shared bottleneck + expert-specific head`、expert 内部 low-rank、`shared low-rank trunk + routed residual experts`、以及更轻的 scorer / router。",
        "  4. 不对称分工的 sparse MoE-like 结构：shared/coarse 分支吃平滑主干，routed experts 只做 residual cleanup，active path 尽量短。",
        "  5. 能明显降低 exceed_alpha_0.50 的结构，即使 total_cost_ratio 有适度上升，只要 objective 更低且仍满足预算。",
        "  6. 能在较小 K 下仍把 exceed 控住的结构，而不是只做线性降 K。",
        f"  7. {mainline_note.strip()}",
        "",
        "当前应下调优先级的方向：",
        "  1. 只改善 FVU、却没有改善 objective_score 的局部 recipe 微调。",
        "  2. 连续在同一 family 上补 `K` / `LATENTS_PER_EXPERT` / rank / lr 插值点，但 objective 没有扩展的细扫。",
        "  3. 任何明显抬高 total_cost_ratio、却没有换回更低 exceed 的结构探针。",
        "  4. 只把 shared / routed 分支换个轻微变体、但没有改变容量与 active path 关系的低信息增量实验。",
        "  5. 在当前目标差距仍很大时，还继续围绕 incumbent 做保守小步扫描。",
        "  6. 把 reference_round 当成默认延续路线，而不是只把它当 patch anchor / 对照基线。",
        "  7. 任何无法导出为静态子库有限加权和的 MoE / dynamic-dictionary 方向。",
    ])


def section_objective_target_gap(
    frontier: dict[str, Any],
    registry: dict[str, str],
    target_objective: float = 0.30,
) -> str:
    """Show what cost/exceed tradeoffs are needed to reach the current stage target."""
    incumbent = _best_compatible_feasible_frontier_entry(frontier, registry)
    if incumbent is None:
        return ""
    _, incumbent_metrics = incumbent

    objective = incumbent_metrics["objective_score"]
    cost = incumbent_metrics["total_cost_ratio"]
    exceed = incumbent_metrics[EXCEED_FIELD]
    if objective is None or cost is None or exceed is None:
        return ""

    gap = float(objective) - target_objective
    parts = [
        f"阶段目标：把 objective_score 压到 {target_objective:.2f} 以下。",
        (
            f"  当前 incumbent 约为 objective={float(objective):.6f} "
            f"(cost={float(cost):.2f}x, exceed={float(exceed):.6f})；"
            f"距离目标仍差 {gap:.6f}。"
        ),
        "  这意味着后续实验不能只满足于很小的局部改进；需要明确回答“这轮是否在逼近 0.30 所需的 tradeoff”。",
    ]

    base_cost = float(cost)
    candidate_costs = sorted({
        max(0.0, round(base_cost - 0.02, 2)),
        round(base_cost, 2),
        round(base_cost + 0.02, 2),
        round(base_cost + 0.04, 2),
    })
    parts.append("  若想达到该目标，不同 cost 档位大致需要满足：")
    for c in candidate_costs:
        need_exceed = target_objective - c
        parts.append(f"    - 若 cost≈{c:.2f}x，则 exceed 需压到 ≤{need_exceed:.2f}")
    upper_probe_cost = candidate_costs[-1]
    upper_need = target_objective - upper_probe_cost
    parts.append(
        f"  因此：若新结构 cost 升到约 {upper_probe_cost:.2f}x，却仍无法把 exceed 压到约 {upper_need:.2f} 附近，则通常不够支撑 {target_objective:.2f} 目标。"
    )
    parts.append("  只有那些明显缩小这一缺口的结构 probe，才值得获得后续 follow-up 预算。")
    return "\n".join(parts)


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
    """Render the retained objective leaderboard."""
    if cost_cache is None:
        cost_cache = {}
    current_target = default_target_profile()
    entries: list[tuple[tuple[float, float, float, float], str]] = []
    saw_any_entry = False

    for key, entry in frontier.items():
        if not isinstance(entry, dict):
            continue
        saw_any_entry = True
        cfg = _entry_config_with_target(entry)
        if not profile_matches(cfg, current_target):
            continue
        arch = entry.get("architecture", "?")
        family_name = str(cfg.get("family_name") or arch).lower()
        if not is_compatible_label(registry.get(family_name)):
            continue
        metrics = entry_objective_metrics(entry, cost_cache=cost_cache)
        rank = objective_rank_tuple(
            metrics["objective_score"],
            metrics["total_cost_ratio"],
            metrics[EXCEED_FIELD],
            metrics["fvu"],
        )
        line = (
            f"  r{str(key).lstrip('r'):>3} "
            f"objective={metrics['objective_score'] if metrics['objective_score'] is not None else '?'} "
            f"= cost {metrics['total_cost_ratio'] if metrics['total_cost_ratio'] is not None else '?'}x "
            f"+ exceed {metrics[EXCEED_FIELD] if metrics[EXCEED_FIELD] is not None else '?'} "
            f"| fvu={metrics['fvu'] if metrics['fvu'] is not None else '?'} "
            f"| arch={arch} K={entry.get('k', '?')} EF={entry.get('ef', '?')} "
            f"| {'FEASIBLE' if metrics['feasible'] else 'OVER_BUDGET'}"
        )
        entries.append((rank, line))

    entries.sort(key=lambda item: item[0])

    parts = ["训练代理 leaderboard（single objective: objective_score = total_cost_ratio + exceed_alpha_0.50；FVU 仅诊断）："]
    if entries:
        for _, line in entries[:limit]:
            parts.append(line)
    else:
        if saw_any_entry:
            parts.append("  （当前 target profile 下暂无 leaderboard 条目；旧 target 历史已隔离）")
        else:
            parts.append("  （暂无条目）")

    return "\n".join(parts)


def section_selection_cost_status(
    frontier: dict[str, Any],
    registry: dict[str, str],
) -> str:
    """Show current incumbent and component leaders under the single objective."""
    current_target = default_target_profile()
    cost_cache: dict[str, dict] = {}
    best_objective: tuple[tuple[float, float, float, float], dict[str, Any], dict[str, Any]] | None = None
    lowest_cost: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    lowest_exceed: tuple[float, dict[str, Any], dict[str, Any]] | None = None

    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        cfg = _entry_config_with_target(entry)
        if not profile_matches(cfg, current_target):
            continue
        family_name = str(cfg.get("family_name") or entry.get("architecture", "")).lower()
        if not is_compatible_label(registry.get(family_name)):
            continue
        metrics = entry_objective_metrics(entry, cost_cache=cost_cache)
        if not metrics["feasible"]:
            continue
        rank = objective_rank_tuple(
            metrics["objective_score"],
            metrics["total_cost_ratio"],
            metrics[EXCEED_FIELD],
            metrics["fvu"],
        )
        if best_objective is None or rank < best_objective[0]:
            best_objective = (rank, entry, metrics)
        if (
            metrics["total_cost_ratio"] is not None
            and (lowest_cost is None or metrics["total_cost_ratio"] < lowest_cost[0])
        ):
            lowest_cost = (float(metrics["total_cost_ratio"]), entry, metrics)
        if (
            metrics[EXCEED_FIELD] is not None
            and (lowest_exceed is None or float(metrics[EXCEED_FIELD]) < lowest_exceed[0])
        ):
            lowest_exceed = (float(metrics[EXCEED_FIELD]), entry, metrics)

    if best_objective is None:
        return ""

    parts = ["目标状态："]

    def _format(label: str, entry: dict[str, Any], metrics: dict[str, Any]) -> None:
        parts.append(
            f"  {label} {entry.get('architecture', '?')} "
            f"(K={entry.get('k', '?')}, EF={entry.get('ef', '?')}):"
        )
        parts.append(
            f"    objective={metrics['objective_score'] if metrics['objective_score'] is not None else '?'} "
            f"= cost {metrics['total_cost_ratio'] if metrics['total_cost_ratio'] is not None else '?'}x "
            f"+ exceed {metrics[EXCEED_FIELD] if metrics[EXCEED_FIELD] is not None else '?'} "
            f"| encoder={metrics['selection_cost_ratio'] if metrics['selection_cost_ratio'] is not None else '?'}x "
            f"| deploy={metrics['deployment_ratio'] if metrics['deployment_ratio'] is not None else '?'}x "
            f"| fvu={metrics['fvu'] if metrics['fvu'] is not None else '?'}"
        )

    _, incumbent_entry, incumbent_metrics = best_objective
    _format("当前 incumbent →", incumbent_entry, incumbent_metrics)

    if lowest_cost is not None and lowest_cost[1] is not incumbent_entry:
        _format("最低 cost leader →", lowest_cost[1], lowest_cost[2])
    if lowest_exceed is not None and lowest_exceed[1] is not incumbent_entry and lowest_exceed[1] is not (lowest_cost[1] if lowest_cost else None):
        _format("最低 exceed leader →", lowest_exceed[1], lowest_exceed[2])

    parts.append("  单目标：objective_score = total_cost_ratio + exceed_alpha_0.50")
    parts.append("  降低 encoder 成本: 降 EF / TRUNK_RANK / NUM_CODES / STAGE1_RATIO / FACTORIZED_HIDDEN_DIM")
    parts.append("  降低部署成本: 降 K / TRUNK_RANK / NUM_CODES")
    parts.append("  降低 exceed_alpha_0.50: 优先减少 tail error / online compensation 需求，而不是只看最终 FVU")
    return "\n".join(parts)


def section_cost_feasibility_table(registry: dict[str, str]) -> str:
    """基础 family 的 encoder 成本速查表。"""
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
        "基础 family 的 Encoder 侧成本速查表（非最终 budget，ratio 指 encoder-only）：",
        f"  {header}",
    ]
    for row in rows:
        parts.append(f"  {row}")
    parts.append("  注意：以上仅为 encoder 选择成本（不依赖 K）。最终 budget 以 total_cost = selection + deployment 为准。")
    parts.append("  部署侧 sparse 路径额外贡献约：K=32 +4%，K=64 +8%，K=128 +16%（按 K×(d+n) 近似）。")
    parts.append("  说明：本表只覆盖基础 family，不覆盖 expert / routed / shared+routed 家族。")
    parts.append("  这些家族的 encoder 成本还受 NUM_EXPERTS / ACTIVE_EXPERTS / LATENTS_PER_EXPERT / FACTORIZED_HIDDEN_DIM 等影响，应优先参考 leaderboard 条目与 reference_round 完整配置。")

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
    frontier: dict[str, Any],
    limit: int = 2,
) -> str:
    """Show recent runnable full env configs that can serve as reference_round anchors."""
    if not round_summaries:
        return (
            "最近可用 reference_round 的完整配置（仅作 patch anchor / 对照基线，不代表默认延续路线）：\n"
            "  只有当你明确选择 param_only patch 时才需要这些完整配置；若本轮是新结构 probe，不必围绕这些点继续微调。\n"
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

    current_target = default_target_profile()
    frontier_rounds: list[int] = []
    frontier_objectives: dict[int, tuple[float, float, float, float]] = {}
    incumbent_round_id: int | None = None
    incumbent_family: str | None = None
    for rid, entry in frontier.items():
        if not isinstance(entry, dict):
            continue
        cfg = _entry_config_with_target(entry)
        if not profile_matches(cfg, current_target):
            continue
        family_name = str(cfg.get("family_name") or entry.get("architecture", "")).lower()
        if family_name and not is_compatible_label(registry.get(family_name)):
            continue
        round_id = _safe_round_id(str(rid).lstrip("r"))
        if round_id is None:
            continue
        metrics = entry_objective_metrics(entry, cost_cache={})
        frontier_rounds.append(round_id)
        frontier_objectives[round_id] = objective_rank_tuple(
            metrics["objective_score"],
            metrics["total_cost_ratio"],
            metrics[EXCEED_FIELD],
            metrics["fvu"],
        )
    incumbent = _best_compatible_feasible_frontier_entry(frontier, registry)
    if incumbent is not None:
        incumbent_entry, _ = incumbent
        incumbent_round_id = _safe_round_id(str(incumbent_entry.get("round", "")).lstrip("r"))
        if incumbent_round_id is None:
            for rid, entry in frontier.items():
                if entry is incumbent_entry:
                    incumbent_round_id = _safe_round_id(str(rid).lstrip("r"))
                    break
        incumbent_family = str(
            incumbent_entry.get("config", {}).get("family_name")
            or incumbent_entry.get("architecture")
            or ""
        ).lower()

    def _priority(summary: dict[str, Any]) -> tuple[int, int]:
        decision = str(summary.get("result", {}).get("decision") or "")
        round_id = int(summary.get("round") or 0)
        band_rank = 0 if round_id in frontier_rounds else 1
        decision_rank = 0 if decision in {"keep", "archive"} else 1
        objective_rank = frontier_objectives.get(round_id, (float("inf"),) * 4)
        return (band_rank, decision_rank, *objective_rank, -round_id)

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

    def _format_frontier_entry_block(entry: dict[str, Any], round_id: int) -> str | None:
        cfg = dict(entry.get("config", {}) or {})
        if not cfg:
            return None
        family_name = str(cfg.get("family_name") or entry.get("architecture") or "").lower()
        if family_name and not is_compatible_label(registry.get(family_name)):
            return None
        decision = "frontier"
        return (
            f"  r{round_id} {family_name or cfg.get('architecture', '?')} "
            f"(decision={decision}, source=frontier_entry)\n{render_env_config(cfg)}"
        )

    ordered = sorted(usable, key=_priority)
    by_round_id: dict[int, dict[str, Any]] = {}
    for summary in ordered:
        try:
            by_round_id[int(summary.get("round") or 0)] = summary
        except (TypeError, ValueError):
            continue

    chosen: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    deferred: list[dict[str, Any]] = []
    incumbent_frontier_entry: dict[str, Any] | None = None
    if incumbent is not None:
        incumbent_frontier_entry, _ = incumbent
    incumbent_frontier_block: str | None = None
    if incumbent_round_id is not None and incumbent_round_id in by_round_id:
        incumbent_summary = by_round_id[incumbent_round_id]
        chosen.append(incumbent_summary)
        if incumbent_family:
            seen_families.add(incumbent_family)
    elif incumbent_frontier_entry is not None and incumbent_round_id is not None:
        incumbent_frontier_block = _format_frontier_entry_block(
            incumbent_frontier_entry, incumbent_round_id
        )
        if incumbent_family:
            seen_families.add(incumbent_family)
    for summary in ordered:
        if summary in chosen:
            continue
        family_name = _family_name(summary)
        if family_name and family_name in seen_families:
            deferred.append(summary)
            continue
        chosen.append(summary)
        if family_name:
            seen_families.add(family_name)

    blocks: list[str] = []
    if incumbent_frontier_block:
        blocks.append(incumbent_frontier_block)
    for summary in [*chosen, *deferred]:
        if len(blocks) >= limit:
            break
        block = _format_block(summary)
        if block:
            blocks.append(block)
    if not blocks:
        if saw_other_target:
            return (
                "最近可用 reference_round 的完整配置（仅作 patch anchor / 对照基线，不代表默认延续路线）：\n"
                "  只有当你明确选择 param_only patch 时才需要这些完整配置；若本轮是新结构 probe，不必围绕这些点继续微调。\n"
                "  （当前 target profile 尚无可用 reference_round；旧 target 结果已隔离）\n"
                "  冷启动阶段请显式返回 reference_round=null，并相对 target_profile_baseline 只改 1 个 env 参数。"
            )
        return ""
    return (
        "最近可用 reference_round 的完整配置（仅作 patch anchor / 对照基线，不代表默认延续路线）：\n"
        "  只有当你明确选择 param_only patch 时才需要这些完整配置；若本轮是新结构 probe，不必围绕这些点继续微调。\n"
        + "\n\n".join(blocks)
    )


def section_tactical_hints(hints: list[dict[str, Any]]) -> str:
    """只保留战术性 hint，过滤掉已经固化到 prompt 模板中的论文灵感。"""
    tactical = []
    current_profile = default_target_profile()
    elevated_tags = {
        "asymmetric-moe-literature",
        "memory-routing",
        "alias-awareness",
    }
    for hint in hints:
        text = str(hint.get("text") or hint.get("message") or "").strip()
        if not text:
            continue
        target_id = hint.get("target_profile_id")
        if target_id not in (None, "", current_profile.profile_id):
            continue
        if any(text.startswith(p) for p in _HARD_CONSTRAINT_HINT_PREFIXES):
            continue
        if str(hint.get("tag") or "") in elevated_tags:
            continue
        tactical.append(text)

    if not tactical:
        return ""
    lines = ["操作提示："]
    for i, t in enumerate(tactical, 1):
        lines.append(f"  {i}. {t}")
    return "\n".join(lines)


def section_literature_inspirations() -> str:
    """Static literature-inspired guidance injected early in the prompt."""
    return """\
论文与结构灵感（当前优先消化）：
最值得吸收的 6 个方向：
1. `DeepSeekMoE`：shared experts + 更细粒度 expert 分片。
   来源：ACL 2024, DeepSeekMoE
   https://aclanthology.org/2024.acl-long.70/
   关键点：
   - 不是只做普通 top-k experts，而是同时把 experts 切得更细，并单独保留 shared experts 负责公共知识。
   - 可转成：`1 个 always-on shared/coarse 分支 + 很多很小的 routed micro-experts`。
   - 重点不是让单个 expert 更大，而是让 expert 更碎、更专门化；每个 token 仍只激活很短路径，但总容量显著变大。
   - 这和当前想要的“不对称分工”最贴近。

2. `Expert Choice Routing`：不是 token 选 expert，而是 expert 选 token。
   来源：Google Research Blog, Expert Choice Routing
   https://research.google/blog/mixture-of-experts-with-expert-choice-routing/?hl=pt_BR
   关键点：
   - 更均衡的 load balancing。
   - 每个 token 可以有可变数量的 experts。
   - 难 token 应允许更多 experts，简单 token 只用 1-2 个。
   - 对当前目标特别 relevant：可以把“小 K 基础路径 + 按难度追加少量 expert / 补偿路径”作为核心思路。
   - 这比固定所有 token 都用同样 `ACTIVE_EXPERTS` 更贴近真实目标。

3. `Product Key Memory`：两级 / 乘积式索引的大容量 memory。
   来源：NeurIPS 2019, Large Memory Layers with Product Keys
   https://papers.nips.cc/paper/9061-large-memory-layers-with-product-keys
   关键点：
   - 用 product keys 做超大 memory。
   - 容量很大，但访问成本仍然很低。
   - 不一定非要 flat top-k 直接从一大坨 latent 里选。
   - 可以做：`coarse codebook A × coarse codebook B -> 候选子库 -> 小 K 精选`。
   - 本质是组合式 dictionary / compositional memory，非常适合“小 K 但想保留大容量”。

4. `Hash Layers`：去掉 learned router，改成 hashing / balanced routing。
   来源：Hash Layers for Large Sparse Models
   https://huggingface.co/papers/2106.04426
   关键点：
   - 不需要额外 routing 参数。
   - 不需要复杂 load balancing loss。
   - 仍然能和 learned routing 竞争。
   - 若 router/scorer 成本不低或训练不稳定，可以试：`hash/bucket -> bucket 内局部 scorer/top-k`。
   - 甚至可以试固定分桶 + 轻量 re-score。

5. `Omni-SMoLA`：shared backbone + 轻量 low-rank experts 做 residual specialization。
   来源：Google Research, Omni-SMoLA
   https://research.google/pubs/omni-smola-boosting-generalist-multimodal-models-with-soft-mixture-of-low-rank-experts/
   关键点：
   - 不复制完整 experts。
   - 用很多轻量 low-rank experts 残差式地补 backbone。
   - 可转成：`shared low-rank trunk + many tiny low-rank expert residuals`。
   - 每个 expert 不是完整 `d_in -> latents`，而是 rank 很小的 residual adapter。

6. Activation sparsity / Top-K thresholding 本身可能带来更稳的性质。
   来源：Google Research, On Emergence of Activation Sparsity in Trained Transformers
   https://research.google/pubs/on-emergence-of-activation-sparsity-in-trained-transformers/
   关键点：
   - transformer 里的 activation sparsity 是自然出现的。
   - 更稀疏的 Top-K thresholding 可能带来更好的鲁棒性 / 校准。
   - 小 K 不一定只是“被迫压成本”，也可能是结构先验的一部分。
   - 关键是要有足够好的 coarse/shared/trunk 去兜住主干，让小 K 只负责 tail。

当前最值得优先尝试的 4 个具体结构想法：
1. `shared coarse library + routed micro-experts`
   - 灵感来自 DeepSeekMoE。
   - shared 分支吃公共部分，routed experts 切得很细，每次只开 1-2 个。
2. `variable-active-experts by difficulty`
   - 灵感来自 Expert Choice。
   - 不是所有 token 都固定 2 experts，而是简单 token 1 个，难 token 2-4 个。
3. `product-key / compositional dictionary`
   - 灵感来自 PKM。
   - 先粗定位，再小 K 精选，换容量而不线性增加 deploy lookup。
4. `shared low-rank trunk + tiny low-rank expert residuals`
   - 灵感来自 Omni-SMoLA。
   - 不是 full expert，而是很多 rank 很小的 expert adapters。

一个重要判断：
- 这些灵感里，和当前目标最匹配的，不是“对称多专家一起干活”，而是：
- `shared / coarse / trunk` 负责主干。
- `tiny routed experts` 负责难点 / residual / tail cleanup。
- `variable active path` 只在困难样本上打开。
- 这和当前 `objective = cost + exceed` 很一致，因为它本质上就是在最小化：
- 普通 token 的基础成本。
- 困难 token 的额外补偿需求。

别名与复用提醒：
- 提出 literature-inspired probe 前，先检查它是不是已经被现有 family 近似覆盖，只是名字不同。
- 例如：`shared coarse + residual experts` 常接近 `shared_two_stage_residual_expert`。
- `shared low-rank trunk + routed experts` 常接近 `shared_lowrank_routed_expert_topk` 或其 residual 变体。
- `cheap trunk + routed experts` 常接近 `lowrank_expert_topk`。
- `shared expert + routed experts` 常接近 `shared_routed_expert_topk`。
- 如果只是重命名或轻微同构变体，不要当成 brand-new architecture；必须明确写出与现有实现的精确差异。

补充一个基线教训：
- `Switch Transformer` 的启发是：很多时候简单路由先赢。
  https://huggingface.co/papers/2101.03961
- 所以如果要加复杂结构，仍应尽量保持：
- `very short active path`
- `very cheap router`
- `very clean decomposition`
"""


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
    """精简记忆：只保留 leaderboard 家族、最近测试家族与近期失败。"""
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

    parts: list[str] = ["架构家族（leaderboard 持有者 + 最近测试）："]
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
        best_objective = _best_objective_from_history(target_history)
        lr = target_history[-1].get("round", "?") if target_history else "?"
        history = target_history[-3:]
        hist_str = "; ".join(
            (
                f"r{tc.get('round','?')} k{tc.get('k','?')} {tc.get('decision','?')}"
                f"{' obj=' + str(tc.get('objective_score')) if tc.get('objective_score') not in (None, '') else ''}"
            )
            for tc in history
        )
        best_str = f" best_objective={best_objective}" if best_objective is not None else ""
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
    digest = _extract_prior_resume_digest(prior_text)
    if len(digest) > 5200:
        digest = digest[:5180] + "\n[truncated]"
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

    这里使用原始 round summary dict，
    而不是先压成字符串后再二次解析。
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
    sections.append(section_high_priority_directions(state.frontier, registry))
    sections.append(section_literature_inspirations())
    sections.append(section_frontier(state.frontier, registry))
    sections.append(section_selection_cost_status(state.frontier, registry))
    sections.append(section_objective_target_gap(state.frontier, registry))
    sections.append(section_cost_feasibility_table(registry))
    sections.append(section_recent_rounds(
        recent_summaries[-8:],
        registry,
    ))
    sections.append(section_tactical_hints(state.get_pending_hints()))
    sections.append(section_reference_configs(
        recent_summaries,
        registry,
        state.frontier,
        limit=2,
    ))

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

    sections.append(f"继续 LUTurbo 研究会话。当前轮次：Round {round_id}。")
    sections.append("返回一个符合 action schema 的 JSON 对象，不要使用 markdown 代码块。")
    sections.append("务必显式包含全部 schema 字段；resume 模式下也不要省略 `needs_sanity`、`reference_round`、`notes_to_memory`、`next_hypotheses` 等字段。")
    sections.append(HARD_CONSTRAINT_REMINDER)
    sections.append(compatibility_hard_rules())

    sections.append(section_policy_guidance(policy_guidance))
    sections.append(section_compatibility_status(registry))
    sections.append(section_agent_state(
        state.round_index, state.consecutive_crashes,
        state.consecutive_no_improve, state.rounds_since_new_family,
    ))
    sections.append(section_target_profile())
    sections.append(section_high_priority_directions(state.frontier, registry))
    sections.append(section_literature_inspirations())
    sections.append(section_frontier(state.frontier, registry, limit=8))
    sections.append(section_selection_cost_status(state.frontier, registry))
    sections.append(section_objective_target_gap(state.frontier, registry))
    sections.append(section_cost_feasibility_table(registry))
    sections.append(section_recent_rounds(
        recent_summaries[-8:],
        registry,
    ))
    sections.append(section_tactical_hints(state.get_pending_hints()))
    sections.append(section_reference_configs(
        recent_summaries,
        registry,
        state.frontier,
        limit=2,
    ))
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
        "repair 返回的 JSON 也必须对应已经实际落盘的代码修改；不要只描述打算怎么修。\n"
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


def _trim_round_summary(summary: dict[str, Any]) -> str:
    action = summary.get("action", {})
    result = summary.get("result", {})
    round_id = summary.get("round", "?")
    family_name = summary.get("family_name", "?")
    change_type = action.get("change_type", "?")
    decision = result.get("decision", "?")
    objective = result.get("objective_score")
    cost_ratio = result.get("total_cost_ratio")
    exceed = result.get(EXCEED_FIELD)
    fvu = result.get("val_fvu")
    duration = summary.get("duration_sec")
    objective_part = f" objective={objective}" if objective not in (None, "") else ""
    cost_part = f" cost={cost_ratio}x" if cost_ratio not in (None, "") else ""
    exceed_part = f" exceed={exceed}" if exceed not in (None, "") else ""
    fvu_part = f" fvu={fvu}" if fvu not in (None, "") else ""
    dur_part = f" {duration}s" if duration not in (None, "") else ""
    return (
        f"r{round_id} {family_name} {change_type} -> {decision}"
        f"{objective_part}{cost_part}{exceed_part}{fvu_part}{dur_part}"
    )
def _truncate(s: str, limit: int) -> str:
    if not isinstance(s, str):
        return str(s)[:limit]
    return s if len(s) <= limit else s[:limit - 3] + "..."


def _safe_round_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _best_objective_from_history(history: list[dict[str, Any]]) -> float | None:
    best: float | None = None
    for item in history:
        try:
            objective = float(item.get("objective_score"))
        except (TypeError, ValueError, AttributeError):
            continue
        if best is None or objective < best:
            best = objective
    return best


_PRIOR_RESUME_HEADINGS: tuple[str, ...] = (
    "## 1. 文档用途",
    "## 1.1 当前 target 的阅读方式",
    "## 3. 可信的继承项",
    "## 4. 不可信的旧结论",
    "## 5. 当前 target 上真正未知的事",
    "## 6. 当前高优先级方向",
    "## 7. 当前默认做法",
)


def _extract_prior_resume_digest(prior_text: str) -> str:
    """Extract the sections that still matter in resume-session prompts.

    Resume prompts already include runtime hard rules and compatibility counts,
    so we skip the long compatibility table here and focus on the parts that
    help the agent reason under the current single objective.
    """
    extracted = _extract_markdown_sections(prior_text, _PRIOR_RESUME_HEADINGS)
    if not extracted:
        return prior_text.strip()
    return (
        "兼容性表已省略；具体 compatibility 以当前 runtime registry 与 hard rules 为准。\n\n"
        + extracted
    ).strip()


def _extract_markdown_sections(text: str, headings: tuple[str, ...]) -> str:
    wanted = set(headings)
    sections: list[str] = []
    current: list[str] = []
    keep = False

    for line in text.splitlines():
        if line.startswith("## "):
            if keep and current:
                sections.append("\n".join(current).rstrip())
            keep = line.strip() in wanted
            current = [line] if keep else []
            continue
        if keep:
            current.append(line)

    if keep and current:
        sections.append("\n".join(current).rstrip())

    return "\n\n".join(section for section in sections if section.strip())
