"""AutoResearch 的策略层。

这个文件只负责三件事：
1. 校验 action 是否违反少数几个硬规则
2. 根据最近的 crash / no-improve 情况判定当前策略模式
3. 把策略模式渲染成一段简洁的 prompt 引导文字

当前 policy 只有五种模式：
- engineering_repair: 最近连续 crash，优先修最近失败实现
- cost_exploration: 当前最佳可行点的 objective 主要受 total_cost_ratio 主导
- exceed_exploration: 当前最佳可行点的 objective 主要受 exceed_alpha_0.50 主导
- mainline: 默认模式；当前 mainline 更偏向结构扩展，而不是默认继续旧 family 局部推进
- architecture_probe: 主线稳定但连续多轮无改进时，允许做 1 个 matched architecture probe

这里刻意不做"恢复到默认主线"的强制回退。
如果新架构写坏了，应该通过 repair loop 修代码，而不是用 policy 逃避问题。
"""

from __future__ import annotations

import subprocess
from typing import Any

from .compatibility import extract_cost_extra_config
from .config_resolution import (
    config_from_round_summary,
    structured_config_from_round_summary,
    render_env_config,
    resolve_action_configs,
    resolve_mainline_snapshot,
    summary_is_usable_reference,
    summary_invalid_reason,
)
from .git_ops import REPO_ROOT
from .objective import (
    EXCEED_FIELD,
    candidate_objective_metrics,
    entry_objective_metrics,
    objective_rank_tuple,
)
from .target_profile import default_target_profile, profile_matches, resolve_target_profile
from .types import Action

MAX_INCUBATING_FAMILIES = 10
MAX_INCUBATING_PROXY_ROUNDS = 3
CRASH_STREAK_FOR_ENGINEERING_REPAIR = 2
NO_IMPROVE_STREAK_FOR_ARCH_PROBE = 5
OBJECTIVE_BALANCE_MARGIN = 0.05

# ---------------------------------------------------------------------------
# Top-level validation
# ---------------------------------------------------------------------------


def validate_action(
    action: Action,
    state: Any,  # StateManager
) -> tuple[Action | None, str | None]:
    """运行少量硬校验。

    设计原则：
    - 尽量少拦，只拦那些明显会让本轮实验含义不清的情况
    - repair 由 repair loop 处理，不在这里强制改写 action
    """
    if action.command != "run":
        return action, "command 只能是 'run'"

    if action.change_type == "no_change":
        return action, "不允许 no_change；每一轮都必须提出可执行的实验或修复"

    resolved = resolve_action_configs(action, state)
    family_name = action.family_name or resolved.candidate_env_config.get("ARCHITECTURE")
    compat_label = state.family_compatibility_label(family_name)
    if compat_label == "incompatible":
        return action, (
            f"family '{family_name}' 在 prior_research_history.md 中被标记为不兼容，"
            "不能继续推进或占据当前 objective leaderboard"
        )

    ok, msg = _check_param_only_single_variable(action, state)
    if not ok:
        return action, msg

    ok, msg = check_incubation_limits(
        state.families,
        action.family_name,
        action.family_stage,
    )
    if not ok:
        return action, f"Incubation limit: {msg}"

    ok, msg = _check_total_cost_feasibility(action, state)
    if not ok:
        return action, msg

    return action, None


# ---------------------------------------------------------------------------
# Incubation management
# ---------------------------------------------------------------------------


def check_incubation_limits(
    families: dict[str, Any],
    family_name: str,
    family_stage: str,
) -> tuple[bool, str]:
    """限制同时存活的孵化 family 数量，并自动淘汰长期无正结果的分支。"""
    name = (family_name or "").lower()
    is_new = name not in families
    is_incubating_stage = family_stage not in ("mainline", "promote_to_mainline")

    active_incubating = sum(
        1 for family in families.values() if family.get("status") == "incubating"
    )

    if is_new and is_incubating_stage and active_incubating >= MAX_INCUBATING_FAMILIES:
        stale = [
            family_name
            for family_name, family in families.items()
            if family.get("status") == "incubating"
            and _proxy_round_count(family) >= MAX_INCUBATING_PROXY_ROUNDS
            and not _has_positive_result(family)
        ]
        return False, (
            f"当前已经有 {active_incubating} 个 incubating family，"
            f"达到上限 {MAX_INCUBATING_FAMILIES}。stale={stale}"
        )

    family = families.get(name)
    if family and family.get("status") == "incubating":
        proxy_rounds = _proxy_round_count(family)
        if proxy_rounds >= MAX_INCUBATING_PROXY_ROUNDS and not _has_positive_result(family):
            return False, f"family '{name}' 已在孵化期尝试 {proxy_rounds} 轮且没有正结果"

    return True, ""


def auto_archive_stale_families(families: dict[str, Any]) -> list[str]:
    """自动归档长期无正结果的 incubating family。"""
    archived: list[str] = []
    for name, family in families.items():
        if family.get("status") != "incubating":
            continue
        if _proxy_round_count(family) >= MAX_INCUBATING_PROXY_ROUNDS and not _has_positive_result(family):
            family["status"] = "archived"
            archived.append(name)
    return archived


# ---------------------------------------------------------------------------
# Policy mode detection
# ---------------------------------------------------------------------------


def detect_stagnation(
    consecutive_no_improve: int,
    consecutive_crashes: int,
    frontier: dict[str, Any] | None = None,
    registry: dict[str, str] | None = None,
) -> dict[str, Any]:
    """把当前轮次归入五种策略模式之一。"""
    objective_status = _objective_focus_status(frontier or {}, registry or {})
    if consecutive_crashes >= CRASH_STREAK_FOR_ENGINEERING_REPAIR:
        mode = "engineering_repair"
        reason = (
            f"最近连续 {consecutive_crashes} 轮 crash，"
            "先判断是不是最近实现写坏了。"
        )
    elif objective_status["incumbent"] is None:
        mode = "cost_exploration"
        reason = (
            "当前 objective leaderboard 没有成本可行的点（均超过 1.5×h×n 预算）。"
            "首要目标是找到可行配置。"
        )
    elif objective_status["dominant_term"] == "cost":
        mode = "cost_exploration"
        reason = objective_status["reason"]
    elif objective_status["dominant_term"] == "exceed":
        mode = "exceed_exploration"
        reason = objective_status["reason"]
    elif consecutive_no_improve >= NO_IMPROVE_STREAK_FOR_ARCH_PROBE:
        mode = "architecture_probe"
        reason = (
            f"主线已经连续 {consecutive_no_improve} 轮没有改进，"
            "可以插入 1 个 matched architecture probe。"
        )
    else:
        mode = "mainline"
        if consecutive_no_improve >= 3:
            reason = (
                f"主线已连续 {consecutive_no_improve} 轮没有改进，"
                "应优先寻找更高信息增量的结构改动，而不是继续在同一 family 上细扫。"
            )
        else:
            reason = "当前没有连续 crash；主线应继续推进单目标改进，并优先考虑结构性信息增量。"

    return {
        "mode": mode,
        "reason": reason,
        "consecutive_no_improve": consecutive_no_improve,
        "consecutive_crashes": consecutive_crashes,
        "has_feasible_frontier": objective_status["incumbent"] is not None,
        "objective_status": objective_status,
    }


# ---------------------------------------------------------------------------
# Policy guidance (for prompt)
# ---------------------------------------------------------------------------


def build_policy_guidance(
    round_id: int,
    state: Any,  # StateManager
    policy_state: dict[str, Any],
) -> str:
    """把当前策略模式渲染成 prompt 中直接可读的中文说明。"""
    mode = policy_state["mode"]
    mainline = _resolve_policy_anchor(state)
    recipe_line = _format_recipe_line(mainline["config"])
    reason = policy_state["reason"]
    if mainline["source"] == "target_profile_baseline":
        role_label = "当前 objective anchor family"
        recipe_label = "当前 anchor 摘要"
    else:
        role_label = "当前 objective anchor family"
        recipe_label = "当前 anchor 摘要"

    anchor_note = (
        "说明：这里只给 matched baseline / 归因校准用的紧凑 anchor。"
        "若本轮要做新的结构 probe，不要把它当默认 continuation plan；"
        "如需 param_only patch，完整 reference_round 配置见后文。"
    )

    if mode == "engineering_repair":
        return "\n".join([
            f"Round {round_id} 策略模式：工程修复",
            f"原因：{reason}",
            f"{role_label}：{mainline['family_name']}",
            f"{recipe_label}：{recipe_line}",
            anchor_note,
            "本轮要求：",
            "1. 优先继续修最近失败实现，不要新开 architecture family。",
            "2. 不要同时换 optimizer、lr、loss、preprocess 等训练 recipe。",
            "3. 如果判断不是代码问题，而是训练链路本身异常，先做最小健康检查确认系统状态。",
        ])

    if mode == "cost_exploration":
        return "\n".join([
            f"Round {round_id} 策略模式：成本探索",
            f"原因：{reason}",
            f"{role_label}：{mainline['family_name']}",
            f"{recipe_label}：{recipe_line}",
            anchor_note,
            "本轮要求：",
            "1. 首要目标是降低 total_cost_ratio，同时保持 total_cost ≤1.5×h×n 的硬预算。",
            "2. 允许为了明显更低的 objective_score 接受轻微 FVU 波动；不要把 FVU 当作唯一目标。",
            "3. 降低 encoder 成本最有效的手段是降低 EXPANSION_FACTOR。降低部署成本靠降 K / TRUNK_RANK / NUM_CODES。",
            "4. 如果某个改动只是把 exceed 换成明显更高的 cost，且 objective 不降，则不值得继续。",
            "5. 可以尝试不同架构；不要因为旧位置结论预先排斥简单结构、低秩结构或更复杂结构。",
            "",
            "单目标：objective_score = total_cost_ratio + exceed_alpha_0.50。",
            "成本硬约束：total_cost (encoder + deployment) 不得超过 1.5×h×n，超过将被拦截。",
        ])

    if mode == "exceed_exploration":
        return "\n".join([
            f"Round {round_id} 策略模式：Exceed 探索",
            f"原因：{reason}",
            f"{role_label}：{mainline['family_name']}",
            f"{recipe_label}：{recipe_line}",
            anchor_note,
            "本轮要求：",
            "1. 首要目标是降低 exceed_alpha_0.50；只要 objective_score 下降，允许适度增加 cost，但必须留在预算内。",
            "2. FVU 只作诊断：当 exceed 明显下降而 FVU 只小幅波动时，不要被旧的 FVU 直觉误导。",
            "3. 优先考虑能减少 tail-error / online compensation 需求的结构，而不是只做线性 cost 压缩。",
            "4. 允许新开 family；但任何新结构都必须仍能导出为静态子库有限加权和。",
            "5. 如果最近几轮都只是同一 family 上的轻微参数插值且 objective 没有实质下降，应优先换结构槽位。",
            "",
            "单目标：objective_score = total_cost_ratio + exceed_alpha_0.50。",
            "成本硬约束：total_cost (encoder + deployment) 不得超过 1.5×h×n，超过将被拦截。",
        ])

    if mode == "architecture_probe":
        return "\n".join([
            f"Round {round_id} 策略模式：架构探针",
            f"原因：{reason}",
            f"{role_label}：{mainline['family_name']}",
            f"{recipe_label}：{recipe_line}",
            anchor_note,
            "本轮要求：",
            "1. 只允许做 1 个 matched architecture probe。",
            "2. 保持主线的 K、OPTIMIZER、LR、EXPANSION_FACTOR 与主要 recipe 不变，只改变 architecture 本身。",
            "3. probe 应优先用于可能显著降低 objective_score 的结构槽位，而不是继续在同一 family 上补局部插值点。",
            "4. 这个 probe 只回答一个问题：该结构槽位本身值不值得继续。",
        ])

    return "\n".join([
        f"Round {round_id} 策略模式：主线推进",
        f"原因：{reason}",
        f"{role_label}：{mainline['family_name']}",
        f"{recipe_label}：{recipe_line}",
        anchor_note,
        "本轮要求：",
        "1. 当前 mainline 配方首先是 objective anchor / matched baseline，不等于后续必须继续围绕它做局部推进。",
        "2. 优先保持归因清晰：先确认当前 incumbent 的 objective 主要受 cost 还是 exceed 主导，然后优先选择信息增量最大的改动。",
        "3. 如果最近 2 轮没有把 objective 明显拉低，默认应切换到新的结构槽位；只有在你能明确说明当前 family 仍有高价值未验证假设时，才继续做局部调整。",
        "4. 降低 encoder 成本靠降 EF。降低部署成本靠降 K / TRUNK_RANK / NUM_CODES。若结构能明显压低 exceed，则允许适度 cost 上升但必须让 objective 真正下降。",
        "5. 调 recipe 时要观察训练曲线形状，不要只看最后一个 F 值。",
        "6. 当前已验证最强主线若是更大 expert 池 + 更小 K 的低成本 routed family，则不要假设更复杂结构天然更优；只有当新结构能明确解释 small-K exceed 为何会更慢退化时，才值得挑战 incumbent。",
        "7. 如果最近的改进来自 architecture probe，而后续 `K / LATENTS / split` 局部 follow-up 多数只是持平或更差，则应把当前 winning line 当作 scaffold / baseline，只保留极少量必要校准，不再让局部细扫继续主导预算。",
        "8. 因此当前优先方向应分成两类：一类是建立在当前 winning scaffold 上、能在近同成本下更强压低 exceed 的结构升级；另一类才是 literature-inspired、但与现有 family 明确不同的结构 probe。",
        "",
        "单目标：objective_score = total_cost_ratio + exceed_alpha_0.50。",
        "成本硬约束：total_cost (encoder + deployment) 不得超过 1.5×h×n，超过将被拦截。",
        "降低选择成本的手段：减小 EXPANSION_FACTOR / TRUNK_RANK / NUM_CODES，使用低秩 scorer 等。",
    ])


def _objective_focus_status(
    frontier: dict[str, Any],
    registry: dict[str, str],
) -> dict[str, Any]:
    """Summarize whether the current incumbent is cost- or exceed-dominated."""
    incumbent = _best_objective_frontier_entry(frontier, registry)
    if incumbent is None:
        return {
            "incumbent": None,
            "dominant_term": None,
            "reason": "当前还没有可用的 objective incumbent。",
        }

    metrics = entry_objective_metrics(incumbent)
    objective = metrics["objective_score"]
    total_cost_ratio = metrics["total_cost_ratio"]
    exceed = metrics[EXCEED_FIELD]

    if total_cost_ratio is None or exceed is None or objective is None:
        return {
            "incumbent": incumbent,
            "dominant_term": None,
            "reason": "当前 incumbent 的 objective 分解不完整，先补齐 cost / exceed 指标。",
        }

    if total_cost_ratio > exceed + OBJECTIVE_BALANCE_MARGIN:
        dominant = "cost"
        reason = (
            f"当前 incumbent objective={objective:.4f}，其中 cost={total_cost_ratio:.4f} "
            f"明显高于 exceed={exceed:.4f}；下一步应优先降 total_cost_ratio。"
        )
    elif exceed > total_cost_ratio + OBJECTIVE_BALANCE_MARGIN:
        dominant = "exceed"
        reason = (
            f"当前 incumbent objective={objective:.4f}，其中 exceed={exceed:.4f} "
            f"明显高于 cost={total_cost_ratio:.4f}；下一步应优先降 exceed_alpha_0.50。"
        )
    else:
        dominant = "balanced"
        reason = (
            f"当前 incumbent objective={objective:.4f}，cost={total_cost_ratio:.4f} "
            f"与 exceed={exceed:.4f} 接近；下一步更适合做结构探针或少量 recipe 校准。"
        )

    return {
        "incumbent": incumbent,
        "dominant_term": dominant,
        "reason": reason,
    }


def _resolve_policy_anchor(state: Any) -> dict[str, Any]:
    """Choose the most relevant anchor config for the current target profile."""
    registry = state.load_compatibility_registry()
    current_profile = default_target_profile()
    best_rank = (float("inf"), float("inf"), float("inf"), float("inf"))
    best_config: dict[str, str] | None = None
    best_family: str | None = None
    for summary in state.recent_round_summaries(limit=100):
        if not isinstance(summary, dict):
            continue
        if not summary_is_usable_reference(summary):
            continue
        cfg = structured_config_from_round_summary(summary)
        if cfg is None or not profile_matches(cfg, current_profile):
            continue
        family_name = str(
            summary.get("family_name")
            or summary.get("action", {}).get("family_name")
            or summary.get("result", {}).get("architecture")
            or cfg.get("architecture")
            or ""
        ).lower()
        if str((registry or {}).get(family_name) or "").lower() == "incompatible":
            continue
        try:
            objective = float(summary.get("result", {}).get("objective_score"))
            total_cost = float(summary.get("result", {}).get("total_cost_ratio"))
            exceed = float(summary.get("result", {}).get(EXCEED_FIELD))
            fvu = float(summary.get("result", {}).get("val_fvu"))
        except (TypeError, ValueError):
            continue
        rank = objective_rank_tuple(objective, total_cost, exceed, fvu)
        if rank < best_rank:
            env_cfg = config_from_round_summary(summary)
            if not env_cfg:
                continue
            best_rank = rank
            best_config = env_cfg
            best_family = family_name or env_cfg.get("ARCHITECTURE", "").lower()
    if best_config is not None:
        return {
            "family_name": best_family or best_config.get("ARCHITECTURE", "topk").lower(),
            "config": best_config,
            "source": "recent_best_summary",
        }
    return resolve_mainline_snapshot(state)


def _best_objective_frontier_entry(
    frontier: dict[str, Any],
    registry: dict[str, str],
) -> dict[str, Any] | None:
    """Return the best compatible feasible entry under the single objective."""
    current_profile = default_target_profile()
    best: dict[str, Any] | None = None
    best_rank = (float("inf"), float("inf"), float("inf"), float("inf"))

    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        cfg = dict(entry.get("config", {}) or {})
        if entry.get("target_profile") is not None and "target_profile" not in cfg:
            cfg["target_profile"] = entry["target_profile"]
        if not profile_matches(cfg, current_profile):
            continue
        family_name = str(
            cfg.get("family_name")
            or entry.get("architecture")
            or ""
        ).lower()
        if str((registry or {}).get(family_name) or "").lower() == "incompatible":
            continue
        metrics = entry_objective_metrics(entry)
        if not metrics["feasible"]:
            continue
        rank = objective_rank_tuple(
            metrics["objective_score"],
            metrics["total_cost_ratio"],
            metrics[EXCEED_FIELD],
            metrics["fvu"],
        )
        if rank < best_rank:
            best_rank = rank
            best = entry

    return best


# ---------------------------------------------------------------------------
# Behavioral diff test
# ---------------------------------------------------------------------------


def behavioral_diff_test(architecture: str, k: int, ef: int) -> dict[str, Any]:
    """Compare candidate architecture encode() output vs baseline topk."""
    if architecture == "topk":
        return {"identical": False, "max_diff": 0.0, "architecture": architecture}

    code = f"""
import sys; sys.path.insert(0, '.')
import torch; torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    import torch_npu
    device = 'npu' if torch.npu.is_available() else device
except ImportError: pass
from sparsify import SparseCoder
from sparsify.config import SparseCoderConfig
x = torch.randn(4, 1024, device=device, dtype=torch.float32)
torch.manual_seed(0)
base = SparseCoder(1024, SparseCoderConfig(architecture='topk', k={k}, expansion_factor={ef}), device=device, dtype=torch.float32)
base_out = base.encode(x)
torch.manual_seed(0)
cand = SparseCoder(1024, SparseCoderConfig(architecture='{architecture}', k={k}, expansion_factor={ef}), device=device, dtype=torch.float32)
cand_out = cand.encode(x)
if hasattr(base_out, 'top_acts'):
    ba, bi = base_out.top_acts, base_out.top_indices
    ca, ci = cand_out.top_acts, cand_out.top_indices
elif isinstance(base_out, tuple) and len(base_out) >= 2:
    ba, bi = base_out[0], base_out[1]
    ca, ci = cand_out[0], cand_out[1]
else:
    print("DIFF:unknown_format"); sys.exit(0)
md = (ba - ca).abs().max().item() if ba.shape == ca.shape else float('inf')
print(f"DIFF:{{'identical' if torch.equal(ba, ca) and torch.equal(bi, ci) else 'different'}}|{{md}}")
"""
    try:
        result = subprocess.run(
            ["python", "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        for line in (result.stdout + result.stderr).splitlines():
            if line.startswith("DIFF:"):
                parts = line[5:].split("|")
                return {
                    "identical": parts[0] == "identical",
                    "max_diff": float(parts[1]) if len(parts) > 1 else 0.0,
                    "architecture": architecture,
                }
        return {
            "identical": False,
            "max_diff": -1.0,
            "architecture": architecture,
            "error": "parse_failed",
        }
    except subprocess.TimeoutExpired:
        return {
            "identical": False,
            "max_diff": -1.0,
            "architecture": architecture,
            "error": "timeout",
        }
    except Exception as exc:
        return {
            "identical": False,
            "max_diff": -1.0,
            "architecture": architecture,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _check_param_only_single_variable(
    action: Action,
    state: Any,  # StateManager
) -> tuple[bool, str]:
    """只对 param_only 做单 env key 校验。"""
    if action.change_type != "param_only":
        return True, ""
    if action.reference_round is None:
        if _has_current_target_reference(state):
            return False, "param_only 必须显式提供 reference_round，明确说明相对哪一轮的完整配置只改一个 env 参数"
        resolved = resolve_action_configs(action, state)
        changed_keys = resolved.changed_keys
        if len(changed_keys) > 1:
            return False, (
                "当前 target 尚无 reference_round，冷启动 param_only 只能相对 target_profile_baseline 改 1 个 env 参数；"
                f"当前同时改了 {', '.join(changed_keys)}"
            )
        if not changed_keys:
            return False, (
                "当前 target 尚无 reference_round；冷启动 param_only 仍必须相对 target_profile_baseline 真正改动 1 个 env 参数"
            )
        return True, ""
    explicit_summary = state.load_round_summary(action.reference_round)
    invalid_reason = summary_invalid_reason(explicit_summary)
    if invalid_reason is not None:
        return False, (
            f"reference_round=r{action.reference_round} 已被标记为无效，"
            f"不能作为单变量锚点。原因：{invalid_reason}"
        )

    resolved = resolve_action_configs(action, state)
    changed_keys = resolved.changed_keys

    if len(changed_keys) > 1:
        return False, (
            "param_only 一次只能改一个 env 参数；"
            f"当前同时改了 {', '.join(changed_keys)}"
        )
    if not changed_keys:
        return False, (
            "param_only 必须相对 reference_round 真的改动 1 个 env 参数；"
            f"当前 reference_source={resolved.reference_source}，未检测到任何变化"
        )

    return True, ""


def _has_current_target_reference(state: Any) -> bool:
    current_profile = default_target_profile()
    for summary in reversed(state.recent_round_summaries(limit=50)):
        if not isinstance(summary, dict):
            continue
        if not summary_is_usable_reference(summary):
            continue
        cfg = structured_config_from_round_summary(summary)
        if cfg is not None and profile_matches(cfg, current_profile):
            return True
    return False


def _check_total_cost_feasibility(
    action: Action,
    state: Any,  # StateManager
) -> tuple[bool, str]:
    """拦截总成本 (encoder + deployment) 超过 1.5×h×n 的配置。

    对 edit_sae_code 类型的 action 跳过 pre-check：代码修改可能正是为了降低成本，
    用修改前的实现去估算成本会错误地拦截合理的降成本提案。
    成本会在代码修改 + sanity check 之后重新评估。
    """
    if action.change_type == "edit_sae_code":
        return True, ""

    cfg = resolve_action_configs(action, state).candidate_env_config
    arch = cfg.get("ARCHITECTURE", "topk").lower()
    k = int(cfg.get("K", 128))
    ef = int(cfg.get("EXPANSION_FACTOR", 12))
    target_profile = resolve_target_profile(cfg)
    metrics = candidate_objective_metrics(
        arch,
        k,
        ef,
        target_profile,
        extra_config=extract_cost_extra_config(cfg),
    )
    if metrics["feasible"]:
        return True, ""

    combined_ratio = metrics.get("total_cost_ratio", 0) or 0
    combined_budget = (
        (metrics.get("total_cost") or 0) / target_profile.budget_accesses()
        if target_profile.budget_accesses() > 0 else 0
    )
    sel_ratio = metrics.get("selection_cost_ratio", 0) or 0
    deploy_ratio = metrics.get("deployment_ratio", 0) or 0
    return False, (
        f"总成本超出预算：{arch} (K={k}, EF={ef}) 的 total_cost 为 {combined_ratio}x 原始矩阵 "
        f"(encoder {sel_ratio}x + deployment {deploy_ratio}x)，"
        f"预算比率 {combined_budget}x（需 ≤1.0x，即 total ≤1.5×h×n）。"
        f"降低 encoder 成本：减小 EF / TRUNK_RANK / NUM_CODES。降低部署成本：减小 K / TRUNK_RANK / NUM_CODES。"
    )


def _format_recipe_line(config: dict[str, str]) -> str:
    """把最关键的 recipe 压成一行。"""
    return (
        f"ARCHITECTURE={config.get('ARCHITECTURE', '?')} "
        f"K={config.get('K', '?')} "
        f"EXPANSION_FACTOR={config.get('EXPANSION_FACTOR', '?')} "
        f"OPTIMIZER={config.get('OPTIMIZER', '?')} "
        f"LR={config.get('LR', '?')}"
    )


def _proxy_round_count(family: dict[str, Any]) -> int:
    return sum(
        1
        for config in family.get("tested_configs", [])
        if config.get("stage") != "mainline"
    )


def _has_positive_result(family: dict[str, Any]) -> bool:
    return any(
        config.get("decision") in ("keep", "promote")
        for config in family.get("tested_configs", [])
    )
