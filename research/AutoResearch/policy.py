"""AutoResearch 的策略层。

这个文件只负责三件事：
1. 校验 action 是否违反少数几个硬规则
2. 根据最近的 crash / no-improve 情况判定当前策略模式
3. 把策略模式渲染成一段简洁的 prompt 引导文字

当前 policy 只有四种模式：
- engineering_repair: 最近连续 crash，优先修最近失败实现
- low_cost_exploration: 在 `<0.25x` 主战场补低成本前沿，而不是回到 `0.5x` 左右做局部打磨
- mainline: 默认模式；当前 mainline 更偏向结构扩展，而不是默认继续旧 family 局部推进
- architecture_probe: 主线稳定但连续多轮无改进时，允许做 1 个 matched architecture probe

这里刻意不做"恢复到默认主线"的强制回退。
如果新架构写坏了，应该通过 repair loop 修代码，而不是用 policy 逃避问题。
"""

from __future__ import annotations

import subprocess
from typing import Any

from .compatibility import compute_selection_cost, is_compatible_label
from .config_resolution import (
    frontier_entry_to_env_config,
    structured_config_from_round_summary,
    render_env_config,
    resolve_action_configs,
    resolve_mainline_snapshot,
    summary_is_usable_reference,
    summary_invalid_reason,
)
from .git_ops import REPO_ROOT
from .target_profile import default_target_profile, profile_matches, resolve_target_profile
from .types import Action

MAX_INCUBATING_FAMILIES = 10
MAX_INCUBATING_PROXY_ROUNDS = 3
CRASH_STREAK_FOR_ENGINEERING_REPAIR = 2
NO_IMPROVE_STREAK_FOR_ARCH_PROBE = 5
PRIMARY_LOW_COST_THRESHOLD = 0.25

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
            "不能继续推进或占据 frontier"
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
    has_feasible_frontier: bool = True,
    frontier: dict[str, Any] | None = None,
    registry: dict[str, str] | None = None,
) -> dict[str, Any]:
    """把当前轮次归入五种策略模式之一。"""
    low_cost_status = _low_cost_frontier_status(frontier or {}, registry or {})
    if consecutive_crashes >= CRASH_STREAK_FOR_ENGINEERING_REPAIR:
        mode = "engineering_repair"
        reason = (
            f"最近连续 {consecutive_crashes} 轮 crash，"
            "先判断是不是最近实现写坏了。"
        )
    elif not has_feasible_frontier:
        mode = "cost_exploration"
        reason = (
            "当前 frontier 没有成本可行的点（均超过 1.5×h×n 预算）。"
            "首要目标是找到可行配置。"
        )
    elif low_cost_status["needs_expansion"]:
        mode = "low_cost_exploration"
        reason = low_cost_status["reason"]
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
                "但还应先把同一 family 的 recipe 与训练曲线看清。"
            )
        else:
            reason = "当前没有连续 crash，主线仍应继续推进。"

    return {
        "mode": mode,
        "reason": reason,
        "consecutive_no_improve": consecutive_no_improve,
        "consecutive_crashes": consecutive_crashes,
        "has_feasible_frontier": has_feasible_frontier,
        "low_cost_status": low_cost_status,
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
    mainline = _resolve_policy_anchor(state, mode)
    recipe_line = _format_recipe_line(mainline["config"])
    recipe_block = render_env_config(mainline["config"])
    reason = policy_state["reason"]
    if mainline["source"] == "target_profile_baseline":
        role_label = "当前参考 family"
        recipe_label = "当前参考配方"
        config_label = "当前参考完整配置"
    elif mainline["source"] == "low_cost_frontier_best":
        role_label = "当前低成本参考 family"
        recipe_label = "当前低成本参考配方"
        config_label = "当前低成本完整配置"
    else:
        role_label = "当前主线 family"
        recipe_label = "当前主线参考配方"
        config_label = "当前主线完整配置"

    if mode == "engineering_repair":
        return "\n".join([
            f"Round {round_id} 策略模式：工程修复",
            f"原因：{reason}",
            f"{role_label}：{mainline['family_name']}",
            f"{recipe_label}：{recipe_line}",
            f"{config_label}（source={mainline['source']}）：\n{recipe_block}",
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
            f"{config_label}（source={mainline['source']}）：\n{recipe_block}",
            "本轮要求：",
            "1. 首要目标：找到 total_cost (encoder + deployment) ≤1.5×h×n 的配置。",
            "2. 降低 encoder 成本最有效的手段是降低 EXPANSION_FACTOR。降低部署成本靠降 K / TRUNK_RANK / NUM_CODES。",
            "3. 选 K 时需权衡 FVU 改善与部署查表开销（K×n），K 过大会推高 total_cost。",
            "4. 参考成本速查表选择可行的 (架构, EF) 组合。",
            "5. 可以尝试不同架构；不要因为旧位置结论预先排斥简单结构、低秩结构或更复杂结构。",
            "6. 允许同时切换 family + 调整 EF，因为当前没有可行点可做 baseline。",
            "",
            "成本硬约束：total_cost (encoder + deployment) 不得超过 1.5×h×n，超过将被拦截。",
        ])

    if mode == "low_cost_exploration":
        return "\n".join([
            f"Round {round_id} 策略模式：低开销探索",
            f"原因：{reason}",
            f"{role_label}：{mainline['family_name']}",
            f"{recipe_label}：{recipe_line}",
            f"{config_label}（source={mainline['source']}）：\n{recipe_block}",
            "本轮要求：",
            "1. 当前阶段重点是 `<0.25x total_cost` 主战场；`0.25x-0.35x` 只作辅助对照，`>0.4x` 不应继续主导预算。",
            "2. 当前低成本 baseline 主要是 `shared_routed_expert_topk` 一线；它现在更适合作为 matched baseline / cost anchor，而不是默认继续细扫的主线。",
            "3. 允许新开 family，但只有在其成本路径明确落在 `<0.25x` 或至少不明显超过 `0.35x` 时才值得优先尝试。",
            "4. MoE-like 方向只有在 router 足够轻、expert 更小、总激活路径仍短、且最终仍能导出为静态子库有限加权和时才值得尝试。",
            "5. 不要让 `0.5x` 左右的成功点重新把注意力拉回中成本区；它们只保留作少量质量锚点。",
            "6. 如果最近几轮都只是同一 family 上的 `K` 或 `LATENTS_PER_EXPERT` 微调且没有形成新的 frontier 扩展，应优先切到 architecture-level probe，而不是继续补线性插值点。",
            "",
            "成本硬约束：total_cost (encoder + deployment) 不得超过 1.5×h×n，超过将被拦截。",
        ])

    if mode == "architecture_probe":
        return "\n".join([
            f"Round {round_id} 策略模式：架构探针",
            f"原因：{reason}",
            f"{role_label}：{mainline['family_name']}",
            f"{recipe_label}：{recipe_line}",
            f"{config_label}（source={mainline['source']}）：\n{recipe_block}",
            "本轮要求：",
            "1. 只允许做 1 个 matched architecture probe。",
            "2. 保持主线的 K、OPTIMIZER、LR、EXPANSION_FACTOR 与主要 recipe 不变，只改变 architecture 本身。",
            "3. probe 应优先用于 `<0.25x` 区域可能成立的轻量 family，例如更轻的 routed / shared+routed / expert 子库结构；不要优先回到 `0.5x` 左右的中成本 family 做重复对照。",
            "4. 这个 probe 只回答一个问题：该结构槽位本身值不值得继续。",
        ])

    return "\n".join([
        f"Round {round_id} 策略模式：主线推进",
        f"原因：{reason}",
        f"{role_label}：{mainline['family_name']}",
        f"{recipe_label}：{recipe_line}",
        f"{config_label}（source={mainline['source']}）：\n{recipe_block}",
        "本轮要求：",
        "1. 当前 mainline 配方首先是 reference anchor，不等于后续必须继续围绕它做局部推进。",
        "2. 当前默认主战场仍是 `<0.25x`；若本轮不直接服务这个区域，应明确说明它只是辅助对照。",
        "3. 优先保持归因清晰：先确认 total_cost 所在区间与结构槽位，再做少量 recipe 或结构参数调整。",
        "   注意：降低 encoder 成本靠降 EF。降低部署成本靠降 K / TRUNK_RANK / NUM_CODES。K 过大推高 total_cost。",
        "4. 调 recipe 时要观察训练曲线形状，不要只看最后一个 F 值。",
        "5. 如果最近几轮都只是同一 family 上的 `K` / `LATENTS_PER_EXPERT` 微调且没有 frontier 扩展，应优先换结构槽位，而不是继续局部细扫。",
        "6. 当前优先的结构方向是新的低成本 routed / shared+routed / low-rank expert probe；`0.5x` 左右的 lowrank+expert+residual 结构只保留作少量质量对照。",
        "",
        "成本硬约束：total_cost (encoder + deployment) 不得超过 1.5×h×n，超过将被拦截。",
        "降低选择成本的手段：减小 EXPANSION_FACTOR / TRUNK_RANK / NUM_CODES，使用低秩 scorer 等。",
    ])


def _low_cost_frontier_status(
    frontier: dict[str, Any],
    registry: dict[str, str],
    threshold_ratio: float = PRIMARY_LOW_COST_THRESHOLD,
) -> dict[str, Any]:
    """Summarize whether the <threshold_ratio total_cost region is underexplored."""
    low_cost_points: list[dict[str, Any]] = []
    feasible_points: list[dict[str, Any]] = []

    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        entry_cfg = dict(entry.get("config", {}) or {})
        if entry.get("target_profile") is not None and "target_profile" not in entry_cfg:
            entry_cfg["target_profile"] = entry["target_profile"]
        entry_family = str(
            entry_cfg.get("family_name")
            or entry.get("architecture")
            or ""
        ).lower()
        if registry and not is_compatible_label(registry.get(entry_family)):
            continue
        try:
            fvu = float(entry.get("fvu", float("inf")))
        except (TypeError, ValueError):
            continue
        tc = entry.get("total_cost")
        if tc is None:
            sel = entry.get("selection_cost")
            deploy = entry.get("deployment_accesses", 0) or 0
            tc = (float(sel) + float(deploy)) if sel is not None else None
        if tc is None:
            continue
        original = resolve_target_profile(entry_cfg).original_matmul_accesses
        ratio = float(tc) / original if original > 0 else float("inf")
        point = {"entry": entry, "fvu": fvu, "ratio": ratio}
        if ratio <= 1.5:
            feasible_points.append(point)
        if ratio < threshold_ratio:
            low_cost_points.append(point)

    best_feasible_fvu = min((p["fvu"] for p in feasible_points), default=float("inf"))
    best_low_cost_fvu = min((p["fvu"] for p in low_cost_points), default=float("inf"))
    needs_expansion = False
    if len(low_cost_points) < 2:
        needs_expansion = True
        reason = f"当前 frontier 在 <{threshold_ratio}x total_cost 区域只有 {len(low_cost_points)} 个点，低开销前沿明显欠覆盖。"
    elif best_feasible_fvu < float("inf") and best_low_cost_fvu > 2.0 * best_feasible_fvu:
        needs_expansion = True
        reason = (
            f"当前 <{threshold_ratio}x 区域最佳 FVU={best_low_cost_fvu:.4f}，"
            f"明显落后于当前最佳可行点 FVU={best_feasible_fvu:.4f}，需要定向补低开销前沿。"
        )
    else:
        reason = f"当前 <{threshold_ratio}x total_cost 区域已有可用前沿点。"

    return {
        "threshold_ratio": threshold_ratio,
        "low_cost_count": len(low_cost_points),
        "best_low_cost_fvu": None if best_low_cost_fvu == float('inf') else best_low_cost_fvu,
        "best_feasible_fvu": None if best_feasible_fvu == float('inf') else best_feasible_fvu,
        "needs_expansion": needs_expansion,
        "reason": reason,
    }


def _resolve_policy_anchor(state: Any, mode: str) -> dict[str, Any]:
    """Choose the most relevant anchor config for the current policy mode."""
    if mode == "low_cost_exploration":
        low_cost = _best_low_cost_frontier_entry(
            state.frontier,
            state.load_compatibility_registry(),
            threshold_ratio=PRIMARY_LOW_COST_THRESHOLD,
        )
        if low_cost is not None:
            config = frontier_entry_to_env_config(low_cost)
            family_name = str(
                low_cost.get("config", {}).get("family_name")
                or low_cost.get("architecture")
                or config.get("ARCHITECTURE", "topk")
            ).lower()
            return {
                "family_name": family_name,
                "config": config,
                "source": "low_cost_frontier_best",
            }
    return resolve_mainline_snapshot(state)


def _best_low_cost_frontier_entry(
    frontier: dict[str, Any],
    registry: dict[str, str],
    threshold_ratio: float = PRIMARY_LOW_COST_THRESHOLD,
) -> dict[str, Any] | None:
    """Return the best-FVU compatible frontier point inside the low-cost band."""
    current_profile = default_target_profile()
    best: dict[str, Any] | None = None
    best_fvu = float("inf")

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
        if registry and not is_compatible_label(registry.get(family_name)):
            continue
        total_cost = entry.get("total_cost")
        if total_cost is None:
            sel = entry.get("selection_cost")
            deploy = entry.get("deployment_accesses", 0) or 0
            total_cost = float(sel) + float(deploy) if sel is not None else None
        if total_cost is None:
            continue
        original = resolve_target_profile(cfg).original_matmul_accesses
        ratio = float(total_cost) / original if original > 0 else float("inf")
        if ratio >= threshold_ratio:
            continue
        try:
            fvu = float(entry.get("fvu", float("inf")))
        except (TypeError, ValueError):
            continue
        if fvu < best_fvu:
            best_fvu = fvu
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

    extra_config: dict[str, Any] = {}
    for env_key, cfg_key in [
        ("TRUNK_RANK", "trunk_rank"),
        ("NUM_CODES", "num_codes"),
        ("STAGE1_RATIO", "stage1_ratio"),
        ("FACTORIZED_HIDDEN_DIM", "factorized_hidden_dim"),
        ("NUM_EXPERTS", "num_experts"),
        ("ACTIVE_EXPERTS", "active_experts"),
        ("LATENTS_PER_EXPERT", "latents_per_expert"),
    ]:
        val = cfg.get(env_key)
        if val is not None and val != "":
            try:
                extra_config[cfg_key] = float(val) if "." in str(val) else int(val)
            except (ValueError, TypeError):
                pass

    cost = compute_selection_cost(
        arch,
        k=k,
        ef=ef,
        d_in=target_profile.d_in,
        n_output=target_profile.n_output,
        extra_config=extra_config or None,
    )
    if "error" in cost:
        return True, ""  # 计算失败时不拦截

    if cost.get("combined_feasible", True):
        return True, ""

    combined_ratio = cost.get("combined_ratio", 0)
    combined_budget = cost.get("combined_budget_ratio", 0)
    sel_ratio = cost.get("ratio", 0)
    deploy_ratio = cost.get("deployment_ratio", 0)
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
