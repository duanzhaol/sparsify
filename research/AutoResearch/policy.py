"""AutoResearch 的策略层。

这个文件只负责三件事：
1. 校验 action 是否违反少数几个硬规则
2. 根据最近的 crash / no-improve 情况判定当前策略模式
3. 把策略模式渲染成一段简洁的 prompt 引导文字

当前 policy 只有四种模式：
- engineering_repair: 最近连续 crash，优先修最近失败实现
- low_cost_exploration: 当前阶段优先补全 <0.5x total_cost 区域
- mainline: 默认模式，在已进入的低成本 family 上做局部推进
- architecture_probe: 主线稳定但连续多轮无改进时，允许做 1 个 matched architecture probe

这里刻意不做"恢复到默认主线"的强制回退。
如果新架构写坏了，应该通过 repair loop 修代码，而不是用 policy 逃避问题。
"""

from __future__ import annotations

import subprocess
from typing import Any

from .compatibility import compute_selection_cost, is_compatible_label
from .git_ops import REPO_ROOT
from .types import Action, BASE_ENV_DEFAULTS

MAX_INCUBATING_FAMILIES = 10
MAX_INCUBATING_PROXY_ROUNDS = 3
CRASH_STREAK_FOR_ENGINEERING_REPAIR = 2
NO_IMPROVE_STREAK_FOR_ARCH_PROBE = 5

_CORE_AXES = {
    "architecture": "ARCHITECTURE",
    "optimizer": "OPTIMIZER",
    "lr": "LR",
    "k": "K",
    "expansion_factor": "EXPANSION_FACTOR",
}

# Map ENV-style names to internal axis names (for primary_variable normalization)
_PRIMARY_VARIABLE_ALIASES: dict[str, str] = {
    env_key.lower(): axis for axis, env_key in _CORE_AXES.items()
}
# Also accept the env-key casing directly
_PRIMARY_VARIABLE_ALIASES.update({env_key: axis for axis, env_key in _CORE_AXES.items()})


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

    family_name = action.family_name or action.effective_config().get("ARCHITECTURE")
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

    ok, msg = _check_total_cost_feasibility(action)
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
    mainline = _resolve_mainline_snapshot(state)
    recipe_line = _format_recipe_line(mainline["config"])
    mode = policy_state["mode"]
    reason = policy_state["reason"]

    if mode == "engineering_repair":
        return "\n".join([
            f"Round {round_id} 策略模式：工程修复",
            f"原因：{reason}",
            f"当前主线 family：{mainline['family_name']}",
            f"当前主线参考配方：{recipe_line}",
            "本轮要求：",
            "1. 优先继续修最近失败实现，不要新开 architecture family。",
            "2. 不要同时换 optimizer、lr、loss、preprocess 等训练 recipe。",
            "3. 如果判断不是代码问题，而是训练链路本身异常，先做最小健康检查确认系统状态。",
        ])

    if mode == "cost_exploration":
        return "\n".join([
            f"Round {round_id} 策略模式：成本探索",
            f"原因：{reason}",
            f"当前主线 family：{mainline['family_name']}",
            f"当前主线参考配方：{recipe_line}",
            "本轮要求：",
            "1. 首要目标：找到 total_cost (encoder + deployment) ≤1.5×h×n 的配置。",
            "2. 降低 encoder 成本最有效的手段是降低 EXPANSION_FACTOR。降低部署成本靠降 K / TRUNK_RANK / NUM_CODES。",
            "3. 选 K 时需权衡 FVU 改善与部署查表开销（K×n），K 过大会推高 total_cost。",
            "4. 参考成本速查表选择可行的 (架构, EF) 组合。",
            "5. 可以尝试不同架构——简单架构在低 EF 下可能是更好的权衡。",
            "6. 允许同时切换 family + 调整 EF，因为当前没有可行点可做 baseline。",
            "",
            "成本硬约束：total_cost (encoder + deployment) 不得超过 1.5×h×n，超过将被拦截。",
        ])

    if mode == "low_cost_exploration":
        return "\n".join([
            f"Round {round_id} 策略模式：低开销探索",
            f"原因：{reason}",
            f"当前参考 family：{mainline['family_name']}",
            f"当前参考配方：{recipe_line}",
            "本轮要求：",
            "1. 当前阶段重点不是继续刷新全局最低 FVU，而是补全 total_cost < 0.5x 区域的 Pareto 前沿。",
            "2. 允许新开 family；不要求继续围绕当前主线 family 做 clean baseline 或邻域微调。",
            "3. 优先尝试天然低成本结构：极低 EF / 极低 K、factorized scorer、更少静态库、更短选择链路、轻量多专家/多子库方案。",
            "4. 在这个区域，topk、factorized_topk、lowrank_residual 等简单架构是合理起点；不要因为它们在高 EF 下表现差就直接排斥。",
            "5. MoE-like 方向只有在 router 足够轻、expert 更小、且最终仍能导出为静态子库有限加权和时才值得尝试。",
            "6. >0.5x 区域只保留少量质量锚点或解释性对照，不应继续作为默认主战场。",
            "",
            "成本硬约束：total_cost (encoder + deployment) 不得超过 1.5×h×n，超过将被拦截。",
        ])

    if mode == "architecture_probe":
        return "\n".join([
            f"Round {round_id} 策略模式：架构探针",
            f"原因：{reason}",
            f"当前主线 family：{mainline['family_name']}",
            f"当前主线参考配方：{recipe_line}",
            "本轮要求：",
            "1. 只允许做 1 个 matched architecture probe。",
            "2. 保持主线的 K、OPTIMIZER、LR、EXPANSION_FACTOR 与主要 recipe 不变，只改变 architecture 本身。",
            "3. 这个 probe 只回答一个问题：该架构本身值不值得继续。",
            "4. probe 完成后下一轮回到主线，不要连续开多个新 family。",
        ])

    return "\n".join([
        f"Round {round_id} 策略模式：主线推进",
        f"原因：{reason}",
        f"当前主线 family：{mainline['family_name']}",
        f"当前主线参考配方：{recipe_line}",
        "本轮要求：",
        "1. 只有在已经进入某个低成本 family 后，才默认围绕该 family 做局部推进。",
        "2. 推荐顺序：先确认 total_cost 落在目标区间，再做必要的 clean baseline，然后调学习率，再换优化器，再看 loss/preprocess。",
        "   注意：降低 encoder 成本靠降 EF。降低部署成本靠降 K / TRUNK_RANK / NUM_CODES。K 过大推高 total_cost。",
        "3. 调 recipe 时要观察训练曲线形状，不要只看最后一个 F 值。",
        "4. 每轮只回答一个问题；如果当前 family 无法逼近低成本目标，应优先切去更便宜的结构。",
        "",
        "成本硬约束：total_cost (encoder + deployment) 不得超过 1.5×h×n，超过将被拦截。",
        "降低选择成本的手段：减小 EXPANSION_FACTOR / TRUNK_RANK / NUM_CODES，使用低秩 scorer 等。",
    ])


def _low_cost_frontier_status(
    frontier: dict[str, Any],
    registry: dict[str, str],
    threshold_ratio: float = 0.5,
    d_in: int = 1024,
) -> dict[str, Any]:
    """Summarize whether the <threshold_ratio total_cost region is underexplored."""
    original = d_in * 4 * d_in
    low_cost_points: list[dict[str, Any]] = []
    feasible_points: list[dict[str, Any]] = []

    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        entry_family = str(
            entry.get("config", {}).get("family_name")
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
    """只对 param_only 做轻量单变量校验。"""
    if action.change_type != "param_only":
        return True, ""

    candidate = _candidate_config(action)
    reference = _resolve_reference_config(action, state)

    changed_axes = [
        axis
        for axis, env_key in _CORE_AXES.items()
        if str(candidate.get(env_key)) != str(reference.get(env_key))
    ]

    if len(changed_axes) > 1:
        return False, (
            "param_only 一次只能改一个主轴；"
            f"当前同时改了 {', '.join(changed_axes)}"
        )

    # Normalize primary_variable: accept both "expansion_factor" and "EXPANSION_FACTOR"
    pv = _PRIMARY_VARIABLE_ALIASES.get(action.primary_variable, action.primary_variable)

    if not changed_axes:
        if pv in _CORE_AXES:
            return False, (
                f"primary_variable={action.primary_variable}，"
                "但和参考配方相比没有看到这个主轴发生变化"
            )
        return True, ""

    changed_axis = changed_axes[0]
    if pv == "other_param":
        return False, (
            f"当前实际改动主轴是 {changed_axis}，"
            "primary_variable 不应写 other_param"
        )
    if pv != changed_axis:
        return False, (
            f"primary_variable={action.primary_variable}（归一化为 {pv}），"
            f"但实际改动主轴是 {changed_axis}"
        )

    return True, ""


def _check_total_cost_feasibility(action: Action) -> tuple[bool, str]:
    """拦截总成本 (encoder + deployment) 超过 1.5×h×n 的配置。

    对 edit_sae_code 类型的 action 跳过 pre-check：代码修改可能正是为了降低成本，
    用修改前的实现去估算成本会错误地拦截合理的降成本提案。
    成本会在代码修改 + sanity check 之后重新评估。
    """
    if action.change_type == "edit_sae_code":
        return True, ""

    cfg = action.effective_config()
    arch = cfg.get("ARCHITECTURE", "topk").lower()
    k = int(cfg.get("K", 128))
    ef = int(cfg.get("EXPANSION_FACTOR", 12))

    extra_config: dict[str, Any] = {}
    for env_key, cfg_key in [
        ("TRUNK_RANK", "trunk_rank"),
        ("NUM_CODES", "num_codes"),
        ("STAGE1_RATIO", "stage1_ratio"),
        ("FACTORIZED_HIDDEN_DIM", "factorized_hidden_dim"),
    ]:
        val = cfg.get(env_key)
        if val is not None and val != "":
            try:
                extra_config[cfg_key] = float(val) if "." in str(val) else int(val)
            except (ValueError, TypeError):
                pass

    cost = compute_selection_cost(arch, k=k, ef=ef, extra_config=extra_config or None)
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


def _candidate_config(action: Action) -> dict[str, str]:
    """得到这轮 action 实际会训练的核心配置。"""
    config = action.effective_config()
    if action.family_name and not any(
        item.get("key") == "ARCHITECTURE" for item in action.env_overrides
    ):
        config["ARCHITECTURE"] = action.family_name.lower()
    return config


def _resolve_reference_config(
    action: Action,
    state: Any,  # StateManager
) -> dict[str, str]:
    """给单变量校验选择一个清晰的参考配方。

    规则：
    1. 如果 action 的 family 就是当前主线 family → 用主线 snapshot（和 prompt 展示一致）
    2. 如果 action 的 family 不同 → 尝试从 frontier 找同 family best entry
    3. 都找不到 → fallback 到主线 snapshot
    """
    mainline = _resolve_mainline_snapshot(state)
    action_family = (action.family_name or "").lower()
    mainline_family = mainline["family_name"]

    # If same family as mainline, use mainline config (matches what agent sees)
    if action_family == mainline_family or not action_family:
        return dict(mainline["config"])

    # Different family: try to find its best feasible frontier entry as reference
    registry = state.load_compatibility_registry()
    family_best = _best_frontier_entry(
        state.frontier,
        family_name=action_family,
        registry=registry,
        prefer_feasible=True,
    )
    if family_best is not None:
        return _frontier_entry_to_env_config(family_best)

    # Fallback to mainline
    return dict(mainline["config"])


def _resolve_mainline_snapshot(state: Any) -> dict[str, Any]:
    """找到当前最像主线的 family 与配方。

    优先选成本可行的 frontier entry，这样 reference config 反映的是
    agent 应该在其上继续改进的基准，而不是不可行的历史最优。
    """
    registry = state.load_compatibility_registry()
    best_entry = _best_frontier_entry(
        state.frontier, registry=registry, prefer_feasible=True,
    )
    if best_entry is not None:
        config = _frontier_entry_to_env_config(best_entry)
        family_name = str(
            best_entry.get("config", {}).get("family_name")
            or best_entry.get("architecture")
            or config.get("ARCHITECTURE", "topk")
        ).lower()
        return {
            "family_name": family_name,
            "config": config,
            "source": "frontier_best",
        }

    active_family_name = _latest_active_family_name(state.families, registry)
    config = dict(BASE_ENV_DEFAULTS)
    config["EXPANSION_FACTOR"] = "12"
    if active_family_name:
        config["ARCHITECTURE"] = active_family_name
        return {
            "family_name": active_family_name,
            "config": config,
            "source": "latest_active_family",
        }

    return {
        "family_name": "topk",
        "config": config,
        "source": "topk_baseline",
    }


def _best_frontier_entry(
    frontier: dict[str, Any],
    family_name: str | None = None,
    registry: dict[str, str] | None = None,
    prefer_feasible: bool = False,
    d_in: int = 1024,
) -> dict[str, Any] | None:
    """按 FVU 选择最佳 frontier entry；可选按 family 过滤。

    当 prefer_feasible=True 时，优先从成本可行的条目中选择；
    仅当没有可行条目时才 fallback 到全部条目。
    """
    budget = 1.5 * d_in * 4 * d_in  # 1.5 × h × n

    def _pick_best(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
        best, best_fvu = None, float("inf")
        for e in entries:
            try:
                fvu = float(e.get("fvu", float("inf")))
            except (TypeError, ValueError):
                continue
            if fvu < best_fvu:
                best_fvu = fvu
                best = e
        return best

    target_family = (family_name or "").lower()
    feasible: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []

    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue

        entry_family = str(
            entry.get("config", {}).get("family_name")
            or entry.get("architecture")
            or ""
        ).lower()

        if target_family:
            if entry_family != target_family:
                continue

        if registry is not None and not is_compatible_label(registry.get(entry_family)):
            continue

        all_candidates.append(entry)

        if prefer_feasible:
            tc = entry.get("total_cost")
            if tc is None:
                sel = entry.get("selection_cost")
                deploy = entry.get("deployment_accesses", 0) or 0
                tc = (float(sel) + float(deploy)) if sel is not None else None
            if tc is not None and float(tc) <= budget:
                feasible.append(entry)

    if prefer_feasible and feasible:
        return _pick_best(feasible)
    return _pick_best(all_candidates)


def _frontier_entry_to_env_config(entry: dict[str, Any]) -> dict[str, str]:
    """把 frontier 里的 config 还原成 env 风格配置。"""
    config = dict(BASE_ENV_DEFAULTS)
    raw = entry.get("config", {})

    mapping = {
        "architecture": "ARCHITECTURE",
        "expansion_factor": "EXPANSION_FACTOR",
        "k": "K",
        "optimizer": "OPTIMIZER",
        "lr": "LR",
        "hookpoints": "HOOKPOINTS",
        "batch_size": "BATCH_SIZE",
        "grad_acc_steps": "GRAD_ACC_STEPS",
        "micro_acc_steps": "MICRO_ACC_STEPS",
        "auxk_alpha": "AUXK_ALPHA",
        "dead_feature_threshold": "DEAD_FEATURE_THRESHOLD",
        "use_hadamard": "USE_HADAMARD",
        "family_name": "FAMILY_NAME",
        "family_stage": "FAMILY_STAGE",
    }
    for src_key, dst_key in mapping.items():
        value = raw.get(src_key)
        if value is not None:
            config[dst_key] = str(value)

    if entry.get("architecture") is not None:
        config["ARCHITECTURE"] = str(entry["architecture"]).lower()
    if entry.get("k") is not None:
        config["K"] = str(entry["k"])
    if entry.get("ef") is not None:
        config["EXPANSION_FACTOR"] = str(entry["ef"])

    return config


def _latest_active_family_name(
    families: dict[str, Any],
    registry: dict[str, str] | None = None,
) -> str | None:
    """在没有 frontier 时，用最近活跃 family 作为主线参考。"""
    best_name: str | None = None
    best_round = -1

    for name, family in families.items():
        if registry is not None and not is_compatible_label(registry.get(str(name).lower())):
            continue
        if family.get("status") != "active":
            continue
        last_round = int(family.get("last_round") or -1)
        if last_round > best_round:
            best_round = last_round
            best_name = name

    return best_name


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
