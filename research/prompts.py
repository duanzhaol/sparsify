"""Prompt construction for the autoresearch agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.state_io import (
    FRONTIER_PATH,
    load_json,
    load_operator_guide_excerpt,
    load_operator_hints,
    recent_round_summaries_trimmed,
    split_hints,
    summarize_results,
)

LANE_HIGH_K_THRESHOLD = 64
ARCHITECTURE_INTEGRATION_SKILL_PATH = Path("/root/.codex/skills/sae-architecture-integration/SKILL.md")


def _parse_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_lane(k_value: int | None) -> str:
    if k_value is not None and k_value < LANE_HIGH_K_THRESHOLD:
        return "low_k_tradeoff"
    return "quality_anchor"


def _normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    k_value = _parse_int(row.get("k"))
    ef_value = _parse_int(row.get("expansion_factor"))
    fvu_value = _parse_float(row.get("val_fvu"))
    return {
        "experiment_id": row.get("experiment_id", ""),
        "tier": row.get("tier", ""),
        "decision": row.get("decision", ""),
        "val_fvu": fvu_value,
        "k": k_value,
        "architecture": row.get("architecture", ""),
        "expansion_factor": ef_value,
        "timestamp": _parse_int(row.get("timestamp")),
        "lane": _row_lane(k_value),
    }


def _lane_label(lane: str) -> str:
    return "quality_anchor" if lane == "quality_anchor" else "low_k_tradeoff"


def lane_grouped_results_digest(rows: list[dict[str, Any]], per_lane_limit: int = 4) -> dict[str, Any]:
    lane_buckets: dict[str, list[dict[str, Any]]] = {
        "quality_anchor": [],
        "low_k_tradeoff": [],
    }
    for row in rows:
        normalized = _normalize_result_row(row)
        lane_buckets[normalized["lane"]].append(normalized)

    payload: dict[str, Any] = {}
    for lane_name, lane_rows in lane_buckets.items():
        by_kef: dict[tuple[int | None, int | None], list[dict[str, Any]]] = {}
        for row in lane_rows:
            key = (row.get("k"), row.get("expansion_factor"))
            by_kef.setdefault(key, []).append(row)

        representatives: list[dict[str, Any]] = []
        for (k_value, ef_value), group in by_kef.items():
            if k_value is None:
                continue
            valid = [item for item in group if item.get("val_fvu") is not None]
            best = min(valid, key=lambda item: item["val_fvu"]) if valid else group[-1]
            last_timestamp = max(item.get("timestamp") or 0 for item in group)
            representatives.append({
                "k": k_value,
                "expansion_factor": ef_value,
                "best_architecture": best.get("architecture"),
                "best_fvu": best.get("val_fvu"),
                "best_decision": best.get("decision"),
                "best_tier": best.get("tier"),
                "recent_architecture": group[-1].get("architecture"),
                "recent_fvu": group[-1].get("val_fvu"),
                "recent_decision": group[-1].get("decision"),
                "runs": len(group),
                "last_timestamp": last_timestamp,
            })

        representatives.sort(
            key=lambda item: (
                -(item.get("last_timestamp") or 0),
                -(item.get("k") or -1),
                item.get("best_fvu") if item.get("best_fvu") is not None else float("inf"),
            ),
        )
        for item in representatives:
            item.pop("last_timestamp", None)
        payload[lane_name] = {
            "lane": _lane_label(lane_name),
            "representative_points": representatives[:per_lane_limit],
        }
    return payload


def infer_active_kef_context(rows: list[dict[str, Any]], lookback: int = 4) -> dict[str, Any]:
    normalized = [_normalize_result_row(row) for row in rows[-lookback:]]
    normalized = [row for row in normalized if row.get("k") is not None]
    if not normalized:
        default_ef = _parse_int(load_json(FRONTIER_PATH, {}).get("config", {}).get("expansion_factor"))
        return {
            "lane": "quality_anchor",
            "target_k": 128,
            "target_ef": default_ef,
        }

    latest = normalized[-1]
    ef_counts: dict[int, int] = {}
    for row in normalized:
        ef_value = row.get("expansion_factor")
        if ef_value is not None:
            ef_counts[ef_value] = ef_counts.get(ef_value, 0) + 1
    target_ef = latest.get("expansion_factor")
    if ef_counts:
        target_ef = max(ef_counts.items(), key=lambda item: (item[1], item[0]))[0]
    return {
        "lane": latest.get("lane"),
        "target_k": latest.get("k"),
        "target_ef": target_ef,
    }


def same_k_ef_local_digest(
    rows: list[dict[str, Any]],
    target_k: int | None,
    target_ef: int | None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    normalized = [_normalize_result_row(row) for row in rows]

    exact = [
        row for row in normalized
        if row.get("k") == target_k and row.get("expansion_factor") == target_ef
    ]
    if len(exact) < limit:
        same_k = [
            row for row in normalized
            if row.get("k") == target_k and row.get("expansion_factor") != target_ef
        ]
        exact.extend(same_k[-(limit - len(exact)):])
    if len(exact) < limit:
        same_ef = [
            row for row in normalized
            if row.get("expansion_factor") == target_ef and row.get("k") != target_k
        ]
        exact.extend(same_ef[-(limit - len(exact)):])

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in exact[-limit:]:
        experiment_id = str(row.get("experiment_id") or "")
        if experiment_id and experiment_id in seen:
            continue
        if experiment_id:
            seen.add(experiment_id)
        deduped.append({
            "experiment_id": experiment_id,
            "tier": row.get("tier"),
            "decision": row.get("decision"),
            "val_fvu": row.get("val_fvu"),
            "k": row.get("k"),
            "expansion_factor": row.get("expansion_factor"),
            "architecture": row.get("architecture"),
        })
    return deduped


def trim_text(value: Any, limit: int = 220) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[: limit - 32].rstrip() + "\n[truncated]"


def load_architecture_integration_checklist() -> str:
    if not ARCHITECTURE_INTEGRATION_SKILL_PATH.exists():
        return ""
    text = ARCHITECTURE_INTEGRATION_SKILL_PATH.read_text().strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2].strip()
    return text


def _error_mentions_architecture_integration(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    summary = str(entry.get("error_summary") or "").lower()
    return any(
        token in summary
        for token in (
            "unknown architecture",
            "factory dispatch",
            "dispatch",
            "registration blocker",
            "constructible",
        )
    )


def should_include_architecture_integration_checklist(
    memory: dict[str, Any],
    recent_summaries: list[dict[str, Any]],
) -> bool:
    for summary in recent_summaries[-2:]:
        if not isinstance(summary, dict):
            continue
        if str(summary.get("change_type") or "") == "edit_sae_code":
            return True
        if (
            str(summary.get("family_stage") or "") == "prototype"
            and str(summary.get("decision") or "") in {"crash", "policy_reject"}
        ):
            return True

    recent_failures = list(memory.get("recent_training_failures", []))[-4:] + list(memory.get("recent_sanity_failures", []))[-2:]
    if any(_error_mentions_architecture_integration(entry) for entry in recent_failures):
        return True

    return False


def baseline_runtime_digest(baseline_runtime: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key, value in baseline_runtime.items():
        if not isinstance(value, dict):
            continue
        entries.append({
            "key": key,
            "tokens_per_sec": value.get("tokens_per_sec"),
            "round": value.get("round"),
            "tier": value.get("tier"),
            "architecture": value.get("architecture"),
            "k": value.get("k"),
        })
    entries.sort(key=lambda item: (item.get("round") or 0, item.get("tokens_per_sec") or 0))
    return entries[-limit:]


def frontier_prompt_digest(frontier: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key, value in frontier.items():
        if not isinstance(value, dict):
            continue
        config = value.get("config", {})
        entries.append({
            "k": int(key) if str(key).isdigit() else value.get("k"),
            "fvu": value.get("fvu"),
            "architecture": value.get("architecture"),
            "expansion_factor": config.get("expansion_factor"),
            "lr": config.get("lr"),
            "optimizer": config.get("optimizer"),
            "family_name": config.get("family_name"),
            "tier": value.get("tier"),
            "round_commit": str(value.get("commit", ""))[:8],
        })
    entries.sort(key=lambda item: ((item.get("k") or 0), item.get("fvu") or float("inf")))
    return entries[:limit]


def pareto_prompt_digest(points: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    digested: list[dict[str, Any]] = []
    for point in points[:limit]:
        if not isinstance(point, dict):
            continue
        config = point.get("config", {})
        digested.append({
            "k": point.get("k"),
            "fvu": point.get("fvu"),
            "architecture": point.get("architecture"),
            "expansion_factor": config.get("expansion_factor"),
            "lr": config.get("lr"),
            "optimizer": config.get("optimizer"),
            "family_name": config.get("family_name"),
            "tier": point.get("tier"),
            "round_commit": str(point.get("commit", ""))[:8],
        })
    return digested


def recent_results_digest(rows: list[dict[str, str]], limit: int = 4) -> list[dict[str, str]]:
    digested: list[dict[str, str]] = []
    for row in rows[-limit:]:
        digested.append({
            "experiment_id": row.get("experiment_id", ""),
            "tier": row.get("tier", ""),
            "decision": row.get("decision", ""),
            "val_fvu": row.get("val_fvu", ""),
            "k": row.get("k", ""),
            "architecture": row.get("architecture", ""),
            "expansion_factor": row.get("expansion_factor", ""),
        })
    return digested


def family_prompt_digest(all_families: dict[str, Any]) -> dict[str, Any]:
    def _family_priority(item: tuple[str, Any]) -> tuple[int, int]:
        _, fdata = item
        status = str(fdata.get("status", ""))
        priority = {
            "promoted": 0,
            "active": 1,
            "incubating": 2,
            "archived": 3,
            "discarded": 4,
        }.get(status, 5)
        last_round = int(fdata.get("last_round") or -1)
        return (priority, -last_round)

    digested: dict[str, Any] = {}
    ranked_families = sorted(all_families.items(), key=_family_priority)
    for fname, fdata in ranked_families[:8]:
        if not isinstance(fdata, dict):
            continue
        status = fdata.get("status")
        base = {
            "status": status,
            "best_proxy_fvu": fdata.get("best_proxy_fvu"),
            "best_full_fvu": fdata.get("best_full_fvu"),
            "last_round": fdata.get("last_round"),
        }
        if status in ("discarded", "archived"):
            digested[fname] = base
            continue
        best_test = fdata.get("best_tested_config")
        if isinstance(best_test, dict):
            base["best_tested_config"] = {
                "round": best_test.get("round"),
                "stage": best_test.get("stage"),
                "k": best_test.get("k"),
                "decision": best_test.get("decision"),
                "val_fvu": best_test.get("val_fvu"),
            }
        next_steps = fdata.get("next_steps", [])
        if isinstance(next_steps, list) and next_steps:
            base["next_step"] = trim_text(next_steps[-1], limit=140)
        known_issues = fdata.get("known_issues", [])
        if isinstance(known_issues, list) and known_issues:
            base["known_issue"] = trim_text(known_issues[-1], limit=140)
        digested[fname] = base
    return digested


def compact_failure_list(items: list[Any], limit: int = 4) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items[-limit:]:
        if not isinstance(item, dict):
            continue
        out.append({
            "round": item.get("round"),
            "family_name": item.get("family_name"),
            "error_type": item.get("error_type"),
            "termination_reason": item.get("termination_reason"),
            "error_summary": trim_text(item.get("error_summary"), limit=160),
        })
    return out


def decision_critical_digest(memory: dict[str, Any]) -> dict[str, Any]:
    families = memory.get("architecture_families", {})
    blocked_family_lessons: list[str] = []
    non_gated_positive_evidence: list[str] = []
    do_not_repeat: list[str] = []
    repair_only_targets: list[str] = []

    for family_name, fdata in families.items():
        if not isinstance(fdata, dict):
            continue
        status = fdata.get("status")
        best_full = fdata.get("best_full_fvu")
        known_issue = ""
        issues = fdata.get("known_issues", [])
        if isinstance(issues, list) and issues:
            known_issue = trim_text(issues[-1], limit=140)

        if status == "archived":
            do_not_repeat.append(trim_text(
                f"{family_name}: archived; avoid generic retuning unless there is a new code-path hypothesis.",
                limit=140,
            ))

        if known_issue and any(token in known_issue.lower() for token in ("unknown architecture", "sanity", "registration", "invalid_env", "timeout")):
            blocked_family_lessons.append(trim_text(f"{family_name}: {known_issue}", limit=160))
            repair_only_targets.append(trim_text(
                f"{family_name}: only justify revisit via narrow code-fix / compatibility repair, not another blind sweep.",
                limit=160,
            ))

        if family_name != "gated" and best_full not in (None, "", "None"):
            non_gated_positive_evidence.append(trim_text(
                f"{family_name}: has real full-signal evidence with best_full_fvu={best_full}.",
                limit=140,
            ))

    recent_failures = compact_failure_list(memory.get("recent_training_failures", []), limit=4)
    for item in recent_failures:
        if item.get("family_name") and item.get("error_summary"):
            blocked_family_lessons.append(trim_text(
                f"{item['family_name']}: recent failure -> {item['error_summary']}",
                limit=160,
            ))

    return {
        "blocked_family_lessons": blocked_family_lessons[-4:],
        "non_gated_positive_evidence": non_gated_positive_evidence[-3:],
        "do_not_repeat": do_not_repeat[-5:],
        "repair_only_targets": repair_only_targets[-4:],
    }


def memory_prompt_digest(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_focus": memory.get("current_focus"),
        "architecture_families": family_prompt_digest(memory.get("architecture_families", {})),
        "recent_insights": [trim_text(item, limit=140) for item in memory.get("recent_insights", [])[-4:]],
        "recent_performance_findings": [trim_text(item, limit=140) for item in memory.get("performance_findings", [])[-4:]],
        "recent_sanity_failures": compact_failure_list(memory.get("recent_sanity_failures", []), limit=3),
        "recent_training_failures": compact_failure_list(memory.get("recent_training_failures", []), limit=4),
        "baseline_runtime": baseline_runtime_digest(memory.get("baseline_runtime", {}), limit=5),
        "next_hypotheses": [trim_text(item, limit=140) for item in memory.get("next_hypotheses", [])[:5]],
        "decision_critical": decision_critical_digest(memory),
    }


def build_prompt(
    state: dict[str, Any],
    memory: dict[str, Any],
    results: list[dict[str, str]],
    policy_context: str = "",
) -> str:
    frontier = load_json(FRONTIER_PATH, state.get("frontier", {}))
    operator_hints, _ = split_hints(load_operator_hints())
    operator_guide_excerpt = load_operator_guide_excerpt()

    memory_digest = memory_prompt_digest(memory)
    active_context = infer_active_kef_context(results)
    lane_context = lane_grouped_results_digest(results, per_lane_limit=4)
    recent_summaries = recent_round_summaries_trimmed(limit=2)
    architecture_integration_checklist = (
        load_architecture_integration_checklist()
        if should_include_architecture_integration_checklist(memory, recent_summaries)
        else ""
    )
    local_context = same_k_ef_local_digest(
        results,
        target_k=active_context.get("target_k"),
        target_ef=active_context.get("target_ef"),
        limit=5,
    )

    context = {
        "best_proxy_frontier": frontier_prompt_digest(state.get("proxy_frontier", {}), limit=4),
        "best_full_frontier": frontier_prompt_digest(state.get("full_frontier", frontier), limit=4),
        "pareto_proxy_frontier": pareto_prompt_digest(state.get("pareto_proxy_frontier", []), limit=4),
        "pareto_full_frontier": pareto_prompt_digest(state.get("pareto_full_frontier", state.get("pareto_frontier", [])), limit=4),
        "active_comparison_context": active_context,
        "quality_anchor_lane": lane_context.get("quality_anchor", {}),
        "low_k_tradeoff_lane": lane_context.get("low_k_tradeoff", {}),
        "same_k_ef_local_context": local_context,
        "agent_state": state.get("agent", {}),
        "rounds_since_new_family": state.get("agent", {}).get("rounds_since_new_family", 0),
        "memory_digest": memory_digest,
        "operator_hints": operator_hints[:8],
        "operator_guide_excerpt": operator_guide_excerpt,
        "architecture_integration_checklist": architecture_integration_checklist,
        "recent_results": recent_results_digest(results, limit=4),
        "recent_round_summaries": recent_summaries,
    }
    policy_section = ""
    if policy_context:
        policy_section = f"\n\nPolicy guidance (from the runtime strategy layer):\n{policy_context}\n"
    return f"""
You are the nightly SAE research agent for this repository.

Primary objective:
- maintain and improve the Pareto frontier across FVU, K, and available cost signals
- K=128 is an initial anchor point, not the only success criterion
- smaller K is intrinsically valuable when it creates a non-dominated tradeoff, even if FVU is somewhat worse
- reducing K is more important than reducing expansion_factor; prefer K exploration before EF exploration when both are open
- reduce FVU where possible, but do not suppress lower-K exploration just because it does not beat the K=128 quality anchor
- improve throughput / memory when that strengthens the Pareto frontier

Execution layer is fixed:
- training entrypoint: scripts/autoresearch_test.sh
- results recorder: research/controller.py
- memory files: research/history/state.json, frontier.json, memory.json, results.tsv

Important rules:
- Read research/program.md before deciding.
- Read and respect any operator_hints in the structured context.
- Read and respect operator_guide_excerpt in the structured context when present.
- Read and follow architecture_integration_checklist when present. It is mandatory for new-family wiring and Unknown architecture repairs.
- You may edit ONLY files under sparsify/.
- Do not edit research/history/*, research/*.py, or scripts/autoresearch_test.sh.
- For parameter-only experiments, use env_overrides instead of editing launch code.
- Make at most ONE coherent hypothesis this round.
- Fill family_name and family_stage when the round is about a specific architecture family.
- Prefer proxy unless there is a strong reason to go straight to full.
- Direct full requests may be coerced back to proxy by the runtime.
- You are allowed to explore the SAE architecture space broadly, not just tune existing defaults.
- Treat `pareto_full_frontier` and `pareto_proxy_frontier` as the main target objects to improve.
- Use the lane summaries as the primary comparison view: `quality_anchor_lane` for higher-K work and `low_k_tradeoff_lane` for lower-K tradeoff work.
- Within a lane, compare architectures primarily at the same `(K, expansion_factor)`. Do not treat different K values inside the same lane as directly interchangeable.
- Use `same_k_ef_local_context` as the first place to judge what changed recently. Only fall back to broader lane/global context when the exact `(K, EF)` neighborhood is sparse.
- Architectural examples such as Sparse-ReLU, Gated, JumpReLU, GroupTopK, MoE-style encoders, factorized encoders, routed encoders, multi-branch encoders, or new sparse activation mechanisms are examples only, not limits.
- If a new architecture family seems promising, you may add it under sparsify/ as long as the change is coherent and compatible with the existing execution layer.
- Encoder design is part of the search space. You may change routing, gating, intermediate width, branch structure, activation form, grouping strategy, or other internal encoder mechanisms when justified.
- When exploring a new architecture family, also consider its own important hyperparameters, for example intermediate width or routing width in an ICE/MoE-style encoder.
- Treat `expansion_factor` as a model-capacity axis, not a neutral parameter. Cross-EF results are not a fair pure architecture comparison.
- When comparing architecture families, prefer the same `expansion_factor` first.
- Recent evidence indicates that `expansion_factor=8` is often too capacity-constrained for the current frontier targets. Do not treat EF=8 as the main default search regime.
- Use `expansion_factor=12` or `16` as the primary comparison band unless memory contains strong contrary evidence for a specific family.
- Treat `expansion_factor=8` mainly as a lower-bound / capacity-floor check, not as the primary place to spend many consecutive rounds.
- The goal for EF is to achieve the lowest practical FVU under the `EF<=16` cap, with most search effort concentrated on `EF in {12, 16}` and only limited validation at `EF=8`.
- Slow runs may indicate implementation bottlenecks rather than bad architectures.
- If a recent run was a performance regression, prefer an edit_perf_code follow-up over drawing a negative architecture conclusion.
- If parameter-only search is not closing the quality gap, escalate to architecture exploration rather than continuing a weak local search.
- A new architecture family may require several rounds of incubation. Do not expect a first prototype to beat the frontier immediately.
- Use family_stage to reflect whether the round is a prototype, stabilization pass, mainline comparison, or promotion attempt.
- If several rounds have passed without proposing a new family or advancing an incubating family, bias toward architecture exploration rather than more local tuning.
- Never stop the session on your own. You must always return command="run".
- If evidence is weak or local tuning looks exhausted, propose the least-bad next informative move instead of stopping.
- Set primary_variable to indicate which single dimension this round changes (architecture, optimizer, lr, k, expansion_factor, other_param, or code_fix). Avoid changing multiple primary variables in one round.
- Return a final JSON object matching the schema exactly.
{policy_section}
Current structured context:
{json.dumps(context, indent=2)}
""".strip()
def build_resume_prompt(
    state: dict[str, Any],
    memory: dict[str, Any],
    results: list[dict[str, str]],
    round_id: int,
    brief: dict[str, Any],
    policy_context: str = "",
) -> str:
    operator_hints, _ = split_hints(load_operator_hints())
    operator_guide_excerpt = load_operator_guide_excerpt()
    memory_digest = memory_prompt_digest(memory)
    active_context = infer_active_kef_context(results)
    lane_context = lane_grouped_results_digest(results, per_lane_limit=4)
    recent_summaries = brief.get("recent_round_summaries", recent_round_summaries_trimmed())
    architecture_integration_checklist = (
        load_architecture_integration_checklist()
        if should_include_architecture_integration_checklist(memory, recent_summaries)
        else ""
    )
    local_context = same_k_ef_local_digest(
        results,
        target_k=active_context.get("target_k"),
        target_ef=active_context.get("target_ef"),
        limit=5,
    )
    latest_pending_hints = operator_hints[:8]
    brief_pending_hints = brief.get("pending_hints", [])
    merged_pending_hints: list[dict[str, Any]] = []
    seen_messages: set[str] = set()
    for hint in latest_pending_hints + brief_pending_hints:
        if not isinstance(hint, dict):
            continue
        message = str(hint.get("message") or "")
        if not message or message in seen_messages:
            continue
        merged_pending_hints.append(hint)
        seen_messages.add(message)
        if len(merged_pending_hints) >= 8:
            break
    payload = {
        "round": round_id,
        "current_focus": brief.get("current_focus") or memory.get("current_focus"),
        "pareto_full_frontier": brief.get("pareto_full_frontier", state.get("pareto_full_frontier", state.get("pareto_frontier", []))),
        "recent_results": brief.get("recent_results", summarize_results(results)),
        "active_comparison_context": active_context,
        "quality_anchor_lane": lane_context.get("quality_anchor", {}),
        "low_k_tradeoff_lane": lane_context.get("low_k_tradeoff", {}),
        "same_k_ef_local_context": local_context,
        "recent_round_summaries": recent_summaries,
        "incubating_families": brief.get("incubating_families", {}),
        "memory_digest": memory_digest,
        "recent_performance_findings": brief.get(
            "recent_performance_findings",
            memory_digest.get("recent_performance_findings", []),
        ),
        "recent_sanity_failures": brief.get(
            "recent_sanity_failures",
            memory_digest.get("recent_sanity_failures", []),
        ),
        "recent_training_failures": brief.get(
            "recent_training_failures",
            memory_digest.get("recent_training_failures", []),
        ),
        "pending_hints": merged_pending_hints,
        "operator_guide_excerpt": operator_guide_excerpt,
        "architecture_integration_checklist": architecture_integration_checklist,
        "next_move_guidance": brief.get("next_move_guidance", memory.get("next_hypotheses", [])[:8]),
        "rounds_since_new_family": state.get("agent", {}).get("rounds_since_new_family", 0),
    }
    policy_section = ""
    if policy_context:
        policy_section = f"\nPolicy guidance (from the runtime strategy layer):\n{policy_context}\n"
    return f"""
Continue the same nightly SAE research session.

This is round {round_id}. Use the existing session context plus the structured update below.
Do not restate policy or explain your reasoning. Return one final JSON object only, with no markdown fences.
The JSON must match the same shape as the established action contract:
- command, hypothesis, summary, change_type, experiment_tier, expected_win
- family_name, family_stage, self_review, needs_sanity
- env_overrides, touched_files, notes_to_memory, next_hypotheses
- primary_variable (which single dimension this round changes: architecture, optimizer, lr, k, expansion_factor, other_param, or code_fix)
Runtime priorities that override weak local heuristics:
- Lower K is the top priority. If K exploration and EF exploration are both plausible, do K first.
- Treat expansion_factor as a capacity axis. Do not claim an architecture win from cross-EF comparisons alone.
- Prefer same-EF architecture comparisons.
- Use the active lane plus the exact `(K, EF)` neighborhood as the main basis for judgment; do not let `K=128` evidence override a low-K tradeoff decision, or vice versa.
- Follow architecture_integration_checklist when it is present in the structured update. It is mandatory for new-family wiring and Unknown architecture repairs.
- Default EF for a new family or refreshed comparison is 12 unless strong prior evidence says 16 is the better mainline band for that family.
- Treat EF=8 as a lower-bound check rather than the primary comparison target.
- Under the current evidence, prefer EF {12, 16} for most meaningful comparisons and use EF=8 sparingly to validate capacity limits.
{policy_section}
Structured update:
{json.dumps(payload, indent=2)}
""".strip()
