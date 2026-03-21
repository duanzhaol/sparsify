"""Prompt construction for the autoresearch agent."""

from __future__ import annotations

import json
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


def trim_text(value: Any, limit: int = 220) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[: limit - 32].rstrip() + "\n[truncated]"


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


def family_prompt_digest(all_families: dict[str, Any]) -> dict[str, Any]:
    digested: dict[str, Any] = {}
    for fname, fdata in all_families.items():
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

    context = {
        "frontier": frontier,
        "proxy_frontier": state.get("proxy_frontier", {}),
        "full_frontier": state.get("full_frontier", frontier),
        "pareto_proxy_frontier": state.get("pareto_proxy_frontier", []),
        "pareto_full_frontier": state.get("pareto_full_frontier", state.get("pareto_frontier", [])),
        "agent_state": state.get("agent", {}),
        "rounds_since_new_family": state.get("agent", {}).get("rounds_since_new_family", 0),
        "memory_digest": memory_digest,
        "operator_hints": operator_hints[:8],
        "operator_guide_excerpt": operator_guide_excerpt,
        "recent_results": summarize_results(results[-6:]),
        "recent_round_summaries": recent_round_summaries_trimmed(limit=3),
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
- You may edit ONLY files under sparsify/.
- Do not edit research/history/*, research/*.py, or scripts/autoresearch_test.sh.
- For parameter-only experiments, use env_overrides instead of editing launch code.
- Make at most ONE coherent hypothesis this round.
- Fill family_name and family_stage when the round is about a specific architecture family.
- Prefer proxy unless there is a strong reason to go straight to full.
- Direct full requests may be coerced back to proxy by the runtime.
- You are allowed to explore the SAE architecture space broadly, not just tune existing defaults.
- Treat `pareto_full_frontier` and `pareto_proxy_frontier` as the main target objects to improve.
- Architectural examples such as Sparse-ReLU, Gated, JumpReLU, GroupTopK, MoE-style encoders, factorized encoders, routed encoders, multi-branch encoders, or new sparse activation mechanisms are examples only, not limits.
- If a new architecture family seems promising, you may add it under sparsify/ as long as the change is coherent and compatible with the existing execution layer.
- Encoder design is part of the search space. You may change routing, gating, intermediate width, branch structure, activation form, grouping strategy, or other internal encoder mechanisms when justified.
- When exploring a new architecture family, also consider its own important hyperparameters, for example intermediate width or routing width in an ICE/MoE-style encoder.
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
    payload = {
        "round": round_id,
        "current_focus": brief.get("current_focus") or memory.get("current_focus"),
        "pareto_full_frontier": brief.get("pareto_full_frontier", state.get("pareto_full_frontier", state.get("pareto_frontier", []))),
        "recent_results": brief.get("recent_results", summarize_results(results)),
        "recent_round_summaries": brief.get("recent_round_summaries", recent_round_summaries_trimmed()),
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
        "pending_hints": brief.get("pending_hints", operator_hints[:8]),
        "operator_guide_excerpt": operator_guide_excerpt,
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
{policy_section}
Structured update:
{json.dumps(payload, indent=2)}
""".strip()
