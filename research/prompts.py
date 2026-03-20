"""Prompt construction for the autoresearch agent."""

from __future__ import annotations

import json
from typing import Any

from research.state_io import (
    FRONTIER_PATH,
    load_json,
    load_operator_hints,
    split_hints,
    summarize_results,
    recent_round_summaries,
    recent_round_summaries_trimmed,
)


def build_prompt(
    state: dict[str, Any],
    memory: dict[str, Any],
    results: list[dict[str, str]],
    policy_context: str = "",
) -> str:
    frontier = load_json(FRONTIER_PATH, state.get("frontier", {}))
    operator_hints, _ = split_hints(load_operator_hints())

    # Trim architecture_families to avoid unbounded prompt growth:
    # keep only non-discarded/non-archived families, and cap tested_configs per family
    all_families = memory.get("architecture_families", {})
    trimmed_families: dict[str, Any] = {}
    for fname, fdata in all_families.items():
        if fdata.get("status") in ("discarded", "archived"):
            # Include a minimal summary for discarded/archived families
            trimmed_families[fname] = {
                "status": fdata.get("status"),
                "best_proxy_fvu": fdata.get("best_proxy_fvu"),
                "best_full_fvu": fdata.get("best_full_fvu"),
            }
        else:
            trimmed = dict(fdata)
            # Cap tested_configs to last 5 entries
            if "tested_configs" in trimmed:
                trimmed["tested_configs"] = trimmed["tested_configs"][-5:]
            # Cap known_issues and next_steps
            if "known_issues" in trimmed:
                trimmed["known_issues"] = trimmed["known_issues"][-3:]
            if "next_steps" in trimmed:
                trimmed["next_steps"] = trimmed["next_steps"][-3:]
            trimmed_families[fname] = trimmed

    baseline_runtime = memory.get("baseline_runtime", {})

    context = {
        "frontier": frontier,
        "proxy_frontier": state.get("proxy_frontier", {}),
        "full_frontier": state.get("full_frontier", frontier),
        "agent_state": state.get("agent", {}),
        "rounds_since_new_family": state.get("agent", {}).get("rounds_since_new_family", 0),
        "current_focus": memory.get("current_focus"),
        "architecture_families": trimmed_families,
        "recent_insights": memory.get("recent_insights", [])[-8:],
        "recent_performance_findings": memory.get("performance_findings", [])[-8:],
        "baseline_runtime": baseline_runtime,
        "operator_hints": operator_hints[:8],
        "next_hypotheses": memory.get("next_hypotheses", [])[:8],
        "recent_results": summarize_results(results[-8:]),
        "recent_round_summaries": recent_round_summaries(),
    }
    policy_section = ""
    if policy_context:
        policy_section = f"\n\nPolicy guidance (from the runtime strategy layer):\n{policy_context}\n"
    return f"""
You are the nightly SAE research agent for this repository.

Primary objective:
- reduce FVU
- K=128 is only an initial quality anchor, not a success criterion
- only reward smaller K after quality has materially beaten the current K=128 baseline
- the quality target is aggressive: aim for configurations that can drive FVU to roughly half of the current K=128 baseline before preferring smaller K or lower cost
- then reduce cost / improve throughput / memory

Execution layer is fixed:
- training entrypoint: scripts/autoresearch_test.sh
- results recorder: research/controller.py
- memory files: research/history/state.json, frontier.json, memory.json, results.tsv

Important rules:
- Read research/program.md before deciding.
- Read and respect any operator_hints in the structured context.
- You may edit ONLY files under sparsify/.
- Do not edit research/history/*, research/*.py, or scripts/autoresearch_test.sh.
- For parameter-only experiments, use env_overrides instead of editing launch code.
- Make at most ONE coherent hypothesis this round.
- Fill family_name and family_stage when the round is about a specific architecture family.
- Prefer proxy unless there is a strong reason to go straight to full.
- Direct full requests may be coerced back to proxy by the runtime.
- You are allowed to explore the SAE architecture space broadly, not just tune existing defaults.
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
- If there is no promising next move, return command="stop".
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
    payload = {
        "round": round_id,
        "current_focus": brief.get("current_focus") or memory.get("current_focus"),
        "best_full_frontier": brief.get("best_full_frontier", state.get("full_frontier", {})),
        "recent_results": brief.get("recent_results", summarize_results(results)),
        "recent_round_summaries": brief.get("recent_round_summaries", recent_round_summaries_trimmed()),
        "incubating_families": brief.get("incubating_families", {}),
        "recent_performance_findings": brief.get("recent_performance_findings", memory.get("performance_findings", [])[-8:]),
        "pending_hints": brief.get("pending_hints", operator_hints[:8]),
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
