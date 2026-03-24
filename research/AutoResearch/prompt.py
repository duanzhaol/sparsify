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


# ---------------------------------------------------------------------------
# Constants — Hard constraints baked into prompt template
# ---------------------------------------------------------------------------

ROLE_AND_OBJECTIVE = """\
You are the SAE research agent for this repository.

Primary objective:
- Maintain and improve the Pareto frontier across FVU and K
- K=128 is an anchor, not the only success criterion
- Smaller K is valuable when it creates a non-dominated tradeoff"""

EXECUTION_LAYER = """\
Execution layer is fixed:
- Training: scripts/autoresearch_test.sh
- Results: research/controller.py
- Memory: research/history/"""

EDIT_RULES = """\
Rules:
- Edit ONLY files under sparsify/
- Use env_overrides for parameter-only experiments
- ONE hypothesis per round
- Set primary_variable to indicate which dimension changes
- Never return command="stop"
- Return a final JSON object matching the action schema"""

K_WHITELIST = "K constraint: K must be one of {4, 8, 16, 24, 32, 64, 96, 128}. No other K value is allowed."

EF_CONSTRAINT = "EF constraint: EXPANSION_FACTOR is fixed at 16 for ALL experiments. Set EXPANSION_FACTOR=16 in every env_overrides."

SINGLE_VARIABLE_PRINCIPLE = "Single-variable principle: Change ONE primary dimension per round."

HARD_CONSTRAINT_REMINDER = (
    "Reminders: K in {4,8,16,24,32,64,96,128}. EF=16 always. "
    "One variable per round. Edit only sparsify/. Return JSON."
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
        EXECUTION_LAYER,
        EDIT_RULES,
        K_WHITELIST,
        EF_CONSTRAINT,
        SINGLE_VARIABLE_PRINCIPLE,
    ])


# ---------------------------------------------------------------------------
# Layer 2: Current Decision State
# ---------------------------------------------------------------------------


def section_policy_guidance(policy_guidance: str) -> str:
    if not policy_guidance:
        return ""
    return f"Policy guidance:\n{policy_guidance}"


def section_agent_state(
    round_index: int,
    consecutive_crashes: int,
    consecutive_no_improve: int,
    rounds_since_new_family: int,
) -> str:
    return (
        f"Agent state: round={round_index}, "
        f"consecutive_crashes={consecutive_crashes}, "
        f"consecutive_no_improve={consecutive_no_improve}, "
        f"rounds_since_new_family={rounds_since_new_family}"
    )


def section_frontier(frontier: dict[str, Any], limit: int = 10) -> str:
    """Frontier snapshot sorted by K, split into EF=16 (main) and EF=12 (legacy)."""
    main: list[tuple[int, str]] = []
    legacy: list[tuple[int, str]] = []

    for _key, entry in frontier.items():
        if not isinstance(entry, dict):
            continue
        k = int(entry.get("k", 0))
        ef = int(entry.get("ef", 0))
        fvu = entry.get("fvu", "?")
        arch = entry.get("architecture", "?")
        cfg = entry.get("config", {})
        lr = cfg.get("lr", "?")
        opt = cfg.get("optimizer", "?")

        if ef == 16:
            main.append((k, f"  k={k:>3d}  fvu={fvu:<12}  arch={arch}  lr={lr} opt={opt}"))
        else:
            legacy.append((k, f"  k={k:>3d}  ef={ef}  fvu={fvu:<12}  arch={arch}"))

    main.sort()
    legacy.sort()

    parts = ["Frontier (EF=16, current regime):"]
    if main:
        for _, line in main[:limit]:
            parts.append(line)
    else:
        parts.append("  (no entries yet)")

    if legacy:
        parts.append("")
        parts.append("Legacy frontier (EF=12, reference only):")
        for _, line in legacy[:6]:
            parts.append(line)

    return "\n".join(parts)


def section_recent_rounds(round_summaries: list[str]) -> str:
    """Recent round summaries as one-liners (already formatted by state)."""
    if not round_summaries:
        return ""
    return f"Recent rounds ({len(round_summaries)}):\n" + "\n".join(
        f"  {s}" for s in round_summaries
    )


def section_tactical_hints(hints: list[dict[str, Any]]) -> str:
    """Tactical operator hints only. Filters out K/EF hard constraints."""
    tactical = []
    for hint in hints:
        text = hint.get("text", "")
        if any(text.startswith(p) for p in _HARD_CONSTRAINT_HINT_PREFIXES):
            continue
        tactical.append(text)

    if not tactical:
        return ""
    lines = ["Operator hints:"]
    for i, t in enumerate(tactical, 1):
        lines.append(f"  {i}. {t}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 3: Rolling Memory
# ---------------------------------------------------------------------------


def section_memory_brief(
    memory: dict[str, Any],
    frontier: dict[str, Any],
    recent_round_limit: int = 10,
) -> str:
    """Slim memory: only frontier-holding + recently tested families, failures, hypotheses."""
    families = memory.get("architecture_families", {})

    # Which families to show
    frontier_families: set[str] = set()
    for entry in frontier.values():
        if isinstance(entry, dict):
            cfg = entry.get("config", {})
            fn = cfg.get("family_name", entry.get("architecture", ""))
            if fn:
                frontier_families.add(fn.lower())

    recent_families: set[str] = set()
    for rr in memory.get("recent_rounds", [])[-recent_round_limit:]:
        fn = rr.get("family_name", "")
        if fn:
            recent_families.add(fn.lower())

    show = frontier_families | recent_families

    parts: list[str] = ["Architecture families (frontier holders + recently tested):"]
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
            parts.append(f"{label}:")
            for f in fails:
                if isinstance(f, dict):
                    parts.append(
                        f"  r{f.get('round','?')} {f.get('family_name','?')} "
                        f"{f.get('error_type','')}: {_truncate(str(f.get('error_summary','')), 80)}"
                    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Layer 4: Reference Digests
# ---------------------------------------------------------------------------


def section_operator_guide_digest(guide_text: str, max_chars: int = 3000) -> str:
    """Extract Runtime Priorities section from operator_guide.md."""
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
        return f"Operator guide (key priorities):\n{text}"

    truncated = guide_text[:max_chars - 20] + "\n[truncated]"
    return f"Operator guide (excerpt):\n{truncated}"


def section_prior_research_digest(prior_text: str, max_chars: int = 2000) -> str:
    """Extract sections 1-2 + phase table from prior_research_history.md."""
    if not prior_text:
        return ""

    lines = prior_text.split("\n")
    extracted: list[str] = []
    in_section = False
    section_depth = 0

    for line in lines:
        if line.startswith("## 1.") or line.startswith("## 2."):
            in_section = True
            section_depth = 2
        elif line.startswith("### 3.1"):
            in_section = True
            section_depth = 3
        elif in_section and line.startswith("## ") and section_depth == 2:
            in_section = False
        elif in_section and line.startswith("### ") and section_depth == 3 and "3.1" not in line:
            in_section = False

        if in_section:
            extracted.append(line)

    if extracted:
        text = "\n".join(extracted)
        if len(text) > max_chars:
            text = text[:max_chars - 20] + "\n[truncated]"
        return f"Prior research (key findings):\n{text}"

    truncated = prior_text[:max_chars - 20] + "\n[truncated]"
    return f"Prior research (excerpt):\n{truncated}"


def section_architecture_checklist(
    checklist_text: str,
    memory: dict[str, Any],
    raw_summaries: list[dict[str, Any]],
) -> str:
    """Conditionally include architecture integration checklist.

    Fixed: uses raw dict summaries from state.recent_round_summaries(),
    not trimmed strings from recent_round_summaries_trimmed().
    """
    if not checklist_text:
        return ""
    if not _should_include_checklist(memory, raw_summaries):
        return ""
    return f"Architecture integration checklist:\n{checklist_text}"


# ---------------------------------------------------------------------------
# Compose functions
# ---------------------------------------------------------------------------


def compose_proposal(state: Any, policy_guidance: str) -> str:
    """Full proposal prompt in 4 layers."""
    sections: list[str] = []

    # Layer 1
    sections.append(section_hard_constraints())

    # Layer 2
    sections.append(section_policy_guidance(policy_guidance))
    sections.append(section_agent_state(
        state.round_index, state.consecutive_crashes,
        state.consecutive_no_improve, state.rounds_since_new_family,
    ))
    sections.append(section_frontier(state.frontier))
    sections.append(section_recent_rounds(
        state.recent_round_summaries_trimmed(limit=5),
    ))
    sections.append(section_tactical_hints(state.get_pending_hints()[:8]))

    # Layer 3
    sections.append(section_memory_brief(state.memory, state.frontier))

    # Layer 4
    sections.append(section_operator_guide_digest(
        state.load_operator_guide_excerpt(),
    ))
    sections.append(section_prior_research_digest(
        state.load_prior_research(),
    ))

    checklist = _load_architecture_checklist()
    if checklist:
        raw = state.recent_round_summaries(limit=2)
        sections.append(section_architecture_checklist(checklist, state.memory, raw))

    return "\n\n".join(s for s in sections if s)


def compose_resume(state: Any, round_id: int, policy_guidance: str) -> str:
    """Lightweight delta prompt for session resume."""
    sections: list[str] = []

    sections.append(f"Continue the SAE research session. Round {round_id}.")
    sections.append("Return one JSON object matching the action schema. No markdown fences.")
    sections.append(HARD_CONSTRAINT_REMINDER)

    sections.append(section_policy_guidance(policy_guidance))
    sections.append(section_agent_state(
        state.round_index, state.consecutive_crashes,
        state.consecutive_no_improve, state.rounds_since_new_family,
    ))
    sections.append(section_frontier(state.frontier, limit=8))
    sections.append(section_recent_rounds(
        state.recent_round_summaries_trimmed(limit=2),
    ))
    sections.append(section_tactical_hints(state.get_pending_hints()[:4]))

    # Open hypotheses
    hypotheses = state.memory.get("next_hypotheses", [])[:5]
    if hypotheses:
        lines = ["Open hypotheses:"]
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
    """Repair prompt: fix engineering blocker without redesigning."""
    sections: list[str] = []

    sections.append(f"Continue round {round_id} in repair mode.")
    sections.append(
        "The previous code-edit failed with an engineering blocker.\n"
        "Do NOT redesign the experiment. Do NOT change family_name or env_overrides.\n"
        "Patch the implementation so the original experiment can run.\n"
        "Stay within sparsify/ only.\n"
        f"Attempt {repair_attempt} of {max_repair_attempts}.\n"
        "Return one final JSON object matching the action schema."
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
    sections.append(f"Repair context:\n{json.dumps(payload, indent=2)}")

    checklist = _load_architecture_checklist()
    if checklist:
        sections.append(f"Architecture integration checklist:\n{checklist}")

    return "\n\n".join(s for s in sections if s)


# ---------------------------------------------------------------------------
# Helpers (moved from agent.py)
# ---------------------------------------------------------------------------


def _load_architecture_checklist() -> str:
    """Load architecture integration checklist from SKILL.md."""
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
    """Determine whether to include the architecture checklist.

    Takes list[dict] (raw round summaries), NOT list[str].
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


def _truncate(s: str, limit: int) -> str:
    if not isinstance(s, str):
        return str(s)[:limit]
    return s if len(s) <= limit else s[:limit - 3] + "..."
