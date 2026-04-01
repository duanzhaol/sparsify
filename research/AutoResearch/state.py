"""Unified persistence layer for the autoresearch framework.

All state file I/O is centralized in ``StateManager``.  No other module
should read or write state/memory/results files directly.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from .compatibility import (
    COST_METRIC_VERSION,
    cost_entry_is_current,
    parse_compatibility_registry,
)
from .config_resolution import resolve_action_configs, summary_invalid_reason
from .objective import (
    EXCEED_FIELD,
    OBJECTIVE_FIELD,
    compute_objective_score,
    entry_objective_metrics,
    extract_step_exceed_alpha_0_50,
    read_latest_step,
)
from .target_profile import resolve_target_profile
from .types import (
    Action,
    BASE_ENV_DEFAULTS,
    FRONTIER_PATH,
    HINTS_PATH,
    HISTORY_DIR,
    LOG_DIR,
    MEMORY_PATH,
    OPERATOR_GUIDE_PATH,
    PRIOR_RESEARCH_PATH,
    RESULTS_COLUMNS,
    RESULTS_PATH,
    ROUND_SUMMARIES_DIR,
    Result,
    RoundContext,
    SESSION_BRIEF_PATH,
    STATE_PATH,
    STATUS_PATH,
    TIMELINE_PATH,
)

# ---------------------------------------------------------------------------
# Low-level JSON helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

_DEFAULT_AGENT_STATE: dict[str, Any] = {
    "round_index": 0,
    "consecutive_crashes": 0,
    "consecutive_no_improve": 0,
    "rounds_since_new_family": 0,
    "last_action_file": None,
    "last_patch_file": None,
    "active_session_id": None,
    "active_session_started_at": None,
    "active_session_rounds": 0,
    "active_session_status": "closed",
    "last_resume_ok_at": None,
    "crash_resets": 0,
    "last_meta_round": 0,
}

_DEFAULT_MEMORY: dict[str, Any] = {
    "current_focus": (
        "Minimize objective_score = total_cost_ratio + exceed_alpha_0_50 under the total_cost budget. "
        "FVU is a diagnostic and tie-break metric, not the primary runtime objective. "
        "Prefer experiments that improve the objective with clean cost/exceed decomposition."
    ),
    "architecture_findings": [],
    "performance_findings": [],
    "failure_patterns": [],
    "baseline_runtime": {},
    "architecture_families": {},
    "recent_rounds": [],
    "recent_insights": [],
    "recent_sanity_failures": [],
    "recent_training_failures": [],
    "next_hypotheses": [
        "Minimize objective_score = total_cost_ratio + exceed_alpha_0_50 instead of treating FVU as the main target.",
        "Use FVU as a diagnostic / tie-break metric while keeping the hard total_cost budget.",
        "Prefer structure changes that materially improve cost or exceed, not only local FVU interpolation.",
    ],
}


class StateManager:
    """Single source of truth for all autoresearch persistent state."""

    def __init__(
        self,
        history_dir: Path = HISTORY_DIR,
        *,
        persist_load_fixes: bool = True,
    ) -> None:
        self.history_dir = history_dir
        self._persist_load_fixes = persist_load_fixes
        self._state: dict[str, Any] = {}
        self._memory: dict[str, Any] = {}
        self._load()

    # -----------------------------------------------------------------------
    # Core state accessors
    # -----------------------------------------------------------------------

    @property
    def agent(self) -> dict[str, Any]:
        return self._state.setdefault("agent", dict(_DEFAULT_AGENT_STATE))

    @property
    def frontier(self) -> dict[str, Any]:
        """Unified frontier (single-tier)."""
        return self._state.setdefault("frontier", {})

    @property
    def memory(self) -> dict[str, Any]:
        return self._memory

    @property
    def round_index(self) -> int:
        return int(self.agent.get("round_index", 0))

    @property
    def consecutive_crashes(self) -> int:
        return int(self.agent.get("consecutive_crashes", 0))

    @property
    def consecutive_no_improve(self) -> int:
        return int(self.agent.get("consecutive_no_improve", 0))

    @property
    def rounds_since_new_family(self) -> int:
        return int(self.agent.get("rounds_since_new_family", 0))

    @property
    def families(self) -> dict[str, Any]:
        return self._memory.setdefault("architecture_families", {})

    # -----------------------------------------------------------------------
    # State mutations
    # -----------------------------------------------------------------------

    def record_round_outcome(
        self,
        round_id: int,
        action: Action,
        result: Result,
        touched_files: list[str],
        patch_path: Path | None,
        ctx: RoundContext,
    ) -> None:
        """Update all state files after a round completes."""
        action_dict = action.to_dict()
        result_dict = result.to_dict()
        family_name = (
            action.family_name
            or (ctx.resolved_candidate_env_config or {}).get("ARCHITECTURE", "")
        ).lower()
        existing_family = self.families.get(family_name)
        is_new_family = existing_family is None or existing_family.get("last_round") is None

        # 1. Update memory (families, insights, failures)
        self._append_memory(action, result, round_id, touched_files, ctx)

        # 2. Update agent counters
        self._update_agent_counters(result, is_new_family)

        # 3. Write round summary
        self._write_round_summary(round_id, action_dict, result_dict, touched_files, patch_path, ctx)

        # 4. Append to results.tsv
        self._append_results_tsv(result_dict)

        # 5. Mark one-shot operator hints as applied
        self._mark_hints_applied(round_id)

        # 6. Persist
        self._save_state()
        self._save_memory()

    def append_timeline_event(self, event: str, **payload: Any) -> str:
        """Append to timeline.jsonl. Returns event_id."""
        event_id = f"evt_{time.time_ns()}"
        record = {
            "event_id": event_id,
            "ts": int(time.time()),
            "event": event,
            **payload,
        }
        timeline_path = self.history_dir / "timeline.jsonl"
        with open(timeline_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return event_id

    def log_round_event(self, ctx: RoundContext, event: str, **payload: Any) -> str:
        """Append timeline event and track in round context."""
        event_id = self.append_timeline_event(
            event,
            round=ctx.round_id,
            session_id=payload.get("session_id", ctx.session_id),
            family_name=payload.get("family_name", ctx.family_name),
            family_stage=payload.get("family_stage", ctx.family_stage),
            status=payload.get("status"),
            decision=payload.get("decision"),
            payload=payload,
        )
        ctx.timeline_event_ids.append(event_id)
        return event_id

    def write_status(self, stage: str, **payload: Any) -> None:
        _save_json(
            self.history_dir / "current_status.json",
            {"timestamp": int(time.time()), "stage": stage, **payload},
        )

    def reset_crash_counters(self) -> None:
        self.agent["consecutive_crashes"] = 0
        self.agent["consecutive_no_improve"] = 0
        self.agent["crash_resets"] = 0
        self._save_state()

    def update_baseline_tps(
        self, key: str, tps: float, round_id: int, config: dict[str, Any]
    ) -> None:
        baselines = self._memory.setdefault("baseline_runtime", {})
        baselines[key] = {
            "tokens_per_sec": tps,
            "round": round_id,
            **{k: config.get(k) for k in ("tier", "architecture", "k") if config.get(k) is not None},
            "updated_at": int(time.time()),
        }
        self._save_memory()

    def get_baseline_tps(self, architecture: str, hookpoints: str) -> float | None:
        key = f"standard|{architecture}|{hookpoints}"
        entry = self._memory.get("baseline_runtime", {}).get(key)
        if entry is None:
            return None
        return entry.get("tokens_per_sec")

    def update_session(self, **fields: Any) -> None:
        """Update session-related fields in agent state."""
        for k, v in fields.items():
            self.agent[k] = v
        self._save_state()

    def load_results(self, limit: int | None = None) -> list[dict[str, str]]:
        results_path = self.history_dir / "results.tsv"
        if not results_path.exists():
            return []
        with open(results_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        return rows[-limit:] if limit is not None else rows

    def recent_round_summaries(self, limit: int = 3) -> list[dict[str, Any]]:
        summaries_dir = self.history_dir / "round_summaries"
        if not summaries_dir.exists():
            return []
        paths = sorted(summaries_dir.glob("round_*.json"))[-limit:]
        return [_load_json(p, {}) for p in paths]

    def load_round_summary(self, round_id: int) -> dict[str, Any]:
        """Load a specific round summary by round id."""
        path = self.history_dir / "round_summaries" / f"round_{round_id:04d}.json"
        return _load_json(path, {})

    def _load_frontier_round_summary(self, frontier_key: str) -> dict[str, Any]:
        round_id = _round_id_from_frontier_key(frontier_key)
        if round_id is None:
            return {}
        return self.load_round_summary(round_id)

    def get_operator_hints(self) -> list[dict[str, Any]]:
        return _load_json(self.history_dir / "operator_hints.json", [])

    def get_pending_hints(self) -> list[dict[str, Any]]:
        return [h for h in self.get_operator_hints() if h.get("status") == "pending"]

    def load_operator_guide_excerpt(self, max_chars: int | None = None) -> str:
        if not OPERATOR_GUIDE_PATH.exists():
            return ""
        text = OPERATOR_GUIDE_PATH.read_text().strip()
        if max_chars is None or len(text) <= max_chars:
            return text
        return text[:max_chars - 64].rstrip() + "\n\n[truncated]"

    def load_prior_research(self) -> str:
        """Load the prior research history summary for prompt injection."""
        if not PRIOR_RESEARCH_PATH.exists():
            return ""
        return PRIOR_RESEARCH_PATH.read_text().strip()

    def load_compatibility_registry(self) -> dict[str, str]:
        """Parse compatibility labels from prior_research_history.md."""
        return parse_compatibility_registry(self.load_prior_research())

    def family_compatibility_label(self, family_name: str | None) -> str:
        if not family_name:
            return "unknown"
        return self.load_compatibility_registry().get(str(family_name).lower(), "unknown")

    # -----------------------------------------------------------------------
    # Directory initialization
    # -----------------------------------------------------------------------

    def ensure_directories(self) -> None:
        for d in [self.history_dir, LOG_DIR, ROUND_SUMMARIES_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        """Load state.json and memory.json from disk."""
        self._state = _load_json(self.history_dir / "state.json", {})
        frontier_before = json.dumps(self._state.get("frontier", {}), sort_keys=True)

        # Ensure agent sub-dict with defaults
        agent = self._state.setdefault("agent", {})
        for k, v in _DEFAULT_AGENT_STATE.items():
            agent.setdefault(k, v)

        # Migrate legacy frontier fields to unified frontier
        if "frontier" not in self._state:
            self._state["frontier"] = self._state.get(
                "full_frontier", self._state.get("proxy_frontier", {})
            )

        # Migrate legacy frontier keys to new format
        frontier = self._state.get("frontier", {})
        self._migrate_frontier_keys(frontier)
        self._backfill_frontier_target_profiles(frontier)
        self._refresh_frontier_objective_metrics(frontier)
        from .controller import compact_frontier
        compacted_frontier = compact_frontier(frontier)
        if compacted_frontier != frontier:
            self._state["frontier"] = compacted_frontier

        self._memory = _load_json(self.history_dir / "memory.json", dict(_DEFAULT_MEMORY))
        for k, v in _DEFAULT_MEMORY.items():
            self._memory.setdefault(k, v)
        frontier_after = json.dumps(self._state.get("frontier", {}), sort_keys=True)
        memory_changed = self._sanitize_memory_against_invalid_rounds()
        if self._persist_load_fixes and frontier_after != frontier_before:
            self._save_state()
        if self._persist_load_fixes and memory_changed:
            self._save_memory()

    def _sanitize_memory_against_invalid_rounds(self) -> bool:
        invalid_rounds: set[int] = set()
        latest_valid_summary: dict[str, Any] | None = None
        latest_summary_invalid = False
        latest_valid_by_family: dict[str, dict[str, Any]] = {}

        for path in sorted((self.history_dir / "round_summaries").glob("round_*.json")):
            summary = _load_json(path, {})
            if not isinstance(summary, dict):
                continue
            round_id = _safe_int(summary.get("round"))
            invalid = summary_invalid_reason(summary) is not None
            if round_id is not None and invalid:
                invalid_rounds.add(round_id)
            latest_summary_invalid = invalid
            if invalid:
                continue
            latest_valid_summary = summary
            family_name = str(
                summary.get("family_name")
                or summary.get("action", {}).get("family_name")
                or summary.get("result", {}).get("architecture")
                or ""
            ).lower()
            if family_name:
                latest_valid_by_family[family_name] = summary

        if not invalid_rounds:
            return False

        changed = False
        round_needles = tuple(f"r{rid}" for rid in sorted(invalid_rounds))

        def _mentions_invalid_round(text: Any) -> bool:
            s = str(text or "")
            return any(needle in s for needle in round_needles)

        recent_rounds = self._memory.get("recent_rounds", [])
        filtered_recent = [
            entry for entry in recent_rounds
            if _safe_int(entry.get("round") if isinstance(entry, dict) else None) not in invalid_rounds
        ]
        if filtered_recent != recent_rounds:
            self._memory["recent_rounds"] = filtered_recent
            changed = True

        for key in ("recent_training_failures", "recent_sanity_failures", "failure_patterns"):
            rows = self._memory.get(key, [])
            filtered_rows = [
                entry for entry in rows
                if _safe_int(entry.get("round") if isinstance(entry, dict) else entry.get("last_round") if isinstance(entry, dict) else None) not in invalid_rounds
            ]
            if filtered_rows != rows:
                self._memory[key] = filtered_rows
                changed = True

        insights = self._memory.get("recent_insights", [])
        filtered_insights = [entry for entry in insights if not _mentions_invalid_round(entry)]
        if filtered_insights != insights:
            self._memory["recent_insights"] = filtered_insights
            changed = True

        families = self._memory.get("architecture_families", {})
        for family_name, family in families.items():
            if not isinstance(family, dict):
                continue
            tested = family.get("tested_configs", [])
            filtered_tested = [
                entry for entry in tested
                if _safe_int(entry.get("round") if isinstance(entry, dict) else None) not in invalid_rounds
            ]
            if filtered_tested != tested:
                family["tested_configs"] = filtered_tested
                changed = True

            latest_valid = latest_valid_by_family.get(str(family_name).lower())
            if latest_valid is not None:
                valid_round = _safe_int(latest_valid.get("round"))
                if valid_round is not None and family.get("last_round") != valid_round:
                    family["last_round"] = valid_round
                    changed = True
                action = latest_valid.get("action", {}) if isinstance(latest_valid.get("action"), dict) else {}
                hypothesis = action.get("hypothesis")
                if hypothesis and family.get("design_hypothesis") != hypothesis:
                    family["design_hypothesis"] = hypothesis
                    changed = True
                next_steps = list(action.get("next_hypotheses", []))[:8]
                if next_steps and family.get("next_steps") != next_steps:
                    family["next_steps"] = next_steps
                    changed = True

        if latest_summary_invalid and latest_valid_summary is not None:
            valid_next = list((latest_valid_summary.get("action", {}) or {}).get("next_hypotheses", []))[:12]
            if valid_next and self._memory.get("next_hypotheses") != valid_next:
                self._memory["next_hypotheses"] = valid_next
                changed = True

        return changed

    def _migrate_frontier_keys(self, frontier: dict[str, Any]) -> None:
        """Migrate legacy frontier key formats to new round-based keys.

        Handles two legacy formats:
        1. K-only keys: "128" → "legacy_128"
        2. K_EF keys: "128_12" → "legacy_128_12"

        Also computes cost fields for entries that lack them.
        """
        legacy_keys = [
            k for k in frontier
            if not k.startswith("r") and not k.startswith("legacy_")
        ]
        for old_key in legacy_keys:
            entry = frontier.pop(old_key)
            if not isinstance(entry, dict):
                continue

            # Ensure k and ef are stored in entry
            if "k" not in entry:
                if old_key.isdigit():
                    entry["k"] = int(old_key)
                elif "_" in old_key:
                    parts = old_key.split("_")
                    entry["k"] = int(parts[0])
            if "ef" not in entry:
                if "_" in old_key:
                    parts = old_key.split("_")
                    if len(parts) >= 2:
                        entry["ef"] = int(parts[1])
                else:
                    cfg = entry.get("config", {})
                    entry["ef"] = int(cfg.get("expansion_factor", cfg.get("EXPANSION_FACTOR", 12)))

            # Compute cost breakdown if missing
            if "total_cost" not in entry:
                cfg = entry.get("config", {})
                target_profile = resolve_target_profile(cfg)
                metrics = entry_objective_metrics(entry)
                entry.setdefault("selection_cost", metrics["selection_cost"])
                entry.setdefault("selection_cost_ratio", metrics["selection_cost_ratio"])
                entry.setdefault("deployment_accesses", metrics["deployment_accesses"])
                entry.setdefault("deployment_ratio", metrics["deployment_ratio"])
                entry["total_cost"] = metrics["total_cost"]
                entry.setdefault("total_cost_ratio", metrics["total_cost_ratio"])
                entry.setdefault("target_profile", target_profile.to_dict())
                entry.setdefault("cost_model_label", target_profile.cost_model_label)
                entry["metric_version"] = COST_METRIC_VERSION

            new_key = f"legacy_{old_key}"
            frontier[new_key] = entry

    def _refresh_frontier_objective_metrics(self, frontier: dict[str, Any]) -> None:
        """Recompute stale or incomplete frontier objective fields using current logic."""
        cost_cache: dict[str, dict[str, Any]] = {}
        for key, entry in frontier.items():
            if not isinstance(entry, dict):
                continue

            cfg = dict(entry.get("config", {}) or {})
            if entry.get("target_profile") is not None and "target_profile" not in cfg:
                cfg["target_profile"] = entry["target_profile"]
            target_profile = resolve_target_profile(cfg)
            if (
                not cost_entry_is_current(entry)
                or entry.get("total_cost") is None
                or entry.get("total_cost_ratio") is None
                or entry.get("selection_cost_ratio") is None
            ):
                metrics = entry_objective_metrics(entry, cost_cache=cost_cache)
                entry["selection_cost"] = metrics["selection_cost"]
                entry["selection_cost_ratio"] = metrics["selection_cost_ratio"]
                entry["deployment_accesses"] = metrics["deployment_accesses"]
                entry["deployment_ratio"] = metrics["deployment_ratio"]
                entry["total_cost"] = metrics["total_cost"]
                entry["total_cost_ratio"] = metrics["total_cost_ratio"]
                entry["target_profile"] = target_profile.to_dict()
                entry["cost_model_label"] = target_profile.cost_model_label
                entry["metric_version"] = COST_METRIC_VERSION

            if entry.get(EXCEED_FIELD) is None:
                summary = self._load_frontier_round_summary(key)
                exceed = _extract_exceed_from_summary(summary, hookpoint=cfg.get("HOOKPOINTS") or cfg.get("hookpoints"))
                if exceed is not None:
                    entry[EXCEED_FIELD] = exceed

            if entry.get(OBJECTIVE_FIELD) is None:
                entry[OBJECTIVE_FIELD] = compute_objective_score(
                    entry.get("total_cost_ratio"),
                    entry.get(EXCEED_FIELD),
                )

    def _backfill_frontier_target_profiles(self, frontier: dict[str, Any]) -> None:
        for entry in frontier.values():
            if not isinstance(entry, dict):
                continue
            cfg = dict(entry.get("config", {}) or {})
            if entry.get("target_profile") is not None and "target_profile" not in cfg:
                cfg["target_profile"] = entry["target_profile"]
            target_profile = resolve_target_profile(cfg)
            entry.setdefault("target_profile", target_profile.to_dict())
            entry.setdefault("cost_model_label", target_profile.cost_model_label)

    def _save_state(self) -> None:
        _save_json(self.history_dir / "state.json", self._state)
        _save_json(self.history_dir / "frontier.json", self._state.get("frontier", {}))

    def _save_memory(self) -> None:
        _save_json(self.history_dir / "memory.json", self._memory)

    def _append_memory(
        self,
        action: Action,
        result: Result,
        round_id: int,
        touched_files: list[str],
        ctx: RoundContext | None = None,
    ) -> None:
        """Update memory with round outcome."""
        m = self._memory
        cfg = (
            ctx.resolved_candidate_env_config
            if ctx is not None and ctx.resolved_candidate_env_config is not None
            else resolve_action_configs(action, self).candidate_env_config
        )
        arch = cfg.get("ARCHITECTURE", BASE_ENV_DEFAULTS["ARCHITECTURE"]).lower()
        family_name = (action.family_name or arch).lower()
        family_stage = action.family_stage or "mainline"

        # Recent round entry
        entry = {
            "round": round_id,
            "hypothesis": action.hypothesis,
            "change_type": action.change_type,
            "expected_win": action.expected_win,
            "family_name": family_name,
            "family_stage": family_stage,
            "decision": result.decision,
            "val_fvu": result.val_fvu,
            EXCEED_FIELD: result.exceed_alpha_0_50,
            "objective_score": result.objective_score,
            "total_cost_ratio": result.total_cost_ratio,
            "observed_fvu": result.observed_fvu,
            "k": result.k,
            "architecture": arch,
            "touched_files": touched_files,
            "run_health": result.run_health,
            "termination_reason": result.termination_reason,
            "tokens_per_sec": result.tokens_per_sec,
            "baseline_ratio": result.baseline_ratio,
        }
        m.setdefault("recent_rounds", []).append(entry)
        m["recent_rounds"] = m["recent_rounds"][-12:]

        # Family tracking
        self._update_family(action, result, round_id, family_name, family_stage, arch)

        # Insights: agent notes preserved as-is, outcome compressed to one-liner
        for note in action.notes_to_memory:
            m.setdefault("recent_insights", []).append(f"NOTE: {note}")
        # Compact outcome line
        objective_part = f" objective={result.objective_score}" if result.objective_score else ""
        exceed_part = f" exceed={result.exceed_alpha_0_50}" if result.exceed_alpha_0_50 else ""
        fvu_part = f" fvu={result.val_fvu}" if result.val_fvu else ""
        health_part = f" [{result.run_health}]" if result.run_health != "normal" else ""
        term_part = f" ({result.termination_reason})" if result.termination_reason != "completed" else ""
        outcome_note = (
            f"r{round_id} {family_name} {action.change_type} k{result.k or '?'}"
            f" -> {result.decision}{objective_part}{exceed_part}{fvu_part}{health_part}{term_part}"
        )
        m.setdefault("recent_insights", []).append(outcome_note)
        m["recent_insights"] = m["recent_insights"][-40:]

        # Performance findings
        if result.run_health == "perf_regression":
            perf_note = (
                f"r{round_id} {arch} perf_regression tps={result.tokens_per_sec} ratio={result.baseline_ratio}"
            )
            m.setdefault("performance_findings", []).append(perf_note)
            m["performance_findings"] = m["performance_findings"][-20:]
        elif action.change_type == "edit_perf_code":
            m.setdefault("performance_findings", []).append(outcome_note)
            m["performance_findings"] = m["performance_findings"][-20:]
        elif action.change_type != "param_only":
            m.setdefault("architecture_findings", []).append(outcome_note)
            m["architecture_findings"] = m["architecture_findings"][-20:]

        # Failure patterns
        if result.decision == "crash" or result.run_health in {"perf_regression", "crash"}:
            m.setdefault("failure_patterns", []).append({
                "pattern": action.hypothesis,
                "count": 1,
                "last_round": round_id,
                "run_health": result.run_health,
                "termination_reason": result.termination_reason,
            })
            m["failure_patterns"] = m["failure_patterns"][-20:]

        if result.decision == "crash":
            training_failure = {
                "round": round_id,
                "family_name": family_name,
                "change_type": action.change_type,
                "hypothesis": action.hypothesis,
                "termination_reason": result.termination_reason,
                "error_type": result.error_type or "",
                "error_summary": result.error_summary or "",
                "traceback_excerpt": result.traceback_excerpt or "",
                "log_excerpt": result.log_excerpt or "",
                "log_path": result.log_path or "",
            }
            failures = m.setdefault("recent_training_failures", [])
            failures.append(training_failure)
            m["recent_training_failures"] = failures[-12:]

        if not _is_invalid_runtime_result(result):
            m["next_hypotheses"] = action.next_hypotheses[:12]

    def _update_family(
        self,
        action: Action,
        result: Result,
        round_id: int,
        family_name: str,
        family_stage: str,
        arch: str,
    ) -> None:
        families = self._memory.setdefault("architecture_families", {})
        family = families.get(family_name)

        # Policy rejection should not create a brand-new incubating family slot.
        if result.decision == "policy_reject":
            if family is not None:
                family["last_round"] = round_id
                family["design_hypothesis"] = action.hypothesis
                family["next_steps"] = action.next_hypotheses[:8]
                family.setdefault("known_issues", []).append(
                    f"round {round_id}: policy_reject | {result.error_summary or result.termination_reason}"
                )
                family["known_issues"] = family["known_issues"][-20:]
            return

        if family is None:
            family = families.setdefault(family_name, {
                "status": "incubating" if family_stage != "mainline" else "active",
                "design_hypothesis": action.hypothesis,
                "tested_configs": [],
                "known_issues": [],
                "next_steps": [],
                "best_objective_score": None,
                "best_fvu": None,
                "last_round": None,
            })

        if _is_invalid_runtime_result(result):
            family.setdefault("known_issues", []).append(
                f"round {round_id}: invalid result ignored | {result.error_summary or result.termination_reason}"
            )
            family["known_issues"] = family["known_issues"][-20:]
            return

        family["last_round"] = round_id
        family["design_hypothesis"] = action.hypothesis
        # Only record actual training runs, not policy rejections
        if result.decision != "policy_reject":
            family.setdefault("tested_configs", []).append({
                "round": round_id,
                "stage": family_stage,
                "k": result.k,
                "decision": result.decision,
                "objective_score": result.objective_score,
                EXCEED_FIELD: result.exceed_alpha_0_50,
                "total_cost_ratio": result.total_cost_ratio,
                "val_fvu": result.val_fvu,
                "run_health": result.run_health,
            })
        family["tested_configs"] = family["tested_configs"][-20:]

        # Update best objective / FVU and family status based on decision
        objective_score = _safe_float(result.objective_score)
        val_fvu = _safe_float(result.val_fvu)
        if result.decision in ("keep", "archive", "promote"):
            if objective_score is not None and (
                family.get("best_objective_score") is None
                or objective_score < float(family["best_objective_score"])
            ):
                family["best_objective_score"] = objective_score
            if val_fvu is not None and (
                family.get("best_fvu") is None or val_fvu < float(family["best_fvu"])
            ):
                family["best_fvu"] = val_fvu

        if result.decision in ("keep", "promote"):
            family["status"] = "active"
        elif result.decision == "archive":
            if family.get("status") != "active":
                family["status"] = "archived"
        elif result.decision == "discard":
            if family.get("status") not in ("active",):
                family["status"] = "discarded"
        elif result.decision == "crash":
            pass  # Don't change family status on crash

        if result.run_health in ("perf_regression", "crash") and result.error_summary:
            family.setdefault("known_issues", []).append(
                f"round {round_id}: {result.error_summary}"
            )
            family["known_issues"] = family["known_issues"][-20:]

        family["next_steps"] = action.next_hypotheses[:8]

    def _update_agent_counters(self, result: Result, is_new_family: bool) -> None:
        agent = self.agent
        agent["round_index"] = self.round_index + 1

        if result.decision == "crash":
            agent["consecutive_crashes"] = self.consecutive_crashes + 1
        else:
            agent["consecutive_crashes"] = 0
            agent["crash_resets"] = 0

        if result.decision == "keep":
            agent["consecutive_no_improve"] = 0
        elif result.decision == "policy_reject":
            pass  # Don't count policy blocks as no-improve
        else:
            agent["consecutive_no_improve"] = self.consecutive_no_improve + 1

        if is_new_family:
            agent["rounds_since_new_family"] = 0
        else:
            agent["rounds_since_new_family"] = self.rounds_since_new_family + 1

    def _write_round_summary(
        self,
        round_id: int,
        action_dict: dict[str, Any],
        result_dict: dict[str, str],
        touched_files: list[str],
        patch_path: Path | None,
        ctx: RoundContext,
    ) -> None:
        from .git_ops import current_git_branch

        summary = {
            "round": round_id,
            "timestamp": int(time.time()),
            "started_at": ctx.started_at,
            "ended_at": int(time.time()),
            "duration_sec": int(time.time()) - ctx.started_at,
            "family_name": ctx.family_name,
            "family_stage": ctx.family_stage,
            "session_id": ctx.session_id,
            "timeline_event_ids": ctx.timeline_event_ids,
            "repair_attempts": ctx.repair_attempts,
            "experiment_branch": current_git_branch(),
            "action": action_dict,
            "result": result_dict,
            "resolved_reference_env_config": ctx.resolved_reference_env_config,
            "resolved_candidate_env_config": ctx.resolved_candidate_env_config,
            "changed_keys": ctx.changed_keys,
            "reference_source": ctx.reference_source,
            "runtime_config_json": ctx.runtime_config_json,
            "runtime_env_config": ctx.runtime_env_config,
            "target_profile": (
                ctx.runtime_config_json.get("target_profile")
                if isinstance(ctx.runtime_config_json, dict)
                else None
            ),
            "cost_model_label": (
                ctx.runtime_config_json.get("cost_model_label")
                if isinstance(ctx.runtime_config_json, dict)
                else None
            ),
            "touched_files": touched_files,
            "patch_path": str(patch_path) if patch_path is not None else None,
        }
        summaries_dir = self.history_dir / "round_summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        _save_json(summaries_dir / f"round_{round_id:04d}.json", summary)

    def _append_results_tsv(self, result_dict: dict[str, str]) -> None:
        results_path = self.history_dir / "results.tsv"
        if results_path.exists():
            with open(results_path, newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                try:
                    header = next(reader)
                except StopIteration:
                    header = []
            if header != RESULTS_COLUMNS:
                existing_rows = self.load_results()
                with open(results_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=RESULTS_COLUMNS, delimiter="\t", extrasaction="ignore"
                    )
                    writer.writeheader()
                    for row in existing_rows:
                        writer.writerow(row)
        write_header = not results_path.exists()
        with open(results_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=RESULTS_COLUMNS, delimiter="\t", extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
            writer.writerow(result_dict)

    def _mark_hints_applied(self, round_id: int) -> None:
        hints_path = self.history_dir / "operator_hints.json"
        hints = _load_json(hints_path, [])
        changed = False
        for hint in hints:
            if hint.get("status") != "pending":
                continue
            if hint.get("scope") == "next_round":
                hint["status"] = "applied"
                hint["applied_at"] = int(time.time())
                hint["applied_in_round"] = round_id
                changed = True
        if changed:
            _save_json(hints_path, hints)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _compact_failure_line(entry: dict[str, Any]) -> str:
    """Compress a failure entry into a single descriptive line."""
    if not isinstance(entry, dict):
        return ""
    r = entry.get("round", "?")
    fam = entry.get("family_name", "?")
    ct = entry.get("change_type", "?")
    et = entry.get("error_type", "")
    tr = entry.get("termination_reason", "")
    summary = str(
        entry.get("error_summary")
        or entry.get("stderr_excerpt")
        or entry.get("traceback_excerpt")
        or ""
    ).strip()
    # Take just the last line / core message, truncated
    if "\n" in summary:
        summary = summary.strip().rsplit("\n", 1)[-1].strip()
    summary = _truncate(summary, 100)
    return f"r{r} {fam} {ct} {et} {tr}: {summary}"


def _truncate(s: str, limit: int) -> str:
    """Truncate a string to limit chars."""
    if not isinstance(s, str):
        return str(s)[:limit]
    if len(s) <= limit:
        return s
    return s[:limit - 3] + "..."


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _round_id_from_frontier_key(key: Any) -> int | None:
    s = str(key or "")
    if s.startswith("r") and s[1:].isdigit():
        return int(s[1:])
    return None


def _extract_exceed_from_summary(
    summary: dict[str, Any] | None,
    hookpoint: str | None = None,
) -> float | None:
    if not isinstance(summary, dict):
        return None
    result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
    direct = _safe_float(result.get(EXCEED_FIELD))
    if direct is not None:
        return direct
    metrics_path = result.get("metrics_path")
    if not metrics_path:
        return None
    latest_step = read_latest_step(metrics_path)
    return extract_step_exceed_alpha_0_50(latest_step, hookpoint=hookpoint)


def _is_invalid_runtime_result(result: Result) -> bool:
    return str(result.error_type or "") == "config_mismatch"
