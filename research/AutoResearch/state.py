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
        "Track a Pareto frontier across reconstruction quality and sparsity/cost. "
        "Treat K=128 as one anchor point, not the only success criterion. "
        "Prefer experiments that add non-dominated tradeoff points, including smaller-K runs "
        "that accept some FVU increase when they improve the overall frontier."
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
        "Maintain a Pareto frontier over FVU and K rather than optimizing only the single best FVU point.",
        "Probe smaller K values even when FVU rises, as long as the new point may improve the tradeoff frontier.",
        "Use K=128 quality anchors to calibrate tradeoffs, not to suppress lower-K exploration.",
    ],
}


class StateManager:
    """Single source of truth for all autoresearch persistent state."""

    def __init__(self, history_dir: Path = HISTORY_DIR) -> None:
        self.history_dir = history_dir
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

        # 1. Update memory (families, insights, failures)
        self._append_memory(action, result, round_id, touched_files)

        # 2. Update agent counters
        is_new_family = action.family_name not in self.families or (
            self.families[action.family_name].get("last_round") is None
        )
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

    # -----------------------------------------------------------------------
    # Read-only digests (for prompt building)
    # -----------------------------------------------------------------------

    def frontier_digest(self, limit: int = 8) -> list[str]:
        """Return frontier as compact one-liner strings, sorted by FVU."""
        points: list[tuple[float, str]] = []
        for key, entry in self.frontier.items():
            if not isinstance(entry, dict):
                continue
            fvu = entry.get("fvu", 999)
            k = entry.get("k", key.split("_")[0] if "_" in key else key)
            ef = entry.get("ef", key.split("_")[1] if "_" in key else "?")
            arch = entry.get("architecture", "?")
            cfg = entry.get("config", {})
            lr = cfg.get("lr", "?")
            opt = cfg.get("optimizer", "?")
            commit = str(entry.get("commit", ""))[:7]
            line = f"k={k} ef={ef} fvu={fvu} arch={arch} lr={lr} opt={opt} @{commit}"
            points.append((float(fvu), line))
        points.sort()
        return [line for _, line in points[:limit]]

    def memory_digest(self) -> dict[str, Any]:
        """Compact memory snapshot for prompts.

        Uses structured one-liners instead of nested dicts to maximize
        information density per token.
        """
        m = self._memory

        # Families: one-liner history + short issues/next
        families_digest: dict[str, dict[str, Any]] = {}
        for name, fam in m.get("architecture_families", {}).items():
            best_fvu = fam.get("best_fvu")
            # Backwards compat: check legacy field names
            if best_fvu is None:
                best_fvu = fam.get("best_full_fvu") or fam.get("best_proxy_fvu")
            best_str = f" best_fvu={best_fvu}" if best_fvu is not None else ""

            # Compress tested_configs into one-liners
            history: list[str] = []
            for tc in fam.get("tested_configs", [])[-5:]:
                r = tc.get("round", "?")
                stage = str(tc.get("stage", "?"))[:5]  # proto/mainl/stab/promo
                k = tc.get("k", "?")
                dec = tc.get("decision", "?")
                fvu_val = tc.get("val_fvu")
                health = tc.get("run_health", "normal")
                fvu_part = f" fvu={fvu_val}" if fvu_val else ""
                health_part = f" [{health}]" if health != "normal" else ""
                history.append(f"r{r} {stage} k{k} {dec}{fvu_part}{health_part}")

            # Compress known_issues: just the core message, truncated
            issues = [_truncate(s, 80) for s in fam.get("known_issues", [])[-3:]]
            next_steps = [_truncate(s, 60) for s in fam.get("next_steps", [])[-3:]]

            families_digest[name] = {
                "status": fam.get("status"),
                "last_round": fam.get("last_round"),
                "best": best_str.strip() or None,
                "history": history,
                "issues": issues if issues else None,
                "next": next_steps if next_steps else None,
            }
            # Drop None values to save space
            families_digest[name] = {
                k: v for k, v in families_digest[name].items() if v is not None
            }

        # Failures: compress to one-liners
        sanity_fails = [
            _compact_failure_line(e) for e in m.get("recent_sanity_failures", [])[-6:]
            if isinstance(e, dict)
        ]
        training_fails = [
            _compact_failure_line(e) for e in m.get("recent_training_failures", [])[-6:]
            if isinstance(e, dict)
        ]

        # Baseline runtime: compact
        baselines = [
            f"{v.get('architecture','?')} k={v.get('k','?')}: {v.get('tokens_per_sec',0):.0f} tok/s (r{v.get('round','?')})"
            for v in list(m.get("baseline_runtime", {}).values())[-4:]
            if isinstance(v, dict)
        ]

        return {
            "current_focus": m.get("current_focus", ""),
            "families": families_digest,
            "insights": m.get("recent_insights", [])[-20:],
            "perf_findings": m.get("performance_findings", [])[-8:],
            "sanity_failures": sanity_fails if sanity_fails else None,
            "training_failures": training_fails if training_fails else None,
            "baselines": baselines if baselines else None,
            "next_hypotheses": m.get("next_hypotheses", [])[:8],
        }

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

    def recent_round_summaries_trimmed(self, limit: int = 3) -> list[str]:
        """Compact round summaries as one-liners for prompts."""
        lines: list[str] = []
        for s in self.recent_round_summaries(limit=limit):
            action = s.get("action", {})
            result = s.get("result", {})
            r = s.get("round", "?")
            fam = s.get("family_name", "?")
            ct = action.get("change_type", "?")
            dec = result.get("decision", "?")
            fvu = result.get("val_fvu", "")
            health = result.get("run_health", "normal")
            dur = s.get("duration_sec", "?")
            hyp = _truncate(str(action.get("hypothesis", "")), 80)
            fvu_part = f" fvu={fvu}" if fvu else ""
            health_part = f" [{health}]" if health != "normal" else ""
            lines.append(f"r{r} {fam} {ct} -> {dec}{fvu_part}{health_part} {dur}s | {hyp}")
        return lines

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

        # Ensure agent sub-dict with defaults
        agent = self._state.setdefault("agent", {})
        for k, v in _DEFAULT_AGENT_STATE.items():
            agent.setdefault(k, v)

        # Migrate legacy frontier fields to unified frontier
        if "frontier" not in self._state:
            self._state["frontier"] = self._state.get(
                "full_frontier", self._state.get("proxy_frontier", {})
            )

        # Migrate old K-only keys ("128") to K_EF keys ("128_32")
        frontier = self._state.get("frontier", {})
        old_keys = [k for k in frontier if k.isdigit()]
        for old_key in old_keys:
            entry = frontier.pop(old_key)
            if not isinstance(entry, dict):
                continue
            k = int(entry.get("k", old_key))
            cfg = entry.get("config", {})
            ef = int(cfg.get("expansion_factor", cfg.get("EXPANSION_FACTOR", 8)))
            entry["k"] = k
            entry["ef"] = ef
            from .controller import frontier_key
            frontier[frontier_key(k, ef)] = entry

        self._memory = _load_json(self.history_dir / "memory.json", dict(_DEFAULT_MEMORY))
        for k, v in _DEFAULT_MEMORY.items():
            self._memory.setdefault(k, v)

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
    ) -> None:
        """Update memory with round outcome."""
        m = self._memory
        cfg = action.effective_config()
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
        fvu_part = f" fvu={result.val_fvu}" if result.val_fvu else ""
        health_part = f" [{result.run_health}]" if result.run_health != "normal" else ""
        term_part = f" ({result.termination_reason})" if result.termination_reason != "completed" else ""
        outcome_note = (
            f"r{round_id} {family_name} {action.change_type} k{result.k or '?'}"
            f" -> {result.decision}{fvu_part}{health_part}{term_part}"
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
                "primary_variable": action.primary_variable,
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
        family = families.setdefault(family_name, {
            "status": "incubating" if family_stage != "mainline" else "active",
            "design_hypothesis": action.hypothesis,
            "tested_configs": [],
            "known_issues": [],
            "next_steps": [],
            "best_fvu": None,
            "last_round": None,
        })

        family["last_round"] = round_id
        family["design_hypothesis"] = action.hypothesis
        # Only record actual training runs, not policy rejections
        if result.decision != "policy_reject":
            family.setdefault("tested_configs", []).append({
                "round": round_id,
                "stage": family_stage,
                "k": result.k,
                "decision": result.decision,
                "val_fvu": result.val_fvu,
                "run_health": result.run_health,
            })
        family["tested_configs"] = family["tested_configs"][-20:]

        # Update best FVU and family status based on decision
        val_fvu = _safe_float(result.val_fvu)
        if result.decision == "keep":
            family["status"] = "active"
            if val_fvu is not None and (
                family.get("best_fvu") is None or val_fvu < float(family["best_fvu"])
            ):
                family["best_fvu"] = val_fvu
        elif result.decision == "archive":
            pass  # Near-frontier, keep current status
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
            "touched_files": touched_files,
            "patch_path": str(patch_path) if patch_path is not None else None,
        }
        summaries_dir = self.history_dir / "round_summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        _save_json(summaries_dir / f"round_{round_id:04d}.json", summary)

    def _append_results_tsv(self, result_dict: dict[str, str]) -> None:
        results_path = self.history_dir / "results.tsv"
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
