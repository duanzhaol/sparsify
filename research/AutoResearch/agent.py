"""Agent module: LLM invocation, prompt construction, and response parsing."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .git_ops import REPO_ROOT
from .types import (
    Action,
    BASE_ENV_DEFAULTS,
    LOG_DIR,
    SCHEMA_PATH,
    LoopConfig,
)

ARCHITECTURE_INTEGRATION_SKILL_PATH = Path(
    "/root/.codex/skills/sae-architecture-integration/SKILL.md"
)


class Agent:
    """Manages codex exec invocations and prompt construction."""

    def __init__(self, config: LoopConfig) -> None:
        self.config = config

    # -------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------

    def propose(
        self,
        state: Any,  # StateManager
        round_id: int,
        policy_guidance: str,
    ) -> tuple[Action, Path]:
        """Ask the agent to propose an experiment.

        Reuses the existing codex session when possible (resume mode sends
        a lightweight delta prompt).  Falls back to a fresh session when
        the session is stale, broken, or the user chose fresh-each-round.

        Returns (action, stdout_log_path).
        Raises RuntimeError if all retries exhausted.
        """
        current_session = state.agent.get("active_session_id")
        need_fresh = self._should_start_fresh_session(state)

        if need_fresh or current_session is None:
            prompt = self._build_proposal_prompt(state, policy_guidance)
            session_to_use = None
        else:
            prompt = self._build_resume_prompt(state, round_id, policy_guidance)
            session_to_use = current_session

        raw, stdout_path, returned_session = self._invoke_with_retry(
            prompt, round_id, session_to_use,
        )

        # Update session tracking
        if returned_session:
            if returned_session != current_session:
                # New session created
                state.update_session(
                    active_session_id=returned_session,
                    active_session_started_at=int(time.time()),
                    active_session_rounds=1,
                    active_session_status="active",
                    last_resume_ok_at=None,
                )
            else:
                # Resumed existing session
                state.update_session(
                    active_session_rounds=state.agent.get("active_session_rounds", 0) + 1,
                    active_session_status="active",
                    last_resume_ok_at=int(time.time()),
                )

        action = Action.from_dict(raw)
        return action, stdout_path

    def request_repair(
        self,
        round_id: int,
        base_action: Action,
        failure_kind: str,
        failure_payload: dict[str, Any],
        repair_attempt: int,
        session_id: str | None,
    ) -> tuple[Action, Path]:
        """Ask the agent to fix code after a failure."""
        prompt = self._build_repair_prompt(
            round_id, base_action, failure_kind, failure_payload, repair_attempt,
        )
        raw, stdout_path, _ = self._invoke_with_retry(
            prompt, round_id, session_id, file_tag=f"repair_{repair_attempt}",
        )
        coerced = self._coerce_repair_action(base_action, raw, repair_attempt)
        return coerced, stdout_path

    def check_backend_reachable(self) -> None:
        """Preflight check that codex CLI exists and proxy is reachable."""
        from .runner import check_agent_backend_reachable
        check_agent_backend_reachable(self.config.agent_proxy)

    # -------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------

    def _should_start_fresh_session(self, state: Any) -> bool:
        """Decide whether to create a new session or resume the existing one."""
        if self.config.session_mode == "fresh-each-round":
            return True

        agent = state.agent
        session_id = agent.get("active_session_id")
        if not session_id:
            return True
        if agent.get("active_session_status") in ("closed", "broken"):
            return True

        # Too many rounds on this session
        rounds = agent.get("active_session_rounds", 0)
        if rounds >= self.config.max_session_rounds:
            return True

        # Session too old
        started = agent.get("active_session_started_at")
        if started and (time.time() - started) / 3600 > self.config.max_session_hours:
            return True

        return False

    # -------------------------------------------------------------------
    # Prompt building
    # -------------------------------------------------------------------

    def _build_proposal_prompt(
        self,
        state: Any,  # StateManager
        policy_guidance: str,
    ) -> str:
        """Construct the full proposal prompt."""
        context = {
            "frontier": state.frontier_digest(limit=8),
            "agent_state": {
                "round_index": state.round_index,
                "consecutive_crashes": state.consecutive_crashes,
                "consecutive_no_improve": state.consecutive_no_improve,
                "rounds_since_new_family": state.rounds_since_new_family,
            },
            "memory_digest": state.memory_digest(),
            "operator_hints": state.get_pending_hints()[:8],
            "operator_guide_excerpt": state.load_operator_guide_excerpt(),
            "prior_research_history": state.load_prior_research(),
            "recent_results": _summarize_results(state.load_results(limit=8)),
            "recent_round_summaries": state.recent_round_summaries_trimmed(limit=3),
        }

        # Architecture integration checklist (conditional)
        checklist = _load_architecture_checklist()
        if checklist and _should_include_checklist(state.memory, context["recent_round_summaries"]):
            context["architecture_integration_checklist"] = checklist

        policy_section = ""
        if policy_guidance:
            policy_section = f"\n\nPolicy guidance:\n{policy_guidance}\n"

        return f"""\
You are the SAE research agent for this repository.

Primary objective:
- Maintain and improve the Pareto frontier across FVU and K
- K=128 is an anchor, not the only success criterion
- Smaller K is valuable when it creates a non-dominated tradeoff
- Reducing K is more important than reducing expansion_factor

Execution layer is fixed:
- Training: scripts/autoresearch_test.sh
- Results: research/controller.py
- Memory: research/history/

Rules:
- Edit ONLY files under sparsify/
- Use env_overrides for parameter-only experiments
- ONE hypothesis per round
- Set primary_variable to indicate which dimension changes
- Never return command="stop"
- Return a final JSON object matching the action schema
{policy_section}
Current context:
{json.dumps(context, indent=2)}"""

    def _build_resume_prompt(
        self,
        state: Any,  # StateManager
        round_id: int,
        policy_guidance: str,
    ) -> str:
        """Lightweight delta prompt for resuming an existing session.

        The agent already has the full system instructions and operator guide
        from the initial prompt.  We only send what changed since last round:
        latest results, updated frontier, and policy guidance.
        """
        update = {
            "round": round_id,
            "frontier": state.frontier_digest(limit=8),
            "latest_results": _summarize_results(state.load_results(limit=3)),
            "latest_round_summary": state.recent_round_summaries_trimmed(limit=1),
            "agent_state": {
                "consecutive_crashes": state.consecutive_crashes,
                "consecutive_no_improve": state.consecutive_no_improve,
                "rounds_since_new_family": state.rounds_since_new_family,
            },
            "pending_hints": state.get_pending_hints()[:4],
            "next_hypotheses": state.memory.get("next_hypotheses", [])[:5],
        }

        # Only include checklist if needed (same logic as full prompt)
        checklist = _load_architecture_checklist()
        summaries = state.recent_round_summaries_trimmed(limit=2)
        if checklist and _should_include_checklist(state.memory, summaries):
            update["architecture_integration_checklist"] = checklist

        policy_section = ""
        if policy_guidance:
            policy_section = f"\nPolicy guidance:\n{policy_guidance}\n"

        return f"""\
Continue the SAE research session. Round {round_id}.
Return one JSON object matching the action schema. No markdown fences.
{policy_section}
Structured update:
{json.dumps(update, indent=2)}"""

    def _build_repair_prompt(
        self,
        round_id: int,
        base_action: Action,
        failure_kind: str,
        failure_payload: dict[str, Any],
        repair_attempt: int,
    ) -> str:
        checklist = _load_architecture_checklist()
        payload = {
            "round": round_id,
            "repair_attempt": repair_attempt,
            "max_repair_attempts": self.config.max_repair_attempts,
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
        checklist_section = f"\nArchitecture integration checklist:\n{checklist}\n" if checklist else ""
        return f"""\
Continue round {round_id} in repair mode.

The previous code-edit failed with an engineering blocker.
Do NOT redesign the experiment. Do NOT change family_name or env_overrides.
Patch the implementation so the original experiment can run.
Stay within sparsify/ only.
Attempt {repair_attempt} of {self.config.max_repair_attempts}.
Return one final JSON object matching the action schema.
{checklist_section}
Repair context:
{json.dumps(payload, indent=2)}"""

    # -------------------------------------------------------------------
    # Codex invocation
    # -------------------------------------------------------------------

    def _invoke_with_retry(
        self,
        prompt: str,
        round_id: int,
        session_id: str | None,
        file_tag: str = "",
    ) -> tuple[dict[str, Any], Path, str | None]:
        """Invoke codex exec with exponential backoff retry.

        On session resume failure, marks the session as broken and
        retries with a fresh session (but keeps the same prompt —
        caller is responsible for choosing the right prompt type).
        """
        max_retries = self.config.agent_max_retries
        last_error: Exception | None = None
        current_session = session_id

        for attempt in range(max_retries + 1):
            try:
                raw, stdout_path, new_session = self._invoke_codex(
                    prompt, round_id, current_session, file_tag=file_tag,
                )
                return raw, stdout_path, new_session
            except Exception as exc:
                last_error = exc
                print(f"Round {round_id}: agent failed (attempt {attempt + 1}/{max_retries + 1}): {exc}")
                # If session resume failed, drop session and retry fresh
                if current_session is not None:
                    print(f"Round {round_id}: session {current_session[:8]}... broken, switching to fresh")
                    current_session = None
                if attempt < max_retries:
                    delay = self.config.agent_retry_base_sec * (2 ** attempt)
                    time.sleep(delay)

        raise RuntimeError(f"Agent invocation failed after {max_retries + 1} attempts: {last_error}")

    def _invoke_codex(
        self,
        prompt: str,
        round_id: int,
        session_id: str | None,
        file_tag: str = "",
    ) -> tuple[dict[str, Any], Path, str | None]:
        """Single codex exec call. Returns (raw_action_dict, stdout_path, session_id)."""
        suffix = f"_{file_tag}" if file_tag else ""
        action_path = LOG_DIR / f"agent_action_{round_id:04d}{suffix}.json"
        stdout_path = LOG_DIR / f"agent_round_{round_id:04d}{suffix}.stdout.log"

        if session_id:
            cmd = [
                "codex", "exec", "resume", session_id, "-",
                "--dangerously-bypass-approvals-and-sandbox",
                "-o", str(action_path),
            ]
        else:
            cmd = [
                "codex", "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--cd", str(REPO_ROOT),
                "--output-schema", str(SCHEMA_PATH),
                "-o", str(action_path),
            ]
        if self.config.model:
            cmd.extend(["--model", self.config.model])

        env = os.environ.copy()
        proxy_keys = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
        if self.config.agent_proxy is not None:
            for k in proxy_keys:
                env.pop(k, None)
        if self.config.agent_proxy:
            for k in proxy_keys:
                env[k] = self.config.agent_proxy

        try:
            result = subprocess.run(
                cmd, input=prompt, text=True, capture_output=True,
                env=env, timeout=self.config.agent_timeout_sec,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"codex exec timed out after {self.config.agent_timeout_sec}s")

        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(
            result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else "")
        )
        if result.returncode != 0:
            raise RuntimeError(f"codex exec failed (exit {result.returncode}): see {stdout_path}")

        # Parse action
        if session_id:
            raw = _extract_json(action_path.read_text() if action_path.exists() else result.stdout)
        else:
            raw = json.loads(action_path.read_text()) if action_path.exists() else _extract_json(result.stdout)

        new_session = _extract_session_id(result.stdout + "\n" + result.stderr) if not session_id else session_id
        return raw, stdout_path, new_session

    # -------------------------------------------------------------------
    # Action coercion
    # -------------------------------------------------------------------

    @staticmethod
    def _coerce_repair_action(
        base_action: Action,
        repair_dict: dict[str, Any],
        repair_attempt: int,
    ) -> Action:
        """Pin repair action to original experiment target."""
        coerced = dict(repair_dict)
        coerced["command"] = base_action.command
        coerced["expected_win"] = base_action.expected_win
        coerced["family_name"] = base_action.family_name
        coerced["family_stage"] = base_action.family_stage
        coerced["env_overrides"] = base_action.env_overrides
        coerced["change_type"] = (
            base_action.change_type
            if base_action.change_type in ("edit_sae_code", "edit_perf_code")
            else "edit_sae_code"
        )
        coerced["needs_sanity"] = True
        coerced["primary_variable"] = "code_fix"
        notes = list(repair_dict.get("notes_to_memory") or [])
        notes.append(f"repair attempt {repair_attempt}: constrained to original experiment target")
        coerced["notes_to_memory"] = notes[-12:]
        return Action.from_dict(coerced)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from text."""
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise RuntimeError("Could not extract JSON object from agent response")


def _extract_session_id(output: str) -> str | None:
    match = re.search(r"session id:\s*([0-9a-fA-F-]{8,})", output)
    return match.group(1) if match else None


def _summarize_results(rows: list[dict[str, str]]) -> list[str]:
    """Compress result rows into one-liner strings."""
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


def _load_architecture_checklist() -> str:
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
    for s in recent_summaries[-2:]:
        if not isinstance(s, dict):
            continue
        if str(s.get("change_type", "")) == "edit_sae_code":
            return True
        if s.get("family_stage") == "prototype" and s.get("decision") in ("crash", "policy_reject"):
            return True
    for entry in list(memory.get("recent_training_failures", []))[-4:] + list(memory.get("recent_sanity_failures", []))[-2:]:
        if isinstance(entry, dict):
            summary = str(entry.get("error_summary") or "").lower()
            if any(kw in summary for kw in ("unknown architecture", "dispatch", "registration")):
                return True
    return False


def coerce_stop_action(action: Action, state: Any, round_id: int) -> Action:
    """Convert a stop action into a runnable fallback."""
    d = action.to_dict()
    # Find best known family
    best_family = BASE_ENV_DEFAULTS["ARCHITECTURE"]
    best_fvu = float("inf")
    for entry in state.frontier.values():
        if isinstance(entry, dict):
            fvu = entry.get("fvu")
            if fvu is not None:
                try:
                    v = float(fvu)
                    if v < best_fvu:
                        best_fvu = v
                        cfg = entry.get("config", {})
                        best_family = str(cfg.get("family_name") or entry.get("architecture") or best_family)
                except (TypeError, ValueError):
                    pass

    d["command"] = "run"
    d["hypothesis"] = d.get("hypothesis") or f"Continue exploring {best_family}"
    d["summary"] = f"Runtime override: continue search. {d.get('summary', '')}"
    d.setdefault("change_type", "param_only")
    d.setdefault("expected_win", "explore_unknown")
    d.setdefault("family_name", best_family)
    d.setdefault("family_stage", "mainline")
    d.setdefault("self_review", "Continuing search (stop not allowed)")
    d.setdefault("primary_variable", "other_param")
    notes = list(d.get("notes_to_memory") or [])
    notes.append("Runtime converted stop to run")
    d["notes_to_memory"] = notes[-12:]
    return Action.from_dict(d)
