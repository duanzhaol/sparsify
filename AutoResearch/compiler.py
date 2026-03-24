"""Compile human-readable Markdown task requests into Task Pack bundles."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import CompileResult

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*\S)\s*$")
_TASK_TYPE_HINT_RE = re.compile(r"^\s*(?:task[_ -]?type|type)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_MAX_ROUNDS_RE = re.compile(r"max[_ -]?rounds\s*:\s*(\d+)", re.IGNORECASE)
_BUDGET_HOURS_RE = re.compile(r"budget[_ -]?hours\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

_DEFAULT_MAX_ROUNDS = 6
_DEFAULT_BUDGET_HOURS = 3.0


def compile_task_markdown(
    source_path: Path,
    output_root: Path,
    *,
    task_id: str | None = None,
    force: bool = False,
) -> CompileResult:
    """Compile a Markdown task request into a human-editable task pack bundle."""
    source_path = source_path.resolve()
    output_root = output_root.resolve()
    text = source_path.read_text()
    spec = _extract_task_spec(text, source_path)

    resolved_task_id = _slugify(task_id or spec["title"] or source_path.stem)
    bundle_dir = output_root / resolved_task_id
    if bundle_dir.exists() and not force:
        raise FileExistsError(f"Task pack directory already exists: {bundle_dir}")

    taskpack = _build_taskpack(spec, resolved_task_id)
    generated_files = _build_bundle_files(spec, taskpack)

    created_files: list[Path] = []
    created_files.append(_write_text(bundle_dir / "task.md", text))
    for relative_path, content in generated_files.items():
        path = bundle_dir / relative_path
        if relative_path.suffix == ".json":
            created_files.append(_write_json(path, content))
        else:
            created_files.append(_write_text(path, str(content)))

    report = _build_compiler_report(spec, resolved_task_id, taskpack, generated_files)
    compiler_report_path = _write_json(bundle_dir / "compiler_report.json", report)
    created_files.append(compiler_report_path)

    return CompileResult(
        task_id=resolved_task_id,
        output_dir=bundle_dir,
        taskpack_path=bundle_dir / "taskpack.json",
        compiler_report_path=compiler_report_path,
        created_files=tuple(created_files),
    )


def _extract_task_spec(text: str, source_path: Path) -> dict[str, Any]:
    title = _extract_title(text, source_path)
    sections = _parse_sections(text)
    task_type = _detect_task_type(text, sections)

    objective = _first_text(
        sections,
        ["objective", "goal", "goals", "summary", "problem", "task"],
        fallback=_first_paragraph(text) or f"Advance the task described in {source_path.name}.",
    )
    success_definition = _first_text(
        sections,
        ["success", "success criteria", "success definition", "done", "definition of done"],
        fallback=_default_success_definition(task_type),
    )
    non_goals = _bullet_list(sections, ["non goals", "non-goals", "out of scope"])
    hard_constraints = _bullet_list(sections, ["constraints", "hard constraints", "requirements"])
    allowed_edit_paths = _bullet_list(sections, ["allowed edit paths", "editable paths", "edit scope"])
    deliverables = _bullet_list(sections, ["deliverables", "outputs", "artifacts"])
    references = _bullet_list(sections, ["references", "materials", "inputs", "context", "resources"])
    requested_skills = _bullet_list(
        sections,
        ["skills", "runtime skills", "needed skills", "tooling skills"],
    )
    requested_prompts = _bullet_list(
        sections,
        ["prompts", "prompt ideas", "prompt requirements"],
    )
    max_rounds, budget_hours = _extract_budget(sections, text)

    assumptions: list[str] = []
    if not non_goals:
        assumptions.append("No non-goals were specified; generated a minimal placeholder non-goal.")
        non_goals = ["Do not optimize outside the stated objective."]
    if not hard_constraints:
        assumptions.append("No hard constraints were specified; generated a conservative placeholder.")
        hard_constraints = ["Preserve correctness and document any uncertainty before execution."]
    if not deliverables:
        assumptions.append("No deliverables were specified; generated a default deliverable list.")
        deliverables = _default_deliverables(task_type)
    if not allowed_edit_paths:
        assumptions.append("No edit boundary was provided; defaulted to a scoped placeholder path.")
        allowed_edit_paths = ["workspace/"]
    if not references:
        assumptions.append("No references section was provided; created an empty reference map for manual fill-in.")
    if not requested_skills:
        assumptions.append("No explicit runtime skills were listed; generated a default task-local skill set.")
    if not requested_prompts:
        assumptions.append("No explicit prompt list was provided; generated a default prompt set.")

    return {
        "title": title,
        "task_type": task_type,
        "objective": objective,
        "success_definition": success_definition,
        "non_goals": non_goals,
        "hard_constraints": hard_constraints,
        "allowed_edit_paths": allowed_edit_paths,
        "deliverables": deliverables,
        "references": references,
        "requested_skills": requested_skills,
        "requested_prompts": requested_prompts,
        "budget": {
            "max_rounds": max_rounds,
            "budget_hours": budget_hours,
        },
        "sections_detected": tuple(sections),
        "source_name": source_path.name,
        "assumptions": assumptions,
    }


def _extract_title(text: str, source_path: Path) -> str:
    for line in text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.lstrip("#").strip()
    return source_path.stem.replace("_", " ").replace("-", " ").strip().title() or "Untitled Task"


def _parse_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_key = "root"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            sections.setdefault(current_key, []).append(content)
        current_lines = []

    for raw_line in text.splitlines():
        match = _HEADING_RE.match(raw_line)
        if match:
            flush()
            current_key = _normalize_heading(match.group(2))
            continue
        current_lines.append(raw_line.rstrip())
    flush()

    return {key: "\n\n".join(parts).strip() for key, parts in sections.items()}


def _normalize_heading(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return normalized or "section"


def _first_text(sections: dict[str, str], aliases: list[str], *, fallback: str) -> str:
    content = _find_section(sections, aliases)
    if not content:
        return fallback

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    bullet_lines = [match.group(1).strip() for line in lines if (match := _BULLET_RE.match(line))]
    if bullet_lines:
        return " ".join(bullet_lines)
    return lines[0]


def _bullet_list(sections: dict[str, str], aliases: list[str]) -> list[str]:
    content = _find_section(sections, aliases)
    if not content:
        return []
    items = [match.group(1).strip() for line in content.splitlines() if (match := _BULLET_RE.match(line))]
    if items:
        return items
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return lines


def _find_section(sections: dict[str, str], aliases: list[str]) -> str:
    normalized_aliases = [_normalize_heading(alias) for alias in aliases]
    for alias in normalized_aliases:
        if alias in sections:
            return sections[alias]
    for key, value in sections.items():
        if any(alias in key for alias in normalized_aliases):
            return value
    return ""


def _first_paragraph(text: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return ""
    candidate = paragraphs[0]
    if candidate.startswith("#"):
        return paragraphs[1] if len(paragraphs) > 1 else ""
    return candidate


def _detect_task_type(text: str, sections: dict[str, str]) -> str:
    explicit = _find_section(sections, ["task type", "type"])
    if explicit:
        normalized = _normalize_heading(explicit.splitlines()[0])
        if "writing" in normalized or "paper" in normalized:
            return "writing_research"
        if "performance" in normalized or "cpu" in normalized or "optimization" in normalized:
            return "performance_optimization"
        if "experiment" in normalized or "training" in normalized:
            return "experiment_research"
        if "idea" in normalized:
            return "idea_research"

    match = _TASK_TYPE_HINT_RE.search(text)
    if match:
        lowered = match.group(1).lower()
        if "writing" in lowered or "paper" in lowered:
            return "writing_research"
        if "performance" in lowered or "cpu" in lowered or "optimization" in lowered:
            return "performance_optimization"
        if "experiment" in lowered or "training" in lowered:
            return "experiment_research"
        if "idea" in lowered:
            return "idea_research"

    lowered_text = text.lower()
    writing_score = sum(keyword in lowered_text for keyword in ("paper", "manuscript", "writing", "draft", "citation"))
    perf_score = sum(keyword in lowered_text for keyword in ("cpu", "latency", "throughput", "operator", "kernel", "optimize", "performance", "benchmark"))
    experiment_score = sum(keyword in lowered_text for keyword in ("experiment", "ablation", "train", "training", "eval", "metric"))
    idea_score = sum(keyword in lowered_text for keyword in ("idea", "hypothesis", "novelty", "research question"))

    scores = {
        "writing_research": writing_score,
        "performance_optimization": perf_score,
        "experiment_research": experiment_score,
        "idea_research": idea_score,
    }
    best_type = max(scores, key=scores.get)
    return best_type if scores[best_type] > 0 else "idea_research"


def _extract_budget(sections: dict[str, str], text: str) -> tuple[int, float]:
    budget_text = _find_section(sections, ["budget", "resources"])
    candidate_text = "\n".join(part for part in (budget_text, text) if part)

    max_rounds_match = _MAX_ROUNDS_RE.search(candidate_text)
    budget_hours_match = _BUDGET_HOURS_RE.search(candidate_text)

    max_rounds = int(max_rounds_match.group(1)) if max_rounds_match else _DEFAULT_MAX_ROUNDS
    budget_hours = float(budget_hours_match.group(1)) if budget_hours_match else _DEFAULT_BUDGET_HOURS
    return max_rounds, budget_hours


def _build_taskpack(spec: dict[str, Any], task_id: str) -> dict[str, Any]:
    worker_role = _worker_role_name(spec["task_type"])
    prompt_library, prompt_role_map, prompt_files = _build_prompt_library(spec, worker_role)
    skill_library, skill_role_map, skill_files = _build_skill_library(spec)

    role_library = {
        "planner": {
            "purpose": "Propose the next highest-value round of work for this task.",
            "prompt_chain": _role_prompt_chain("planner", prompt_role_map),
            "skills": _role_skills("planner", skill_role_map),
            "allowed_tools": ["tool_search"],
            "context_view": "planner_view",
            "input_contract": "task brief + prior rounds + objective digest",
            "output_schema": "proposal",
            "behavioral_rules": [
                "Keep each round focused on one core hypothesis or one bounded deliverable.",
                "Preserve task constraints and edit boundaries from the mission.",
            ],
        },
        "reviewer": {
            "purpose": "Review the proposed round using internal rules plus MCP feedback.",
            "prompt_chain": _role_prompt_chain("reviewer", prompt_role_map),
            "skills": _role_skills("reviewer", skill_role_map),
            "allowed_tools": [],
            "context_view": "reviewer_view",
            "input_contract": "proposal + external review + checklist + prior findings",
            "output_schema": "review",
            "behavioral_rules": [
                "Produce structured findings and a clear verdict.",
                "Escalate uncertainty instead of silently weakening constraints.",
            ],
        },
        worker_role: {
            "purpose": _worker_purpose(spec["task_type"]),
            "prompt_chain": _role_prompt_chain(worker_role, prompt_role_map),
            "skills": _role_skills(worker_role, skill_role_map),
            "allowed_tools": ["tool_task_executor", "tool_search"],
            "context_view": "worker_view",
            "input_contract": "approved proposal + working context + task-local skills",
            "output_schema": "execution",
            "behavioral_rules": [
                "Stay within the approved proposal scope.",
                "Record assumptions and open questions explicitly.",
            ],
        },
        "repairer": {
            "purpose": "Revise the proposal after review feedback or a rejected round.",
            "prompt_chain": _role_prompt_chain("repairer", prompt_role_map),
            "skills": _role_skills("repairer", skill_role_map),
            "allowed_tools": [],
            "context_view": "repair_view",
            "input_contract": "proposal + review findings + previous assumptions",
            "output_schema": "proposal",
            "behavioral_rules": [
                "Only change what the findings require.",
                "Keep the revised proposal human-auditable.",
            ],
        },
    }

    taskpack = {
        "version": "1",
        "meta": {
            "id": task_id,
            "name": spec["title"],
            "version": "0.1.0",
            "extends": "base.taskpack",
            "tags": [
                "research",
                spec["task_type"],
            ],
            "owner": "user",
            "generated_by": "AutoResearch compiler",
            "source_spec": "task.md",
        },
        "mission": {
            "task_type": spec["task_type"],
            "objective": spec["objective"],
            "success_definition": spec["success_definition"],
            "non_goals": spec["non_goals"],
            "hard_constraints": spec["hard_constraints"],
            "allowed_edit_paths": spec["allowed_edit_paths"],
            "deliverables": spec["deliverables"],
            "budget": spec["budget"],
        },
        "knowledge": {
            "descriptions": [
                {
                    "id": "task_source",
                    "kind": "file",
                    "path": "task.md",
                    "roles": sorted(role_library),
                },
                {
                    "id": "task_brief",
                    "kind": "file",
                    "path": "docs/task_brief.md",
                    "roles": sorted(role_library),
                },
            ],
            "references": [
                {
                    "id": "reference_map",
                    "kind": "file",
                    "path": "docs/reference_map.md",
                    "roles": sorted(role_library),
                }
            ],
            "checklists": [
                {
                    "id": "review_checklist",
                    "kind": "file",
                    "path": "docs/review_checklist.md",
                    "roles": ["reviewer", "repairer"],
                }
            ],
            "prompt_library": prompt_library,
            "skill_library": skill_library,
        },
        "field_library": _field_library(spec["task_type"]),
        "schemas": {
            "proposal": "schemas/proposal.schema.json",
            "review": "schemas/review.schema.json",
            "execution": "schemas/execution.schema.json",
            "evaluation": "schemas/evaluation.schema.json",
            "decision": "schemas/decision.schema.json",
            "report": "schemas/report.schema.json",
        },
        "role_library": role_library,
        "state_model": {
            "stores": {
                "timeline": {
                    "path": "runtime/history/timeline.jsonl",
                    "format": "jsonl",
                },
                "memory": {
                    "path": "runtime/history/memory.json",
                    "format": "json",
                },
                "artifacts": {
                    "path": "runtime/history/artifacts",
                    "format": "dir",
                },
                "reports": {
                    "path": "runtime/history/reports",
                    "format": "dir",
                },
            },
            "views": {
                "planner_view": {
                    "sections": [
                        "objective_digest",
                        "prior_rounds",
                        "open_questions",
                    ]
                },
                "reviewer_view": {
                    "sections": [
                        "proposal",
                        "external_review",
                        "checklists",
                        "prior_findings",
                    ]
                },
                "worker_view": {
                    "sections": [
                        "approved_proposal",
                        "allowed_edit_paths",
                        "task_skills",
                    ]
                },
                "repair_view": {
                    "sections": [
                        "proposal",
                        "review",
                        "external_review",
                        "open_risks",
                    ]
                },
            },
            "retention": {
                "recent_rounds_limit": 20,
            },
            "compression": {
                "enable_summaries": True,
            },
        },
        "objective_model": _objective_model(spec["task_type"]),
        "adapter_registry": {
            "agents": {
                "default_agent": {
                    "type": "llm_agent",
                    "config": {
                        "model": "replace-me",
                    },
                }
            },
            "tools": {
                "tool_search": {
                    "type": "search",
                },
                "tool_task_executor": {
                    "type": "task_executor",
                    "config": {
                        "mode": spec["task_type"],
                    },
                },
            },
            "evaluators": {
                "default_evaluator": {
                    "type": "rubric_evaluator",
                }
            },
            "summarizers": {
                "default_summarizer": {
                    "type": "llm_summary",
                }
            },
            "recorders": {
                "default_recorder": {
                    "type": "jsonl_recorder",
                }
            },
            "gates": {
                "review_gate": {
                    "type": "verdict_gate",
                    "config": {
                        "accept": ["approve"],
                        "revise": ["revise"],
                        "reject": ["reject"],
                    },
                }
            },
            "reviewers": {
                "external_mcp_review": {
                    "type": "mcp_reviewer",
                    "config": {
                        "server": "replace-me",
                        "tool": "replace-me",
                    },
                }
            },
        },
        "workflow": {
            "entry_node": "draft_proposal",
            "nodes": {
                "draft_proposal": {
                    "kind": "agent",
                    "role": "planner",
                    "uses": "default_agent",
                    "input": {
                        "state": "planner_view",
                    },
                    "output": "proposal",
                },
                "mcp_review": {
                    "kind": "mcp_review",
                    "uses": "external_mcp_review",
                    "input": {
                        "proposal": "proposal",
                    },
                    "output": "external_review",
                },
                "internal_review": {
                    "kind": "agent",
                    "role": "reviewer",
                    "uses": "default_agent",
                    "input": {
                        "proposal": "proposal",
                        "external_review": "external_review",
                        "state": "reviewer_view",
                    },
                    "output": "review",
                },
                "review_gate": {
                    "kind": "gate",
                    "uses": "review_gate",
                    "input": {
                        "review": "review",
                    },
                    "output": "gate_result",
                },
                "repair_plan": {
                    "kind": "agent",
                    "role": "repairer",
                    "uses": "default_agent",
                    "input": {
                        "proposal": "proposal",
                        "review": "review",
                        "external_review": "external_review",
                        "state": "repair_view",
                    },
                    "output": "proposal",
                },
                "execute_round": {
                    "kind": "tool",
                    "uses": "tool_task_executor",
                    "input": {
                        "proposal": "proposal",
                        "state": "worker_view",
                    },
                    "output": "execution_result",
                },
                "evaluate_result": {
                    "kind": "evaluator",
                    "uses": "default_evaluator",
                    "input": {
                        "execution_result": "execution_result",
                        "review": "review",
                    },
                    "output": "evaluation",
                },
                "record_round": {
                    "kind": "record",
                    "uses": "default_recorder",
                    "input": {
                        "proposal": "proposal",
                        "external_review": "external_review",
                        "review": "review",
                        "evaluation": "evaluation",
                    },
                    "output": "recorded_round",
                },
            },
            "edges": [
                {"from": "draft_proposal", "to": "mcp_review"},
                {"from": "mcp_review", "to": "internal_review"},
                {"from": "internal_review", "to": "review_gate"},
                {"from": "review_gate", "to": "execute_round", "when": "gate_result == 'accept'"},
                {"from": "review_gate", "to": "repair_plan", "when": "gate_result == 'revise'"},
                {"from": "review_gate", "to": "record_round", "when": "gate_result == 'reject'"},
                {"from": "repair_plan", "to": "mcp_review"},
                {"from": "execute_round", "to": "evaluate_result"},
                {"from": "evaluate_result", "to": "record_round"},
            ],
            "loop_policies": [
                {
                    "name": "review_repair_loop",
                    "max_retries": 3,
                }
            ],
            "failure_policies": [
                {
                    "on": "execution_error",
                    "action": "route_to_repair_plan",
                }
            ],
            "stop_conditions": [
                "budget exhausted",
                "objective reached",
                "no valuable next step remains",
            ],
        },
        "reporting": {
            "outputs": [
                "round_summary",
                "final_report",
            ],
            "final_report_schema": "report",
        },
    }

    taskpack["_generated_assets"] = {
        "prompt_files": sorted(str(path) for path in prompt_files),
        "skill_files": sorted(str(path) for path in skill_files),
    }
    return taskpack


def _build_bundle_files(spec: dict[str, Any], taskpack: dict[str, Any]) -> dict[Path, Any]:
    worker_role = _worker_role_name(spec["task_type"])
    files: dict[Path, Any] = {
        Path("taskpack.json"): taskpack,
        Path("README.md"): _bundle_readme(spec),
        Path("docs/task_brief.md"): _task_brief(spec),
        Path("docs/reference_map.md"): _reference_map(spec),
        Path("docs/review_checklist.md"): _review_checklist(spec),
    }

    for relative_path, content in _prompt_file_contents(
        spec,
        taskpack["knowledge"]["prompt_library"],
        worker_role,
    ).items():
        files[relative_path] = content
    for relative_path, content in _skill_file_contents(
        spec,
        taskpack["knowledge"]["skill_library"],
    ).items():
        files[relative_path] = content
    for relative_path, content in _schema_file_contents(spec["task_type"]).items():
        files[relative_path] = content

    return files


def _build_prompt_library(
    spec: dict[str, Any],
    worker_role: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], tuple[Path, ...]]:
    prompt_library: dict[str, dict[str, Any]] = {
        "global_system": {"kind": "file", "path": "prompts/global_system.md"},
        "planner_prompt": {"kind": "file", "path": "prompts/planner.md"},
        "reviewer_prompt": {"kind": "file", "path": "prompts/reviewer.md"},
        f"{worker_role}_prompt": {"kind": "file", "path": f"prompts/{worker_role}.md"},
        "repair_prompt": {"kind": "file", "path": "prompts/repair.md"},
    }
    role_map = {
        "planner": ["global_system", "planner_prompt"],
        "reviewer": ["global_system", "reviewer_prompt"],
        worker_role: ["global_system", f"{worker_role}_prompt"],
        "repairer": ["global_system", "repair_prompt"],
    }
    created_paths = {
        Path("prompts/global_system.md"),
        Path("prompts/planner.md"),
        Path("prompts/reviewer.md"),
        Path(f"prompts/{worker_role}.md"),
        Path("prompts/repair.md"),
    }

    for raw_name in spec["requested_prompts"]:
        prompt_id = _make_unique_id(_slugify(raw_name).replace("-", "_"), prompt_library)
        prompt_path = Path("prompts") / f"{prompt_id}.md"
        prompt_library[prompt_id] = {"kind": "file", "path": str(prompt_path)}
        created_paths.add(prompt_path)
        for role_name in _prompt_targets(raw_name, worker_role):
            role_map.setdefault(role_name, []).append(prompt_id)

    return prompt_library, role_map, tuple(sorted(created_paths))


def _build_skill_library(
    spec: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], tuple[Path, ...]]:
    skill_library: dict[str, dict[str, Any]] = {
        "domain_context": {
            "kind": "domain_skill",
            "path": "skills/domain_context.md",
            "description": "Task-specific domain context extracted from the request.",
        },
        "process_playbook": {
            "kind": "process_skill",
            "path": "skills/process_playbook.md",
            "description": "Default round structure, revision loop, and record-keeping rules.",
        },
        "tooling_guide": {
            "kind": "tool_skill",
            "path": "skills/tooling_guide.md",
            "description": "How this task should use runtime tools and execution adapters.",
        },
        "review_guardrails": {
            "kind": "safety_skill",
            "path": "skills/review_guardrails.md",
            "description": "Review, safety, and escalation rules for the task.",
        },
    }
    role_map = {
        "planner": ["domain_context", "process_playbook"],
        "reviewer": ["review_guardrails", "process_playbook"],
        _worker_role_name(spec["task_type"]): ["domain_context", "tooling_guide"],
        "repairer": ["process_playbook", "review_guardrails"],
    }
    created_paths = {
        Path("skills/domain_context.md"),
        Path("skills/process_playbook.md"),
        Path("skills/tooling_guide.md"),
        Path("skills/review_guardrails.md"),
    }

    for raw_name in spec["requested_skills"]:
        skill_id = _make_unique_id(_slugify(raw_name).replace("-", "_"), skill_library)
        skill_kind = _skill_kind(raw_name)
        skill_path = Path("skills") / f"{skill_id}.md"
        skill_library[skill_id] = {
            "kind": skill_kind,
            "path": str(skill_path),
            "description": f"Task-local skill generated from request: {raw_name}",
        }
        created_paths.add(skill_path)
        for role_name in _skill_targets(raw_name, spec["task_type"]):
            role_map.setdefault(role_name, []).append(skill_id)

    return skill_library, role_map, tuple(sorted(created_paths))


def _role_prompt_chain(role_name: str, prompt_role_map: dict[str, list[str]]) -> list[str]:
    return list(dict.fromkeys(prompt_role_map.get(role_name, [])))


def _role_skills(role_name: str, skill_role_map: dict[str, list[str]]) -> list[str]:
    return list(dict.fromkeys(skill_role_map.get(role_name, [])))


def _prompt_targets(prompt_name: str, worker_role: str) -> tuple[str, ...]:
    lowered = prompt_name.lower()
    if any(token in lowered for token in ("review", "critic", "audit")):
        return ("reviewer", "repairer")
    if any(token in lowered for token in ("repair", "revise")):
        return ("repairer",)
    if any(token in lowered for token in ("write", "draft", "execute", "worker", worker_role)):
        return (worker_role,)
    if "system" in lowered or "global" in lowered:
        return ("planner", "reviewer", worker_role, "repairer")
    return ("planner", worker_role)


def _skill_targets(skill_name: str, task_type: str) -> tuple[str, ...]:
    worker_role = _worker_role_name(task_type)
    lowered = skill_name.lower()
    if any(token in lowered for token in ("review", "safety", "guardrail", "critic")):
        return ("reviewer", "repairer")
    if any(token in lowered for token in ("benchmark", "tool", "execute", "citation", "profile", "kernel", "experiment")):
        return (worker_role,)
    return ("planner", worker_role)


def _make_unique_id(base: str, registry: dict[str, Any]) -> str:
    candidate = base or "generated_item"
    index = 2
    while candidate in registry:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def _skill_kind(raw_name: str) -> str:
    lowered = raw_name.lower()
    if any(token in lowered for token in ("review", "safety", "guardrail", "critic")):
        return "safety_skill"
    if any(token in lowered for token in ("tool", "benchmark", "profile", "citation", "executor")):
        return "tool_skill"
    if any(token in lowered for token in ("process", "workflow", "loop", "playbook")):
        return "process_skill"
    return "domain_skill"


def _worker_role_name(task_type: str) -> str:
    if task_type == "writing_research":
        return "writer"
    if task_type == "performance_optimization":
        return "implementer"
    return "researcher"


def _worker_purpose(task_type: str) -> str:
    if task_type == "writing_research":
        return "Draft, revise, and structure human-readable research artifacts."
    if task_type == "performance_optimization":
        return "Execute bounded optimization work and collect concrete performance evidence."
    if task_type == "experiment_research":
        return "Run bounded experiments and summarize the observed evidence."
    return "Execute the approved research action and capture the resulting evidence."


def _default_success_definition(task_type: str) -> str:
    if task_type == "writing_research":
        return "Produce a materially improved draft with clearer structure, stronger evidence, and explicit open questions."
    if task_type == "performance_optimization":
        return "Produce a measured improvement or a well-supported rejection of the optimization path."
    if task_type == "experiment_research":
        return "Produce an executed experiment with interpretable results and a clear next decision."
    return "Produce a concrete, reviewable advance on the research question with explicit rationale."


def _default_deliverables(task_type: str) -> list[str]:
    if task_type == "writing_research":
        return ["Revised outline or draft section", "Review notes", "Next-step recommendation"]
    if task_type == "performance_optimization":
        return ["Optimization proposal", "Measured comparison", "Decision on whether to keep the change"]
    if task_type == "experiment_research":
        return ["Experiment plan", "Result summary", "Decision on follow-up experiment"]
    return ["Proposal", "Evidence summary", "Decision and next step"]


def _objective_model(task_type: str) -> dict[str, Any]:
    if task_type == "performance_optimization":
        return {
            "optimization_mode": "pareto",
            "metrics": [
                {"name": "latency_ms", "direction": "min", "unit": "ms"},
                {"name": "throughput", "direction": "max", "unit": "units_per_sec"},
                {"name": "correctness_score", "direction": "max"},
            ],
            "constraints": [
                "Do not regress correctness.",
                "Record the benchmark setup alongside every claimed improvement.",
            ],
            "decision_policy": {
                "accept_if": "review.verdict == 'approve' and evaluation.primary_outcome == 'improved'",
            },
            "stop_conditions": [
                "No credible optimization path remains.",
                "Budget exhausted.",
            ],
        }
    if task_type == "writing_research":
        return {
            "optimization_mode": "rubric",
            "metrics": [
                {"name": "quality_score", "direction": "max"},
                {"name": "evidence_coverage", "direction": "max"},
            ],
            "rubric": [
                "clarity",
                "structure",
                "grounding",
                "citation discipline",
            ],
            "decision_policy": {
                "accept_if": "review.verdict == 'approve'",
            },
            "stop_conditions": [
                "Requested writing artifact is coherent and review-approved.",
                "Budget exhausted.",
            ],
        }
    return {
        "optimization_mode": "rubric",
        "metrics": [
            {"name": "quality_score", "direction": "max"},
        ],
        "rubric": [
            "clarity",
            "grounding",
            "feasibility",
            "novelty",
        ],
        "decision_policy": {
            "accept_if": "review.verdict == 'approve'",
        },
        "stop_conditions": [
            "Objective reached.",
            "Budget exhausted.",
        ],
    }


def _field_library(task_type: str) -> dict[str, Any]:
    records = {
        "proposal": {
            "description": "Plan for one research round.",
            "fields": {
                "hypothesis": {
                    "type": "string",
                    "description": "Single core hypothesis or bounded objective for this round.",
                    "required": True,
                },
                "summary": {
                    "type": "string",
                    "description": "Human-readable summary of the proposed work.",
                    "required": True,
                },
                "change_mode": {
                    "type": "string",
                    "description": "How the round intends to make progress.",
                    "enum_ref": "change_mode",
                },
            },
        },
        "review": {
            "description": "Structured proposal or result review.",
            "fields": {
                "verdict": {
                    "type": "string",
                    "description": "Review verdict.",
                    "enum_ref": "review_verdict",
                    "required": True,
                },
                "summary": {
                    "type": "string",
                    "description": "Concise review summary.",
                    "required": True,
                },
            },
        },
        "execution": {
            "description": "Structured execution summary.",
            "fields": {
                "status": {
                    "type": "string",
                    "enum_ref": "execution_status",
                    "required": True,
                },
                "summary": {
                    "type": "string",
                    "required": True,
                },
                "artifacts": {
                    "type": "array",
                    "description": "Artifacts or files produced by the round.",
                },
            },
        },
        "evaluation": {
            "description": "Evaluation outcome for one round.",
            "fields": {
                "primary_outcome": {
                    "type": "string",
                    "required": True,
                },
                "summary": {
                    "type": "string",
                    "required": True,
                },
            },
        },
        "decision": {
            "description": "Decision about whether the round result should be kept.",
            "fields": {
                "label": {
                    "type": "string",
                    "enum_ref": "decision_label",
                    "required": True,
                },
                "rationale": {
                    "type": "string",
                    "required": True,
                },
            },
        },
        "report": {
            "description": "Human-readable report artifact.",
            "fields": {
                "title": {
                    "type": "string",
                    "required": True,
                },
                "summary": {
                    "type": "string",
                    "required": True,
                },
                "next_steps": {
                    "type": "array",
                    "description": "Recommended next actions.",
                },
            },
        },
    }

    units = {
        "latency_ms": "ms",
        "throughput": "units_per_sec",
    }
    aliases = {
        "proposal.hypothesis": "The single highest-value thing this round tries to verify or produce.",
        "review.verdict": "The structured review verdict after internal or MCP review.",
    }

    if task_type == "performance_optimization":
        aliases["evaluation.primary_outcome"] = "Whether the optimization improved, regressed, or stayed neutral."
    elif task_type == "writing_research":
        aliases["report.summary"] = "Human-readable writing progress summary."

    return {
        "enums": {
            "change_mode": [
                "analysis_only",
                "writing_change",
                "experiment_change",
                "code_change",
            ],
            "review_verdict": [
                "approve",
                "revise",
                "reject",
            ],
            "decision_label": [
                "keep",
                "discard",
                "rerun",
                "reject",
            ],
            "execution_status": [
                "planned",
                "completed",
                "blocked",
                "failed",
            ],
        },
        "units": units,
        "aliases": aliases,
        "records": records,
    }


def _task_brief(spec: dict[str, Any]) -> str:
    lines = [
        f"# {spec['title']}",
        "",
        "## Objective",
        spec["objective"],
        "",
        "## Task Type",
        spec["task_type"],
        "",
        "## Success Definition",
        spec["success_definition"],
        "",
        "## Deliverables",
    ]
    lines.extend(f"- {item}" for item in spec["deliverables"])
    lines.extend(
        [
            "",
            "## Hard Constraints",
        ]
    )
    lines.extend(f"- {item}" for item in spec["hard_constraints"])
    lines.extend(
        [
            "",
            "## Non-Goals",
        ]
    )
    lines.extend(f"- {item}" for item in spec["non_goals"])
    lines.extend(
        [
            "",
            "## Allowed Edit Paths",
        ]
    )
    lines.extend(f"- {item}" for item in spec["allowed_edit_paths"])
    return "\n".join(lines) + "\n"


def _reference_map(spec: dict[str, Any]) -> str:
    lines = [
        "# Reference Map",
        "",
        "Use this file to list papers, docs, benchmarks, datasets, or repo paths that the runtime should treat as first-class references.",
        "",
    ]
    if spec["references"]:
        lines.append("## Requested References")
        lines.extend(f"- {item}" for item in spec["references"])
    else:
        lines.extend(
            [
                "## Requested References",
                "- No explicit references were found in the source task request.",
                "- Add repo paths, URLs, paper titles, or benchmark notes here before running.",
            ]
        )
    return "\n".join(lines) + "\n"


def _review_checklist(spec: dict[str, Any]) -> str:
    generic_checks = [
        "Does the proposal directly advance the stated objective?",
        "Does it stay within the allowed edit paths and hard constraints?",
        "Is the round small enough to review and attribute?",
        "Does the output schema match what the runtime expects?",
    ]
    type_specific = {
        "writing_research": [
            "Will the output improve clarity, structure, or evidence quality?",
            "Are citations or references handled explicitly?",
        ],
        "performance_optimization": [
            "Does the proposal include a concrete measurement plan?",
            "Does it preserve correctness and define a regression threshold?",
        ],
        "experiment_research": [
            "Is the experiment design falsifiable and bounded?",
            "Are success and failure outcomes both informative?",
        ],
        "idea_research": [
            "Does the proposal reduce uncertainty rather than merely restate the problem?",
            "Is the claimed novelty grounded in available evidence?",
        ],
    }
    lines = [
        "# Review Checklist",
        "",
        "Use this checklist before approving a round.",
        "",
    ]
    lines.extend(f"- {item}" for item in generic_checks)
    lines.extend(f"- {item}" for item in type_specific.get(spec["task_type"], []))
    return "\n".join(lines) + "\n"


def _bundle_readme(spec: dict[str, Any]) -> str:
    lines = [
        f"# {spec['title']}",
        "",
        "This directory is a human-editable Task Pack bundle generated from `task.md`.",
        "",
        "## Edit First",
        "- `taskpack.json`: high-level workflow, roles, runtime adapters, and store model",
        "- `docs/`: task brief, references, and review checklist",
        "- `prompts/`: runtime prompts for each role",
        "- `skills/`: task-local runtime skills",
        "",
        "## Typical Workflow",
        "1. Review `compiler_report.json` to see what the compiler inferred.",
        "2. Edit prompts, skills, docs, or `taskpack.json` as needed.",
        "3. Run `python -m AutoResearch validate --taskpack taskpacks/<task_id>/taskpack.json`.",
        "4. Run `python -m AutoResearch run --taskpack taskpacks/<task_id>/taskpack.json --dry-run` before live execution.",
        "",
        "## Compiler Defaults",
        f"- task_type: `{spec['task_type']}`",
        f"- max_rounds: `{spec['budget']['max_rounds']}`",
        f"- budget_hours: `{spec['budget']['budget_hours']}`",
        "",
        "## Manual Editing Notes",
        "- Long-form prompts and skills live in Markdown files on purpose.",
        "- Runtime path references are relative to this directory.",
        "- Generated task-local skills are private by default; promote them manually if they become reusable.",
    ]
    return "\n".join(lines) + "\n"


def _prompt_file_contents(
    spec: dict[str, Any],
    prompt_library: dict[str, dict[str, Any]],
    worker_role: str,
) -> dict[Path, str]:
    contents = {
        Path("prompts/global_system.md"): "\n".join(
            [
                "# Global System Prompt",
                "",
                "You are operating inside a Task Pack runtime.",
                "Read the task brief, keep outputs structured, and preserve human-auditable reasoning.",
                "Follow the role-specific prompt and the task-local skills.",
                "",
            ]
        ),
        Path("prompts/planner.md"): "\n".join(
            [
                "# Planner Prompt",
                "",
                f"Task type: `{spec['task_type']}`",
                "",
                "Produce one bounded next-round proposal.",
                "Prioritize high-information, low-ambiguity progress.",
                "Return only content that matches the `proposal` schema.",
                "",
            ]
        ),
        Path("prompts/reviewer.md"): "\n".join(
            [
                "# Reviewer Prompt",
                "",
                "Review the incoming proposal together with the MCP review result.",
                "Use the review checklist and return only content matching the `review` schema.",
                "",
            ]
        ),
        Path(f"prompts/{worker_role}.md"): "\n".join(
            [
                f"# {worker_role.title()} Prompt",
                "",
                f"Task type: `{spec['task_type']}`",
                "",
                _worker_execution_instructions(spec["task_type"]),
                "Return only content matching the `execution` schema.",
                "",
            ]
        ),
        Path("prompts/repair.md"): "\n".join(
            [
                "# Repair Prompt",
                "",
                "Revise the proposal in response to review findings.",
                "Keep accepted parts unchanged and only modify what the findings require.",
                "Return only content matching the `proposal` schema.",
                "",
            ]
        ),
    }

    default_paths = set(contents)
    default_ids = {
        "global_system",
        "planner_prompt",
        "reviewer_prompt",
        f"{worker_role}_prompt",
        "repair_prompt",
    }
    for prompt_id, entry in prompt_library.items():
        if prompt_id in default_ids:
            continue
        path = Path(entry["path"])
        if path in default_paths:
            continue
        contents[path] = "\n".join(
            [
                f"# {prompt_id.replace('_', ' ').title()}",
                "",
                "This prompt file was generated because the source task request explicitly asked for it.",
                "Refine it manually before relying on it in a live run.",
                "",
                "## Suggested Focus",
                f"- Task type: `{spec['task_type']}`",
                f"- Objective: {spec['objective']}",
                "",
            ]
        )
    return contents


def _worker_execution_instructions(task_type: str) -> str:
    if task_type == "writing_research":
        return "Draft or revise the requested writing artifact with explicit structure, evidence, and open issues."
    if task_type == "performance_optimization":
        return "Apply the smallest credible optimization step and preserve a measurable benchmark plan."
    if task_type == "experiment_research":
        return "Run the bounded experiment and summarize the evidence in a way that supports a next decision."
    return "Execute the approved research action and capture the evidence needed for the next decision."


def _skill_file_contents(
    spec: dict[str, Any],
    skill_library: dict[str, dict[str, Any]],
) -> dict[Path, str]:
    contents = {
        Path("skills/domain_context.md"): "\n".join(
            [
                "# Domain Context",
                "",
                f"Task title: {spec['title']}",
                f"Task type: `{spec['task_type']}`",
                "",
                "## Objective",
                spec["objective"],
                "",
                "## Requested Deliverables",
                *[f"- {item}" for item in spec["deliverables"]],
                "",
            ]
        ),
        Path("skills/process_playbook.md"): "\n".join(
            [
                "# Process Playbook",
                "",
                "Each round should follow this sequence:",
                "1. Propose one bounded round.",
                "2. Run MCP review.",
                "3. Run internal review and verdict gate.",
                "4. Execute only approved work.",
                "5. Evaluate and record the result.",
                "",
                "If the review verdict is `revise`, route to the repair prompt instead of starting from scratch.",
                "",
            ]
        ),
        Path("skills/tooling_guide.md"): "\n".join(
            [
                "# Tooling Guide",
                "",
                "Runtime tools are task-pack specific. Keep tool usage explicit and reviewable.",
                "",
                "## Required Rules",
                "- Use search for context gathering and external references.",
                "- Use the task executor only after the proposal is approved.",
                "- Record artifacts and benchmark setup alongside execution results.",
                "",
            ]
        ),
        Path("skills/review_guardrails.md"): "\n".join(
            [
                "# Review Guardrails",
                "",
                "Reviewers should reject or revise work that violates constraints, is too large, or hides uncertainty.",
                "",
                "## Hard Constraints",
                *[f"- {item}" for item in spec["hard_constraints"]],
                "",
            ]
        ),
    }

    default_paths = set(contents)
    default_ids = {
        "domain_context",
        "process_playbook",
        "tooling_guide",
        "review_guardrails",
    }
    for skill_id, entry in skill_library.items():
        if skill_id in default_ids:
            continue
        path = Path(entry["path"])
        if path in default_paths:
            continue
        contents[path] = "\n".join(
            [
                f"# {skill_id.replace('_', ' ').title()}",
                "",
                "This task-local skill was generated from the source task request.",
                "Keep it private to this task pack until it proves reusable.",
                "",
                "## Suggested Contents",
                f"- Why it matters for `{spec['title']}`",
                "- How the worker role should apply it",
                "- What evidence or artifacts it should leave behind",
                "",
            ]
        )
    return contents


def _schema_file_contents(task_type: str) -> dict[Path, dict[str, Any]]:
    proposal_schema = _schema_object(
        title="Proposal",
        required=["hypothesis", "summary"],
        properties={
            "hypothesis": {"type": "string"},
            "summary": {"type": "string"},
            "change_mode": {"type": "string"},
        },
    )
    review_schema = _schema_object(
        title="Review",
        required=["verdict", "summary"],
        properties={
            "verdict": {"type": "string"},
            "summary": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    )
    execution_schema = _schema_object(
        title="Execution",
        required=["status", "summary"],
        properties={
            "status": {"type": "string"},
            "summary": {"type": "string"},
            "artifacts": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    )
    evaluation_schema = _schema_object(
        title="Evaluation",
        required=["primary_outcome", "summary"],
        properties={
            "primary_outcome": {"type": "string"},
            "summary": {"type": "string"},
            "metrics": {
                "type": "object",
                "additionalProperties": {
                    "type": ["number", "string"],
                },
            },
        },
    )
    decision_schema = _schema_object(
        title="Decision",
        required=["label", "rationale"],
        properties={
            "label": {"type": "string"},
            "rationale": {"type": "string"},
        },
    )
    report_schema = _schema_object(
        title="Report",
        required=["title", "summary"],
        properties={
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "next_steps": {
                "type": "array",
                "items": {"type": "string"},
            },
            "task_type": {"type": "string", "const": task_type},
        },
    )
    return {
        Path("schemas/proposal.schema.json"): proposal_schema,
        Path("schemas/review.schema.json"): review_schema,
        Path("schemas/execution.schema.json"): execution_schema,
        Path("schemas/evaluation.schema.json"): evaluation_schema,
        Path("schemas/decision.schema.json"): decision_schema,
        Path("schemas/report.schema.json"): report_schema,
    }


def _schema_object(
    *,
    title: str,
    required: list[str],
    properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": True,
    }


def _build_compiler_report(
    spec: dict[str, Any],
    task_id: str,
    taskpack: dict[str, Any],
    generated_files: dict[Path, Any],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "title": spec["title"],
        "task_type": spec["task_type"],
        "generated_at": datetime.now(UTC).isoformat(),
        "source_spec": spec["source_name"],
        "sections_detected": list(spec["sections_detected"]),
        "requested_skills": list(spec["requested_skills"]),
        "requested_prompts": list(spec["requested_prompts"]),
        "assumptions": list(spec["assumptions"]),
        "bundle": {
            "entrypoint": "taskpack.json",
            "generated_files": sorted(str(path) for path in generated_files),
        },
        "runtime_summary": {
            "roles": sorted(taskpack["role_library"]),
            "workflow_nodes": sorted(taskpack["workflow"]["nodes"]),
            "workflow_entry_node": taskpack["workflow"]["entry_node"],
            "review_nodes": [
                name
                for name, node in taskpack["workflow"]["nodes"].items()
                if node.get("kind") in {"mcp_review", "gate"}
            ],
        },
    }


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _write_json(path: Path, content: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")
    return path


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "task"
