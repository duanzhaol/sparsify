"""Task Pack loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import TaskPack, ValidationMessage
from .paths import SCHEMA_PATH
from .workflow import unreachable_nodes

_TOP_LEVEL_REQUIRED = [
    "version",
    "meta",
    "mission",
    "knowledge",
    "field_library",
    "schemas",
    "role_library",
    "state_model",
    "objective_model",
    "adapter_registry",
    "workflow",
]


def load_taskpack(path: Path) -> TaskPack:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Task pack must be a JSON object: {path}")
    return TaskPack(path=path.resolve(), raw=raw)


def validate_taskpack(
    taskpack: TaskPack,
    *,
    check_schema: bool = True,
    check_external_paths: bool = True,
) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []

    messages.extend(_validate_top_level_shape(taskpack.raw))
    if check_schema:
        messages.extend(_validate_json_schema(taskpack.raw))

    if any(message.level == "error" for message in messages):
        return messages

    messages.extend(_validate_knowledge(taskpack, check_external_paths=check_external_paths))
    messages.extend(_validate_field_library(taskpack))
    messages.extend(_validate_schema_refs(taskpack, check_external_paths=check_external_paths))
    messages.extend(_validate_roles(taskpack))
    messages.extend(_validate_workflow(taskpack))
    messages.extend(_validate_state(taskpack))
    return messages


def describe_taskpack(taskpack: TaskPack) -> dict[str, Any]:
    messages = validate_taskpack(taskpack, check_schema=False, check_external_paths=False)
    return {
        "id": str(taskpack.meta.get("id", "")),
        "name": str(taskpack.meta.get("name", "")),
        "task_type": str(taskpack.mission.get("task_type", "")),
        "optimization_mode": str(taskpack.objective_model.get("optimization_mode", "")),
        "roles": sorted(taskpack.role_library),
        "prompts": sorted(taskpack.knowledge.get("prompt_library", {})),
        "skills": sorted(taskpack.knowledge.get("skill_library", {})),
        "stores": sorted(taskpack.state_model.get("stores", {})),
        "reviewers": sorted(taskpack.adapter_registry.get("reviewers", {})),
        "workflow_node_count": len(taskpack.workflow.get("nodes", {})),
        "workflow_edge_count": len(taskpack.workflow.get("edges", [])),
        "warnings": [m.render() for m in messages if m.level == "warning"],
    }


def _validate_top_level_shape(raw: dict[str, Any]) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    for key in _TOP_LEVEL_REQUIRED:
        if key not in raw:
            messages.append(ValidationMessage("error", "missing required top-level section", key))
    return messages


def _validate_json_schema(raw: dict[str, Any]) -> list[ValidationMessage]:
    try:
        import jsonschema
    except ImportError:
        return [ValidationMessage("info", "jsonschema is not installed; skipped schema validation")]

    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    messages: list[ValidationMessage] = []
    for error in sorted(validator.iter_errors(raw), key=str):
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        messages.append(ValidationMessage("error", error.message, location))
    return messages


def _validate_knowledge(
    taskpack: TaskPack,
    *,
    check_external_paths: bool,
) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    knowledge = taskpack.knowledge

    for section in ("descriptions", "references", "checklists"):
        for index, entry in enumerate(knowledge.get(section, [])):
            kind = entry.get("kind")
            path = entry.get("path")
            if not check_external_paths or kind in {"inline", "url"} or not path:
                continue
            resolved = taskpack.resolve_asset_path(path)
            if not resolved.exists():
                messages.append(
                    ValidationMessage(
                        "warning",
                        f"referenced asset does not exist: {path}",
                        f"knowledge.{section}[{index}]",
                    )
                )

    for section_name in ("prompt_library", "skill_library"):
        library = knowledge.get(section_name, {})
        for key, entry in library.items():
            path = entry.get("path")
            if not check_external_paths or not path:
                continue
            resolved = taskpack.resolve_asset_path(path)
            if not resolved.exists():
                messages.append(
                    ValidationMessage(
                        "warning",
                        f"referenced asset does not exist: {path}",
                        f"knowledge.{section_name}.{key}",
                    )
                )

    return messages


def _validate_roles(taskpack: TaskPack) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    prompts = taskpack.knowledge.get("prompt_library", {})
    skills = taskpack.knowledge.get("skill_library", {})
    tools = taskpack.adapter_registry.get("tools", {})
    views = taskpack.state_model.get("views", {})
    schema_names = taskpack.schemas

    for role_name, role in taskpack.role_library.items():
        for prompt_name in role.get("prompt_chain", []):
            if prompt_name not in prompts:
                messages.append(
                    ValidationMessage(
                        "error",
                        f"unknown prompt reference: {prompt_name}",
                        f"role_library.{role_name}.prompt_chain",
                    )
                )
        for skill_name in role.get("skills", []):
            if skill_name not in skills:
                messages.append(
                    ValidationMessage(
                        "error",
                        f"unknown skill reference: {skill_name}",
                        f"role_library.{role_name}.skills",
                    )
                )
        for tool_name in role.get("allowed_tools", []):
            if tool_name not in tools:
                messages.append(
                    ValidationMessage(
                        "error",
                        f"unknown tool reference: {tool_name}",
                        f"role_library.{role_name}.allowed_tools",
                    )
                )
        context_view = role.get("context_view")
        if context_view and context_view not in views:
            messages.append(
                ValidationMessage(
                    "error",
                    f"unknown context view: {context_view}",
                    f"role_library.{role_name}.context_view",
                )
            )
        output_schema = role.get("output_schema")
        if output_schema and output_schema not in schema_names:
            messages.append(
                ValidationMessage(
                    "error",
                    f"unknown schema reference: {output_schema}",
                    f"role_library.{role_name}.output_schema",
                )
            )

    return messages


def _validate_field_library(taskpack: TaskPack) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    field_library = taskpack.raw.get("field_library", {})
    enums = field_library.get("enums", {})
    records = field_library.get("records", {})

    for record_name, record in records.items():
        fields = record.get("fields", {})
        for field_name, field in fields.items():
            enum_ref = field.get("enum_ref")
            if enum_ref and enum_ref not in enums:
                messages.append(
                    ValidationMessage(
                        "error",
                        f"unknown enum reference: {enum_ref}",
                        f"field_library.records.{record_name}.fields.{field_name}.enum_ref",
                    )
                )

    return messages


def _validate_schema_refs(
    taskpack: TaskPack,
    *,
    check_external_paths: bool,
) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    if not check_external_paths:
        return messages

    for schema_name, schema_path in taskpack.schemas.items():
        resolved = taskpack.resolve_asset_path(schema_path)
        if not resolved.exists():
            messages.append(
                ValidationMessage(
                    "warning",
                    f"referenced schema does not exist: {schema_path}",
                    f"schemas.{schema_name}",
                )
            )
    return messages


def _validate_workflow(taskpack: TaskPack) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    workflow = taskpack.workflow
    nodes = workflow.get("nodes", {})
    edges = workflow.get("edges", [])
    entry = workflow.get("entry_node")

    if entry not in nodes:
        messages.append(ValidationMessage("error", "entry_node is not defined in workflow.nodes", "workflow.entry_node"))
        return messages

    registries = taskpack.adapter_registry
    for node_name, node in nodes.items():
        kind = node.get("kind")
        if kind == "agent":
            role_name = node.get("role")
            uses = node.get("uses")
            if role_name not in taskpack.role_library:
                messages.append(ValidationMessage("error", f"unknown role: {role_name}", f"workflow.nodes.{node_name}.role"))
            if uses not in registries.get("agents", {}):
                messages.append(ValidationMessage("error", f"unknown agent adapter: {uses}", f"workflow.nodes.{node_name}.uses"))
        elif kind == "tool":
            uses = node.get("uses")
            if uses not in registries.get("tools", {}):
                messages.append(ValidationMessage("error", f"unknown tool adapter: {uses}", f"workflow.nodes.{node_name}.uses"))
        elif kind == "evaluator":
            uses = node.get("uses")
            if uses not in registries.get("evaluators", {}):
                messages.append(ValidationMessage("error", f"unknown evaluator adapter: {uses}", f"workflow.nodes.{node_name}.uses"))
        elif kind == "gate":
            uses = node.get("uses")
            if uses not in registries.get("gates", {}):
                messages.append(ValidationMessage("error", f"unknown gate adapter: {uses}", f"workflow.nodes.{node_name}.uses"))
        elif kind == "record":
            uses = node.get("uses")
            if uses not in registries.get("recorders", {}):
                messages.append(ValidationMessage("error", f"unknown recorder adapter: {uses}", f"workflow.nodes.{node_name}.uses"))
        elif kind == "mcp_review":
            uses = node.get("uses")
            if uses not in registries.get("reviewers", {}):
                messages.append(
                    ValidationMessage(
                        "error",
                        f"unknown reviewer adapter: {uses}",
                        f"workflow.nodes.{node_name}.uses",
                    )
                )
        elif kind not in {"router", "reducer"}:
            messages.append(
                ValidationMessage(
                    "error",
                    f"unsupported workflow node kind: {kind}",
                    f"workflow.nodes.{node_name}.kind",
                )
            )

    for index, edge in enumerate(edges):
        src = edge.get("from")
        dst = edge.get("to")
        if src not in nodes:
            messages.append(ValidationMessage("error", f"unknown source node: {src}", f"workflow.edges[{index}].from"))
        if dst not in nodes:
            messages.append(ValidationMessage("error", f"unknown destination node: {dst}", f"workflow.edges[{index}].to"))

    for node_name in unreachable_nodes(str(entry), nodes, edges):
        messages.append(ValidationMessage("warning", "node is unreachable from entry_node", f"workflow.nodes.{node_name}"))

    return messages


def _validate_state(taskpack: TaskPack) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    stores = taskpack.state_model.get("stores", {})
    if not stores:
        messages.append(ValidationMessage("warning", "no state stores declared", "state_model.stores"))

    for store_name, store in stores.items():
        if "path" not in store:
            messages.append(ValidationMessage("error", "store is missing path", f"state_model.stores.{store_name}"))
        if "format" not in store:
            messages.append(ValidationMessage("error", "store is missing format", f"state_model.stores.{store_name}"))

    return messages
