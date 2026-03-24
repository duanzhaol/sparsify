"""Workflow dry-run and stub-run helpers for Task Packs."""

from __future__ import annotations

import json
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import RunReport, TaskPack, WorkflowNodePreview
from .storage import initialize_runtime, resolve_store_paths


def build_workflow_preview(taskpack: TaskPack) -> tuple[WorkflowNodePreview, ...]:
    """Build a stable reachable-node preview from the workflow graph."""
    workflow = taskpack.workflow
    nodes = workflow.get("nodes", {})
    edges = workflow.get("edges", [])
    entry_node = str(workflow.get("entry_node", ""))
    edges_by_src = _edges_by_source(edges)

    if entry_node not in nodes:
        return ()

    ordered_nodes: list[WorkflowNodePreview] = []
    seen: set[str] = set()
    queue: deque[str] = deque([entry_node])

    while queue:
        name = queue.popleft()
        if name in seen or name not in nodes:
            continue
        seen.add(name)
        node = nodes[name]
        outgoing = edges_by_src.get(name, [])
        ordered_nodes.append(
            WorkflowNodePreview(
                name=name,
                kind=str(node.get("kind", "")),
                uses=node.get("uses"),
                role=node.get("role"),
                input_keys=tuple(sorted(node.get("input", {}).keys())),
                output=node.get("output"),
                branches=tuple(
                    {
                        "to": str(edge.get("to", "")),
                        "when": str(edge.get("when", "always")),
                    }
                    for edge in outgoing
                ),
            )
        )
        for edge in outgoing:
            destination = str(edge.get("to", ""))
            if destination and destination not in seen:
                queue.append(destination)

    return tuple(ordered_nodes)


def run_taskpack(taskpack: TaskPack, runtime_root: Path, *, dry_run: bool) -> RunReport:
    """Run a dry-run or stub-run over the workflow graph."""
    preview = build_workflow_preview(taskpack)
    runtime_root = runtime_root.resolve()
    created_paths: tuple[Path, ...] = ()
    report_path: Path | None = None

    if not dry_run:
        created_paths = tuple(initialize_runtime(taskpack, runtime_root))
        report_path = _persist_stub_run(taskpack, runtime_root, preview)

    return RunReport(
        dry_run=dry_run,
        taskpack_path=taskpack.path,
        runtime_root=runtime_root,
        entry_node=str(taskpack.workflow.get("entry_node", "")),
        nodes=preview,
        created_paths=created_paths,
        report_path=report_path,
    )


def run_report_to_dict(report: RunReport) -> dict[str, Any]:
    """Serialize a run report to a JSON-friendly dictionary."""
    return {
        "dry_run": report.dry_run,
        "taskpack_path": str(report.taskpack_path),
        "runtime_root": str(report.runtime_root),
        "entry_node": report.entry_node,
        "created_paths": [str(path) for path in report.created_paths],
        "report_path": str(report.report_path) if report.report_path else None,
        "nodes": [
            {
                "name": node.name,
                "kind": node.kind,
                "uses": node.uses,
                "role": node.role,
                "input_keys": list(node.input_keys),
                "output": node.output,
                "branches": list(node.branches),
            }
            for node in report.nodes
        ],
    }


def _persist_stub_run(
    taskpack: TaskPack,
    runtime_root: Path,
    preview: tuple[WorkflowNodePreview, ...],
) -> Path | None:
    stores = resolve_store_paths(taskpack, runtime_root)
    timestamp = datetime.now(UTC).isoformat()

    timeline_path = stores.get("timeline")
    if timeline_path:
        with timeline_path.open("a", encoding="utf-8") as handle:
            for index, node in enumerate(preview, start=1):
                event = {
                    "timestamp": timestamp,
                    "sequence": index,
                    "node": node.name,
                    "kind": node.kind,
                    "status": "stubbed",
                }
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    report_dir = stores.get("reports")
    if report_dir and report_dir.is_dir():
        path = report_dir / "stub_run_report.json"
        path.write_text(
            json.dumps(
                {
                    "generated_at": timestamp,
                    "mode": "stub_run",
                    "taskpack_path": str(taskpack.path),
                    "nodes": [
                        {
                            "name": node.name,
                            "kind": node.kind,
                            "uses": node.uses,
                            "role": node.role,
                            "branches": list(node.branches),
                        }
                        for node in preview
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        return path

    return None


def _edges_by_source(edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        grouped.setdefault(str(edge.get("from", "")), []).append(edge)
    return grouped
