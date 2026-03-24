"""High-level helpers for the generic Task Pack runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .compiler import compile_task_markdown
from .execution import run_report_to_dict, run_taskpack
from .models import CompileResult, RunReport, TaskPack, ValidationMessage
from .storage import initialize_runtime as _initialize_runtime
from .taskpack import describe_taskpack, load_taskpack, validate_taskpack


def load_and_validate_taskpack(
    path: Path,
    *,
    check_schema: bool = True,
    check_external_paths: bool = True,
) -> tuple[TaskPack, list[ValidationMessage]]:
    taskpack = load_taskpack(path)
    messages = validate_taskpack(
        taskpack,
        check_schema=check_schema,
        check_external_paths=check_external_paths,
    )
    return taskpack, messages


def describe_taskpack_file(
    path: Path,
    *,
    check_schema: bool = True,
    check_external_paths: bool = True,
) -> tuple[dict[str, Any], list[ValidationMessage]]:
    taskpack, messages = load_and_validate_taskpack(
        path,
        check_schema=check_schema,
        check_external_paths=check_external_paths,
    )
    return describe_taskpack(taskpack), messages


def initialize_runtime(taskpack: TaskPack, runtime_root: Path) -> list[Path]:
    return _initialize_runtime(taskpack, runtime_root)


def compile_task_request(
    source_path: Path,
    output_root: Path,
    *,
    task_id: str | None = None,
    force: bool = False,
) -> CompileResult:
    return compile_task_markdown(source_path, output_root, task_id=task_id, force=force)


def run_taskpack_file(
    path: Path,
    runtime_root: Path,
    *,
    dry_run: bool,
    check_schema: bool = True,
    check_external_paths: bool = True,
) -> tuple[RunReport, list[ValidationMessage]]:
    taskpack, messages = load_and_validate_taskpack(
        path,
        check_schema=check_schema,
        check_external_paths=check_external_paths,
    )
    if any(message.level == "error" for message in messages):
        return (
            RunReport(
                dry_run=dry_run,
                taskpack_path=path.resolve(),
                runtime_root=runtime_root.resolve(),
                entry_node="",
                nodes=(),
                created_paths=(),
            ),
            messages,
        )
    return run_taskpack(taskpack, runtime_root, dry_run=dry_run), messages


def run_taskpack_summary(report: RunReport) -> dict[str, Any]:
    return run_report_to_dict(report)
