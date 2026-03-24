"""High-level helpers for the generic Task Pack runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import TaskPack, ValidationMessage
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
