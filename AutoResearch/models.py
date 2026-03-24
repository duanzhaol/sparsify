"""Core datatypes for the generic AutoResearch runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationMessage:
    """Single validation message."""

    level: str
    message: str
    location: str | None = None

    def render(self) -> str:
        prefix = self.level.upper()
        if self.location:
            return f"[{prefix}] {self.location}: {self.message}"
        return f"[{prefix}] {self.message}"


@dataclass(frozen=True)
class TaskPack:
    """Loaded Task Pack plus its source path."""

    path: Path
    raw: dict[str, Any]

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    @property
    def meta(self) -> dict[str, Any]:
        return self.raw.get("meta", {})

    @property
    def mission(self) -> dict[str, Any]:
        return self.raw.get("mission", {})

    @property
    def knowledge(self) -> dict[str, Any]:
        return self.raw.get("knowledge", {})

    @property
    def schemas(self) -> dict[str, str]:
        return self.raw.get("schemas", {})

    @property
    def role_library(self) -> dict[str, dict[str, Any]]:
        return self.raw.get("role_library", {})

    @property
    def state_model(self) -> dict[str, Any]:
        return self.raw.get("state_model", {})

    @property
    def objective_model(self) -> dict[str, Any]:
        return self.raw.get("objective_model", {})

    @property
    def adapter_registry(self) -> dict[str, dict[str, Any]]:
        return self.raw.get("adapter_registry", {})

    @property
    def workflow(self) -> dict[str, Any]:
        return self.raw.get("workflow", {})

    def resolve_asset_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()


@dataclass(frozen=True)
class CompileResult:
    """Result of compiling a Markdown task request into a task pack."""

    task_id: str
    output_dir: Path
    taskpack_path: Path
    compiler_report_path: Path
    created_files: tuple[Path, ...]


@dataclass(frozen=True)
class WorkflowNodePreview:
    """Single workflow node in a dry-run or stub-run report."""

    name: str
    kind: str
    uses: str | None
    role: str | None
    input_keys: tuple[str, ...]
    output: str | None
    branches: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class RunReport:
    """Summarized workflow run plan."""

    dry_run: bool
    taskpack_path: Path
    runtime_root: Path
    entry_node: str
    nodes: tuple[WorkflowNodePreview, ...]
    created_paths: tuple[Path, ...]
    report_path: Path | None = None
