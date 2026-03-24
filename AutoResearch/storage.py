"""Generic runtime storage initialization."""

from __future__ import annotations

from pathlib import Path

from .models import TaskPack

_EMPTY_BY_FORMAT = {
    "json": "{}\n",
    "jsonl": "",
    "tsv": "",
    "md": "",
}


def initialize_runtime(taskpack: TaskPack, runtime_root: Path) -> list[Path]:
    """Create declared stores under the given runtime root."""
    created: list[Path] = []
    runtime_root = runtime_root.resolve()
    stores = taskpack.state_model.get("stores", {})

    for store in stores.values():
        rel_path = store.get("path")
        fmt = store.get("format")
        if not rel_path or not fmt:
            continue

        path = (runtime_root / rel_path).resolve()
        if fmt == "dir":
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(_EMPTY_BY_FORMAT.get(fmt, ""))
        created.append(path)

    return created
