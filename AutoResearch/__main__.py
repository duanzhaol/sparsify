"""CLI entry point for the generic AutoResearch Task Pack runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .paths import DEFAULT_TASKPACK_FILENAME, SCHEMA_PATH, TEMPLATE_PATH
from .runtime import describe_taskpack_file, initialize_runtime, load_and_validate_taskpack


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AutoResearch Task Pack runtime",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate a task pack.")
    validate.add_argument("--taskpack", default=DEFAULT_TASKPACK_FILENAME)
    validate.add_argument("--no-schema-check", action="store_true")
    validate.add_argument("--no-path-check", action="store_true")

    describe = subparsers.add_parser("describe", help="Describe a task pack.")
    describe.add_argument("--taskpack", default=DEFAULT_TASKPACK_FILENAME)
    describe.add_argument("--json", action="store_true")
    describe.add_argument("--no-schema-check", action="store_true")
    describe.add_argument("--no-path-check", action="store_true")

    init_runtime = subparsers.add_parser(
        "init-runtime",
        help="Create the state stores declared by a task pack.",
    )
    init_runtime.add_argument("--taskpack", default=DEFAULT_TASKPACK_FILENAME)
    init_runtime.add_argument("--runtime-root", default=".")
    init_runtime.add_argument("--no-schema-check", action="store_true")
    init_runtime.add_argument("--no-path-check", action="store_true")

    template = subparsers.add_parser("template", help="Show the bundled task pack template path.")
    template.add_argument("--print", action="store_true")

    schema = subparsers.add_parser("schema", help="Show the bundled task pack schema path.")
    schema.add_argument("--print", action="store_true")

    return parser


def _print_file(path: Path) -> None:
    print(path.read_text(), end="")


def _run_validate(args: argparse.Namespace) -> int:
    _, messages = load_and_validate_taskpack(
        Path(args.taskpack),
        check_schema=not args.no_schema_check,
        check_external_paths=not args.no_path_check,
    )
    for message in messages:
        print(message.render())
    return 1 if any(message.level == "error" for message in messages) else 0


def _run_describe(args: argparse.Namespace) -> int:
    summary, messages = describe_taskpack_file(
        Path(args.taskpack),
        check_schema=not args.no_schema_check,
        check_external_paths=not args.no_path_check,
    )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"id: {summary['id']}")
        print(f"name: {summary['name']}")
        print(f"task_type: {summary['task_type']}")
        print(f"optimization_mode: {summary['optimization_mode']}")
        print(f"roles: {', '.join(summary['roles']) or '(none)'}")
        print(f"workflow_nodes: {summary['workflow_node_count']}")
        print(f"workflow_edges: {summary['workflow_edge_count']}")
        print(f"stores: {', '.join(summary['stores']) or '(none)'}")
        print(f"prompts: {', '.join(summary['prompts']) or '(none)'}")
        print(f"skills: {', '.join(summary['skills']) or '(none)'}")
        if summary["warnings"]:
            print("warnings:")
            for warning in summary["warnings"]:
                print(f"  - {warning}")
    return 1 if any(message.level == "error" for message in messages) else 0


def _run_init_runtime(args: argparse.Namespace) -> int:
    taskpack, messages = load_and_validate_taskpack(
        Path(args.taskpack),
        check_schema=not args.no_schema_check,
        check_external_paths=not args.no_path_check,
    )
    for message in messages:
        print(message.render())
    if any(message.level == "error" for message in messages):
        return 1

    created = initialize_runtime(taskpack, Path(args.runtime_root))
    for path in created:
        print(f"created: {path}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        return _run_validate(args)
    if args.command == "describe":
        return _run_describe(args)
    if args.command == "init-runtime":
        return _run_init_runtime(args)
    if args.command == "template":
        if args.print:
            _print_file(TEMPLATE_PATH)
        else:
            print(TEMPLATE_PATH)
        return 0
    if args.command == "schema":
        if args.print:
            _print_file(SCHEMA_PATH)
        else:
            print(SCHEMA_PATH)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
