from __future__ import annotations

import json
from pathlib import Path

from AutoResearch import compile_task_request, load_and_validate_taskpack, run_taskpack_file, run_taskpack_summary


def _write_task_md(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "# CPU Operator Optimization",
                "",
                "## Objective",
                "Optimize a CPU operator implementation and keep the task auditable.",
                "",
                "## Success Criteria",
                "- Reduce latency with a concrete measurement plan.",
                "- Preserve correctness.",
                "",
                "## Constraints",
                "- No correctness regression.",
                "- Keep changes bounded.",
                "",
                "## Skills",
                "- benchmark harness",
                "- review checklist",
                "",
                "## Prompts",
                "- reviewer audit prompt",
                "",
                "## Budget",
                "max_rounds: 4",
                "budget_hours: 2.5",
                "",
            ]
        )
        + "\n"
    )
    return path


def test_compile_generates_taskpack_bundle(tmp_path: Path) -> None:
    task_md = _write_task_md(tmp_path / "task.md")

    result = compile_task_request(task_md, tmp_path / "taskpacks")

    assert result.task_id == "cpu-operator-optimization"
    assert result.taskpack_path.exists()
    assert (result.output_dir / "docs" / "task_brief.md").exists()
    assert (result.output_dir / "prompts" / "reviewer_audit_prompt.md").exists()
    assert (result.output_dir / "skills" / "benchmark_harness.md").exists()
    assert result.compiler_report_path.exists()

    taskpack, messages = load_and_validate_taskpack(result.taskpack_path, check_schema=False)
    errors = [message for message in messages if message.level == "error"]
    assert not errors
    assert taskpack.workflow["nodes"]["mcp_review"]["kind"] == "mcp_review"
    assert "external_mcp_review" in taskpack.adapter_registry["reviewers"]


def test_runtime_dry_run_and_stub_run(tmp_path: Path) -> None:
    task_md = _write_task_md(tmp_path / "task.md")
    result = compile_task_request(task_md, tmp_path / "taskpacks")

    dry_report, dry_messages = run_taskpack_file(
        result.taskpack_path,
        tmp_path / "runtime",
        dry_run=True,
        check_schema=False,
    )
    assert not [message for message in dry_messages if message.level == "error"]
    dry_summary = run_taskpack_summary(dry_report)
    assert dry_summary["dry_run"] is True
    assert any(node["kind"] == "mcp_review" for node in dry_summary["nodes"])

    stub_report, stub_messages = run_taskpack_file(
        result.taskpack_path,
        tmp_path / "runtime",
        dry_run=False,
        check_schema=False,
    )
    assert not [message for message in stub_messages if message.level == "error"]
    stub_summary = run_taskpack_summary(stub_report)
    assert stub_summary["dry_run"] is False
    assert stub_summary["report_path"] is not None

    report_path = Path(stub_summary["report_path"])
    timeline_path = tmp_path / "runtime" / "runtime" / "history" / "timeline.jsonl"
    assert report_path.exists()
    assert timeline_path.exists()

    report_data = json.loads(report_path.read_text())
    assert report_data["nodes"][1]["kind"] == "mcp_review"
