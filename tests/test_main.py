from __future__ import annotations

import sys

from sparsify.__main__ import (
    model_path_uses_qwen3_5_text_loader,
    suppress_stdout_for_nonzero_rank,
)


def test_suppress_stdout_for_nonzero_rank_keeps_stdout_file_like(capsys):
    with suppress_stdout_for_nonzero_rank(1):
        assert sys.stdout is not None
        assert hasattr(sys.stdout, "write")
        print("hidden-from-rank1")

    captured = capsys.readouterr()
    assert "hidden-from-rank1" not in captured.out


def test_suppress_stdout_for_nonzero_rank_keeps_rank0_output(capsys):
    with suppress_stdout_for_nonzero_rank(0):
        print("visible-from-rank0")

    captured = capsys.readouterr()
    assert "visible-from-rank0" in captured.out


def test_model_path_uses_qwen3_5_text_loader_for_local_config(tmp_path):
    model_dir = tmp_path / "qwen3_5"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "qwen3_5"}')

    assert model_path_uses_qwen3_5_text_loader(str(model_dir))
