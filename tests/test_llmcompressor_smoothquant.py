import json
import subprocess
import sys

import pytest
from datasets import Dataset

from quantization.llmcompressor_smoothquant import (
    SmoothQuantRecipeConfig,
    build_smoothquant_w8a8_recipe,
    import_llmcompressor_symbols,
    prepare_tokenized_calibration_dataset,
    write_smoothquant_export_manifest,
)


class _FakeSmoothQuantModifier:
    def __init__(self, *, smoothing_strength):
        self.smoothing_strength = smoothing_strength


class _FakeGPTQModifier:
    def __init__(self, *, targets, scheme, ignore):
        self.targets = targets
        self.scheme = scheme
        self.ignore = ignore


def test_build_smoothquant_w8a8_recipe_uses_expected_defaults():
    recipe = build_smoothquant_w8a8_recipe(
        SmoothQuantRecipeConfig(),
        smoothquant_modifier_cls=_FakeSmoothQuantModifier,
        gptq_modifier_cls=_FakeGPTQModifier,
    )

    assert len(recipe) == 2
    assert recipe[0].smoothing_strength == 0.8
    assert recipe[1].targets == "Linear"
    assert recipe[1].scheme == "W8A8"
    assert recipe[1].ignore == ["lm_head"]


def test_prepare_tokenized_calibration_dataset_truncates_and_limits(tmp_path):
    dataset = Dataset.from_dict(
        {
            "input_ids": [
                [1, 2, 3, 4, 5],
                [6, 7],
                [8, 9, 10, 11],
            ],
            "meta": ["a", "b", "c"],
        }
    )
    dataset.save_to_disk(tmp_path / "calib")

    prepared = prepare_tokenized_calibration_dataset(
        str(tmp_path / "calib"),
        num_samples=2,
        max_seq_length=3,
        shuffle_seed=0,
    )

    assert len(prepared) == 2
    assert prepared.column_names == ["input_ids"]
    assert all(len(row["input_ids"]) <= 3 for row in prepared)


def test_prepare_tokenized_calibration_dataset_requires_input_ids(tmp_path):
    dataset = Dataset.from_dict({"text": ["a", "b"]})
    dataset.save_to_disk(tmp_path / "calib")

    with pytest.raises(ValueError, match="input_ids"):
        prepare_tokenized_calibration_dataset(
            str(tmp_path / "calib"),
            num_samples=2,
            max_seq_length=8,
        )


def test_prepare_tokenized_calibration_dataset_avoids_map_and_filter(tmp_path, monkeypatch):
    dataset = Dataset.from_dict(
        {
            "input_ids": [
                [1, 2, 3, 4],
                [5, 6, 7],
            ],
            "meta": ["a", "b"],
        }
    )
    dataset.save_to_disk(tmp_path / "calib")

    def fail_map(*args, **kwargs):
        raise AssertionError("map should not be called")

    def fail_filter(*args, **kwargs):
        raise AssertionError("filter should not be called")

    monkeypatch.setattr(Dataset, "map", fail_map)
    monkeypatch.setattr(Dataset, "filter", fail_filter)

    prepared = prepare_tokenized_calibration_dataset(
        str(tmp_path / "calib"),
        num_samples=2,
        max_seq_length=2,
    )

    assert len(prepared) == 2
    assert prepared.column_names == ["input_ids"]
    assert prepared[0]["input_ids"] in ([1, 2], [5, 6])


def test_write_smoothquant_export_manifest_records_recipe(tmp_path):
    manifest_path = write_smoothquant_export_manifest(
        tmp_path,
        model_path="/root/models/Qwen3-0.6B",
        dataset_path="/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048",
        calibration_size=64,
        max_seq_length=2048,
        recipe_cfg=SmoothQuantRecipeConfig(smoothing_strength=0.75),
    )

    data = json.loads(manifest_path.read_text())
    assert data["model_path"] == "/root/models/Qwen3-0.6B"
    assert data["calibration_size"] == 64
    assert data["recipe"]["smoothing_strength"] == 0.75


def test_import_llmcompressor_symbols_when_installed():
    oneshot, gptq_modifier_cls, smoothquant_modifier_cls = import_llmcompressor_symbols()

    assert callable(oneshot)
    assert gptq_modifier_cls.__name__ == "GPTQModifier"
    assert smoothquant_modifier_cls.__name__ == "SmoothQuantModifier"


def test_export_script_help_runs_as_top_level_script():
    result = subprocess.run(
        [
            sys.executable,
            "quantization/export_llmcompressor_smoothquant_w8a8_teacher.py",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Export a Qwen3 W8A8 teacher" in result.stdout
