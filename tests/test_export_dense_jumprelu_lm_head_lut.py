from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

EXPORTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "export"
    / "export_dense_jumprelu_lm_head_lut.py"
)


class FakeModel(torch.nn.Module):
    def __init__(self, lm_head: torch.nn.Module):
        super().__init__()
        self.lm_head = lm_head
        self.config = type(
            "Config",
            (),
            {
                "to_dict": lambda self: {
                    "model_type": "tiny",
                    "hidden_size": 3,
                    "vocab_size": 5,
                }
            },
        )()


def _load_exporter():
    assert EXPORTER_PATH.exists(), f"missing exporter script: {EXPORTER_PATH}"
    spec = importlib.util.spec_from_file_location(
        "export_dense_jumprelu_lm_head_lut",
        EXPORTER_PATH,
    )
    assert spec is not None and spec.loader is not None
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    return exporter


def _write_dense_jumprelu_checkpoint(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    d_in = 3
    num_latents = 4

    raw_threshold = torch.tensor([-1.0, 0.0, 0.5, 1.0], dtype=torch.float32)
    tensors = {
        "encoder.weight": torch.arange(
            num_latents * d_in,
            dtype=torch.float32,
        ).reshape(num_latents, d_in)
        / 10.0,
        "encoder.bias": torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float32),
        "log_threshold": raw_threshold,
        "W_dec": torch.arange(num_latents * d_in, dtype=torch.float32).reshape(
            num_latents,
            d_in,
        )
        / 20.0,
        "b_dec": torch.tensor([0.5, -0.25, 0.75], dtype=torch.float32),
    }
    save_file(tensors, str(checkpoint_dir / "sae.safetensors"))

    with open(checkpoint_dir / "cfg.json", "w") as f:
        json.dump(
            {
                "architecture": "jumprelu",
                "d_in": d_in,
                "k": 2,
                "expansion_factor": 2,
                "num_latents": 0,
                "jumprelu_init_threshold": 0.0,
                "jumprelu_bandwidth": 0.1,
            },
            f,
        )

    return tensors


def _make_lm_head() -> torch.nn.Linear:
    weight = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.0, -0.5],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5], dtype=torch.float32)
    layer = torch.nn.Linear(3, 5, bias=True)
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    return layer


def test_load_dense_jumprelu_checkpoint_maps_runtime_keys(tmp_path: Path):
    exporter = _load_exporter()
    checkpoint_dir = tmp_path / "lm_head"
    source_tensors = _write_dense_jumprelu_checkpoint(checkpoint_dir)

    loaded = exporter.load_dense_jumprelu_checkpoint(checkpoint_dir)

    assert loaded["encoder_weight"].shape == (4, 3)
    assert loaded["encoder_bias"].shape == (4,)
    assert loaded["threshold"].shape == (4,)
    assert loaded["decoder_weight"].shape == (4, 3)
    assert loaded["decoder_bias"].shape == (3,)
    assert torch.allclose(
        loaded["threshold"],
        F.softplus(source_tensors["log_threshold"]),
    )
    assert loaded["config"]["architecture"] == "jumprelu"


def test_export_lm_head_writes_dense_runtime_bundle(tmp_path: Path):
    exporter = _load_exporter()
    checkpoint_dir = tmp_path / "run" / "best" / "lm_head"
    source_tensors = _write_dense_jumprelu_checkpoint(checkpoint_dir)
    output_dir = tmp_path / "lut"
    model = FakeModel(_make_lm_head())

    layer_info = exporter.export_lm_head(
        model=model,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        target_dtype=torch.float16,
        device="cpu",
        batch_size=None,
    )

    output_path = output_dir / "lm_head.lut.safetensors"
    assert output_path.exists()
    assert layer_info["input_dim"] == 3
    assert layer_info["output_dim"] == 5
    assert layer_info["operator"] == "lm_head"
    assert layer_info["encoder_architecture"] == "jumprelu"

    lm_head = model.lm_head
    with safe_open(str(output_path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        assert keys == {
            "encoder_weight",
            "encoder_bias",
            "threshold",
            "decoder_weight",
            "decoder_bias",
            "precomputed_products",
            "bias_product",
            "compensation_weight_t",
        }
        assert f.get_tensor("encoder_weight").dtype == torch.float16
        assert torch.allclose(
            f.get_tensor("precomputed_products").float(),
            source_tensors["W_dec"] @ lm_head.weight.detach().T,
            atol=1e-3,
            rtol=1e-3,
        )
        assert torch.allclose(
            f.get_tensor("bias_product").float(),
            source_tensors["b_dec"] @ lm_head.weight.detach().T + lm_head.bias.detach(),
            atol=1e-3,
            rtol=1e-3,
        )
        assert torch.allclose(
            f.get_tensor("compensation_weight_t").float(),
            lm_head.weight.detach().T,
            atol=1e-3,
            rtol=1e-3,
        )


def test_resolve_lm_head_checkpoint_prefers_best_dir(tmp_path: Path):
    exporter = _load_exporter()
    run_dir = tmp_path / "run"
    direct = run_dir / "lm_head"
    best = run_dir / "best" / "lm_head"
    _write_dense_jumprelu_checkpoint(direct)
    _write_dense_jumprelu_checkpoint(best)

    assert exporter.resolve_lm_head_checkpoint(run_dir) == best
    assert exporter.resolve_lm_head_checkpoint(direct) == direct
