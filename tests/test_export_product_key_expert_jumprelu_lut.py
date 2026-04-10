from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

import export_product_key_expert_jumprelu_lut as exporter


class FakeModel:
    def __init__(self, modules: dict[str, torch.nn.Module]):
        self._modules = modules

    def get_submodule(self, path: str) -> torch.nn.Module:
        return self._modules[path]


def _write_product_key_checkpoint(
    layer_dir: Path,
    *,
    d_in: int = 4,
    num_experts: int = 2,
    latents_per_expert: int = 3,
) -> dict[str, torch.Tensor]:
    layer_dir.mkdir(parents=True, exist_ok=True)

    num_latents = num_experts * latents_per_expert
    raw_threshold = torch.tensor(
        [[0.0, 0.5, -0.25], [0.25, -0.75, 1.0]],
        dtype=torch.float32,
    )
    tensors = {
        "left_router.weight": torch.arange(8, dtype=torch.float32).reshape(2, d_in),
        "left_router.bias": torch.tensor([0.5, -0.5], dtype=torch.float32),
        "right_router.weight": torch.arange(12, dtype=torch.float32).reshape(3, d_in),
        "right_router.bias": torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32),
        "pair_left_index": torch.tensor([0, 1], dtype=torch.int64),
        "pair_right_index": torch.tensor([1, 2], dtype=torch.int64),
        "expert_encoders": torch.arange(
            num_latents * d_in,
            dtype=torch.float32,
        ).reshape(num_experts, latents_per_expert, d_in),
        "expert_encoder_bias": torch.arange(
            num_latents,
            dtype=torch.float32,
        ).reshape(num_experts, latents_per_expert),
        "log_threshold": raw_threshold,
        "W_dec": torch.arange(
            num_latents * d_in,
            dtype=torch.float32,
        ).reshape(num_latents, d_in)
        / 10.0,
        "b_dec": torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float32),
    }
    save_file(tensors, str(layer_dir / "sae.safetensors"))

    cfg = {
        "architecture": "product_key_expert_jumprelu",
        "d_in": d_in,
        "k": 4,
        "num_experts": num_experts,
        "active_experts": 2,
        "latents_per_expert": latents_per_expert,
        "jumprelu_init_threshold": 0.0,
        "jumprelu_bandwidth": 0.1,
    }
    with open(layer_dir / "cfg.json", "w") as f:
        json.dump(cfg, f)

    return tensors


def _make_linear(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.nn.Linear:
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias is not None:
            layer.bias.copy_(bias)
    return layer


def test_load_product_key_checkpoint_maps_runtime_keys(tmp_path: Path):
    checkpoint_dir = tmp_path / "layers.14.self_attn.q_proj"
    source_tensors = _write_product_key_checkpoint(checkpoint_dir)

    loaded = exporter.load_product_key_checkpoint(checkpoint_dir)

    assert loaded["router_left_weight"].shape == (2, 4)
    assert loaded["router_right_weight"].shape == (3, 4)
    assert loaded["expert_encoder_weight"].shape == (2, 3, 4)
    assert loaded["decoder_weight"].shape == (6, 4)
    assert loaded["expert_threshold"].shape == (2, 3)
    assert torch.allclose(
        loaded["expert_threshold"],
        F.softplus(source_tensors["log_threshold"]),
    )
    assert loaded["config"]["architecture"] == "product_key_expert_jumprelu"


def test_export_projection_layer_writes_complete_runtime_bundle(tmp_path: Path):
    checkpoint_dir = tmp_path / "layers.14.self_attn.q_proj"
    source_tensors = _write_product_key_checkpoint(checkpoint_dir)
    output_dir = tmp_path / "lut"

    modules = {
        "model.layers.14.self_attn.q_proj": _make_linear(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]),
            torch.tensor([0.1, -0.2]),
        ),
        "model.layers.14.self_attn.k_proj": _make_linear(
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            None,
        ),
        "model.layers.14.self_attn.v_proj": _make_linear(
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            torch.tensor([0.3]),
        ),
    }
    model = FakeModel(modules)

    layer_info = exporter.export_projection_layer(
        model=model,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        layer_idx=14,
        operator_name="self_attn.qkv_proj",
        output_name="layers.14.self_attn.qkv_proj",
        target_modules=["q_proj", "k_proj", "v_proj"],
        target_dtype=torch.float16,
        device="cpu",
        batch_size=None,
    )

    output_path = output_dir / "layers.14.self_attn.qkv_proj.lut.safetensors"
    assert output_path.exists()
    assert layer_info["input_dim"] == 4
    assert layer_info["output_dim"] == 4
    assert layer_info["operator"] == "self_attn.qkv_proj"

    fused_weight = torch.cat(
        [
            modules["model.layers.14.self_attn.q_proj"].weight.detach(),
            modules["model.layers.14.self_attn.k_proj"].weight.detach(),
            modules["model.layers.14.self_attn.v_proj"].weight.detach(),
        ],
        dim=0,
    )
    fused_bias = torch.cat(
        [
            modules["model.layers.14.self_attn.q_proj"].bias.detach(),
            torch.zeros(1),
            modules["model.layers.14.self_attn.v_proj"].bias.detach(),
        ]
    )

    with safe_open(str(output_path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        assert keys == {
            "router_left_weight",
            "router_left_bias",
            "router_right_weight",
            "router_right_bias",
            "pair_left_index",
            "pair_right_index",
            "expert_encoder_weight",
            "expert_encoder_bias",
            "expert_threshold",
            "decoder_weight",
            "decoder_bias",
            "precomputed_products",
            "bias_product",
            "compensation_weight_t",
        }
        assert f.get_tensor("router_left_weight").dtype == torch.float16
        assert f.get_tensor("pair_left_index").dtype == torch.int32
        assert torch.allclose(
            f.get_tensor("precomputed_products").float(),
            source_tensors["W_dec"] @ fused_weight.T,
            atol=1e-3,
            rtol=1e-3,
        )
        assert torch.allclose(
            f.get_tensor("bias_product").float(),
            source_tensors["b_dec"] @ fused_weight.T + fused_bias,
            atol=1e-3,
            rtol=1e-3,
        )
        assert torch.allclose(
            f.get_tensor("compensation_weight_t").float(),
            fused_weight.T,
            atol=1e-3,
            rtol=1e-3,
        )


def test_build_metadata_uses_required_product_key_schema():
    layer_info = {
        "layers.14.self_attn.qkv_proj": {
            "file": "layers.14.self_attn.qkv_proj.lut.safetensors",
            "input_dim": 1024,
            "output_dim": 4096,
            "operator": "self_attn.qkv_proj",
            "encoder_architecture": "product_key_expert_jumprelu",
            "architecture_config": {
                "k": 32,
                "num_experts": 512,
                "active_experts": 2,
                "latents_per_expert": 56,
                "left_keys": 22,
                "right_keys": 24,
            },
        }
    }

    metadata = exporter.build_metadata(
        model_path="/root/models/Qwen3-0.6B",
        model_config={"model_type": "qwen3", "num_hidden_layers": 28, "hidden_size": 1024},
        layer_info=layer_info,
        compensation_ratio=0.25,
        dtype_name="bfloat16",
    )

    assert metadata["architecture"] == "product_key_expert_jumprelu"
    assert metadata["runtime_target"] == "gpu_decode_only"
    assert metadata["operators"] == ["self_attn.qkv_proj", "mlp.gate_up_proj"]
    assert metadata["compensation"] == {"mode": "ratio", "ratio": 0.25}
    assert metadata["dtype"] == "bfloat16"
    assert metadata["layers"]["layers.14.self_attn.qkv_proj"]["encoder_architecture"] == (
        "product_key_expert_jumprelu"
    )


def test_resolve_layer_indices_uses_shared_layers_by_default(tmp_path: Path):
    q_best_dir = tmp_path / "q" / "best"
    up_best_dir = tmp_path / "up" / "best"
    (q_best_dir / "layers.14.self_attn.q_proj").mkdir(parents=True)
    (q_best_dir / "layers.15.self_attn.q_proj").mkdir(parents=True)
    (up_best_dir / "layers.15.mlp.up_proj").mkdir(parents=True)
    (up_best_dir / "layers.16.mlp.up_proj").mkdir(parents=True)

    assert exporter.resolve_layer_indices(q_best_dir, up_best_dir, layers=None) == [15]
