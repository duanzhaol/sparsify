import pytest
import torch

from quantization.eval_utils import (
    build_summary,
    compute_reconstruction_metrics,
    load_elbow_thresholds_for_hookpoints,
    resolve_checkpoint_paths,
    resolve_matching_hookpoints,
)
from quantization.quant_utils import (
    quantize_activation_per_token_symmetric,
    quantize_weight_per_row_symmetric,
    simulate_w8a8_linear,
    simulate_w8a8_matmul,
)


class TestQuantUtils:
    def test_quantize_weight_per_row_symmetric_returns_int8_and_scales(self):
        weight = torch.tensor(
            [[[1.0, -2.0, 0.5], [0.0, 4.0, -4.0]]],
            dtype=torch.float32,
        )

        q_weight, scales = quantize_weight_per_row_symmetric(weight)

        assert q_weight.dtype == torch.int8
        assert q_weight.shape == weight.shape
        assert scales.shape == (1, 2, 1)
        assert torch.all(scales > 0)
        restored = q_weight.to(torch.float32) * scales
        assert torch.allclose(restored, weight, atol=0.05)

    def test_quantize_activation_per_token_symmetric_returns_int8_and_scales(self):
        acts = torch.tensor(
            [[1.0, -2.0, 0.5], [0.0, 4.0, -4.0]],
            dtype=torch.float32,
        )

        q_acts, scales = quantize_activation_per_token_symmetric(acts)

        assert q_acts.dtype == torch.int8
        assert q_acts.shape == acts.shape
        assert scales.shape == (2, 1)
        restored = q_acts.to(torch.float32) * scales
        assert torch.allclose(restored, acts, atol=0.05)

    def test_simulate_w8a8_matmul_matches_manual_reference(self):
        acts = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
        weight = torch.tensor(
            [[
                [[1.5, 0.0, -0.5], [-1.0, 2.0, 0.25]],
                [[0.25, -0.5, 1.0], [2.0, -1.0, 0.5]],
            ]],
            dtype=torch.float32,
        )

        simulated = simulate_w8a8_matmul(acts, weight)
        reference = torch.einsum("bd,bald->bal", acts, weight)

        assert simulated.shape == reference.shape
        assert torch.allclose(simulated, reference, atol=0.1)

    def test_simulate_w8a8_linear_matches_torch_linear(self):
        acts = torch.tensor(
            [[1.0, -2.0, 0.5], [-0.5, 1.5, 2.0]],
            dtype=torch.float32,
        )
        weight = torch.tensor(
            [[1.5, 0.0, -0.5], [-1.0, 2.0, 0.25]],
            dtype=torch.float32,
        )
        bias = torch.tensor([0.25, -0.75], dtype=torch.float32)

        simulated = simulate_w8a8_linear(acts, weight, bias)
        reference = torch.nn.functional.linear(acts, weight, bias)

        assert simulated.shape == reference.shape
        assert torch.allclose(simulated, reference, atol=0.1)


class TestEvalUtils:
    def test_resolve_matching_hookpoints_expands_range_patterns(self):
        available = [
            "layers.0.self_attn.q_proj",
            "layers.1.self_attn.q_proj",
            "layers.2.self_attn.q_proj",
            "layers.2.mlp.up_proj",
        ]

        matched = resolve_matching_hookpoints(
            ["layers.[0-2].self_attn.q_proj"],
            available,
        )

        assert matched == [
            "layers.0.self_attn.q_proj",
            "layers.1.self_attn.q_proj",
            "layers.2.self_attn.q_proj",
        ]

    def test_load_elbow_thresholds_for_hookpoints_matches_component_style_keys(self, tmp_path):
        path = tmp_path / "thresholds.json"
        path.write_text(
            '{"layer_0/self_attn_q_proj": {"elbow_value": 0.25}, '
            '"layer_1/mlp_up_proj": {"elbow_value": 0.5}}'
        )

        result = load_elbow_thresholds_for_hookpoints(
            path,
            ["layers.0.self_attn.q_proj", "layers.1.mlp.up_proj"],
        )

        assert result == {
            "layers.0.self_attn.q_proj": 0.25,
            "layers.1.mlp.up_proj": 0.5,
        }

    def test_resolve_checkpoint_paths_requires_per_hookpoint_directories(self, tmp_path):
        hook = "layers.0.self_attn.q_proj"
        expected = tmp_path / hook
        expected.mkdir(parents=True)
        (expected / "cfg.json").write_text("{}")

        resolved = resolve_checkpoint_paths(tmp_path, [hook])

        assert resolved == {hook: expected}

    def test_build_summary_computes_aggregate_deltas(self):
        records = [
            {
                "hookpoint": "layers.0.self_attn.q_proj",
                "fvu_base": 0.10,
                "fvu_w8a8": 0.12,
                "fvu_delta": 0.02,
                "exceed_alpha_0.50_base": 0.30,
                "exceed_alpha_0.50_w8a8": 0.35,
                "exceed_alpha_0.50_delta": 0.05,
            },
            {
                "hookpoint": "layers.1.self_attn.q_proj",
                "fvu_base": 0.20,
                "fvu_w8a8": 0.21,
                "fvu_delta": 0.01,
                "exceed_alpha_0.50_base": 0.40,
                "exceed_alpha_0.50_w8a8": 0.45,
                "exceed_alpha_0.50_delta": 0.05,
            },
        ]

        summary = build_summary(records)

        assert summary["aggregate"]["num_hookpoints"] == 2
        assert summary["aggregate"]["mean_fvu_delta"] == pytest.approx(0.015)
        assert summary["aggregate"]["mean_exceed_alpha_0.50_delta"] == pytest.approx(0.05)
        assert summary["aggregate"]["worst_fvu_hookpoint"] == "layers.0.self_attn.q_proj"

    def test_compute_reconstruction_metrics_returns_fvu_and_exceed(self):
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        recon = torch.tensor([[1.0, 1.0], [2.0, 4.0]], dtype=torch.float32)

        metrics = compute_reconstruction_metrics(target, recon, elbow_value=1.5)

        total_variance = (target - target.mean(0)).pow(2).sum()
        expected_fvu = ((target - recon).pow(2).sum() / total_variance).item()
        expected_exceed = (torch.abs(target - recon) > 0.75).float().mean().item()

        assert metrics["fvu"] == pytest.approx(expected_fvu)
        assert metrics["exceed_alpha_0.50"] == pytest.approx(expected_exceed)
