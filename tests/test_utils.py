from __future__ import annotations

from types import SimpleNamespace

import torch

from sparsify.utils import get_layer_list, get_max_layer_index


class _NestedTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
        )


class _Wrapper:
    def __init__(self):
        self.config = SimpleNamespace(
            num_hidden_layers=None,
            text_config=SimpleNamespace(num_hidden_layers=2),
        )
        self.base_model = _NestedTextModel()


def test_get_layer_list_uses_text_config_for_nested_qwen3_5_models():
    name, layers = get_layer_list(_Wrapper())

    assert name == "language_model.layers"
    assert len(layers) == 2


def test_get_max_layer_index_handles_nested_layers_name():
    assert get_max_layer_index(
        [
            "language_model.layers.3.linear_attn.in_proj_qkv",
            "language_model.layers.11.mlp.up_proj",
        ],
        "language_model.layers",
    ) == 11


def test_get_max_layer_index_keeps_legacy_layers_name_behavior():
    assert get_max_layer_index(
        ["layers.3.self_attn.q_proj", "layers.7.mlp.up_proj"],
        "layers",
    ) == 7
