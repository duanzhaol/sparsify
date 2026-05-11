from dataclasses import dataclass
from types import SimpleNamespace

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from compute_elbow_thresholds import ActivationCollector, load_threshold_dataset


@dataclass
class DummyArgs:
    dataset: str
    ctx_len: int = 2048
    num_tokens: int = 4096


def test_load_threshold_dataset_supports_save_to_disk_directory(tmp_path):
    output = tmp_path / "tokenized"
    Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "overflow_to_sample_mapping": [0, 1],
        }
    ).save_to_disk(output)

    dataset, dataset_type = load_threshold_dataset(
        DummyArgs(str(output)),
        tokenizer=None,
    )

    assert dataset_type == "hf_tokenized"
    assert len(dataset) == 2
    assert dataset.column_names == ["input_ids", "overflow_to_sample_mapping"]


class _TinyModelWithTopLevelHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=None,
            text_config=SimpleNamespace(num_hidden_layers=1),
        )
        self.base_model = torch.nn.Module()
        self.base_model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.lm_head = _CountingHead(2, 5)

    def forward(self, input_ids):
        hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 2)
        return self.lm_head(hidden)


class _CountingHead(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)
        self.calls = 0

    def forward(self, input):
        self.calls += 1
        return super().forward(input)


def test_activation_collector_registers_top_level_lm_head_hookpoint():
    model = _TinyModelWithTopLevelHead()
    collector = ActivationCollector(model, ["lm_head"], max_tokens=4)
    dataloader = DataLoader(
        [{"input_ids": torch.ones(1, 2, dtype=torch.long)}],
        batch_size=None,
    )

    collector.collect(dataloader, torch.device("cpu"))

    assert model.lm_head.calls == 0
    assert collector.activations["lm_head"].shape == (2, 2)
