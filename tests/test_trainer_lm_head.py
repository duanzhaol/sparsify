from __future__ import annotations

from types import SimpleNamespace

import torch
from datasets import Dataset

from sparsify.config import SparseCoderConfig, TrainConfig
from sparsify.trainer import Trainer


class _CountingHead(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)
        self.calls = 0

    def forward(self, input):
        self.calls += 1
        return super().forward(input)


class _TinyBackbone(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.embed = torch.nn.Embedding(8, hidden_size)

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _TinyCausalLM(torch.nn.Module):
    def __init__(self, hidden_size: int = 4, vocab_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.base_model = _TinyBackbone(hidden_size)
        self.lm_head = _CountingHead(hidden_size, vocab_size)
        self.dummy_inputs = {"input_ids": torch.tensor([[1, 2]])}

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids):
        hidden = self.base_model(input_ids)
        return self.lm_head(hidden)


def test_lm_head_training_captures_input_without_running_head(tmp_path):
    model = _TinyCausalLM()
    dataset = Dataset.from_dict({"input_ids": [[1, 2]]}).with_format("torch")
    cfg = TrainConfig(
        sae=SparseCoderConfig(
            architecture="topk",
            expansion_factor=2,
            normalize_decoder=False,
            k=1,
        ),
        hookpoints=["lm_head"],
        batch_size=1,
        grad_acc_steps=1,
        max_tokens=2,
        log_to_wandb=False,
        save_metrics_jsonl=False,
        save_dir=str(tmp_path),
        save_every=100,
        wandb_log_frequency=1,
        auxk_alpha=0.0,
        exceed_alphas=[],
    )

    trainer = Trainer(cfg, dataset, model)
    assert model.lm_head.calls == 1

    trainer.fit()

    assert model.lm_head.calls == 1
