from dataclasses import replace
from typing import Literal, Protocol

import torch
from torch import Tensor, nn

from lowrank_encoder import LowRankSparseCoder
from ..sparse_coder import ForwardOutput, SparseCoder
from ..tiled_sparse_coder import TiledSparseCoder
from .two_stage import TwoStageConfig, TwoStageEncoder
from .pca import select_pca_from_bundle


class EncoderStrategy(Protocol):
    def forward(self, x: Tensor, y: Tensor | None) -> ForwardOutput:
        ...


class FullEncoderStrategy:
    def __init__(self, sae: nn.Module):
        self.sae = sae

    def forward(self, x: Tensor, y: Tensor | None) -> ForwardOutput:
        return self.sae(x=x, y=y)


class TwoStageEncoderStrategy:
    def __init__(self, sae: SparseCoder | LowRankSparseCoder, cfg: TwoStageConfig):
        self.encoder = TwoStageEncoder(sae, cfg)

    def forward(self, x: Tensor, y: Tensor | None) -> ForwardOutput:
        return self.encoder.forward(x, y)


def build_encoder_strategies(
    saes: dict[str, nn.Module],
    mode: Literal["full", "two_stage"],
    *,
    two_stage_cfg: TwoStageConfig | None = None,
    pca_bundle=None,
    pca_mean_path: str | None = None,
) -> dict[str, EncoderStrategy]:
    strategies: dict[str, EncoderStrategy] = {}

    if mode == "full":
        for key, sae in saes.items():
            strategies[key] = FullEncoderStrategy(sae)
        return strategies

    if two_stage_cfg is None:
        raise ValueError("two_stage_cfg is required for mode='two_stage'")

    for key, sae in saes.items():
        if isinstance(sae, TiledSparseCoder):
            raise ValueError("two_stage encoder does not support TiledSparseCoder")
        if not isinstance(sae, (SparseCoder, LowRankSparseCoder)):
            raise ValueError(f"Unsupported SAE type for two_stage: {type(sae)}")

        cfg = two_stage_cfg
        if two_stage_cfg.projection == "pca":
            if pca_bundle is None:
                raise ValueError("pca_bundle is required for projection='pca'")
            hook_name = key.partition("/")[0]
            matrix, mean = select_pca_from_bundle(
                pca_bundle, hook_name, mean_path=pca_mean_path
            )
            cfg = replace(two_stage_cfg, pca_matrix=matrix, pca_mean=mean)

        strategies[key] = TwoStageEncoderStrategy(sae, cfg)

    return strategies
