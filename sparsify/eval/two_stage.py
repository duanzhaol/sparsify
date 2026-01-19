from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from ..sparse_coder import ForwardOutput, SparseCoder
from lowrank_encoder import LowRankSparseCoder


@dataclass
class TwoStageConfig:
    low_dim: int = 128
    k_coarse: int = 1000
    projection: Literal["slice", "random", "pca"] = "slice"
    seed: int = 0
    pca_matrix: Tensor | None = None
    pca_mean: Tensor | None = None


class TwoStageEncoder:
    def __init__(self, sae: SparseCoder | LowRankSparseCoder, cfg: TwoStageConfig):
        if sae.cfg.activation != "topk":
            raise ValueError("two-stage encoder only supports activation='topk'")
        if cfg.low_dim <= 0:
            raise ValueError("low_dim must be positive")
        if cfg.k_coarse <= 0:
            raise ValueError("k_coarse must be positive")
        if cfg.k_coarse < sae.cfg.k:
            raise ValueError(
                f"k_coarse ({cfg.k_coarse}) must be >= k ({sae.cfg.k})"
            )

        self.sae = sae
        self.cfg = cfg
        self.device = sae.device
        self.dtype = sae.dtype
        self.low_dim = min(cfg.low_dim, sae.d_in)
        self.is_lowrank = isinstance(sae, LowRankSparseCoder)

        self.pca_mean = None
        self.proj = None
        if cfg.projection == "random":
            gen = torch.Generator(device=self.device)
            gen.manual_seed(cfg.seed)
            self.proj = torch.randn(
                sae.d_in, self.low_dim, device=self.device, dtype=self.dtype, generator=gen
            ) / (self.low_dim ** 0.5)
        elif cfg.projection == "pca":
            if cfg.pca_matrix is None:
                raise ValueError("pca_matrix is required for projection='pca'")
            proj = cfg.pca_matrix.to(device=self.device, dtype=self.dtype)
            if proj.ndim != 2:
                raise ValueError("pca_matrix must be 2D")
            if proj.shape[0] == self.low_dim and proj.shape[1] == sae.d_in:
                proj = proj.T
            if proj.shape[0] != sae.d_in:
                raise ValueError(
                    f"pca_matrix first dim must be {sae.d_in}, got {proj.shape[0]}"
                )
            if proj.shape[1] < self.low_dim:
                raise ValueError(
                    f"pca_matrix second dim must be >= {self.low_dim}, got {proj.shape[1]}"
                )
            self.proj = proj[:, : self.low_dim]
            if cfg.pca_mean is not None:
                mean = cfg.pca_mean.to(device=self.device, dtype=self.dtype)
                if mean.ndim == 2 and mean.shape[0] == 1:
                    mean = mean.squeeze(0)
                if mean.ndim != 1 or mean.shape[0] != sae.d_in:
                    raise ValueError("pca_mean must be shape [d_in]")
                self.pca_mean = mean

        if self.is_lowrank:
            B = sae.encoder_B.weight
            if self.proj is None:
                self.B_low = B[:, : self.low_dim]
            else:
                self.B_low = B @ self.proj
            self.A = sae.encoder_A.weight
            self.bias = sae.encoder_A.bias
        else:
            W = sae.encoder.weight
            if self.proj is None:
                self.W_low = W[:, : self.low_dim]
            else:
                self.W_low = W @ self.proj
            self.W_full = W
            self.bias = sae.encoder.bias

    def _center_input(self, x: Tensor) -> Tensor:
        x = x.to(self.dtype)
        if self.sae.transcoder:
            return x
        return x - self.sae.b_dec

    def _coarse_scores(self, x_centered: Tensor) -> Tensor:
        x_proj = x_centered
        if self.pca_mean is not None:
            x_proj = x_proj - self.pca_mean

        if self.proj is None:
            x_low = x_proj[:, : self.low_dim]
        else:
            x_low = x_proj @ self.proj

        if self.is_lowrank:
            intermediate = x_low @ self.B_low.T
            scores = intermediate @ self.A.T + self.bias
        else:
            scores = x_low @ self.W_low.T + self.bias
        return torch.relu(scores)

    def _refine_scores(self, x_centered: Tensor, candidates: Tensor) -> Tensor:
        if self.is_lowrank:
            xB = x_centered @ self.sae.encoder_B.weight.T
            A_cand = self.A[candidates]
            b_cand = self.bias[candidates]
            scores = torch.einsum("nkr,nr->nk", A_cand, xB) + b_cand
        else:
            W_cand = self.W_full[candidates]
            b_cand = self.bias[candidates]
            scores = torch.einsum("nkd,nd->nk", W_cand, x_centered) + b_cand
        return torch.relu(scores)

    def forward(self, x: Tensor, y: Tensor | None = None) -> ForwardOutput:
        x_centered = self._center_input(x)
        k_coarse = self.cfg.k_coarse

        coarse_scores = self._coarse_scores(x_centered)
        if k_coarse > coarse_scores.shape[-1]:
            k_coarse = coarse_scores.shape[-1]
        _, coarse_indices = coarse_scores.topk(k_coarse, dim=-1, sorted=False)

        refine_scores = self._refine_scores(x_centered, coarse_indices)
        k = self.sae.cfg.k
        top_acts, top_pos = refine_scores.topk(k, dim=-1, sorted=False)
        top_indices = coarse_indices.gather(1, top_pos)

        sae_out = self.sae.decode(top_acts, top_indices)
        if self.sae.W_skip is not None:
            sae_out = sae_out + x.to(self.dtype) @ self.sae.W_skip.mT

        zero = sae_out.new_tensor(0.0)
        return ForwardOutput(
            sae_out=sae_out,
            latent_acts=top_acts,
            latent_indices=top_indices,
            fvu=zero,
            auxk_loss=zero,
            multi_topk_fvu=zero,
        )
