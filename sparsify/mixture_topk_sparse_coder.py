"""Soft mixture-of-experts TopK sparse coder."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput
from .sparse_coder import SparseCoder


class MixtureTopKSparseCoder(SparseCoder):
    """TopK SAE with a softly routed two-expert encoder.

    A small router mixes two expert projections per token before top-k selection.
    This increases encoder expressivity while avoiding the hard capacity limits and
    optimizer friction seen in grouped routing variants.
    """

    num_experts: int = 2

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.experts = nn.ModuleList(
            nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
            for _ in range(self.num_experts)
        )
        for expert in self.experts:
            expert.bias.data.zero_()

        self.router = nn.Linear(d_in, self.num_experts, device=device, dtype=dtype)
        self.router.bias.data.zero_()

        if decoder:
            self.W_dec = nn.Parameter(self.experts[0].weight.data.clone())
            if cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode with a soft expert mixture before latent top-k."""
        x = x - self.b_dec

        router_weights = torch.softmax(self.router(x), dim=-1)
        expert_pre_acts = torch.stack(
            [F.relu(expert(x)) for expert in self.experts], dim=1
        )
        pre_acts = (expert_pre_acts * router_weights.unsqueeze(-1)).sum(dim=1)

        top_acts, top_indices = pre_acts.topk(self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, pre_acts)
