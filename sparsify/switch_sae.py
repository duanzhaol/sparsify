"""Switch SAE: Mixture-of-Experts sparse autoencoder."""

import torch
from torch import nn, Tensor
from typing import NamedTuple

from .sparse_coder import SparseCoder, ForwardOutput
from .config import SparseCoderConfig


class SwitchForwardOutput(NamedTuple):
    sae_out: Tensor
    latent_acts: Tensor
    latent_indices: Tensor
    fvu: Tensor
    auxk_loss: Tensor
    multi_topk_fvu: Tensor
    expert_ids: Tensor
    load_balance_loss: Tensor


class SwitchSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        num_experts: int = 8,
        load_balance_alpha: float = 0.01,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__()
        self.d_in = d_in
        self.num_experts = num_experts
        self.load_balance_alpha = load_balance_alpha
        self.cfg = cfg

        # Router
        self.router = nn.Linear(d_in, num_experts, device=device, dtype=dtype)

        # Experts: each has 1/N of the total dictionary size
        expert_cfg = SparseCoderConfig(
            activation=cfg.activation,
            expansion_factor=max(1, cfg.expansion_factor // num_experts),
            normalize_decoder=cfg.normalize_decoder,
            num_latents=cfg.num_latents // num_experts if cfg.num_latents > 0 else 0,
            k=cfg.k,
            multi_topk=cfg.multi_topk,
            skip_connection=cfg.skip_connection,
        )

        self.experts = nn.ModuleList([
            SparseCoder(d_in, expert_cfg, device=device, dtype=dtype, **kwargs)
            for _ in range(num_experts)
        ])

        # Total latents across all experts (needed for Trainer compatibility)
        self.num_latents = self.experts[0].num_latents * num_experts

    # Wrapping the forward in bf16 autocast improves performance by almost 2x
    @torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_bf16_supported(),
    )
    def forward(
        self,
        x: Tensor,
        y: Tensor | None = None,
        *,
        dead_mask: Tensor | None = None
    ) -> SwitchForwardOutput:
        B, D = x.shape

        # Routing: select expert for each token
        router_logits = self.router(x)  # [B, N]
        expert_ids = router_logits.argmax(dim=-1)  # [B]
        router_probs = torch.softmax(router_logits, dim=-1)  # [B, N]
        selected_probs = router_probs.gather(-1, expert_ids.unsqueeze(-1)).squeeze(-1)  # [B]

        # Load balancing loss: encourage uniform expert usage
        expert_counts = torch.bincount(expert_ids, minlength=self.num_experts).float()
        expert_freq = expert_counts / B
        target_freq = 1.0 / self.num_experts
        load_balance_loss = ((expert_freq - target_freq) ** 2).sum()

        # Expert computation (simple loop version)
        sae_out = torch.zeros_like(x)
        # Initialize as tensors to preserve gradients (use float32 for stability)
        total_fvu = torch.zeros(1, device=x.device, dtype=torch.float32)
        total_auxk = torch.zeros(1, device=x.device, dtype=torch.float32)
        total_multi_topk_fvu = torch.zeros(1, device=x.device, dtype=torch.float32)

        # Collect latent info (use first expert's shape as template)
        latent_acts = torch.zeros(B, self.cfg.k, device=x.device, dtype=x.dtype)
        latent_indices = torch.zeros(B, self.cfg.k, device=x.device, dtype=torch.long)

        for i in range(self.num_experts):
            mask = (expert_ids == i)
            if not mask.any():
                continue

            x_i = x[mask]
            y_i = y[mask] if y is not None else None

            # Forward through expert
            out_i: ForwardOutput = self.experts[i](x_i, y_i, dead_mask=dead_mask)

            # Weight output by routing probability (enables gradient flow to router)
            weighted_out = out_i.sae_out * selected_probs[mask].unsqueeze(-1)
            sae_out[mask] = weighted_out.type_as(sae_out)  # Ensure dtype matches

            # Accumulate losses (weighted by token count, preserve gradients)
            # Skip if FVU is inf/nan (happens when expert gets only 1 token)
            n_tokens = mask.sum()
            if not (torch.isinf(out_i.fvu).any() or torch.isnan(out_i.fvu).any()):
                total_fvu = total_fvu + out_i.fvu * n_tokens
                total_auxk = total_auxk + out_i.auxk_loss * n_tokens
                total_multi_topk_fvu = total_multi_topk_fvu + out_i.multi_topk_fvu * n_tokens
            else:
                # Note: Still produce output but don't add to loss for numerical stability
                pass

            # Store latent info
            latent_acts[mask] = out_i.latent_acts.type_as(latent_acts)
            latent_indices[mask] = out_i.latent_indices

        return SwitchForwardOutput(
            sae_out=sae_out,
            latent_acts=latent_acts,
            latent_indices=latent_indices,
            fvu=(total_fvu / B).squeeze(),
            auxk_loss=(total_auxk / B).squeeze(),
            multi_topk_fvu=(total_multi_topk_fvu / B).squeeze(),
            expert_ids=expert_ids,
            load_balance_loss=load_balance_loss,
        )

    @property
    def device(self):
        return self.router.weight.device

    @property
    def dtype(self):
        return self.router.weight.dtype

    def save_to_disk(self, path):
        from pathlib import Path
        import json
        from safetensors.torch import save_model

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save router + all experts
        save_model(self, str(path / "switch_sae.safetensors"))

        # Save config
        with open(path / "cfg.json", "w") as f:
            json.dump({
                **self.cfg.to_dict(),
                "d_in": self.d_in,
                "num_experts": self.num_experts,
                "load_balance_alpha": self.load_balance_alpha,
                "switch_sae": True,
            }, f)

    @staticmethod
    def load_from_disk(path, device="cpu", **kwargs):
        from pathlib import Path
        import json
        from safetensors.torch import load_model

        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            num_experts = cfg_dict.pop("num_experts", 8)
            load_balance_alpha = cfg_dict.pop("load_balance_alpha", 0.01)
            cfg_dict.pop("switch_sae", None)
            cfg = SparseCoderConfig.from_dict(cfg_dict, drop_extra_fields=True)

        switch_sae = SwitchSAE(d_in, cfg, num_experts, load_balance_alpha, device=device)
        load_model(switch_sae, str(path / "switch_sae.safetensors"), device=str(device))

        return switch_sae

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Normalize decoder weights of all experts."""
        for expert in self.experts:
            expert.set_decoder_norm_to_unit_norm()

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """Remove parallel gradients for all experts (only those with gradients)."""
        for expert in self.experts:
            # Only process experts that have gradients (were used in this batch)
            if expert.W_dec is not None and expert.W_dec.grad is not None:
                expert.remove_gradient_parallel_to_decoder_directions()
