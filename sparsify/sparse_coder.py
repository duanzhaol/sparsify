import json
import math
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

import einops
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors import safe_open
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from .config import SparseCoderConfig
from .device import device_autocast
from .fused_encoder import EncoderOutput, fused_encoder
from .utils import decoder_impl


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    auxk_loss: Tensor
    """AuxK loss, if applicable."""


class SparseCoder(nn.Module):
    def __new__(
        cls,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        if cls is SparseCoder:
            architecture = getattr(cfg, "architecture", "topk")
            target_cls = _get_sae_class(architecture)
            if target_cls is not SparseCoder:
                return super().__new__(target_cls)
        return super().__new__(cls)

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        if decoder:
            self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    @staticmethod
    def load_many(
        name: str,
        local: bool = False,
        layers: list[str] | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        pattern: str | None = None,
    ) -> dict[str, "SparseCoder"]:
        """Load sparse coders for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        if local:
            repo_path = Path(name)
        else:
            repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        if layers is not None:
            return {
                layer: SparseCoder.load_any(
                    repo_path / layer, device=device, decoder=decoder
                )
                for layer in natsorted(layers)
            }
        files = [
            f
            for f in repo_path.iterdir()
            if f.is_dir() and (pattern is None or fnmatch(f.name, pattern))
        ]
        return {
            f.name: SparseCoder.load_any(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.name)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        hookpoint: str | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "SparseCoder":
        # Download from the HuggingFace Hub
        repo_path = Path(
            snapshot_download(
                name,
                allow_patterns=f"{hookpoint}/*" if hookpoint is not None else None,
            )
        )
        if hookpoint is not None:
            repo_path = repo_path / hookpoint

        # No layer specified, and there are multiple layers
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError("No config file found; try specifying a layer.")

        return SparseCoder.load_any(repo_path, device=device, decoder=decoder)

    @classmethod
    def load_from_disk(
        cls,
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "SparseCoder":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SparseCoderConfig.from_dict(cfg_dict, drop_extra_fields=True)

        safetensors_path = str(path / "sae.safetensors")

        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            first_key = next(iter(f.keys()))
            reference_dtype = f.get_tensor(first_key).dtype

        sae = cls(
            d_in, cfg, device=device, decoder=decoder, dtype=reference_dtype
        )

        load_model(
            model=sae,
            filename=safetensors_path,
            device=str(device),
            # TODO: Maybe be more fine-grained about this in the future?
            strict=decoder,
        )
        return sae

    @staticmethod
    def load_any(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "SparseCoder":
        """Factory: load any SAE variant by reading architecture from cfg.json."""
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)

        architecture = cfg_dict.get("architecture", "topk")
        cls = _get_sae_class(architecture)
        return cls.load_from_disk(path, device=device, decoder=decoder)

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                },
                f,
            )

    @property
    def device(self):
        return self.b_dec.device

    @property
    def dtype(self):
        return self.b_dec.dtype

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """Return optimizer parameter groups. Subclasses can override for per-component LR."""
        return [{"params": self.parameters(), "lr": base_lr}]

    # ------------------------------------------------------------------
    # Selection cost estimation (for LUTurbo deployment feasibility)
    # ------------------------------------------------------------------

    def _encoder_linear_layers(self) -> list[tuple[str, "nn.Linear"]]:
        """Return all nn.Linear layers on the encode path. Override in subclasses."""
        return [("encoder", self.encoder)]

    def _extra_encode_accesses(self) -> list[tuple[str, int, str]]:
        """Return extra encode-path accesses not captured by linear layers.

        Each entry: (name, num_accesses, shape_description).
        Override in subclasses with codebook matmuls, etc.
        """
        return []

    def _deployment_lookup_accesses(self, n_output: int) -> list[tuple[str, int, str]]:
        """Return deployment-side memory accesses for LUTurbo lookup phase.

        These represent the cost of deploying each static library element:
        read the input-side atom/value (size d_in) and the output-side lookup
        result (size n_output). For each static vector library, this proxy
        counts count × (d_in + n_output) accesses.
        Override in subclasses that add trunk / codebook / extra libraries.
        """
        return [
            (
                "sparse_lookup",
                self._deploy_library_accesses(self.cfg.k, n_output),
                self._deploy_library_shape(self.cfg.k, n_output, label=f"K={self.cfg.k}"),
            )
        ]

    def _deploy_library_accesses(self, num_vectors: int, n_output: int) -> int:
        """Cost proxy for one deploy-time static library of `num_vectors` entries."""
        return int(num_vectors * (self.d_in + n_output))

    def _deploy_library_shape(
        self,
        num_vectors: int,
        n_output: int,
        *,
        label: str,
    ) -> str:
        return f"{label}×(d={self.d_in}+n={n_output})"

    def selection_cost_estimate(self, n_output: int | None = None) -> dict:
        """Estimate encoder-side, deployment-side, and combined memory accesses.

        Args:
            n_output: output dim of original weight matrix. Default 4*d_in.

        Returns:
            dict with three sets of metrics:
            - Encoder (selection): total_accesses, ratio, budget_ratio, feasible
            - Deployment (lookup): deployment_accesses, deployment_ratio
            - Combined (total): combined_accesses, combined_ratio,
              combined_budget_ratio, combined_feasible
        """
        if n_output is None:
            n_output = 4 * self.d_in
        original = self.d_in * n_output
        budget = 1.5 * original

        breakdown = []
        total = 0
        for name, layer in self._encoder_linear_layers():
            acc = layer.in_features * layer.out_features
            breakdown.append({
                "name": name,
                "accesses": acc,
                "shape": f"{layer.in_features}x{layer.out_features}",
            })
            total += acc
        for name, acc, shape in self._extra_encode_accesses():
            breakdown.append({"name": name, "accesses": acc, "shape": shape})
            total += acc

        ratio = total / original if original > 0 else float("inf")
        budget_ratio = total / budget if budget > 0 else float("inf")

        # Deployment-side lookup cost
        deploy_breakdown = []
        deploy_total = 0
        for name, acc, shape in self._deployment_lookup_accesses(n_output):
            deploy_breakdown.append({"name": name, "accesses": acc, "shape": shape})
            deploy_total += acc
        deploy_ratio = deploy_total / original if original > 0 else float("inf")

        # Combined (encoder + deployment)
        combined = total + deploy_total
        combined_ratio = combined / original if original > 0 else float("inf")
        combined_budget_ratio = combined / budget if budget > 0 else float("inf")

        return {
            # --- encoder-only (selection) ---
            "total_accesses": total,
            "original_matmul_accesses": original,
            "ratio": round(ratio, 2),
            "budget_ratio": round(budget_ratio, 2),
            "feasible": budget_ratio <= 1.0,
            "breakdown": breakdown,
            # --- deployment (lookup) ---
            "deployment_accesses": deploy_total,
            "deployment_ratio": round(deploy_ratio, 2),
            "deployment_breakdown": deploy_breakdown,
            # --- combined = encoder + deployment ---
            "combined_accesses": combined,
            "combined_ratio": round(combined_ratio, 2),
            "combined_budget_ratio": round(combined_budget_ratio, 2),
            "combined_feasible": combined_budget_ratio <= 1.0,
        }

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        x = x - self.b_dec
        return fused_encoder(x, self.encoder.weight, self.encoder.bias, self.cfg.k)

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    # Wrapping the forward in bf16 autocast improves performance by almost 2x
    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        top_acts, top_indices, pre_acts = self.encode(x)

        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = x

        # Decode
        sae_out = self.decode(top_acts, top_indices)

        # Compute the residual
        e = y - sae_out

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (y - y.mean(0)).pow(2).sum()

        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = y.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        if self.W_dec.grad is None:
            return

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )


def _get_sae_class(architecture: str) -> type:
    """Return the SparseCoder subclass for the given architecture string."""
    if architecture == "topk":
        return SparseCoder
    if architecture == "batch_topk":
        return BatchTopKSparseCoder
    if architecture == "adaptive_budget_topk":
        return AdaptiveBudgetTopKSparseCoder
    if architecture == "bucketed_topk":
        return BucketedTopKSparseCoder
    if architecture == "codebook_topk":
        return CodebookTopKSparseCoder
    if architecture == "residual_vq":
        return ResidualVQSparseCoder
    if architecture == "two_code_residual_vq":
        return TwoCodeResidualVQSparseCoder
    if architecture == "lowrank_residual_vq":
        return LowRankResidualVQSparseCoder
    if architecture == "whitened_topk":
        return WhitenedTopKSparseCoder
    if architecture == "jumprelu":
        return JumpReLUSparseCoder
    if architecture == "gated":
        return GatedSparseCoder
    if architecture == "routed":
        return RoutedSparseCoder
    if architecture == "expert_topk":
        return ExpertTopKSparseCoder
    if architecture == "expert_jumprelu":
        return ExpertJumpReLUSparseCoder
    if architecture == "adaptive_active_expert_jumprelu":
        return AdaptiveActiveExpertJumpReLUSparseCoder
    if architecture == "factorized_router_expert_topk":
        return FactorizedRouterExpertTopKSparseCoder
    if architecture == "shared_routed_expert_topk":
        return SharedRoutedExpertTopKSparseCoder
    if architecture == "shared_two_stage_residual_expert":
        return SharedTwoStageResidualExpertSparseCoder
    if architecture == "factorized_expert_topk":
        return FactorizedExpertTopKSparseCoder
    if architecture == "shared_routed_factorized_expert_topk":
        return SharedRoutedFactorizedExpertTopKSparseCoder
    if architecture == "shared_routed_factorized_expert_residual":
        return SharedRoutedFactorizedExpertResidualSparseCoder
    if architecture == "shared_lowrank_routed_expert_topk":
        return SharedLowRankRoutedExpertTopKSparseCoder
    if architecture == "shared_lowrank_two_stage_residual_expert":
        return SharedLowRankTwoStageResidualExpertSparseCoder
    if architecture == "shared_lowrank_routed_expert_residual":
        return SharedLowRankRoutedExpertResidualSparseCoder
    if architecture == "lowrank_expert_topk":
        return LowRankExpertTopKSparseCoder
    if architecture == "lowrank_expert_jumprelu":
        return LowRankExpertJumpReLUSparseCoder
    if architecture == "lowrank_expert_residual":
        return LowRankExpertResidualSparseCoder
    if architecture == "two_stage_residual_expert":
        return TwoStageResidualExpertSparseCoder
    if architecture == "group_topk":
        return GroupTopKSparseCoder
    if architecture == "factorized_topk":
        return FactorizedTopKSparseCoder
    if architecture == "lowrank_residual":
        return LowRankResidualSparseCoder
    if architecture == "lowrank_two_stage_residual":
        return LowRankTwoStageResidualSparseCoder
    if architecture == "routed_lowrank_two_stage_residual":
        return RoutedLowRankTwoStageResidualSparseCoder
    if architecture == "bucketed_lowrank_residual":
        return BucketedLowRankResidualSparseCoder
    if architecture == "whitened_lowrank_residual":
        return WhitenedLowRankResidualSparseCoder
    if architecture == "lowrank_adaptive_budget_residual":
        return LowRankAdaptiveBudgetResidualSparseCoder
    if architecture == "lowrank_gated_residual":
        return LowRankGatedResidualSparseCoder
    if architecture == "lowrank_jumprelu_residual":
        return LowRankJumpReLUResidualSparseCoder
    if architecture == "lowrank_multi_branch_residual":
        return LowRankMultiBranchResidualSparseCoder
    if architecture == "lowrank_factorized_residual":
        return LowRankFactorizedResidualSparseCoder
    if architecture == "lowrank_soft_codebook_residual":
        return LowRankSoftCodebookResidualSparseCoder
    if architecture == "lowrank_gated_soft_codebook_residual":
        return LowRankGatedSoftCodebookResidualSparseCoder
    if architecture == "lowrank_grouped_soft_codebook_residual":
        return LowRankGroupedSoftCodebookResidualSparseCoder
    if architecture == "lowrank_two_stage_soft_codebook_residual":
        return LowRankTwoStageSoftCodebookResidualSparseCoder
    if architecture == "bucketed_lowrank_two_stage_soft_codebook_residual":
        return BucketedLowRankTwoStageSoftCodebookResidualSparseCoder
    if architecture == "whitened_lowrank_two_stage_soft_codebook_residual":
        return WhitenedLowRankTwoStageSoftCodebookResidualSparseCoder
    if architecture == "lowrank_asymmetric_two_stage_soft_codebook_residual":
        return LowRankAsymmetricTwoStageSoftCodebookResidualSparseCoder
    if architecture == "routed_lowrank_asymmetric_two_stage_soft_codebook_residual":
        return RoutedLowRankAsymmetricTwoStageSoftCodebookResidualSparseCoder
    if architecture == "routed_lowrank_two_stage_soft_codebook_residual":
        return RoutedLowRankTwoStageSoftCodebookResidualSparseCoder
    if architecture == "whitened_lowrank_gated_residual":
        return WhitenedLowRankGatedResidualSparseCoder
    if architecture == "lowrank_grouped_residual":
        return LowRankGroupedResidualSparseCoder
    if architecture == "two_stage_residual":
        return TwoStageResidualSparseCoder
    if architecture == "multi_branch_gated":
        return MultiBranchGatedSparseCoder
    raise ValueError(f"Unknown architecture: {architecture!r}")


# Allow for alternate naming conventions
Sae = SparseCoder


class WhitenedTopKSparseCoder(SparseCoder):
    """Top-k SAE with normalized low-rank input preconditioning."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        hidden_dim = min(d_in, max(cfg.k * 2, d_in // 4))
        self.preconditioner_down = nn.Linear(
            d_in, hidden_dim, bias=False, device=device, dtype=dtype
        )
        self.preconditioner_up = nn.Linear(
            hidden_dim, d_in, bias=False, device=device, dtype=dtype
        )

        # Keep the family close to a stable normalized identity map while
        # still making step-0 behavior observably different from plain top-k.
        self.preconditioner_down.weight.data.zero_()
        self.preconditioner_up.weight.data.zero_()
        diagonal = min(d_in, hidden_dim)
        self.preconditioner_down.weight.data[:diagonal, :diagonal] = torch.eye(
            diagonal, device=device, dtype=dtype
        )
        self.preconditioner_up.weight.data[:diagonal, :diagonal] = 0.05 * torch.eye(
            diagonal, device=device, dtype=dtype
        )

    def _encoder_linear_layers(self):
        return [("encoder", self.encoder), ("preconditioner_down", self.preconditioner_down), ("preconditioner_up", self.preconditioner_up)]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        centered = x - x.mean(dim=-1, keepdim=True)
        rms = centered.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        normalized = centered * rms
        correction = self.preconditioner_up(self.preconditioner_down(normalized))
        mixed = normalized + 0.25 * torch.roll(normalized, shifts=1, dims=-1)
        whitened = mixed + correction
        return fused_encoder(
            whitened, self.encoder.weight, self.encoder.bias, self.cfg.k
        )


class AdaptiveBudgetTopKSparseCoder(SparseCoder):
    """Allocate a fixed batch-level feature budget across samples dynamically."""

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        acts = F.relu(F.linear(x, self.encoder.weight, self.encoder.bias))

        batch_size = acts.shape[0]
        if batch_size == 1:
            top_acts, top_indices = torch.topk(
                acts, self.cfg.k, dim=-1, sorted=False
            )
            return EncoderOutput(top_acts, top_indices, acts)

        difficulty = acts.detach().pow(2).mean(dim=-1)
        difficulty_sum = difficulty.sum().clamp_min(torch.finfo(acts.dtype).tiny)
        total_budget = batch_size * self.cfg.k

        raw_quota = difficulty / difficulty_sum * total_budget
        quotas = torch.floor(raw_quota).to(dtype=torch.long)
        quotas = quotas.clamp(min=1, max=self.cfg.k)

        remaining = total_budget - int(quotas.sum().item())
        if remaining > 0:
            order = torch.argsort(raw_quota - quotas.to(raw_quota.dtype), descending=True)
            for idx in order.tolist():
                if remaining == 0:
                    break
                if quotas[idx] < self.cfg.k:
                    quotas[idx] += 1
                    remaining -= 1
        elif remaining < 0:
            order = torch.argsort(raw_quota - quotas.to(raw_quota.dtype))
            for idx in order.tolist():
                if remaining == 0:
                    break
                if quotas[idx] > 1:
                    quotas[idx] -= 1
                    remaining += 1

        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        rank = torch.arange(self.cfg.k, device=acts.device)
        active_mask = rank.unsqueeze(0) < quotas.unsqueeze(1)
        top_acts = top_acts * active_mask.to(top_acts.dtype)
        return EncoderOutput(top_acts, top_indices, acts)


class BatchTopKSparseCoder(SparseCoder):
    """Select top activations under a global batch-level budget."""

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        acts = F.relu(F.linear(x, self.encoder.weight, self.encoder.bias))

        batch_size = acts.shape[0]
        if batch_size == 1:
            top_acts, top_indices = torch.topk(
                acts, self.cfg.k, dim=-1, sorted=False
            )
            return EncoderOutput(top_acts, top_indices, acts)

        flat = acts.reshape(-1)
        total_budget = batch_size * self.cfg.k
        total_budget = min(total_budget, flat.numel())
        _, flat_indices = torch.topk(flat, total_budget, sorted=False)

        selected = acts.new_zeros(acts.shape)
        selected.view(-1).scatter_(0, flat_indices, flat.index_select(0, flat_indices))

        top_acts, top_indices = torch.topk(selected, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)


class BucketedTopKSparseCoder(SparseCoder):
    """Route samples between two dictionaries based on activation norm."""

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

        self.low_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.high_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.low_encoder.bias.data.zero_()
        self.high_encoder.bias.data.zero_()
        self.high_encoder.weight.data.mul_(1.05)

        self.bucket_scale = nn.Parameter(
            torch.tensor(2.0, device=device, dtype=dtype)
        )
        self.bucket_bias = nn.Parameter(
            torch.tensor(0.0, device=device, dtype=dtype)
        )

        if decoder:
            decoder_init = 0.5 * (
                self.low_encoder.weight.data + self.high_encoder.weight.data
            )
            self.W_dec = nn.Parameter(decoder_init)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("low_encoder", self.low_encoder), ("high_encoder", self.high_encoder)]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        low_acts = F.relu(F.linear(x, self.low_encoder.weight, self.low_encoder.bias))
        high_acts = F.relu(
            F.linear(x, self.high_encoder.weight, self.high_encoder.bias)
        )

        norms = x.norm(dim=-1, keepdim=True)
        centered_norms = norms - norms.mean()
        gate = torch.sigmoid(self.bucket_scale * centered_norms + self.bucket_bias)
        acts = (1.0 - gate) * low_acts + gate * high_acts

        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)


class CodebookTopKSparseCoder(SparseCoder):
    """Reconstruct a coarse codebook vector, then sparsify the residual."""

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
        self.num_codes = min(256, max(32, cfg.k * 2))

        self.codebook = nn.Parameter(
            torch.randn(self.num_codes, d_in, device=device, dtype=dtype) * 0.02
        )
        self.code_router = nn.Linear(d_in, self.num_codes, device=device, dtype=dtype)
        self.code_router.bias.data.zero_()

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        if decoder:
            self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("code_router", self.code_router), ("encoder", self.encoder)]

    def _extra_encode_accesses(self):
        return [("codebook_matmul", self.num_codes * self.d_in, f"{self.num_codes}x{self.d_in}")]

    def _deployment_lookup_accesses(self, n_output):
        base = super()._deployment_lookup_accesses(n_output)
        return [
            (
                "codebook_deploy",
                self._deploy_library_accesses(self.num_codes, n_output),
                self._deploy_library_shape(self.num_codes, n_output, label=f"codes={self.num_codes}"),
            )
        ] + base

    def _select_code(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.code_router(x)
        code_indices = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)
        hard_assign = F.one_hot(code_indices, num_classes=self.num_codes).to(
            dtype=probs.dtype
        )
        # Keep the routed code hard in the forward pass while preserving a gradient
        # path through the router so DDP does not see unused parameters.
        routing = hard_assign + probs - probs.detach()
        coarse = routing @ self.codebook
        return coarse, logits

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        coarse, _ = self._select_code(x)
        residual = x - coarse
        return fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        coarse, _ = self._select_code(x_centered)
        residual = x_centered - coarse
        top_acts, top_indices, pre_acts = fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

        assert self.W_dec is not None, "Decoder weight was not initialized."
        sparse_residual = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        sae_out = coarse + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            aux_sparse = decoder_impl(auxk_indices, auxk_acts.to(self.dtype), self.W_dec.mT)
            e_hat = coarse + aux_sparse + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class ResidualVQSparseCoder(SparseCoder):
    """Hard codebook reconstruction with sparse residual correction."""

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
        self.num_codes = min(512, max(64, cfg.k * 4))

        self.codebook = nn.Parameter(
            torch.randn(self.num_codes, d_in, device=device, dtype=dtype) * 0.02
        )
        self.code_router = nn.Linear(d_in, self.num_codes, device=device, dtype=dtype)
        self.code_router.bias.data.zero_()

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        if decoder:
            self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("code_router", self.code_router), ("encoder", self.encoder)]

    def _extra_encode_accesses(self):
        return [("codebook_matmul", self.num_codes * self.d_in, f"{self.num_codes}x{self.d_in}")]

    def _deployment_lookup_accesses(self, n_output):
        base = super()._deployment_lookup_accesses(n_output)
        return [
            (
                "codebook_deploy",
                self._deploy_library_accesses(self.num_codes, n_output),
                self._deploy_library_shape(self.num_codes, n_output, label=f"codes={self.num_codes}"),
            )
        ] + base

    def _select_code(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.code_router(x)
        code_indices = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)
        hard_assign = F.one_hot(code_indices, num_classes=self.num_codes).to(
            dtype=probs.dtype
        )
        routing = hard_assign + probs - probs.detach()
        coarse = routing @ self.codebook
        return coarse, logits

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        coarse, _ = self._select_code(x)
        residual = x - coarse
        return fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        coarse, _ = self._select_code(x_centered)
        residual = x_centered - coarse
        top_acts, top_indices, pre_acts = fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

        assert self.W_dec is not None, "Decoder weight was not initialized."
        sparse_residual = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        sae_out = coarse + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            aux_sparse = decoder_impl(auxk_indices, auxk_acts.to(self.dtype), self.W_dec.mT)
            e_hat = coarse + aux_sparse + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class TwoCodeResidualVQSparseCoder(ResidualVQSparseCoder):
    """Two hard codebook reconstructions followed by sparse residual correction."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.second_codebook = nn.Parameter(
            torch.randn(self.num_codes, d_in, device=device, dtype=dtype) * 0.02
        )
        self.second_code_router = nn.Linear(
            d_in, self.num_codes, device=device, dtype=dtype
        )
        self.second_code_router.bias.data.zero_()

    def _encoder_linear_layers(self):
        return [("code_router", self.code_router), ("second_code_router", self.second_code_router), ("encoder", self.encoder)]

    def _extra_encode_accesses(self):
        return [
            ("codebook_matmul", self.num_codes * self.d_in, f"{self.num_codes}x{self.d_in}"),
            ("second_codebook_matmul", self.num_codes * self.d_in, f"{self.num_codes}x{self.d_in}"),
        ]

    def _deployment_lookup_accesses(self, n_output):
        base = super()._deployment_lookup_accesses(n_output)
        return base + [
            (
                "codebook2_deploy",
                self._deploy_library_accesses(self.num_codes, n_output),
                self._deploy_library_shape(self.num_codes, n_output, label=f"codes={self.num_codes}"),
            )
        ]

    def _select_code_from(
        self, x: Tensor, router: nn.Linear, codebook: Tensor
    ) -> tuple[Tensor, Tensor]:
        logits = router(x)
        code_indices = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)
        hard_assign = F.one_hot(code_indices, num_classes=self.num_codes).to(
            dtype=probs.dtype
        )
        routing = hard_assign + probs - probs.detach()
        coarse = routing @ codebook
        return coarse, logits

    def _select_codes(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        first_coarse, first_logits = self._select_code_from(
            x, self.code_router, self.codebook
        )
        second_input = x - first_coarse
        second_coarse, second_logits = self._select_code_from(
            second_input, self.second_code_router, self.second_codebook
        )
        combined = first_coarse + second_coarse
        return combined, first_logits, second_logits

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        coarse, _, _ = self._select_codes(x)
        residual = x - coarse
        return fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        coarse, _, _ = self._select_codes(x_centered)
        residual = x_centered - coarse
        top_acts, top_indices, pre_acts = fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

        assert self.W_dec is not None, "Decoder weight was not initialized."
        sparse_residual = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        sae_out = coarse + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            aux_sparse = decoder_impl(auxk_indices, auxk_acts.to(self.dtype), self.W_dec.mT)
            e_hat = coarse + aux_sparse + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class JumpReLUSparseCoder(SparseCoder):
    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)

        threshold = torch.full(
            (self.num_latents,),
            cfg.jumprelu_init_threshold,
            device=self.device,
            dtype=self.dtype,
        ).clamp_min(torch.finfo(self.dtype).tiny)
        init = torch.full(
            (self.num_latents,), 0.0, device=self.device, dtype=self.dtype
        )
        init.copy_(torch.log(torch.expm1(threshold)))
        self.log_threshold = nn.Parameter(init)

    @property
    def threshold(self) -> Tensor:
        return F.softplus(self.log_threshold)

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        pre_acts = F.linear(x, self.encoder.weight, self.encoder.bias)
        positive = F.relu(pre_acts)
        gate = torch.sigmoid((positive - self.threshold) / self.cfg.jumprelu_bandwidth)
        acts = positive * gate
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)


class GatedSparseCoder(SparseCoder):
    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.gate_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )

        # Start close to plain ReLU top-k, then let the gate branch learn support.
        self.gate_encoder.weight.data.zero_()
        self.gate_encoder.bias.data.fill_(cfg.gated_init_logit)

    def _encoder_linear_layers(self):
        return [("encoder", self.encoder), ("gate_encoder", self.gate_encoder)]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        pre_acts = F.linear(x, self.encoder.weight, self.encoder.bias)
        positive = F.relu(pre_acts)
        gate_logits = self.gate_encoder(x) / self.cfg.gated_temperature
        gate = torch.sigmoid(gate_logits)
        acts = positive * gate
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)


class RoutedSparseCoder(SparseCoder):
    """Gated SAE with a separate router used only for support selection."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.gate_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )

        # Initialize the router independently so routed support is observably
        # different from plain top-k at step 0.
        nn.init.kaiming_uniform_(self.gate_encoder.weight, a=5**0.5)
        self.gate_encoder.bias.data.zero_()

    def _encoder_linear_layers(self):
        return [("encoder", self.encoder), ("gate_encoder", self.gate_encoder)]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        pre_acts = F.linear(x, self.encoder.weight, self.encoder.bias)
        positive = F.relu(pre_acts)
        gate_logits = self.gate_encoder(x) / self.cfg.gated_temperature
        gate = torch.sigmoid(gate_logits)

        # Use the router to perturb support selection directly while decoding
        # with gated magnitudes, so routing is not a monotone rescaling of the
        # base activations.
        acts = positive * gate
        scores = acts + 0.1 * torch.tanh(gate_logits)
        _, top_indices = torch.topk(scores, self.cfg.k, dim=-1, sorted=False)
        top_acts = acts.gather(-1, top_indices)
        return EncoderOutput(top_acts, top_indices, acts)


def _resolve_expert_layout(
    cfg: SparseCoderConfig,
    d_in: int,
) -> tuple[int, int, int, int]:
    base_num_latents = cfg.num_latents or d_in * cfg.expansion_factor
    if cfg.num_experts is None:
        num_experts = 4
        default_latents_per_expert = math.ceil(base_num_latents / num_experts)
    else:
        # Explicit NUM_EXPERTS switches expert families into sparse-capacity
        # expansion mode. LATENTS_PER_EXPERT can then shrink or expand the
        # active-path width independently of the routed expert count.
        num_experts = cfg.num_experts
        default_latents_per_expert = base_num_latents

    latents_per_expert = cfg.latents_per_expert or default_latents_per_expert

    active_experts = cfg.active_experts or 1
    if active_experts > num_experts:
        raise ValueError(
            "active_experts must be <= num_experts, "
            f"got active_experts={active_experts} and num_experts={num_experts}"
        )

    return base_num_latents, num_experts, latents_per_expert, active_experts


def _select_active_expert_indices(
    router_logits: Tensor,
    active_experts: int,
) -> tuple[Tensor, Tensor]:
    router_probs = torch.softmax(router_logits, dim=-1)
    active_probs, active_indices = torch.topk(
        router_probs, active_experts, dim=-1, sorted=False
    )
    norm = active_probs.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(active_probs.dtype).eps
    )
    return active_indices, active_probs / norm


def _select_adaptive_active_expert_indices(
    router_logits: Tensor,
    active_experts: int,
    *,
    easy_share_threshold: float = 0.8,
) -> tuple[Tensor, Tensor]:
    """Use one expert for confident tokens, otherwise keep the full routed set."""
    if active_experts <= 1:
        return _select_active_expert_indices(router_logits, active_experts)

    router_probs = torch.softmax(router_logits, dim=-1)
    active_probs, active_indices = torch.topk(
        router_probs, active_experts, dim=-1, sorted=True
    )
    eps = torch.finfo(active_probs.dtype).eps
    top1 = active_probs[:, :1]
    tail = active_probs[:, 1:].sum(dim=-1, keepdim=True)
    top1_share = top1 / (top1 + tail).clamp_min(eps)
    use_single = top1_share >= easy_share_threshold

    adaptive_probs = active_probs.clone()
    adaptive_probs[:, 1:] = torch.where(
        use_single,
        torch.zeros_like(adaptive_probs[:, 1:]),
        adaptive_probs[:, 1:],
    )
    norm = adaptive_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return active_indices, adaptive_probs / norm


def _finalize_routed_expert_acts(
    acts: Tensor,
    selected_expert_idx: Tensor,
    total_k: int,
    latents_per_expert: int,
    num_latents: int,
    *,
    index_offset: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    flat_acts = acts.reshape(acts.shape[0], -1)
    top_acts, top_pos = torch.topk(flat_acts, total_k, dim=-1, sorted=False)

    local_offsets = torch.arange(
        latents_per_expert, device=acts.device
    ).view(1, 1, latents_per_expert)
    flat_indices = (
        selected_expert_idx.unsqueeze(-1) * latents_per_expert + local_offsets
    ).reshape(acts.shape[0], -1)
    top_indices = flat_indices.gather(1, top_pos) + index_offset

    full_acts = acts.new_zeros(acts.shape[0], num_latents)
    full_acts.scatter_(1, flat_indices, flat_acts)
    return top_acts, top_indices, full_acts


class ExpertTopKSparseCoder(SparseCoder):
    """Route each token to one expert-local dictionary before local top-k."""

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
        (
            _base_num_latents,
            self.num_experts,
            self.latents_per_expert,
            self.active_experts,
        ) = _resolve_expert_layout(cfg, d_in)
        self.num_latents = self.num_experts * self.latents_per_expert
        if cfg.k > self.active_experts * self.latents_per_expert:
            raise ValueError(
                "expert_topk requires k <= active_experts * latents_per_expert, "
                f"got k={cfg.k}, active_experts={self.active_experts}, "
                f"latents_per_expert={self.latents_per_expert}"
            )

        self.router = nn.Linear(d_in, self.num_experts, device=device, dtype=dtype)
        self.expert_encoders = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.latents_per_expert,
                d_in,
                device=device,
                dtype=dtype,
            )
        )
        self.expert_encoder_bias = nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        self.router.bias.data.zero_()
        nn.init.kaiming_uniform_(self.expert_encoders, a=5**0.5)

        if decoder:
            decoder_init = self.expert_encoders.data.reshape(self.num_latents, d_in)
            self.W_dec = nn.Parameter(decoder_init.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("router", self.router)]

    def _extra_encode_accesses(self) -> list[tuple[str, int, str]]:
        return [
            (
                "active_expert_encoder",
                self.active_experts * self.d_in * self.latents_per_expert,
                f"active={self.active_experts}×{self.d_in}x{self.latents_per_expert}",
            )
        ]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        original_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.d_in)

        router_logits = self.router(flat_x)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.expert_encoders[selected_expert_idx]
        selected_bias = self.expert_encoder_bias[selected_expert_idx]
        pre_acts = torch.einsum("bd,bald->bal", flat_x, selected_weight) + selected_bias
        acts = F.relu(pre_acts) * selected_probs.unsqueeze(-1)
        top_acts, top_indices, full_acts = _finalize_routed_expert_acts(
            acts,
            selected_expert_idx,
            self.cfg.k,
            self.latents_per_expert,
            self.num_latents,
        )

        target_shape = (*original_shape, self.cfg.k)
        acts_shape = (*original_shape, self.num_latents)
        return EncoderOutput(
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )


class ExpertJumpReLUSparseCoder(ExpertTopKSparseCoder):
    """Expert routing with JumpReLU-gated local supports inside selected experts."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)

        param_dtype = self.expert_encoder_bias.dtype
        threshold = torch.full(
            (self.num_experts, self.latents_per_expert),
            cfg.jumprelu_init_threshold,
            device=self.expert_encoder_bias.device,
            dtype=param_dtype,
        ).clamp_min(torch.finfo(param_dtype).tiny)
        init = torch.log(torch.expm1(threshold))
        self.log_threshold = nn.Parameter(init)

    @property
    def threshold(self) -> Tensor:
        return F.softplus(self.log_threshold)

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        original_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.d_in)

        router_logits = self.router(flat_x)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.expert_encoders[selected_expert_idx]
        selected_bias = self.expert_encoder_bias[selected_expert_idx]
        selected_threshold = self.threshold[selected_expert_idx]
        pre_acts = torch.einsum("bd,bald->bal", flat_x, selected_weight) + selected_bias
        positive = F.relu(pre_acts)
        gate = torch.sigmoid(
            (positive - selected_threshold) / self.cfg.jumprelu_bandwidth
        )
        acts = positive * gate * selected_probs.unsqueeze(-1)
        top_acts, top_indices, full_acts = _finalize_routed_expert_acts(
            acts,
            selected_expert_idx,
            self.cfg.k,
            self.latents_per_expert,
            self.num_latents,
        )

        target_shape = (*original_shape, self.cfg.k)
        acts_shape = (*original_shape, self.num_latents)
        return EncoderOutput(
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )


class AdaptiveActiveExpertJumpReLUSparseCoder(ExpertJumpReLUSparseCoder):
    """Confidence-routed JumpReLU experts with adaptive 1-or-2 expert paths."""

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        original_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.d_in)

        router_logits = self.router(flat_x)
        selected_expert_idx, selected_probs = _select_adaptive_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.expert_encoders[selected_expert_idx]
        selected_bias = self.expert_encoder_bias[selected_expert_idx]
        selected_threshold = self.threshold[selected_expert_idx]
        pre_acts = torch.einsum("bd,bald->bal", flat_x, selected_weight) + selected_bias
        positive = F.relu(pre_acts)
        gate = torch.sigmoid(
            (positive - selected_threshold) / self.cfg.jumprelu_bandwidth
        )
        acts = positive * gate * selected_probs.unsqueeze(-1)
        top_acts, top_indices, full_acts = _finalize_routed_expert_acts(
            acts,
            selected_expert_idx,
            self.cfg.k,
            self.latents_per_expert,
            self.num_latents,
        )

        target_shape = (*original_shape, self.cfg.k)
        acts_shape = (*original_shape, self.num_latents)
        return EncoderOutput(
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )


class FactorizedRouterExpertTopKSparseCoder(ExpertTopKSparseCoder):
    """Expert-local top-k with a low-rank router but full expert heads."""

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
        (
            _base_num_latents,
            self.num_experts,
            self.latents_per_expert,
            self.active_experts,
        ) = _resolve_expert_layout(cfg, d_in)
        self.num_latents = self.num_experts * self.latents_per_expert
        if cfg.k > self.active_experts * self.latents_per_expert:
            raise ValueError(
                "factorized_router_expert_topk requires "
                "k <= active_experts * latents_per_expert, "
                f"got k={cfg.k}, active_experts={self.active_experts}, "
                f"latents_per_expert={self.latents_per_expert}"
            )

        max_router_hidden = max(32, self.num_experts // 2)
        requested_hidden = (
            cfg.factorized_hidden_dim
            if cfg.factorized_hidden_dim is not None
            else max_router_hidden
        )
        self.router_hidden_dim = min(d_in, max(1, min(requested_hidden, max_router_hidden)))
        self.router_down = nn.Linear(
            d_in, self.router_hidden_dim, device=device, dtype=dtype
        )
        self.router = nn.Linear(
            self.router_hidden_dim, self.num_experts, device=device, dtype=dtype
        )
        self.expert_encoders = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.latents_per_expert,
                d_in,
                device=device,
                dtype=dtype,
            )
        )
        self.expert_encoder_bias = nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.kaiming_uniform_(self.router_down.weight, a=5**0.5)
        self.router_down.bias.data.zero_()
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        self.router.bias.data.zero_()
        nn.init.kaiming_uniform_(self.expert_encoders, a=5**0.5)

        if decoder:
            decoder_init = self.expert_encoders.data.reshape(self.num_latents, d_in)
            self.W_dec = nn.Parameter(decoder_init.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("router_down", self.router_down), ("router", self.router)]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        original_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.d_in)

        router_hidden = F.relu(self.router_down(flat_x))
        router_logits = self.router(router_hidden)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.expert_encoders[selected_expert_idx]
        selected_bias = self.expert_encoder_bias[selected_expert_idx]
        pre_acts = torch.einsum("bd,bald->bal", flat_x, selected_weight) + selected_bias
        acts = F.relu(pre_acts) * selected_probs.unsqueeze(-1)
        top_acts, top_indices, full_acts = _finalize_routed_expert_acts(
            acts,
            selected_expert_idx,
            self.cfg.k,
            self.latents_per_expert,
            self.num_latents,
        )

        target_shape = (*original_shape, self.cfg.k)
        acts_shape = (*original_shape, self.num_latents)
        return EncoderOutput(
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )


class SharedRoutedExpertTopKSparseCoder(SparseCoder):
    """One always-on shared expert plus a small routed expert set."""

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
        (
            _base_num_latents,
            self.num_routed_experts,
            self.latents_per_expert,
            self.active_experts,
        ) = _resolve_expert_layout(cfg, d_in)
        self.num_shared_experts = 1
        self.shared_num_latents = self.num_shared_experts * self.latents_per_expert
        self.routed_num_latents = (
            self.num_routed_experts * self.latents_per_expert
        )
        self.num_latents = self.shared_num_latents + self.routed_num_latents
        max_active_latents = (
            self.num_shared_experts + self.active_experts
        ) * self.latents_per_expert
        if cfg.k > max_active_latents:
            raise ValueError(
                "shared_routed_expert_topk requires "
                "k <= (shared_experts + active_experts) * latents_per_expert, "
                f"got k={cfg.k}, shared_experts={self.num_shared_experts}, "
                f"active_experts={self.active_experts}, "
                f"latents_per_expert={self.latents_per_expert}"
            )

        self.router = nn.Linear(
            d_in, self.num_routed_experts, device=device, dtype=dtype
        )
        self.shared_encoders = nn.Parameter(
            torch.empty(
                self.num_shared_experts,
                self.latents_per_expert,
                d_in,
                device=device,
                dtype=dtype,
            )
        )
        self.shared_encoder_bias = nn.Parameter(
            torch.zeros(
                self.num_shared_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        self.routed_expert_encoders = nn.Parameter(
            torch.empty(
                self.num_routed_experts,
                self.latents_per_expert,
                d_in,
                device=device,
                dtype=dtype,
            )
        )
        self.routed_expert_encoder_bias = nn.Parameter(
            torch.zeros(
                self.num_routed_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        self.router.bias.data.zero_()
        nn.init.kaiming_uniform_(self.shared_encoders, a=5**0.5)
        nn.init.kaiming_uniform_(self.routed_expert_encoders, a=5**0.5)

        if decoder:
            decoder_init = torch.cat(
                (
                    self.shared_encoders.data.reshape(self.shared_num_latents, d_in),
                    self.routed_expert_encoders.data.reshape(
                        self.routed_num_latents, d_in
                    ),
                ),
                dim=0,
            )
            self.W_dec = nn.Parameter(decoder_init.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("router", self.router)]

    def _extra_encode_accesses(self) -> list[tuple[str, int, str]]:
        return [
            (
                "shared_expert_encoder",
                self.num_shared_experts * self.d_in * self.latents_per_expert,
                f"shared={self.num_shared_experts}×{self.d_in}x{self.latents_per_expert}",
            ),
            (
                "active_routed_expert_encoder",
                self.active_experts * self.d_in * self.latents_per_expert,
                f"active={self.active_experts}×{self.d_in}x{self.latents_per_expert}",
            ),
        ]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        original_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.d_in)
        batch = flat_x.shape[0]

        shared_pre_acts = (
            torch.einsum("bd,sld->bsl", flat_x, self.shared_encoders)
            + self.shared_encoder_bias.unsqueeze(0)
        )
        shared_acts = F.relu(shared_pre_acts)

        router_logits = self.router(flat_x)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.routed_expert_encoders[selected_expert_idx]
        selected_bias = self.routed_expert_encoder_bias[selected_expert_idx]
        routed_pre_acts = (
            torch.einsum("bd,bald->bal", flat_x, selected_weight) + selected_bias
        )
        routed_acts = F.relu(routed_pre_acts) * selected_probs.unsqueeze(-1)

        flat_shared_acts = shared_acts.reshape(batch, -1)
        flat_routed_acts = routed_acts.reshape(batch, -1)
        flat_acts = torch.cat((flat_shared_acts, flat_routed_acts), dim=-1)

        shared_offsets = torch.arange(
            self.shared_num_latents, device=flat_x.device
        ).view(1, -1)
        local_offsets = torch.arange(
            self.latents_per_expert, device=flat_x.device
        ).view(1, 1, self.latents_per_expert)
        routed_offsets = (
            self.shared_num_latents
            + selected_expert_idx.unsqueeze(-1) * self.latents_per_expert
            + local_offsets
        ).reshape(batch, -1)
        flat_indices = torch.cat(
            (shared_offsets.expand(batch, -1), routed_offsets), dim=-1
        )

        top_acts, top_pos = torch.topk(flat_acts, self.cfg.k, dim=-1, sorted=False)
        top_indices = flat_indices.gather(1, top_pos)

        full_acts = flat_acts.new_zeros(batch, self.num_latents)
        full_acts.scatter_(1, flat_indices, flat_acts)

        target_shape = (*original_shape, self.cfg.k)
        acts_shape = (*original_shape, self.num_latents)
        return EncoderOutput(
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )


class SharedTwoStageResidualExpertSparseCoder(SparseCoder):
    """Shared sparse coarse stage followed by expert-routed residual cleanup."""

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
        (
            _base_num_latents,
            self.num_experts,
            self.latents_per_expert,
            self.active_experts,
        ) = _resolve_expert_layout(cfg, d_in)
        self.num_shared_experts = 1
        self.shared_num_latents = self.num_shared_experts * self.latents_per_expert
        self.expert_num_latents = self.num_experts * self.latents_per_expert
        self.num_latents = self.shared_num_latents + self.expert_num_latents

        if cfg.stage1_ratio is not None:
            self.stage1_k = max(1, round(cfg.k * cfg.stage1_ratio))
        else:
            self.stage1_k = max(1, cfg.k // 2)
        self.stage2_k = max(1, cfg.k - self.stage1_k)

        if self.stage1_k > self.shared_num_latents:
            raise ValueError(
                "shared_two_stage_residual_expert requires "
                "stage1_k <= shared_experts * latents_per_expert, "
                f"got stage1_k={self.stage1_k}, "
                f"shared_experts={self.num_shared_experts}, "
                f"latents_per_expert={self.latents_per_expert}"
            )
        if self.stage2_k > self.active_experts * self.latents_per_expert:
            raise ValueError(
                "shared_two_stage_residual_expert requires "
                "stage2_k <= active_experts * latents_per_expert, "
                f"got stage2_k={self.stage2_k}, "
                f"active_experts={self.active_experts}, "
                f"latents_per_expert={self.latents_per_expert}"
            )

        self.shared_encoders = nn.Parameter(
            torch.empty(
                self.num_shared_experts,
                self.latents_per_expert,
                d_in,
                device=device,
                dtype=dtype,
            )
        )
        self.shared_encoder_bias = nn.Parameter(
            torch.zeros(
                self.num_shared_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        self.router = nn.Linear(d_in, self.num_experts, device=device, dtype=dtype)
        self.expert_encoders = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.latents_per_expert,
                d_in,
                device=device,
                dtype=dtype,
            )
        )
        self.expert_encoder_bias = nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )

        nn.init.kaiming_uniform_(self.shared_encoders, a=5**0.5)
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        self.router.bias.data.zero_()
        nn.init.kaiming_uniform_(self.expert_encoders, a=5**0.5)

        if decoder:
            shared_decoder = self.shared_encoders.data.reshape(
                self.shared_num_latents, d_in
            )
            expert_decoder = self.expert_encoders.data.reshape(
                self.expert_num_latents, d_in
            )
            decoder_init = torch.cat((shared_decoder, expert_decoder), dim=0)
            self.W_dec = nn.Parameter(decoder_init.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("router", self.router)]

    def _extra_encode_accesses(self) -> list[tuple[str, int, str]]:
        return [
            (
                "shared_expert_encoder",
                self.num_shared_experts * self.d_in * self.latents_per_expert,
                f"shared={self.num_shared_experts}×{self.d_in}x{self.latents_per_expert}",
            ),
            (
                "active_expert_encoder",
                self.active_experts * self.d_in * self.latents_per_expert,
                f"active={self.active_experts}×{self.d_in}x{self.latents_per_expert}",
            ),
        ]

    def _decode_sparse(self, acts: Tensor, indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        return decoder_impl(indices, acts.to(self.dtype), self.W_dec.mT)

    def _encode_shared_stage(
        self, x_centered: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = x_centered.shape[:-1]
        flat_x = x_centered.reshape(-1, self.d_in)
        shared_pre_acts = (
            torch.einsum("bd,sld->bsl", flat_x, self.shared_encoders)
            + self.shared_encoder_bias.unsqueeze(0)
        )
        shared_acts = F.relu(shared_pre_acts).reshape(-1, self.shared_num_latents)

        top_acts, top_indices = torch.topk(
            shared_acts, self.stage1_k, dim=-1, sorted=False
        )
        target_shape = (*original_shape, self.stage1_k)
        acts_shape = (*original_shape, self.shared_num_latents)
        return (
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            shared_acts.reshape(acts_shape),
        )

    def _encode_expert_stage(
        self, residual: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = residual.shape[:-1]
        flat_x = residual.reshape(-1, self.d_in)

        router_logits = self.router(flat_x)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.expert_encoders[selected_expert_idx]
        selected_bias = self.expert_encoder_bias[selected_expert_idx]
        pre_acts = torch.einsum("bd,bald->bal", flat_x, selected_weight) + selected_bias
        acts = F.relu(pre_acts) * selected_probs.unsqueeze(-1)
        top_acts, top_indices, full_acts = _finalize_routed_expert_acts(
            acts,
            selected_expert_idx,
            self.stage2_k,
            self.latents_per_expert,
            self.expert_num_latents,
            index_offset=self.shared_num_latents,
        )

        target_shape = (*original_shape, self.stage2_k)
        acts_shape = (*original_shape, self.expert_num_latents)
        return (
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )

    def encode(self, x: Tensor) -> EncoderOutput:
        x_centered = x - self.b_dec
        stage1_acts, stage1_indices, stage1_full = self._encode_shared_stage(
            x_centered
        )
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        residual = x_centered - stage1_out
        stage2_acts, stage2_indices, stage2_full = self._encode_expert_stage(
            residual
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_full = torch.cat((stage1_full, stage2_full), dim=-1)
        return EncoderOutput(combined_acts, combined_indices, combined_full)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        stage1_acts, stage1_indices, stage1_full = self._encode_shared_stage(
            x_centered
        )
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        residual = x_centered - stage1_out
        stage2_acts, stage2_indices, stage2_full = self._encode_expert_stage(
            residual
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_full = torch.cat((stage1_full, stage2_full), dim=-1)

        sparse_out = self._decode_sparse(combined_acts, combined_indices)
        sae_out = sparse_out + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], combined_full, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = self._decode_sparse(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class FactorizedExpertTopKSparseCoder(SparseCoder):
    """Route into an expert-local top-k head after a shared low-rank projection."""

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
        (
            _base_num_latents,
            self.num_experts,
            self.latents_per_expert,
            self.active_experts,
        ) = _resolve_expert_layout(cfg, d_in)
        self.num_latents = self.num_experts * self.latents_per_expert
        if cfg.k > self.active_experts * self.latents_per_expert:
            raise ValueError(
                "factorized_expert_topk requires k <= active_experts * latents_per_expert, "
                f"got k={cfg.k}, active_experts={self.active_experts}, "
                f"latents_per_expert={self.latents_per_expert}"
            )

        hidden_dim = (
            cfg.factorized_hidden_dim
            if cfg.factorized_hidden_dim is not None
            else min(self.latents_per_expert, max(d_in // 2, cfg.k * 4))
        )
        self.factor_encoder = nn.Linear(
            d_in, hidden_dim, device=device, dtype=dtype
        )
        self.router = nn.Linear(
            hidden_dim, self.num_experts, device=device, dtype=dtype
        )
        self.expert_heads = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.latents_per_expert,
                hidden_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.expert_head_bias = nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        self.factor_encoder.bias.data.zero_()
        self.router.bias.data.zero_()
        nn.init.kaiming_uniform_(self.factor_encoder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.expert_heads, a=5**0.5)

        if decoder:
            effective_encoder = torch.einsum(
                "elh,hd->eld", self.expert_heads.data, self.factor_encoder.weight.data
            ).reshape(self.num_latents, d_in)
            self.W_dec = nn.Parameter(effective_encoder.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("factor_encoder", self.factor_encoder), ("router", self.router)]

    def _extra_encode_accesses(self) -> list[tuple[str, int, str]]:
        return [
            (
                "active_expert_head",
                self.active_experts
                * self.factor_encoder.out_features
                * self.latents_per_expert,
                f"active={self.active_experts}×{self.factor_encoder.out_features}x{self.latents_per_expert}",
            )
        ]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        original_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.d_in)

        hidden = F.relu(self.factor_encoder(flat_x))
        router_logits = self.router(hidden)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.expert_heads[selected_expert_idx]
        selected_bias = self.expert_head_bias[selected_expert_idx]
        pre_acts = torch.einsum("bh,balh->bal", hidden, selected_weight) + selected_bias
        acts = F.relu(pre_acts) * selected_probs.unsqueeze(-1)
        top_acts, top_indices, full_acts = _finalize_routed_expert_acts(
            acts,
            selected_expert_idx,
            self.cfg.k,
            self.latents_per_expert,
            self.num_latents,
        )

        target_shape = (*original_shape, self.cfg.k)
        acts_shape = (*original_shape, self.num_latents)
        return EncoderOutput(
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )


class SharedRoutedFactorizedExpertTopKSparseCoder(SparseCoder):
    """Always-on shared factorized expert plus routed factorized experts."""

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
        (
            _base_num_latents,
            self.num_routed_experts,
            self.latents_per_expert,
            self.active_experts,
        ) = _resolve_expert_layout(cfg, d_in)
        self.num_shared_experts = 1
        self.shared_num_latents = self.num_shared_experts * self.latents_per_expert
        self.routed_num_latents = (
            self.num_routed_experts * self.latents_per_expert
        )
        self.num_latents = self.shared_num_latents + self.routed_num_latents
        max_active_latents = (
            self.num_shared_experts + self.active_experts
        ) * self.latents_per_expert
        if cfg.k > max_active_latents:
            raise ValueError(
                "shared_routed_factorized_expert_topk requires "
                "k <= (shared_experts + active_experts) * latents_per_expert, "
                f"got k={cfg.k}, shared_experts={self.num_shared_experts}, "
                f"active_experts={self.active_experts}, "
                f"latents_per_expert={self.latents_per_expert}"
            )

        hidden_dim = (
            cfg.factorized_hidden_dim
            if cfg.factorized_hidden_dim is not None
            else min(self.latents_per_expert, max(d_in // 2, cfg.k * 4))
        )
        self.factor_encoder = nn.Linear(
            d_in, hidden_dim, device=device, dtype=dtype
        )
        self.router = nn.Linear(
            hidden_dim, self.num_routed_experts, device=device, dtype=dtype
        )
        self.shared_heads = nn.Parameter(
            torch.empty(
                self.num_shared_experts,
                self.latents_per_expert,
                hidden_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.shared_head_bias = nn.Parameter(
            torch.zeros(
                self.num_shared_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        self.routed_heads = nn.Parameter(
            torch.empty(
                self.num_routed_experts,
                self.latents_per_expert,
                hidden_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.routed_head_bias = nn.Parameter(
            torch.zeros(
                self.num_routed_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        self.factor_encoder.bias.data.zero_()
        self.router.bias.data.zero_()
        nn.init.kaiming_uniform_(self.factor_encoder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.shared_heads, a=5**0.5)
        nn.init.kaiming_uniform_(self.routed_heads, a=5**0.5)

        if decoder:
            shared_decoder = torch.einsum(
                "slh,hd->sld",
                self.shared_heads.data,
                self.factor_encoder.weight.data,
            ).reshape(self.shared_num_latents, d_in)
            routed_decoder = torch.einsum(
                "elh,hd->eld",
                self.routed_heads.data,
                self.factor_encoder.weight.data,
            ).reshape(self.routed_num_latents, d_in)
            decoder_init = torch.cat((shared_decoder, routed_decoder), dim=0)
            self.W_dec = nn.Parameter(decoder_init.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("factor_encoder", self.factor_encoder), ("router", self.router)]

    def _extra_encode_accesses(self) -> list[tuple[str, int, str]]:
        hidden_dim = self.factor_encoder.out_features
        return [
            (
                "shared_factorized_head",
                self.num_shared_experts * hidden_dim * self.latents_per_expert,
                f"shared={self.num_shared_experts}×{hidden_dim}x{self.latents_per_expert}",
            ),
            (
                "active_routed_factorized_head",
                self.active_experts * hidden_dim * self.latents_per_expert,
                f"active={self.active_experts}×{hidden_dim}x{self.latents_per_expert}",
            ),
        ]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        original_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.d_in)
        batch = flat_x.shape[0]

        hidden = F.relu(self.factor_encoder(flat_x))
        shared_pre_acts = (
            torch.einsum("bh,slh->bsl", hidden, self.shared_heads)
            + self.shared_head_bias.unsqueeze(0)
        )
        shared_acts = F.relu(shared_pre_acts)

        router_logits = self.router(hidden)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.routed_heads[selected_expert_idx]
        selected_bias = self.routed_head_bias[selected_expert_idx]
        routed_pre_acts = (
            torch.einsum("bh,balh->bal", hidden, selected_weight) + selected_bias
        )
        routed_acts = F.relu(routed_pre_acts) * selected_probs.unsqueeze(-1)

        flat_shared_acts = shared_acts.reshape(batch, -1)
        flat_routed_acts = routed_acts.reshape(batch, -1)
        flat_acts = torch.cat((flat_shared_acts, flat_routed_acts), dim=-1)

        shared_offsets = torch.arange(
            self.shared_num_latents, device=flat_x.device
        ).view(1, -1)
        local_offsets = torch.arange(
            self.latents_per_expert, device=flat_x.device
        ).view(1, 1, self.latents_per_expert)
        routed_offsets = (
            self.shared_num_latents
            + selected_expert_idx.unsqueeze(-1) * self.latents_per_expert
            + local_offsets
        ).reshape(batch, -1)
        flat_indices = torch.cat(
            (shared_offsets.expand(batch, -1), routed_offsets), dim=-1
        )

        top_acts, top_pos = torch.topk(flat_acts, self.cfg.k, dim=-1, sorted=False)
        top_indices = flat_indices.gather(1, top_pos)

        full_acts = flat_acts.new_zeros(batch, self.num_latents)
        full_acts.scatter_(1, flat_indices, flat_acts)

        target_shape = (*original_shape, self.cfg.k)
        acts_shape = (*original_shape, self.num_latents)
        return EncoderOutput(
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )


class SharedLowRankRoutedExpertTopKSparseCoder(SparseCoder):
    """Low-rank shared library plus full routed expert-local sparse heads."""

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
        (
            _base_num_latents,
            self.num_routed_experts,
            self.latents_per_expert,
            self.active_experts,
        ) = _resolve_expert_layout(cfg, d_in)
        self.num_shared_experts = 1
        self.shared_num_latents = self.num_shared_experts * self.latents_per_expert
        self.routed_num_latents = (
            self.num_routed_experts * self.latents_per_expert
        )
        self.num_latents = self.shared_num_latents + self.routed_num_latents
        max_active_latents = (
            self.num_shared_experts + self.active_experts
        ) * self.latents_per_expert
        if cfg.k > max_active_latents:
            raise ValueError(
                "shared_lowrank_routed_expert_topk requires "
                "k <= (shared_experts + active_experts) * latents_per_expert, "
                f"got k={cfg.k}, shared_experts={self.num_shared_experts}, "
                f"active_experts={self.active_experts}, "
                f"latents_per_expert={self.latents_per_expert}"
            )

        hidden_dim = (
            cfg.factorized_hidden_dim
            if cfg.factorized_hidden_dim is not None
            else min(self.latents_per_expert, max(d_in // 2, cfg.k * 4))
        )
        self.factor_encoder = nn.Linear(
            d_in, hidden_dim, device=device, dtype=dtype
        )
        self.router = nn.Linear(
            hidden_dim, self.num_routed_experts, device=device, dtype=dtype
        )
        self.shared_heads = nn.Parameter(
            torch.empty(
                self.num_shared_experts,
                self.latents_per_expert,
                hidden_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.shared_head_bias = nn.Parameter(
            torch.zeros(
                self.num_shared_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        self.routed_expert_encoders = nn.Parameter(
            torch.empty(
                self.num_routed_experts,
                self.latents_per_expert,
                d_in,
                device=device,
                dtype=dtype,
            )
        )
        self.routed_expert_encoder_bias = nn.Parameter(
            torch.zeros(
                self.num_routed_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        self.factor_encoder.bias.data.zero_()
        self.router.bias.data.zero_()
        nn.init.kaiming_uniform_(self.factor_encoder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.shared_heads, a=5**0.5)
        nn.init.kaiming_uniform_(self.routed_expert_encoders, a=5**0.5)

        if decoder:
            shared_decoder = torch.einsum(
                "slh,hd->sld",
                self.shared_heads.data,
                self.factor_encoder.weight.data,
            ).reshape(self.shared_num_latents, d_in)
            routed_decoder = self.routed_expert_encoders.data.reshape(
                self.routed_num_latents, d_in
            )
            decoder_init = torch.cat((shared_decoder, routed_decoder), dim=0)
            self.W_dec = nn.Parameter(decoder_init.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("factor_encoder", self.factor_encoder), ("router", self.router)]

    def _extra_encode_accesses(self) -> list[tuple[str, int, str]]:
        hidden_dim = self.factor_encoder.out_features
        return [
            (
                "shared_lowrank_head",
                self.num_shared_experts * hidden_dim * self.latents_per_expert,
                f"shared={self.num_shared_experts}×{hidden_dim}x{self.latents_per_expert}",
            ),
            (
                "active_routed_expert_encoder",
                self.active_experts * self.d_in * self.latents_per_expert,
                f"active={self.active_experts}×{self.d_in}x{self.latents_per_expert}",
            ),
        ]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        original_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.d_in)
        batch = flat_x.shape[0]

        hidden = F.relu(self.factor_encoder(flat_x))
        shared_pre_acts = (
            torch.einsum("bh,slh->bsl", hidden, self.shared_heads)
            + self.shared_head_bias.unsqueeze(0)
        )
        shared_acts = F.relu(shared_pre_acts)

        router_logits = self.router(hidden)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.routed_expert_encoders[selected_expert_idx]
        selected_bias = self.routed_expert_encoder_bias[selected_expert_idx]
        routed_pre_acts = (
            torch.einsum("bd,bald->bal", flat_x, selected_weight) + selected_bias
        )
        routed_acts = F.relu(routed_pre_acts) * selected_probs.unsqueeze(-1)

        flat_shared_acts = shared_acts.reshape(batch, -1)
        flat_routed_acts = routed_acts.reshape(batch, -1)
        flat_acts = torch.cat((flat_shared_acts, flat_routed_acts), dim=-1)

        shared_offsets = torch.arange(
            self.shared_num_latents, device=flat_x.device
        ).view(1, -1)
        local_offsets = torch.arange(
            self.latents_per_expert, device=flat_x.device
        ).view(1, 1, self.latents_per_expert)
        routed_offsets = (
            self.shared_num_latents
            + selected_expert_idx.unsqueeze(-1) * self.latents_per_expert
            + local_offsets
        ).reshape(batch, -1)
        flat_indices = torch.cat(
            (shared_offsets.expand(batch, -1), routed_offsets), dim=-1
        )

        top_acts, top_pos = torch.topk(flat_acts, self.cfg.k, dim=-1, sorted=False)
        top_indices = flat_indices.gather(1, top_pos)

        full_acts = flat_acts.new_zeros(batch, self.num_latents)
        full_acts.scatter_(1, flat_indices, flat_acts)

        target_shape = (*original_shape, self.cfg.k)
        acts_shape = (*original_shape, self.num_latents)
        return EncoderOutput(
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )


class SharedRoutedFactorizedExpertResidualSparseCoder(SparseCoder):
    """Shared+routed factorized expert stage followed by global sparse residual cleanup."""

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
        (
            base_num_latents,
            self.num_routed_experts,
            self.latents_per_expert,
            self.active_experts,
        ) = _resolve_expert_layout(cfg, d_in)
        self.num_shared_experts = 1
        self.shared_num_latents = self.num_shared_experts * self.latents_per_expert
        self.routed_num_latents = (
            self.num_routed_experts * self.latents_per_expert
        )
        self.stage1_num_latents = self.shared_num_latents + self.routed_num_latents
        self.residual_num_latents = base_num_latents
        self.num_latents = self.stage1_num_latents + self.residual_num_latents

        if cfg.stage1_ratio is not None:
            self.stage1_k = max(1, round(cfg.k * cfg.stage1_ratio))
        else:
            self.stage1_k = max(1, cfg.k // 2)
        self.stage2_k = max(1, cfg.k - self.stage1_k)

        max_stage1_latents = (
            self.num_shared_experts + self.active_experts
        ) * self.latents_per_expert
        if self.stage1_k > max_stage1_latents:
            raise ValueError(
                "shared_routed_factorized_expert_residual requires "
                "stage1_k <= (shared_experts + active_experts) * latents_per_expert, "
                f"got stage1_k={self.stage1_k}, "
                f"shared_experts={self.num_shared_experts}, "
                f"active_experts={self.active_experts}, "
                f"latents_per_expert={self.latents_per_expert}"
            )

        hidden_dim = (
            cfg.factorized_hidden_dim
            if cfg.factorized_hidden_dim is not None
            else min(self.latents_per_expert, max(d_in // 2, cfg.k * 4))
        )
        self.factor_encoder = nn.Linear(
            d_in, hidden_dim, device=device, dtype=dtype
        )
        self.router = nn.Linear(
            hidden_dim, self.num_routed_experts, device=device, dtype=dtype
        )
        self.shared_heads = nn.Parameter(
            torch.empty(
                self.num_shared_experts,
                self.latents_per_expert,
                hidden_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.shared_head_bias = nn.Parameter(
            torch.zeros(
                self.num_shared_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        self.routed_heads = nn.Parameter(
            torch.empty(
                self.num_routed_experts,
                self.latents_per_expert,
                hidden_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.routed_head_bias = nn.Parameter(
            torch.zeros(
                self.num_routed_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        self.residual_encoder = nn.Linear(
            d_in, self.residual_num_latents, device=device, dtype=dtype
        )

        self.factor_encoder.bias.data.zero_()
        self.router.bias.data.zero_()
        self.residual_encoder.bias.data.zero_()
        nn.init.kaiming_uniform_(self.factor_encoder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.shared_heads, a=5**0.5)
        nn.init.kaiming_uniform_(self.routed_heads, a=5**0.5)

        if decoder:
            shared_decoder = torch.einsum(
                "slh,hd->sld",
                self.shared_heads.data,
                self.factor_encoder.weight.data,
            ).reshape(self.shared_num_latents, d_in)
            routed_decoder = torch.einsum(
                "elh,hd->eld",
                self.routed_heads.data,
                self.factor_encoder.weight.data,
            ).reshape(self.routed_num_latents, d_in)
            decoder_init = torch.cat(
                (
                    shared_decoder,
                    routed_decoder,
                    self.residual_encoder.weight.data.clone(),
                ),
                dim=0,
            )
            self.W_dec = nn.Parameter(decoder_init)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [
            ("factor_encoder", self.factor_encoder),
            ("router", self.router),
            ("residual_encoder", self.residual_encoder),
        ]

    def _extra_encode_accesses(self) -> list[tuple[str, int, str]]:
        hidden_dim = self.factor_encoder.out_features
        return [
            (
                "shared_factorized_head",
                self.num_shared_experts * hidden_dim * self.latents_per_expert,
                f"shared={self.num_shared_experts}×{hidden_dim}x{self.latents_per_expert}",
            ),
            (
                "active_routed_factorized_head",
                self.active_experts * hidden_dim * self.latents_per_expert,
                f"active={self.active_experts}×{hidden_dim}x{self.latents_per_expert}",
            ),
        ]

    def _decode_sparse(self, acts: Tensor, indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        return decoder_impl(indices, acts.to(self.dtype), self.W_dec.mT)

    def _deployment_lookup_accesses(self, n_output):
        return [
            (
                "sparse_lookup",
                self._deploy_library_accesses(self.cfg.k, n_output),
                self._deploy_library_shape(self.cfg.k, n_output, label=f"K={self.cfg.k}"),
            ),
        ]

    def _encode_stage1(
        self, x_centered: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = x_centered.shape[:-1]
        flat_x = x_centered.reshape(-1, self.d_in)
        batch = flat_x.shape[0]

        hidden = F.relu(self.factor_encoder(flat_x))
        shared_pre_acts = (
            torch.einsum("bh,slh->bsl", hidden, self.shared_heads)
            + self.shared_head_bias.unsqueeze(0)
        )
        shared_acts = F.relu(shared_pre_acts)

        router_logits = self.router(hidden)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.routed_heads[selected_expert_idx]
        selected_bias = self.routed_head_bias[selected_expert_idx]
        routed_pre_acts = (
            torch.einsum("bh,balh->bal", hidden, selected_weight) + selected_bias
        )
        routed_acts = F.relu(routed_pre_acts) * selected_probs.unsqueeze(-1)

        flat_shared_acts = shared_acts.reshape(batch, -1)
        flat_routed_acts = routed_acts.reshape(batch, -1)
        flat_acts = torch.cat((flat_shared_acts, flat_routed_acts), dim=-1)

        shared_offsets = torch.arange(
            self.shared_num_latents, device=flat_x.device
        ).view(1, -1)
        local_offsets = torch.arange(
            self.latents_per_expert, device=flat_x.device
        ).view(1, 1, self.latents_per_expert)
        routed_offsets = (
            self.shared_num_latents
            + selected_expert_idx.unsqueeze(-1) * self.latents_per_expert
            + local_offsets
        ).reshape(batch, -1)
        flat_indices = torch.cat(
            (shared_offsets.expand(batch, -1), routed_offsets), dim=-1
        )

        top_acts, top_pos = torch.topk(
            flat_acts, self.stage1_k, dim=-1, sorted=False
        )
        top_indices = flat_indices.gather(1, top_pos)

        full_acts = flat_acts.new_zeros(batch, self.stage1_num_latents)
        full_acts.scatter_(1, flat_indices, flat_acts)

        target_shape = (*original_shape, self.stage1_k)
        acts_shape = (*original_shape, self.stage1_num_latents)
        return (
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )

    def _encode_residual_stage(
        self, residual: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = residual.shape[:-1]
        flat_x = residual.reshape(-1, self.d_in)
        pre_acts = F.linear(
            flat_x, self.residual_encoder.weight, self.residual_encoder.bias
        )
        acts = F.relu(pre_acts)
        top_acts, local_indices = torch.topk(
            acts, self.stage2_k, dim=-1, sorted=False
        )
        top_indices = local_indices + self.stage1_num_latents

        target_shape = (*original_shape, self.stage2_k)
        acts_shape = (*original_shape, self.residual_num_latents)
        return (
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            acts.reshape(acts_shape),
        )

    def encode(self, x: Tensor) -> EncoderOutput:
        x_centered = x - self.b_dec
        stage1_acts, stage1_indices, stage1_full = self._encode_stage1(x_centered)
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        stage2_input = x_centered - stage1_out
        stage2_acts, stage2_indices, stage2_full = self._encode_residual_stage(
            stage2_input
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_full = torch.cat((stage1_full, stage2_full), dim=-1)
        return EncoderOutput(combined_acts, combined_indices, combined_full)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        stage1_acts, stage1_indices, stage1_full = self._encode_stage1(x_centered)
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        stage2_input = x_centered - stage1_out
        stage2_acts, stage2_indices, stage2_full = self._encode_residual_stage(
            stage2_input
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_full = torch.cat((stage1_full, stage2_full), dim=-1)

        sparse_out = self._decode_sparse(combined_acts, combined_indices)
        sae_out = sparse_out + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], combined_full, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = self._decode_sparse(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class SharedLowRankTwoStageResidualExpertSparseCoder(SparseCoder):
    """Shared low-rank coarse stage followed by expert-routed residual cleanup."""

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
        (
            _base_num_latents,
            self.num_experts,
            self.latents_per_expert,
            self.active_experts,
        ) = _resolve_expert_layout(cfg, d_in)
        self.num_shared_experts = 1
        self.shared_num_latents = self.num_shared_experts * self.latents_per_expert
        self.expert_num_latents = self.num_experts * self.latents_per_expert
        self.num_latents = self.shared_num_latents + self.expert_num_latents

        if cfg.stage1_ratio is not None:
            self.stage1_k = max(1, round(cfg.k * cfg.stage1_ratio))
        else:
            self.stage1_k = max(1, cfg.k // 2)
        self.stage2_k = max(1, cfg.k - self.stage1_k)

        if self.stage1_k > self.shared_num_latents:
            raise ValueError(
                "shared_lowrank_two_stage_residual_expert requires "
                "stage1_k <= shared_experts * latents_per_expert, "
                f"got stage1_k={self.stage1_k}, "
                f"shared_experts={self.num_shared_experts}, "
                f"latents_per_expert={self.latents_per_expert}"
            )
        if self.stage2_k > self.active_experts * self.latents_per_expert:
            raise ValueError(
                "shared_lowrank_two_stage_residual_expert requires "
                "stage2_k <= active_experts * latents_per_expert, "
                f"got stage2_k={self.stage2_k}, "
                f"active_experts={self.active_experts}, "
                f"latents_per_expert={self.latents_per_expert}"
            )

        hidden_dim = (
            cfg.factorized_hidden_dim
            if cfg.factorized_hidden_dim is not None
            else min(self.latents_per_expert, max(d_in // 2, cfg.k * 4))
        )
        self.factor_encoder = nn.Linear(
            d_in, hidden_dim, device=device, dtype=dtype
        )
        self.shared_heads = nn.Parameter(
            torch.empty(
                self.num_shared_experts,
                self.latents_per_expert,
                hidden_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.shared_head_bias = nn.Parameter(
            torch.zeros(
                self.num_shared_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        self.router = nn.Linear(
            hidden_dim, self.num_experts, device=device, dtype=dtype
        )
        self.expert_encoders = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.latents_per_expert,
                d_in,
                device=device,
                dtype=dtype,
            )
        )
        self.expert_encoder_bias = nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )

        self.factor_encoder.bias.data.zero_()
        self.router.bias.data.zero_()
        nn.init.kaiming_uniform_(self.factor_encoder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.shared_heads, a=5**0.5)
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.expert_encoders, a=5**0.5)

        if decoder:
            shared_decoder = torch.einsum(
                "slh,hd->sld",
                self.shared_heads.data,
                self.factor_encoder.weight.data,
            ).reshape(self.shared_num_latents, d_in)
            expert_decoder = self.expert_encoders.data.reshape(
                self.expert_num_latents, d_in
            )
            decoder_init = torch.cat((shared_decoder, expert_decoder), dim=0)
            self.W_dec = nn.Parameter(decoder_init.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("factor_encoder", self.factor_encoder), ("router", self.router)]

    def _extra_encode_accesses(self) -> list[tuple[str, int, str]]:
        hidden_dim = self.factor_encoder.out_features
        return [
            (
                "shared_lowrank_head",
                self.num_shared_experts * hidden_dim * self.latents_per_expert,
                f"shared={self.num_shared_experts}×{hidden_dim}x{self.latents_per_expert}",
            ),
            (
                "active_expert_encoder",
                self.active_experts * self.d_in * self.latents_per_expert,
                f"active={self.active_experts}×{self.d_in}x{self.latents_per_expert}",
            ),
        ]

    def _decode_sparse(self, acts: Tensor, indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        return decoder_impl(indices, acts.to(self.dtype), self.W_dec.mT)

    def _deployment_lookup_accesses(self, n_output):
        return [
            (
                "sparse_lookup",
                self._deploy_library_accesses(self.cfg.k, n_output),
                self._deploy_library_shape(self.cfg.k, n_output, label=f"K={self.cfg.k}"),
            ),
        ]

    def _encode_shared_stage(
        self, x_centered: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        original_shape = x_centered.shape[:-1]
        flat_x = x_centered.reshape(-1, self.d_in)
        hidden = F.relu(self.factor_encoder(flat_x))
        shared_pre_acts = (
            torch.einsum("bh,slh->bsl", hidden, self.shared_heads)
            + self.shared_head_bias.unsqueeze(0)
        )
        shared_acts = F.relu(shared_pre_acts).reshape(-1, self.shared_num_latents)

        top_acts, top_indices = torch.topk(
            shared_acts, self.stage1_k, dim=-1, sorted=False
        )
        target_shape = (*original_shape, self.stage1_k)
        acts_shape = (*original_shape, self.shared_num_latents)
        return (
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            shared_acts.reshape(acts_shape),
            hidden,
        )

    def _encode_expert_stage(
        self, residual: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = residual.shape[:-1]
        flat_residual = residual.reshape(-1, self.d_in)

        router_logits = self.router(hidden)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.expert_encoders[selected_expert_idx]
        selected_bias = self.expert_encoder_bias[selected_expert_idx]
        pre_acts = (
            torch.einsum("bd,bald->bal", flat_residual, selected_weight)
            + selected_bias
        )
        acts = F.relu(pre_acts) * selected_probs.unsqueeze(-1)
        top_acts, top_indices, full_acts = _finalize_routed_expert_acts(
            acts,
            selected_expert_idx,
            self.stage2_k,
            self.latents_per_expert,
            self.expert_num_latents,
            index_offset=self.shared_num_latents,
        )

        target_shape = (*original_shape, self.stage2_k)
        acts_shape = (*original_shape, self.expert_num_latents)
        return (
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )

    def encode(self, x: Tensor) -> EncoderOutput:
        x_centered = x - self.b_dec
        stage1_acts, stage1_indices, stage1_full, hidden = self._encode_shared_stage(
            x_centered
        )
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        residual = x_centered - stage1_out
        stage2_acts, stage2_indices, stage2_full = self._encode_expert_stage(
            residual, hidden
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_full = torch.cat((stage1_full, stage2_full), dim=-1)
        return EncoderOutput(combined_acts, combined_indices, combined_full)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        stage1_acts, stage1_indices, stage1_full, hidden = self._encode_shared_stage(
            x_centered
        )
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        residual = x_centered - stage1_out
        stage2_acts, stage2_indices, stage2_full = self._encode_expert_stage(
            residual, hidden
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_full = torch.cat((stage1_full, stage2_full), dim=-1)

        sparse_out = self._decode_sparse(combined_acts, combined_indices)
        sae_out = sparse_out + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], combined_full, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = self._decode_sparse(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class SharedLowRankRoutedExpertResidualSparseCoder(SparseCoder):
    """Shared low-rank expert stage followed by global sparse residual cleanup."""

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
        (
            base_num_latents,
            self.num_routed_experts,
            self.latents_per_expert,
            self.active_experts,
        ) = _resolve_expert_layout(cfg, d_in)
        self.num_shared_experts = 1
        self.shared_num_latents = self.num_shared_experts * self.latents_per_expert
        self.routed_num_latents = (
            self.num_routed_experts * self.latents_per_expert
        )
        self.stage1_num_latents = self.shared_num_latents + self.routed_num_latents
        self.residual_num_latents = base_num_latents
        self.num_latents = self.stage1_num_latents + self.residual_num_latents

        if cfg.stage1_ratio is not None:
            self.stage1_k = max(1, round(cfg.k * cfg.stage1_ratio))
        else:
            self.stage1_k = max(1, cfg.k // 2)
        self.stage2_k = max(1, cfg.k - self.stage1_k)

        max_stage1_latents = (
            self.num_shared_experts + self.active_experts
        ) * self.latents_per_expert
        if self.stage1_k > max_stage1_latents:
            raise ValueError(
                "shared_lowrank_routed_expert_residual requires "
                "stage1_k <= (shared_experts + active_experts) * latents_per_expert, "
                f"got stage1_k={self.stage1_k}, "
                f"shared_experts={self.num_shared_experts}, "
                f"active_experts={self.active_experts}, "
                f"latents_per_expert={self.latents_per_expert}"
            )

        hidden_dim = (
            cfg.factorized_hidden_dim
            if cfg.factorized_hidden_dim is not None
            else min(self.latents_per_expert, max(d_in // 2, cfg.k * 4))
        )
        self.factor_encoder = nn.Linear(
            d_in, hidden_dim, device=device, dtype=dtype
        )
        self.router = nn.Linear(
            hidden_dim, self.num_routed_experts, device=device, dtype=dtype
        )
        self.shared_heads = nn.Parameter(
            torch.empty(
                self.num_shared_experts,
                self.latents_per_expert,
                hidden_dim,
                device=device,
                dtype=dtype,
            )
        )
        self.shared_head_bias = nn.Parameter(
            torch.zeros(
                self.num_shared_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        self.routed_expert_encoders = nn.Parameter(
            torch.empty(
                self.num_routed_experts,
                self.latents_per_expert,
                d_in,
                device=device,
                dtype=dtype,
            )
        )
        self.routed_expert_encoder_bias = nn.Parameter(
            torch.zeros(
                self.num_routed_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        self.residual_encoder = nn.Linear(
            d_in, self.residual_num_latents, device=device, dtype=dtype
        )
        self.factor_encoder.bias.data.zero_()
        self.router.bias.data.zero_()
        self.residual_encoder.bias.data.zero_()
        nn.init.kaiming_uniform_(self.factor_encoder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.shared_heads, a=5**0.5)
        nn.init.kaiming_uniform_(self.routed_expert_encoders, a=5**0.5)

        if decoder:
            shared_decoder = torch.einsum(
                "slh,hd->sld",
                self.shared_heads.data,
                self.factor_encoder.weight.data,
            ).reshape(self.shared_num_latents, d_in)
            routed_decoder = self.routed_expert_encoders.data.reshape(
                self.routed_num_latents, d_in
            )
            decoder_init = torch.cat(
                (
                    shared_decoder,
                    routed_decoder,
                    self.residual_encoder.weight.data.clone(),
                ),
                dim=0,
            )
            self.W_dec = nn.Parameter(decoder_init)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [
            ("factor_encoder", self.factor_encoder),
            ("router", self.router),
            ("residual_encoder", self.residual_encoder),
        ]

    def _extra_encode_accesses(self) -> list[tuple[str, int, str]]:
        hidden_dim = self.factor_encoder.out_features
        return [
            (
                "shared_lowrank_head",
                self.num_shared_experts * hidden_dim * self.latents_per_expert,
                f"shared={self.num_shared_experts}×{hidden_dim}x{self.latents_per_expert}",
            ),
            (
                "active_routed_expert_encoder",
                self.active_experts * self.d_in * self.latents_per_expert,
                f"active={self.active_experts}×{self.d_in}x{self.latents_per_expert}",
            ),
        ]

    def _decode_sparse(self, acts: Tensor, indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        return decoder_impl(indices, acts.to(self.dtype), self.W_dec.mT)

    def _deployment_lookup_accesses(self, n_output):
        return [
            (
                "sparse_lookup",
                self._deploy_library_accesses(self.cfg.k, n_output),
                self._deploy_library_shape(self.cfg.k, n_output, label=f"K={self.cfg.k}"),
            ),
        ]

    def _encode_stage1(
        self, x_centered: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = x_centered.shape[:-1]
        flat_x = x_centered.reshape(-1, self.d_in)
        batch = flat_x.shape[0]

        hidden = F.relu(self.factor_encoder(flat_x))
        shared_pre_acts = (
            torch.einsum("bh,slh->bsl", hidden, self.shared_heads)
            + self.shared_head_bias.unsqueeze(0)
        )
        shared_acts = F.relu(shared_pre_acts)

        router_logits = self.router(hidden)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.routed_expert_encoders[selected_expert_idx]
        selected_bias = self.routed_expert_encoder_bias[selected_expert_idx]
        routed_pre_acts = (
            torch.einsum("bd,bald->bal", flat_x, selected_weight) + selected_bias
        )
        routed_acts = F.relu(routed_pre_acts) * selected_probs.unsqueeze(-1)

        flat_shared_acts = shared_acts.reshape(batch, -1)
        flat_routed_acts = routed_acts.reshape(batch, -1)
        flat_acts = torch.cat((flat_shared_acts, flat_routed_acts), dim=-1)

        shared_offsets = torch.arange(
            self.shared_num_latents, device=flat_x.device
        ).view(1, -1)
        local_offsets = torch.arange(
            self.latents_per_expert, device=flat_x.device
        ).view(1, 1, self.latents_per_expert)
        routed_offsets = (
            self.shared_num_latents
            + selected_expert_idx.unsqueeze(-1) * self.latents_per_expert
            + local_offsets
        ).reshape(batch, -1)
        flat_indices = torch.cat(
            (shared_offsets.expand(batch, -1), routed_offsets), dim=-1
        )

        top_acts, top_pos = torch.topk(
            flat_acts, self.stage1_k, dim=-1, sorted=False
        )
        top_indices = flat_indices.gather(1, top_pos)

        full_acts = flat_acts.new_zeros(batch, self.stage1_num_latents)
        full_acts.scatter_(1, flat_indices, flat_acts)

        target_shape = (*original_shape, self.stage1_k)
        acts_shape = (*original_shape, self.stage1_num_latents)
        return (
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )

    def _encode_residual_stage(
        self, residual: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = residual.shape[:-1]
        flat_x = residual.reshape(-1, self.d_in)
        pre_acts = F.linear(
            flat_x, self.residual_encoder.weight, self.residual_encoder.bias
        )
        acts = F.relu(pre_acts)
        top_acts, local_indices = torch.topk(
            acts, self.stage2_k, dim=-1, sorted=False
        )
        top_indices = local_indices + self.stage1_num_latents

        target_shape = (*original_shape, self.stage2_k)
        acts_shape = (*original_shape, self.residual_num_latents)
        return (
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            acts.reshape(acts_shape),
        )

    def encode(self, x: Tensor) -> EncoderOutput:
        x_centered = x - self.b_dec
        stage1_acts, stage1_indices, stage1_full = self._encode_stage1(x_centered)
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        stage2_input = x_centered - stage1_out
        stage2_acts, stage2_indices, stage2_full = self._encode_residual_stage(
            stage2_input
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_full = torch.cat((stage1_full, stage2_full), dim=-1)
        return EncoderOutput(combined_acts, combined_indices, combined_full)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        stage1_acts, stage1_indices, stage1_full = self._encode_stage1(x_centered)
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        stage2_input = x_centered - stage1_out
        stage2_acts, stage2_indices, stage2_full = self._encode_residual_stage(
            stage2_input
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_full = torch.cat((stage1_full, stage2_full), dim=-1)

        sparse_out = self._decode_sparse(combined_acts, combined_indices)
        sae_out = sparse_out + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], combined_full, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = self._decode_sparse(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class LowRankExpertTopKSparseCoder(SparseCoder):
    """Low-rank trunk followed by expert-local sparse residual selection."""

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
        (
            _base_num_latents,
            self.num_experts,
            self.latents_per_expert,
            self.active_experts,
        ) = _resolve_expert_layout(cfg, d_in)
        self.num_latents = self.num_experts * self.latents_per_expert
        if cfg.k > self.active_experts * self.latents_per_expert:
            raise ValueError(
                "lowrank_expert_topk requires k <= active_experts * latents_per_expert, "
                f"got k={cfg.k}, active_experts={self.active_experts}, "
                f"latents_per_expert={self.latents_per_expert}"
            )

        trunk_rank = (
            cfg.trunk_rank
            if cfg.trunk_rank is not None
            else min(d_in, max(cfg.k * 2, d_in // 4))
        )
        self.trunk_encoder = nn.Linear(
            d_in, trunk_rank, bias=False, device=device, dtype=dtype
        )
        self.trunk_decoder = nn.Linear(
            trunk_rank, d_in, bias=False, device=device, dtype=dtype
        )
        self.router = nn.Linear(d_in, self.num_experts, device=device, dtype=dtype)
        self.expert_encoders = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.latents_per_expert,
                d_in,
                device=device,
                dtype=dtype,
            )
        )
        self.expert_encoder_bias = nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )

        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        self.router.bias.data.zero_()
        nn.init.kaiming_uniform_(self.expert_encoders, a=5**0.5)

        if decoder:
            decoder_init = self.expert_encoders.data.reshape(self.num_latents, d_in)
            self.W_dec = nn.Parameter(decoder_init.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [
            ("trunk_encoder", self.trunk_encoder),
            ("trunk_decoder", self.trunk_decoder),
            ("router", self.router),
        ]

    def _extra_encode_accesses(self) -> list[tuple[str, int, str]]:
        return [
            (
                "active_expert_encoder",
                self.active_experts * self.d_in * self.latents_per_expert,
                f"active={self.active_experts}×{self.d_in}x{self.latents_per_expert}",
            )
        ]

    def _deployment_lookup_accesses(self, n_output):
        return [
            (
                "trunk_deploy",
                self._deploy_library_accesses(self.trunk_encoder.out_features, n_output),
                self._deploy_library_shape(
                    self.trunk_encoder.out_features,
                    n_output,
                    label=f"r={self.trunk_encoder.out_features}",
                ),
            ),
            (
                "sparse_lookup",
                self._deploy_library_accesses(self.cfg.k, n_output),
                self._deploy_library_shape(self.cfg.k, n_output, label=f"K={self.cfg.k}"),
            ),
        ]

    def _encode_residual(
        self, residual: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = residual.shape[:-1]
        flat_x = residual.reshape(-1, self.d_in)

        router_logits = self.router(flat_x)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.expert_encoders[selected_expert_idx]
        selected_bias = self.expert_encoder_bias[selected_expert_idx]
        pre_acts = torch.einsum("bd,bald->bal", flat_x, selected_weight) + selected_bias
        acts = F.relu(pre_acts) * selected_probs.unsqueeze(-1)
        top_acts, top_indices, full_acts = _finalize_routed_expert_acts(
            acts,
            selected_expert_idx,
            self.cfg.k,
            self.latents_per_expert,
            self.num_latents,
        )

        target_shape = (*original_shape, self.cfg.k)
        acts_shape = (*original_shape, self.num_latents)
        return (
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        return EncoderOutput(*self._encode_residual(residual))

    def decode_residual(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        return decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        top_acts, top_indices, full_acts = self._encode_residual(residual)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], full_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankExpertJumpReLUSparseCoder(LowRankExpertTopKSparseCoder):
    """Low-rank trunk with JumpReLU-smoothed routed expert residuals."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)

        param_dtype = self.expert_encoder_bias.dtype
        threshold = torch.full(
            (self.num_experts, self.latents_per_expert),
            cfg.jumprelu_init_threshold,
            device=self.expert_encoder_bias.device,
            dtype=param_dtype,
        ).clamp_min(torch.finfo(param_dtype).tiny)
        init = torch.log(torch.expm1(threshold))
        self.log_threshold = nn.Parameter(init)

    @property
    def threshold(self) -> Tensor:
        return F.softplus(self.log_threshold)

    def _encode_residual(
        self, residual: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = residual.shape[:-1]
        flat_x = residual.reshape(-1, self.d_in)

        router_logits = self.router(flat_x)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.expert_encoders[selected_expert_idx]
        selected_bias = self.expert_encoder_bias[selected_expert_idx]
        selected_threshold = self.threshold[selected_expert_idx]
        pre_acts = torch.einsum("bd,bald->bal", flat_x, selected_weight) + selected_bias
        positive = F.relu(pre_acts)
        gate = torch.sigmoid(
            (positive - selected_threshold) / self.cfg.jumprelu_bandwidth
        )
        acts = positive * gate * selected_probs.unsqueeze(-1)
        top_acts, top_indices, full_acts = _finalize_routed_expert_acts(
            acts,
            selected_expert_idx,
            self.cfg.k,
            self.latents_per_expert,
            self.num_latents,
        )

        target_shape = (*original_shape, self.cfg.k)
        acts_shape = (*original_shape, self.num_latents)
        return (
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )


class LowRankExpertResidualSparseCoder(SparseCoder):
    """Low-rank trunk, expert-local sparse pass, then global residual cleanup."""

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
        (
            base_num_latents,
            self.num_experts,
            self.latents_per_expert,
            self.active_experts,
        ) = _resolve_expert_layout(cfg, d_in)

        self.expert_num_latents = self.num_experts * self.latents_per_expert
        self.residual_num_latents = base_num_latents
        self.num_latents = self.expert_num_latents + self.residual_num_latents

        if cfg.stage1_ratio is not None:
            self.stage1_k = max(1, round(cfg.k * cfg.stage1_ratio))
        else:
            self.stage1_k = max(1, cfg.k // 2)
        self.stage2_k = max(1, cfg.k - self.stage1_k)

        if self.stage1_k > self.active_experts * self.latents_per_expert:
            raise ValueError(
                "lowrank_expert_residual requires stage1_k <= active_experts * latents_per_expert, "
                f"got stage1_k={self.stage1_k} and "
                f"active_experts={self.active_experts} and "
                f"latents_per_expert={self.latents_per_expert}"
            )

        trunk_rank = (
            cfg.trunk_rank
            if cfg.trunk_rank is not None
            else min(d_in, max(cfg.k * 2, d_in // 4))
        )
        self.trunk_encoder = nn.Linear(
            d_in, trunk_rank, bias=False, device=device, dtype=dtype
        )
        self.trunk_decoder = nn.Linear(
            trunk_rank, d_in, bias=False, device=device, dtype=dtype
        )
        self.router = nn.Linear(d_in, self.num_experts, device=device, dtype=dtype)
        self.expert_encoders = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.latents_per_expert,
                d_in,
                device=device,
                dtype=dtype,
            )
        )
        self.expert_encoder_bias = nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )
        self.residual_encoder = nn.Linear(
            d_in, self.residual_num_latents, device=device, dtype=dtype
        )
        self.residual_encoder.bias.data.zero_()

        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        self.router.bias.data.zero_()
        nn.init.kaiming_uniform_(self.expert_encoders, a=5**0.5)

        if decoder:
            expert_decoder = self.expert_encoders.data.reshape(
                self.expert_num_latents, d_in
            )
            decoder_init = torch.cat(
                (expert_decoder, self.residual_encoder.weight.data.clone()), dim=0
            )
            self.W_dec = nn.Parameter(decoder_init)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [
            ("trunk_encoder", self.trunk_encoder),
            ("trunk_decoder", self.trunk_decoder),
            ("router", self.router),
            ("residual_encoder", self.residual_encoder),
        ]

    def _extra_encode_accesses(self) -> list[tuple[str, int, str]]:
        return [
            (
                "active_expert_encoder",
                self.active_experts * self.d_in * self.latents_per_expert,
                f"active={self.active_experts}×{self.d_in}x{self.latents_per_expert}",
            )
        ]

    def _deployment_lookup_accesses(self, n_output):
        return [
            (
                "trunk_deploy",
                self._deploy_library_accesses(self.trunk_encoder.out_features, n_output),
                self._deploy_library_shape(
                    self.trunk_encoder.out_features,
                    n_output,
                    label=f"r={self.trunk_encoder.out_features}",
                ),
            ),
            (
                "sparse_lookup",
                self._deploy_library_accesses(self.cfg.k, n_output),
                self._deploy_library_shape(self.cfg.k, n_output, label=f"K={self.cfg.k}"),
            ),
        ]

    def _decode_sparse(self, acts: Tensor, indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        return decoder_impl(indices, acts.to(self.dtype), self.W_dec.mT)

    def _encode_expert_stage(
        self, residual: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = residual.shape[:-1]
        flat_x = residual.reshape(-1, self.d_in)

        router_logits = self.router(flat_x)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.expert_encoders[selected_expert_idx]
        selected_bias = self.expert_encoder_bias[selected_expert_idx]
        pre_acts = torch.einsum("bd,bald->bal", flat_x, selected_weight) + selected_bias
        acts = F.relu(pre_acts) * selected_probs.unsqueeze(-1)
        top_acts, top_indices, full_acts = _finalize_routed_expert_acts(
            acts,
            selected_expert_idx,
            self.stage1_k,
            self.latents_per_expert,
            self.expert_num_latents,
        )

        target_shape = (*original_shape, self.stage1_k)
        acts_shape = (*original_shape, self.expert_num_latents)
        return (
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )

    def _encode_residual_stage(
        self, residual: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = residual.shape[:-1]
        flat_x = residual.reshape(-1, self.d_in)
        pre_acts = F.linear(
            flat_x, self.residual_encoder.weight, self.residual_encoder.bias
        )
        acts = F.relu(pre_acts)
        top_acts, local_indices = torch.topk(
            acts, self.stage2_k, dim=-1, sorted=False
        )
        top_indices = local_indices + self.expert_num_latents

        target_shape = (*original_shape, self.stage2_k)
        acts_shape = (*original_shape, self.residual_num_latents)
        return (
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            acts.reshape(acts_shape),
        )

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk

        stage1_acts, stage1_indices, stage1_full = self._encode_expert_stage(residual)
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        stage2_input = residual - stage1_out
        stage2_acts, stage2_indices, stage2_full = self._encode_residual_stage(
            stage2_input
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_full = torch.cat((stage1_full, stage2_full), dim=-1)
        return EncoderOutput(combined_acts, combined_indices, combined_full)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk

        stage1_acts, stage1_indices, stage1_full = self._encode_expert_stage(residual)
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        stage2_input = residual - stage1_out
        stage2_acts, stage2_indices, stage2_full = self._encode_residual_stage(
            stage2_input
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_full = torch.cat((stage1_full, stage2_full), dim=-1)

        sparse_residual = self._decode_sparse(combined_acts, combined_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], combined_full, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self._decode_sparse(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class TwoStageResidualExpertSparseCoder(SparseCoder):
    """Global sparse pass followed by expert-routed sparse residual cleanup."""

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
        (
            base_num_latents,
            self.num_experts,
            self.latents_per_expert,
            self.active_experts,
        ) = _resolve_expert_layout(cfg, d_in)

        self.stage1_num_latents = base_num_latents
        self.expert_num_latents = self.num_experts * self.latents_per_expert
        self.num_latents = self.stage1_num_latents + self.expert_num_latents

        if cfg.stage1_ratio is not None:
            self.stage1_k = max(1, round(cfg.k * cfg.stage1_ratio))
        else:
            self.stage1_k = max(1, cfg.k // 2)
        self.stage2_k = max(1, cfg.k - self.stage1_k)

        if self.stage2_k > self.active_experts * self.latents_per_expert:
            raise ValueError(
                "two_stage_residual_expert requires stage2_k <= active_experts * latents_per_expert, "
                f"got stage2_k={self.stage2_k} and "
                f"active_experts={self.active_experts} and "
                f"latents_per_expert={self.latents_per_expert}"
            )

        self.encoder = nn.Linear(
            d_in, self.stage1_num_latents, device=device, dtype=dtype
        )
        self.encoder.bias.data.zero_()

        self.router = nn.Linear(d_in, self.num_experts, device=device, dtype=dtype)
        self.expert_encoders = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.latents_per_expert,
                d_in,
                device=device,
                dtype=dtype,
            )
        )
        self.expert_encoder_bias = nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.latents_per_expert,
                device=device,
                dtype=dtype,
            )
        )

        nn.init.kaiming_uniform_(self.router.weight, a=5**0.5)
        self.router.bias.data.zero_()
        nn.init.kaiming_uniform_(self.expert_encoders, a=5**0.5)

        if decoder:
            expert_decoder = self.expert_encoders.data.reshape(
                self.expert_num_latents, d_in
            )
            decoder_init = torch.cat(
                (self.encoder.weight.data.clone(), expert_decoder.clone()), dim=0
            )
            self.W_dec = nn.Parameter(decoder_init)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("encoder", self.encoder), ("router", self.router)]

    def _extra_encode_accesses(self) -> list[tuple[str, int, str]]:
        return [
            (
                "active_expert_encoder",
                self.active_experts * self.d_in * self.latents_per_expert,
                f"active={self.active_experts}×{self.d_in}x{self.latents_per_expert}",
            )
        ]

    def _decode_sparse(self, acts: Tensor, indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        return decoder_impl(indices, acts.to(self.dtype), self.W_dec.mT)

    def _encode_expert_stage(
        self, residual: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        original_shape = residual.shape[:-1]
        flat_x = residual.reshape(-1, self.d_in)

        router_logits = self.router(flat_x)
        selected_expert_idx, selected_probs = _select_active_expert_indices(
            router_logits, self.active_experts
        )
        selected_weight = self.expert_encoders[selected_expert_idx]
        selected_bias = self.expert_encoder_bias[selected_expert_idx]
        pre_acts = torch.einsum("bd,bald->bal", flat_x, selected_weight) + selected_bias
        acts = F.relu(pre_acts) * selected_probs.unsqueeze(-1)
        top_acts, top_indices, full_acts = _finalize_routed_expert_acts(
            acts,
            selected_expert_idx,
            self.stage2_k,
            self.latents_per_expert,
            self.expert_num_latents,
            index_offset=self.stage1_num_latents,
        )

        target_shape = (*original_shape, self.stage2_k)
        acts_shape = (*original_shape, self.expert_num_latents)
        return (
            top_acts.reshape(target_shape),
            top_indices.reshape(target_shape),
            full_acts.reshape(acts_shape),
        )

    def encode(self, x: Tensor) -> EncoderOutput:
        x_centered = x - self.b_dec

        stage1_acts, stage1_indices, stage1_full = fused_encoder(
            x_centered, self.encoder.weight, self.encoder.bias, self.stage1_k
        )
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        residual = x_centered - stage1_out
        stage2_acts, stage2_indices, stage2_full = self._encode_expert_stage(residual)

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_full = torch.cat((stage1_full, stage2_full), dim=-1)
        return EncoderOutput(combined_acts, combined_indices, combined_full)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec

        stage1_acts, stage1_indices, stage1_full = fused_encoder(
            x_centered, self.encoder.weight, self.encoder.bias, self.stage1_k
        )
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        residual = x_centered - stage1_out
        stage2_acts, stage2_indices, stage2_full = self._encode_expert_stage(residual)

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_full = torch.cat((stage1_full, stage2_full), dim=-1)

        sparse_out = self._decode_sparse(combined_acts, combined_indices)
        sae_out = sparse_out + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], combined_full, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = self._decode_sparse(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class GroupTopKSparseCoder(SparseCoder):
    """Select one winner per local group before the global top-k competition."""

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        acts = F.relu(F.linear(x, self.encoder.weight, self.encoder.bias))

        group_size = self.cfg.group_topk_size
        if self.num_latents % group_size != 0:
            raise ValueError(
                "group_topk requires num_latents divisible by group_topk_size, "
                f"got num_latents={self.num_latents} and "
                f"group_topk_size={group_size}"
            )

        num_groups = self.num_latents // group_size
        if self.cfg.k > num_groups:
            raise ValueError(
                "group_topk requires k <= number of groups, "
                f"got k={self.cfg.k} and num_groups={num_groups}"
            )

        grouped = acts.view(*acts.shape[:-1], num_groups, group_size)
        winner_acts, winner_offsets = grouped.max(dim=-1)
        top_group_acts, top_group_indices = torch.topk(
            winner_acts, self.cfg.k, dim=-1, sorted=False
        )

        top_offsets = winner_offsets.gather(-1, top_group_indices)
        top_indices = top_group_indices * group_size + top_offsets
        return EncoderOutput(top_group_acts, top_indices, acts)


class FactorizedTopKSparseCoder(SparseCoder):
    """Top-k SAE with a low-rank ReLU mixing stage before latent scoring."""

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

        hidden_dim = cfg.factorized_hidden_dim if cfg.factorized_hidden_dim is not None else min(self.num_latents, max(d_in // 2, cfg.k * 4))
        self.factor_encoder = nn.Linear(
            d_in, hidden_dim, device=device, dtype=dtype
        )
        self.encoder = nn.Linear(
            hidden_dim, self.num_latents, device=device, dtype=dtype
        )
        self.factor_encoder.bias.data.zero_()
        self.encoder.bias.data.zero_()

        if decoder:
            effective_encoder = self.encoder.weight.data @ self.factor_encoder.weight.data
            self.W_dec = nn.Parameter(effective_encoder)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("factor_encoder", self.factor_encoder), ("encoder", self.encoder)]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        hidden = F.relu(
            F.linear(x, self.factor_encoder.weight, self.factor_encoder.bias)
        )
        acts = F.relu(F.linear(hidden, self.encoder.weight, self.encoder.bias))
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)


class LowRankResidualSparseCoder(SparseCoder):
    """Model a dense low-rank trunk, then sparsely code the remaining residual."""

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

        # Keep a meaningfully expressive dense trunk before sparse residual
        # coding. Tying trunk rank too tightly to expansion factor made the
        # prototype collapse to a minimal rank under the strong EF=16 recipe.
        trunk_rank = cfg.trunk_rank if cfg.trunk_rank is not None else min(d_in, max(cfg.k * 2, d_in // 4))
        self.trunk_encoder = nn.Linear(
            d_in, trunk_rank, bias=False, device=device, dtype=dtype
        )
        self.trunk_decoder = nn.Linear(
            trunk_rank, d_in, bias=False, device=device, dtype=dtype
        )
        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        if decoder:
            self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("trunk_encoder", self.trunk_encoder), ("trunk_decoder", self.trunk_decoder), ("encoder", self.encoder)]

    def _deployment_lookup_accesses(self, n_output):
        base = super()._deployment_lookup_accesses(n_output)
        r = self.trunk_encoder.out_features
        return [
            (
                "trunk_deploy",
                self._deploy_library_accesses(r, n_output),
                self._deploy_library_shape(r, n_output, label=f"r={r}"),
            )
        ] + base

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        return fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

    def decode_residual(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        return decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        top_acts, top_indices, pre_acts = fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankAdaptiveBudgetResidualSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk with adaptive per-sample residual feature budgets."""

    @staticmethod
    def _apply_adaptive_budget(acts: Tensor, k: int) -> tuple[Tensor, Tensor]:
        batch_size = acts.shape[0]
        if batch_size == 1:
            return torch.topk(acts, k, dim=-1, sorted=False)

        difficulty = acts.detach().pow(2).mean(dim=-1)
        difficulty_sum = difficulty.sum().clamp_min(torch.finfo(acts.dtype).tiny)
        total_budget = batch_size * k

        raw_quota = difficulty / difficulty_sum * total_budget
        quotas = torch.floor(raw_quota).to(dtype=torch.long)
        quotas = quotas.clamp(min=1, max=k)

        remaining = total_budget - int(quotas.sum().item())
        if remaining > 0:
            order = torch.argsort(
                raw_quota - quotas.to(raw_quota.dtype), descending=True
            )
            for idx in order.tolist():
                if remaining == 0:
                    break
                if quotas[idx] < k:
                    quotas[idx] += 1
                    remaining -= 1
        elif remaining < 0:
            order = torch.argsort(raw_quota - quotas.to(raw_quota.dtype))
            for idx in order.tolist():
                if remaining == 0:
                    break
                if quotas[idx] > 1:
                    quotas[idx] -= 1
                    remaining += 1

        top_acts, top_indices = torch.topk(acts, k, dim=-1, sorted=False)
        rank = torch.arange(k, device=acts.device)
        active_mask = rank.unsqueeze(0) < quotas.unsqueeze(1)
        top_acts = top_acts * active_mask.to(top_acts.dtype)
        return top_acts, top_indices

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        pre_acts = F.linear(residual, self.encoder.weight, self.encoder.bias)
        acts = F.relu(pre_acts)
        top_acts, top_indices = self._apply_adaptive_budget(acts, self.cfg.k)
        return EncoderOutput(top_acts, top_indices, acts)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        pre_acts = F.linear(residual, self.encoder.weight, self.encoder.bias)
        acts = F.relu(pre_acts)
        top_acts, top_indices = self._apply_adaptive_budget(acts, self.cfg.k)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankTwoStageResidualSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk followed by two residual sparse refinement passes."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=False)
        if cfg.stage1_ratio is not None:
            self.stage1_k = max(1, round(cfg.k * cfg.stage1_ratio))
        else:
            self.stage1_k = max(1, cfg.k // 2)
        self.stage2_k = max(1, cfg.k - self.stage1_k)
        self.residual_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.residual_encoder.bias.data.zero_()

        if decoder:
            decoder_init = 0.5 * (
                self.encoder.weight.data + self.residual_encoder.weight.data
            )
            self.W_dec = nn.Parameter(decoder_init)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

    def _encoder_linear_layers(self):
        return [("trunk_encoder", self.trunk_encoder), ("trunk_decoder", self.trunk_decoder), ("encoder", self.encoder), ("residual_encoder", self.residual_encoder)]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk

        stage1_acts, stage1_indices, stage1_pre = fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.stage1_k
        )
        stage1_out = self.decode_residual(stage1_acts, stage1_indices)

        stage2_input = residual - stage1_out
        stage2_acts, stage2_indices, stage2_pre = fused_encoder(
            stage2_input,
            self.residual_encoder.weight,
            self.residual_encoder.bias,
            self.stage2_k,
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_pre = torch.maximum(stage1_pre, stage2_pre)
        return EncoderOutput(combined_acts, combined_indices, combined_pre)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk

        stage1_acts, stage1_indices, stage1_pre = fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.stage1_k
        )
        stage1_out = self.decode_residual(stage1_acts, stage1_indices)

        stage2_input = residual - stage1_out
        stage2_acts, stage2_indices, stage2_pre = fused_encoder(
            stage2_input,
            self.residual_encoder.weight,
            self.residual_encoder.bias,
            self.stage2_k,
        )
        stage2_out = self.decode_residual(stage2_acts, stage2_indices)

        sae_out = trunk + stage1_out + stage2_out + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            aux_stage1 = torch.where(dead_mask[None], stage1_pre, -torch.inf)
            aux_stage2 = torch.where(dead_mask[None], stage2_pre, -torch.inf)

            aux1_k = min(self.stage1_k, k_aux)
            aux2_k = min(self.stage2_k, max(1, k_aux - aux1_k))

            aux1_acts, aux1_indices = aux_stage1.topk(aux1_k, sorted=False)
            aux1_out = self.decode_residual(aux1_acts, aux1_indices)

            aux2_acts, aux2_indices = aux_stage2.topk(aux2_k, sorted=False)
            aux2_out = self.decode_residual(aux2_acts, aux2_indices)

            e_hat = trunk + aux1_out + aux2_out + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class RoutedLowRankTwoStageResidualSparseCoder(LowRankTwoStageResidualSparseCoder):
    """Two-stage low-rank residual SAE with explicit router-driven support selection."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.stage1_router = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.stage2_router = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )

        nn.init.kaiming_uniform_(self.stage1_router.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.stage2_router.weight, a=5**0.5)
        self.stage1_router.bias.data.zero_()
        self.stage2_router.bias.data.zero_()

    def _encoder_linear_layers(self):
        return super()._encoder_linear_layers() + [("stage1_router", self.stage1_router), ("stage2_router", self.stage2_router)]

    def _route_stage(
        self,
        inputs: Tensor,
        weight: Tensor,
        bias: Tensor,
        router: nn.Linear,
        k: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pre_acts = F.linear(inputs, weight, bias)
        acts = F.relu(pre_acts)
        router_logits = router(inputs) / self.cfg.gated_temperature
        router_gate = torch.sigmoid(router_logits)
        # Keep routing on the reconstruction path so DDP does not treat the
        # router weights as unused when only the selected support is decoded.
        routed_acts = acts * router_gate
        scores = routed_acts + 0.1 * torch.tanh(router_logits)
        _, top_indices = torch.topk(scores, k, dim=-1, sorted=False)
        top_acts = routed_acts.gather(-1, top_indices)
        return top_acts, top_indices, scores, routed_acts

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk

        stage1_acts, stage1_indices, stage1_scores, _ = self._route_stage(
            residual,
            self.encoder.weight,
            self.encoder.bias,
            self.stage1_router,
            self.stage1_k,
        )
        stage1_out = self.decode_residual(stage1_acts, stage1_indices)

        stage2_input = residual - stage1_out
        stage2_acts, stage2_indices, stage2_scores, _ = self._route_stage(
            stage2_input,
            self.residual_encoder.weight,
            self.residual_encoder.bias,
            self.stage2_router,
            self.stage2_k,
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_scores = torch.maximum(stage1_scores, stage2_scores)
        return EncoderOutput(combined_acts, combined_indices, combined_scores)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk

        stage1_acts, stage1_indices, stage1_scores, stage1_routed = self._route_stage(
            residual,
            self.encoder.weight,
            self.encoder.bias,
            self.stage1_router,
            self.stage1_k,
        )
        stage1_out = self.decode_residual(stage1_acts, stage1_indices)

        stage2_input = residual - stage1_out
        stage2_acts, stage2_indices, stage2_scores, _ = self._route_stage(
            stage2_input,
            self.residual_encoder.weight,
            self.residual_encoder.bias,
            self.stage2_router,
            self.stage2_k,
        )
        stage2_out = self.decode_residual(stage2_acts, stage2_indices)

        sae_out = trunk + stage1_out + stage2_out + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            aux_stage1 = torch.where(dead_mask[None], stage1_scores, -torch.inf)

            aux1_k = min(self.stage1_k, k_aux)
            aux2_k = min(self.stage2_k, max(1, k_aux - aux1_k))

            _, aux1_indices = aux_stage1.topk(aux1_k, sorted=False)
            aux1_acts = stage1_routed.gather(-1, aux1_indices)
            aux1_out = self.decode_residual(aux1_acts, aux1_indices)

            aux2_input = residual - aux1_out
            aux2_logits = F.linear(
                aux2_input,
                self.residual_encoder.weight,
                self.residual_encoder.bias,
            )
            aux2_full = F.relu(aux2_logits)
            aux2_router_logits = (
                self.stage2_router(aux2_input) / self.cfg.gated_temperature
            )
            aux2_routed = aux2_full * torch.sigmoid(aux2_router_logits)
            aux_stage2 = torch.where(
                dead_mask[None],
                aux2_routed + 0.1 * torch.tanh(aux2_router_logits),
                -torch.inf,
            )
            _, aux2_indices = aux_stage2.topk(aux2_k, sorted=False)
            aux2_acts = aux2_routed.gather(-1, aux2_indices)
            aux2_out = self.decode_residual(aux2_acts, aux2_indices)

            e_hat = trunk + aux1_out + aux2_out + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class BucketedLowRankResidualSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk with norm-routed residual dictionaries."""

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

        trunk_rank = cfg.trunk_rank if cfg.trunk_rank is not None else min(d_in, max(cfg.k * 2, d_in // 4))
        self.trunk_encoder = nn.Linear(
            d_in, trunk_rank, bias=False, device=device, dtype=dtype
        )
        self.trunk_decoder = nn.Linear(
            trunk_rank, d_in, bias=False, device=device, dtype=dtype
        )
        self.low_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.high_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.low_encoder.bias.data.zero_()
        self.high_encoder.bias.data.zero_()
        self.high_encoder.weight.data.mul_(1.05)

        self.bucket_scale = nn.Parameter(
            torch.tensor(2.0, device=device, dtype=dtype)
        )
        self.bucket_bias = nn.Parameter(
            torch.tensor(0.0, device=device, dtype=dtype)
        )

        if decoder:
            decoder_init = 0.5 * (
                self.low_encoder.weight.data + self.high_encoder.weight.data
            )
            self.W_dec = nn.Parameter(decoder_init)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("trunk_encoder", self.trunk_encoder), ("trunk_decoder", self.trunk_decoder), ("low_encoder", self.low_encoder), ("high_encoder", self.high_encoder)]

    def _compute_bucketed_acts(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        low_acts = F.relu(
            F.linear(residual, self.low_encoder.weight, self.low_encoder.bias)
        )
        high_acts = F.relu(
            F.linear(residual, self.high_encoder.weight, self.high_encoder.bias)
        )
        norms = residual.norm(dim=-1, keepdim=True)
        centered_norms = norms - norms.mean()
        gate = torch.sigmoid(self.bucket_scale * centered_norms + self.bucket_bias)
        acts = (1.0 - gate) * low_acts + gate * high_acts
        return acts, gate

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        acts, _ = self._compute_bucketed_acts(residual)
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        acts, _ = self._compute_bucketed_acts(residual)
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankResidualVQSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk with hard codebook residual modeling before sparse correction."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.num_codes = min(512, max(64, cfg.k * 4))
        self.codebook = nn.Parameter(
            torch.randn(self.num_codes, d_in, device=device, dtype=dtype) * 0.02
        )
        self.code_router = nn.Linear(d_in, self.num_codes, device=device, dtype=dtype)
        self.code_router.bias.data.zero_()

    def _encoder_linear_layers(self):
        return [("trunk_encoder", self.trunk_encoder), ("trunk_decoder", self.trunk_decoder), ("code_router", self.code_router), ("encoder", self.encoder)]

    def _extra_encode_accesses(self):
        return [("codebook_matmul", self.num_codes * self.d_in, f"{self.num_codes}x{self.d_in}")]

    def _deployment_lookup_accesses(self, n_output):
        base = super()._deployment_lookup_accesses(n_output)
        return [
            (
                "codebook_deploy",
                self._deploy_library_accesses(self.num_codes, n_output),
                self._deploy_library_shape(self.num_codes, n_output, label=f"codes={self.num_codes}"),
            )
        ] + base

    def _select_code(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.code_router(residual)
        code_indices = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)
        hard_assign = F.one_hot(code_indices, num_classes=self.num_codes).to(
            dtype=probs.dtype
        )
        routing = hard_assign + probs - probs.detach()
        coarse = routing @ self.codebook
        return coarse, logits

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        coarse, _ = self._select_code(residual)
        vq_residual = residual - coarse
        return fused_encoder(
            vq_residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        coarse, _ = self._select_code(residual)
        vq_residual = residual - coarse
        top_acts, top_indices, pre_acts = fused_encoder(
            vq_residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + coarse + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = (
                trunk
                + coarse
                + self.decode_residual(auxk_acts, auxk_indices)
                + self.b_dec
            )
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankMultiBranchResidualSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk with a learned mixture of residual scoring branches."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=False)
        self.num_branches = 3
        self.branch_encoders = nn.ModuleList(
            [
                nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
                for _ in range(self.num_branches)
            ]
        )
        self.branch_mix_logits = nn.Linear(
            d_in, self.num_branches, device=device, dtype=dtype
        )

        for encoder in self.branch_encoders:
            encoder.bias.data.zero_()
        self.branch_mix_logits.weight.data.zero_()
        self.branch_mix_logits.bias.data.zero_()

        if decoder:
            decoder_init = torch.stack(
                [branch.weight.data for branch in self.branch_encoders], dim=0
            ).mean(dim=0)
            self.W_dec = nn.Parameter(decoder_init)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

    def _encoder_linear_layers(self):
        layers = [("trunk_encoder", self.trunk_encoder), ("trunk_decoder", self.trunk_decoder)]
        layers.append(("encoder", self.encoder))
        for i, branch in enumerate(self.branch_encoders):
            layers.append((f"branch_encoder_{i}", branch))
        layers.append(("branch_mix_logits", self.branch_mix_logits))
        return layers

    def _mixed_acts(self, residual: Tensor) -> Tensor:
        # Keep the inherited residual encoder on the reconstruction path so
        # DDP does not mark it unused while the routed branches specialize.
        anchor_acts = F.relu(F.linear(residual, self.encoder.weight, self.encoder.bias))
        branch_weights = torch.softmax(self.branch_mix_logits(residual), dim=-1)
        branch_acts = []
        for branch_idx, encoder in enumerate(self.branch_encoders):
            acts = F.relu(F.linear(residual, encoder.weight, encoder.bias))
            branch_weight = branch_weights[..., branch_idx].unsqueeze(-1)
            branch_acts.append(acts * branch_weight)
        routed_acts = torch.stack(branch_acts, dim=0).sum(dim=0)
        anchor_scale = 1.0 / (self.num_branches + 1)
        return routed_acts * (1.0 - anchor_scale) + anchor_acts * anchor_scale

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        acts = self._mixed_acts(residual)
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        acts = self._mixed_acts(residual)
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankFactorizedResidualSparseCoder(SparseCoder):
    """Low-rank trunk followed by a factorized residual scorer before top-k."""

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

        trunk_rank = cfg.trunk_rank if cfg.trunk_rank is not None else min(d_in, max(cfg.k * 2, d_in // 4))
        self.trunk_encoder = nn.Linear(
            d_in, trunk_rank, bias=False, device=device, dtype=dtype
        )
        self.trunk_decoder = nn.Linear(
            trunk_rank, d_in, bias=False, device=device, dtype=dtype
        )

        hidden_dim = cfg.factorized_hidden_dim if cfg.factorized_hidden_dim is not None else min(self.num_latents, max(d_in // 2, cfg.k * 4))
        self.factor_encoder = nn.Linear(
            d_in, hidden_dim, device=device, dtype=dtype
        )
        self.encoder = nn.Linear(
            hidden_dim, self.num_latents, device=device, dtype=dtype
        )
        self.factor_encoder.bias.data.zero_()
        self.encoder.bias.data.zero_()

        if decoder:
            effective_encoder = self.encoder.weight.data @ self.factor_encoder.weight.data
            self.W_dec = nn.Parameter(effective_encoder)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("trunk_encoder", self.trunk_encoder), ("trunk_decoder", self.trunk_decoder), ("factor_encoder", self.factor_encoder), ("encoder", self.encoder)]

    def _deployment_lookup_accesses(self, n_output):
        base = super()._deployment_lookup_accesses(n_output)
        r = self.trunk_encoder.out_features
        return [
            (
                "trunk_deploy",
                self._deploy_library_accesses(r, n_output),
                self._deploy_library_shape(r, n_output, label=f"r={r}"),
            )
        ] + base

    def decode_residual(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        return decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)

    def _encode_residual(self, residual: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = F.relu(
            F.linear(residual, self.factor_encoder.weight, self.factor_encoder.bias)
        )
        pre_acts = F.relu(F.linear(hidden, self.encoder.weight, self.encoder.bias))
        top_acts, top_indices = torch.topk(pre_acts, self.cfg.k, dim=-1, sorted=False)
        return top_acts, top_indices, pre_acts

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        top_acts, top_indices, pre_acts = self._encode_residual(residual)
        return EncoderOutput(top_acts, top_indices, pre_acts)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        top_acts, top_indices, pre_acts = self._encode_residual(residual)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankSoftCodebookResidualSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk with soft codebook residual projection before sparse correction."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.num_codes = cfg.num_codes if cfg.num_codes is not None else min(256, max(32, cfg.k * 2))
        self.codebook = nn.Parameter(
            torch.randn(self.num_codes, d_in, device=device, dtype=dtype) * 0.02
        )
        self.code_router = nn.Linear(d_in, self.num_codes, device=device, dtype=dtype)
        self.code_router.bias.data.zero_()

    def _encoder_linear_layers(self):
        return [("trunk_encoder", self.trunk_encoder), ("trunk_decoder", self.trunk_decoder), ("code_router", self.code_router), ("encoder", self.encoder)]

    def _extra_encode_accesses(self):
        return [("codebook_matmul", self.num_codes * self.d_in, f"{self.num_codes}x{self.d_in}")]

    def _deployment_lookup_accesses(self, n_output):
        base = super()._deployment_lookup_accesses(n_output)
        return [
            (
                "codebook_deploy",
                self._deploy_library_accesses(self.num_codes, n_output),
                self._deploy_library_shape(self.num_codes, n_output, label=f"codes={self.num_codes}"),
            )
        ] + base

    def _project_codebook(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.code_router(residual)
        routing = logits.softmax(dim=-1)
        coarse = routing @ self.codebook
        return coarse, logits

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse
        return fused_encoder(
            code_residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse
        top_acts, top_indices, pre_acts = fused_encoder(
            code_residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + coarse + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = (
                trunk
                + coarse
                + self.decode_residual(auxk_acts, auxk_indices)
                + self.b_dec
            )
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankGroupedSoftCodebookResidualSparseCoder(
    LowRankSoftCodebookResidualSparseCoder
):
    """Soft-codebook residual model with group-local sparse competition."""

    def _group_topk(self, acts: Tensor) -> tuple[Tensor, Tensor]:
        group_size = self.cfg.group_topk_size
        if self.num_latents % group_size != 0:
            raise ValueError(
                "lowrank_grouped_soft_codebook_residual requires num_latents "
                f"divisible by group_topk_size, got num_latents={self.num_latents} "
                f"and group_topk_size={group_size}"
            )

        num_groups = self.num_latents // group_size
        if self.cfg.k > num_groups:
            raise ValueError(
                "lowrank_grouped_soft_codebook_residual requires k <= number of "
                f"groups, got k={self.cfg.k} and num_groups={num_groups}"
            )

        grouped = acts.view(*acts.shape[:-1], num_groups, group_size)
        winner_acts, winner_offsets = grouped.max(dim=-1)
        top_group_acts, top_group_indices = torch.topk(
            winner_acts, self.cfg.k, dim=-1, sorted=False
        )
        top_offsets = winner_offsets.gather(-1, top_group_indices)
        top_indices = top_group_indices * group_size + top_offsets
        return top_group_acts, top_indices

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse
        acts = F.relu(F.linear(code_residual, self.encoder.weight, self.encoder.bias))
        top_acts, top_indices = self._group_topk(acts)
        return EncoderOutput(top_acts, top_indices, acts)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse
        acts = F.relu(F.linear(code_residual, self.encoder.weight, self.encoder.bias))
        top_acts, top_indices = self._group_topk(acts)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + coarse + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = (
                trunk
                + coarse
                + self.decode_residual(auxk_acts, auxk_indices)
                + self.b_dec
            )
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankGatedSoftCodebookResidualSparseCoder(
    LowRankSoftCodebookResidualSparseCoder
):
    """Soft-codebook residual model with gated sparse support selection."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.gate_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.gate_encoder.weight.data.zero_()
        self.gate_encoder.bias.data.fill_(cfg.gated_init_logit)

    def _encoder_linear_layers(self):
        return super()._encoder_linear_layers() + [("gate_encoder", self.gate_encoder)]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse
        pre_acts = F.linear(
            code_residual, self.encoder.weight, self.encoder.bias
        )
        positive = F.relu(pre_acts)
        gate_logits = self.gate_encoder(code_residual) / self.cfg.gated_temperature
        gate = torch.sigmoid(gate_logits)
        acts = positive * gate
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse

        pre_acts = F.linear(
            code_residual, self.encoder.weight, self.encoder.bias
        )
        positive = F.relu(pre_acts)
        gate_logits = self.gate_encoder(code_residual) / self.cfg.gated_temperature
        gate = torch.sigmoid(gate_logits)
        acts = positive * gate
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + coarse + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = (
                trunk
                + coarse
                + self.decode_residual(auxk_acts, auxk_indices)
                + self.b_dec
            )
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankTwoStageSoftCodebookResidualSparseCoder(
    LowRankTwoStageResidualSparseCoder
):
    """Low-rank trunk with soft codebook projection before two-stage sparse refinement."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.num_codes = cfg.num_codes if cfg.num_codes is not None else min(256, max(32, cfg.k * 2))
        self.codebook = nn.Parameter(
            torch.randn(self.num_codes, d_in, device=device, dtype=dtype) * 0.02
        )
        self.code_router = nn.Linear(d_in, self.num_codes, device=device, dtype=dtype)
        self.code_router.bias.data.zero_()
        self.factorized_hidden_dim = cfg.factorized_hidden_dim
        if self.factorized_hidden_dim is not None:
            self.encoder.weight.requires_grad_(False)
            self.encoder.bias.requires_grad_(False)
            self.residual_encoder.weight.requires_grad_(False)
            self.residual_encoder.bias.requires_grad_(False)

            self.stage1_factor_encoder = nn.Linear(
                d_in, self.factorized_hidden_dim, device=device, dtype=dtype
            )
            self.stage1_factor_projector = nn.Linear(
                self.factorized_hidden_dim,
                self.num_latents,
                device=device,
                dtype=dtype,
            )
            self.stage2_factor_encoder = nn.Linear(
                d_in, self.factorized_hidden_dim, device=device, dtype=dtype
            )
            self.stage2_factor_projector = nn.Linear(
                self.factorized_hidden_dim,
                self.num_latents,
                device=device,
                dtype=dtype,
            )
            self.stage1_factor_encoder.bias.data.zero_()
            self.stage1_factor_projector.bias.data.zero_()
            self.stage2_factor_encoder.bias.data.zero_()
            self.stage2_factor_projector.bias.data.zero_()

    def _encoder_linear_layers(self):
        if self.factorized_hidden_dim is not None:
            return [
                ("trunk_encoder", self.trunk_encoder),
                ("trunk_decoder", self.trunk_decoder),
                ("code_router", self.code_router),
                ("stage1_factor_encoder", self.stage1_factor_encoder),
                ("stage1_factor_projector", self.stage1_factor_projector),
                ("stage2_factor_encoder", self.stage2_factor_encoder),
                ("stage2_factor_projector", self.stage2_factor_projector),
            ]
        return super()._encoder_linear_layers() + [("code_router", self.code_router)]

    def _extra_encode_accesses(self):
        return [("codebook_matmul", self.num_codes * self.d_in, f"{self.num_codes}x{self.d_in}")]

    def _deployment_lookup_accesses(self, n_output):
        base = super()._deployment_lookup_accesses(n_output)
        return [
            (
                "codebook_deploy",
                self._deploy_library_accesses(self.num_codes, n_output),
                self._deploy_library_shape(self.num_codes, n_output, label=f"codes={self.num_codes}"),
            )
        ] + base

    def _project_codebook(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.code_router(residual)
        routing = logits.softmax(dim=-1)
        coarse = routing @ self.codebook
        return coarse, logits

    def _encode_stage(
        self,
        residual: Tensor,
        dense_encoder: nn.Linear,
        factor_encoder: nn.Linear | None,
        factor_projector: nn.Linear | None,
        k: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if factor_encoder is None or factor_projector is None:
            return fused_encoder(
                residual, dense_encoder.weight, dense_encoder.bias, k
            )

        hidden = F.relu(F.linear(residual, factor_encoder.weight, factor_encoder.bias))
        acts = F.relu(F.linear(hidden, factor_projector.weight, factor_projector.bias))
        top_acts, top_indices = torch.topk(acts, k, dim=-1, sorted=False)
        return top_acts, top_indices, acts

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse

        stage1_acts, stage1_indices, stage1_pre = self._encode_stage(
            code_residual,
            self.encoder,
            getattr(self, "stage1_factor_encoder", None),
            getattr(self, "stage1_factor_projector", None),
            self.stage1_k,
        )
        stage1_out = self.decode_residual(stage1_acts, stage1_indices)

        stage2_input = code_residual - stage1_out
        stage2_acts, stage2_indices, stage2_pre = self._encode_stage(
            stage2_input,
            self.residual_encoder,
            getattr(self, "stage2_factor_encoder", None),
            getattr(self, "stage2_factor_projector", None),
            self.stage2_k,
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_pre = torch.maximum(stage1_pre, stage2_pre)
        return EncoderOutput(combined_acts, combined_indices, combined_pre)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse

        stage1_acts, stage1_indices, stage1_pre = self._encode_stage(
            code_residual,
            self.encoder,
            getattr(self, "stage1_factor_encoder", None),
            getattr(self, "stage1_factor_projector", None),
            self.stage1_k,
        )
        stage1_out = self.decode_residual(stage1_acts, stage1_indices)

        stage2_input = code_residual - stage1_out
        stage2_acts, stage2_indices, stage2_pre = self._encode_stage(
            stage2_input,
            self.residual_encoder,
            getattr(self, "stage2_factor_encoder", None),
            getattr(self, "stage2_factor_projector", None),
            self.stage2_k,
        )
        stage2_out = self.decode_residual(stage2_acts, stage2_indices)

        sae_out = trunk + coarse + stage1_out + stage2_out + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            aux_stage1 = torch.where(dead_mask[None], stage1_pre, -torch.inf)
            aux_stage2 = torch.where(dead_mask[None], stage2_pre, -torch.inf)

            aux1_k = min(self.stage1_k, k_aux)
            aux2_k = min(self.stage2_k, max(1, k_aux - aux1_k))

            aux1_acts, aux1_indices = aux_stage1.topk(aux1_k, sorted=False)
            aux1_out = self.decode_residual(aux1_acts, aux1_indices)

            aux2_acts, aux2_indices = aux_stage2.topk(aux2_k, sorted=False)
            aux2_out = self.decode_residual(aux2_acts, aux2_indices)

            e_hat = trunk + coarse + aux1_out + aux2_out + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class BucketedLowRankTwoStageSoftCodebookResidualSparseCoder(
    LowRankTwoStageSoftCodebookResidualSparseCoder
):
    """Soft-codebook two-stage residual SAE with norm-bucketed stage scorers."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        # This variant replaces the inherited stage scorers with bucketed
        # low/high mixtures, so keep the parent scorer weights out of DDP.
        self.encoder.weight.requires_grad_(False)
        self.encoder.bias.requires_grad_(False)
        self.residual_encoder.weight.requires_grad_(False)
        self.residual_encoder.bias.requires_grad_(False)
        self.stage1_low_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.stage1_high_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.stage2_low_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.stage2_high_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.stage1_low_encoder.bias.data.zero_()
        self.stage1_high_encoder.bias.data.zero_()
        self.stage2_low_encoder.bias.data.zero_()
        self.stage2_high_encoder.bias.data.zero_()

        self.stage1_low_encoder.weight.data.copy_(self.encoder.weight.data)
        self.stage1_high_encoder.weight.data.copy_(self.encoder.weight.data * 1.05)
        self.stage2_low_encoder.weight.data.copy_(self.residual_encoder.weight.data)
        self.stage2_high_encoder.weight.data.copy_(
            self.residual_encoder.weight.data * 1.05
        )

        self.stage1_bucket_scale = nn.Parameter(
            torch.tensor(2.0, device=device, dtype=dtype)
        )
        self.stage1_bucket_bias = nn.Parameter(
            torch.tensor(0.0, device=device, dtype=dtype)
        )
        self.stage2_bucket_scale = nn.Parameter(
            torch.tensor(2.0, device=device, dtype=dtype)
        )
        self.stage2_bucket_bias = nn.Parameter(
            torch.tensor(0.0, device=device, dtype=dtype)
        )

    def _encoder_linear_layers(self):
        return [
            ("trunk_encoder", self.trunk_encoder),
            ("trunk_decoder", self.trunk_decoder),
            ("code_router", self.code_router),
            ("stage1_low_encoder", self.stage1_low_encoder),
            ("stage1_high_encoder", self.stage1_high_encoder),
            ("stage2_low_encoder", self.stage2_low_encoder),
            ("stage2_high_encoder", self.stage2_high_encoder),
        ]

    def _extra_encode_accesses(self):
        return [("codebook_matmul", self.num_codes * self.d_in, f"{self.num_codes}x{self.d_in}")]

    def _bucketed_stage_acts(
        self,
        residual: Tensor,
        low_encoder: nn.Linear,
        high_encoder: nn.Linear,
        bucket_scale: Tensor,
        bucket_bias: Tensor,
    ) -> Tensor:
        low_acts = F.relu(F.linear(residual, low_encoder.weight, low_encoder.bias))
        high_acts = F.relu(F.linear(residual, high_encoder.weight, high_encoder.bias))
        norms = residual.norm(dim=-1, keepdim=True)
        centered_norms = norms - norms.mean()
        gate = torch.sigmoid(bucket_scale * centered_norms + bucket_bias)
        return (1.0 - gate) * low_acts + gate * high_acts

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse

        stage1_pre = self._bucketed_stage_acts(
            code_residual,
            self.stage1_low_encoder,
            self.stage1_high_encoder,
            self.stage1_bucket_scale,
            self.stage1_bucket_bias,
        )
        stage1_acts, stage1_indices = torch.topk(
            stage1_pre, self.stage1_k, dim=-1, sorted=False
        )
        stage1_out = self.decode_residual(stage1_acts, stage1_indices)

        stage2_input = code_residual - stage1_out
        stage2_pre = self._bucketed_stage_acts(
            stage2_input,
            self.stage2_low_encoder,
            self.stage2_high_encoder,
            self.stage2_bucket_scale,
            self.stage2_bucket_bias,
        )
        stage2_acts, stage2_indices = torch.topk(
            stage2_pre, self.stage2_k, dim=-1, sorted=False
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_pre = torch.maximum(stage1_pre, stage2_pre)
        return EncoderOutput(combined_acts, combined_indices, combined_pre)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse

        stage1_pre = self._bucketed_stage_acts(
            code_residual,
            self.stage1_low_encoder,
            self.stage1_high_encoder,
            self.stage1_bucket_scale,
            self.stage1_bucket_bias,
        )
        stage1_acts, stage1_indices = torch.topk(
            stage1_pre, self.stage1_k, dim=-1, sorted=False
        )
        stage1_out = self.decode_residual(stage1_acts, stage1_indices)

        stage2_input = code_residual - stage1_out
        stage2_pre = self._bucketed_stage_acts(
            stage2_input,
            self.stage2_low_encoder,
            self.stage2_high_encoder,
            self.stage2_bucket_scale,
            self.stage2_bucket_bias,
        )
        stage2_acts, stage2_indices = torch.topk(
            stage2_pre, self.stage2_k, dim=-1, sorted=False
        )
        stage2_out = self.decode_residual(stage2_acts, stage2_indices)

        sae_out = trunk + coarse + stage1_out + stage2_out + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            aux_stage1 = torch.where(dead_mask[None], stage1_pre, -torch.inf)
            aux_stage2 = torch.where(dead_mask[None], stage2_pre, -torch.inf)

            aux1_k = min(self.stage1_k, k_aux)
            aux2_k = min(self.stage2_k, max(1, k_aux - aux1_k))

            aux1_acts, aux1_indices = aux_stage1.topk(aux1_k, sorted=False)
            aux1_out = self.decode_residual(aux1_acts, aux1_indices)

            aux2_acts, aux2_indices = aux_stage2.topk(aux2_k, sorted=False)
            aux2_out = self.decode_residual(aux2_acts, aux2_indices)

            e_hat = trunk + coarse + aux1_out + aux2_out + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class WhitenedLowRankTwoStageSoftCodebookResidualSparseCoder(
    LowRankTwoStageSoftCodebookResidualSparseCoder
):
    """Soft-codebook two-stage residual SAE with normalized residual preconditioning."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        mix_rank = min(d_in, max(cfg.k * 2, d_in // 8))
        self.preconditioner_down = nn.Linear(
            d_in, mix_rank, bias=False, device=device, dtype=dtype
        )
        self.preconditioner_up = nn.Linear(
            mix_rank, d_in, bias=False, device=device, dtype=dtype
        )
        self.preconditioner_down.weight.data.zero_()
        self.preconditioner_up.weight.data.zero_()
        diagonal = min(d_in, mix_rank)
        self.preconditioner_down.weight.data[:diagonal, :diagonal] = torch.eye(
            diagonal, device=device, dtype=dtype
        )
        self.preconditioner_up.weight.data[:diagonal, :diagonal] = 0.05 * torch.eye(
            diagonal, device=device, dtype=dtype
        )

    def _encoder_linear_layers(self):
        return super()._encoder_linear_layers() + [("preconditioner_down", self.preconditioner_down), ("preconditioner_up", self.preconditioner_up)]

    def _precondition_residual(self, residual: Tensor) -> Tensor:
        centered = residual - residual.mean(dim=-1, keepdim=True)
        rms = centered.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        normalized = centered * rms
        mixed = normalized + 0.25 * torch.roll(normalized, shifts=1, dims=-1)
        correction = self.preconditioner_up(self.preconditioner_down(normalized))
        return mixed + correction

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse

        stage1_input = self._precondition_residual(code_residual)
        stage1_acts, stage1_indices, stage1_pre = fused_encoder(
            stage1_input, self.encoder.weight, self.encoder.bias, self.stage1_k
        )
        stage1_out = self.decode_residual(stage1_acts, stage1_indices)

        stage2_input = self._precondition_residual(code_residual - stage1_out)
        stage2_acts, stage2_indices, stage2_pre = fused_encoder(
            stage2_input,
            self.residual_encoder.weight,
            self.residual_encoder.bias,
            self.stage2_k,
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_pre = torch.maximum(stage1_pre, stage2_pre)
        return EncoderOutput(combined_acts, combined_indices, combined_pre)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse

        stage1_input = self._precondition_residual(code_residual)
        stage1_acts, stage1_indices, stage1_pre = fused_encoder(
            stage1_input, self.encoder.weight, self.encoder.bias, self.stage1_k
        )
        stage1_out = self.decode_residual(stage1_acts, stage1_indices)

        stage2_input = self._precondition_residual(code_residual - stage1_out)
        stage2_acts, stage2_indices, stage2_pre = fused_encoder(
            stage2_input,
            self.residual_encoder.weight,
            self.residual_encoder.bias,
            self.stage2_k,
        )
        stage2_out = self.decode_residual(stage2_acts, stage2_indices)

        sae_out = trunk + coarse + stage1_out + stage2_out + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            aux_stage1 = torch.where(dead_mask[None], stage1_pre, -torch.inf)
            aux_stage2 = torch.where(dead_mask[None], stage2_pre, -torch.inf)

            aux1_k = min(self.stage1_k, k_aux)
            aux2_k = min(self.stage2_k, max(1, k_aux - aux1_k))

            aux1_acts, aux1_indices = aux_stage1.topk(aux1_k, sorted=False)
            aux1_out = self.decode_residual(aux1_acts, aux1_indices)

            aux2_acts, aux2_indices = aux_stage2.topk(aux2_k, sorted=False)
            aux2_out = self.decode_residual(aux2_acts, aux2_indices)

            e_hat = trunk + coarse + aux1_out + aux2_out + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class LowRankAsymmetricTwoStageSoftCodebookResidualSparseCoder(
    LowRankTwoStageSoftCodebookResidualSparseCoder
):
    """Soft-codebook two-stage residual SAE with a front-loaded sparse budget."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        # Favor the first refinement pass at smaller K so most support is spent
        # on the coarse residual after codebook projection, leaving a smaller
        # cleanup budget for the second pass.
        self.stage1_k = max(1, math.ceil(cfg.k * 0.75))
        self.stage2_k = max(1, cfg.k - self.stage1_k)


class RoutedLowRankTwoStageSoftCodebookResidualSparseCoder(
    LowRankTwoStageSoftCodebookResidualSparseCoder
):
    """Soft-codebook two-stage residual SAE with explicit router-driven support selection."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.stage1_router = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.stage2_router = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )

        nn.init.kaiming_uniform_(self.stage1_router.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.stage2_router.weight, a=5**0.5)
        self.stage1_router.bias.data.zero_()
        self.stage2_router.bias.data.zero_()

    def _encoder_linear_layers(self):
        return super()._encoder_linear_layers() + [("stage1_router", self.stage1_router), ("stage2_router", self.stage2_router)]

    @staticmethod
    def _straight_through_top_acts(
        acts: Tensor, routed_acts: Tensor, top_indices: Tensor
    ) -> Tensor:
        top_raw = acts.gather(-1, top_indices)
        top_routed = routed_acts.gather(-1, top_indices)
        # Preserve router gradients while decoding the unshrunk activations.
        return top_routed + (top_raw - top_routed).detach()

    def _route_stage(
        self,
        inputs: Tensor,
        weight: Tensor,
        bias: Tensor,
        router: nn.Linear,
        k: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        pre_acts = F.linear(inputs, weight, bias)
        acts = F.relu(pre_acts)
        router_logits = router(inputs) / self.cfg.gated_temperature
        router_gate = torch.sigmoid(router_logits)
        routed_acts = acts * router_gate
        scores = routed_acts + 0.1 * torch.tanh(router_logits)
        _, top_indices = torch.topk(scores, k, dim=-1, sorted=False)
        top_acts = self._straight_through_top_acts(acts, routed_acts, top_indices)
        return top_acts, top_indices, scores, acts, routed_acts

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse

        stage1_acts, stage1_indices, stage1_scores, _, _ = self._route_stage(
            code_residual,
            self.encoder.weight,
            self.encoder.bias,
            self.stage1_router,
            self.stage1_k,
        )
        stage1_out = self.decode_residual(stage1_acts, stage1_indices)

        stage2_input = code_residual - stage1_out
        stage2_acts, stage2_indices, stage2_scores, _, _ = self._route_stage(
            stage2_input,
            self.residual_encoder.weight,
            self.residual_encoder.bias,
            self.stage2_router,
            self.stage2_k,
        )

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)
        combined_scores = torch.maximum(stage1_scores, stage2_scores)
        return EncoderOutput(combined_acts, combined_indices, combined_scores)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        coarse, _ = self._project_codebook(residual)
        code_residual = residual - coarse

        stage1_acts, stage1_indices, stage1_scores, stage1_full, stage1_routed = self._route_stage(
            code_residual,
            self.encoder.weight,
            self.encoder.bias,
            self.stage1_router,
            self.stage1_k,
        )
        stage1_out = self.decode_residual(stage1_acts, stage1_indices)

        stage2_input = code_residual - stage1_out
        stage2_acts, stage2_indices, stage2_scores, _, _ = self._route_stage(
            stage2_input,
            self.residual_encoder.weight,
            self.residual_encoder.bias,
            self.stage2_router,
            self.stage2_k,
        )
        stage2_out = self.decode_residual(stage2_acts, stage2_indices)

        sae_out = trunk + coarse + stage1_out + stage2_out + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            aux_stage1 = torch.where(dead_mask[None], stage1_scores, -torch.inf)

            aux1_k = min(self.stage1_k, k_aux)
            aux2_k = min(self.stage2_k, max(1, k_aux - aux1_k))

            _, aux1_indices = aux_stage1.topk(aux1_k, sorted=False)
            aux1_acts = self._straight_through_top_acts(
                stage1_full, stage1_routed, aux1_indices
            )
            aux1_out = self.decode_residual(aux1_acts, aux1_indices)

            aux2_input = code_residual - aux1_out
            aux2_logits = F.linear(
                aux2_input,
                self.residual_encoder.weight,
                self.residual_encoder.bias,
            )
            aux2_full = F.relu(aux2_logits)
            aux2_router_logits = (
                self.stage2_router(aux2_input) / self.cfg.gated_temperature
            )
            aux2_routed = aux2_full * torch.sigmoid(aux2_router_logits)
            aux_stage2 = torch.where(
                dead_mask[None],
                aux2_routed + 0.1 * torch.tanh(aux2_router_logits),
                -torch.inf,
            )
            _, aux2_indices = aux_stage2.topk(aux2_k, sorted=False)
            aux2_acts = self._straight_through_top_acts(
                aux2_full, aux2_routed, aux2_indices
            )
            aux2_out = self.decode_residual(aux2_acts, aux2_indices)

            e_hat = trunk + coarse + aux1_out + aux2_out + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class RoutedLowRankAsymmetricTwoStageSoftCodebookResidualSparseCoder(
    RoutedLowRankTwoStageSoftCodebookResidualSparseCoder
):
    """Routed soft-codebook two-stage residual SAE with a front-loaded sparse budget."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.stage1_k = max(1, math.ceil(cfg.k * 0.75))
        self.stage2_k = max(1, cfg.k - self.stage1_k)


class LowRankGatedResidualSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk with a gated sparse residual selector."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.gate_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )

        # Start near the winning low-rank residual recipe, then let the gate
        # branch learn which residual features deserve support at low K.
        self.gate_encoder.weight.data.zero_()
        self.gate_encoder.bias.data.fill_(cfg.gated_init_logit)

    def _encoder_linear_layers(self):
        return super()._encoder_linear_layers() + [("gate_encoder", self.gate_encoder)]

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        pre_acts = F.linear(residual, self.encoder.weight, self.encoder.bias)
        positive = F.relu(pre_acts)
        gate_logits = self.gate_encoder(residual) / self.cfg.gated_temperature
        gate = torch.sigmoid(gate_logits)
        acts = positive * gate
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk

        pre_acts = F.linear(residual, self.encoder.weight, self.encoder.bias)
        positive = F.relu(pre_acts)
        gate_logits = self.gate_encoder(residual) / self.cfg.gated_temperature
        gate = torch.sigmoid(gate_logits)
        acts = positive * gate
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankJumpReLUResidualSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk with JumpReLU-smoothed sparse residual selection."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)

        threshold = torch.full(
            (self.num_latents,),
            cfg.jumprelu_init_threshold,
            device=self.device,
            dtype=self.dtype,
        ).clamp_min(torch.finfo(self.dtype).tiny)
        init = torch.full(
            (self.num_latents,), 0.0, device=self.device, dtype=self.dtype
        )
        init.copy_(torch.log(torch.expm1(threshold)))
        self.log_threshold = nn.Parameter(init)

    @property
    def threshold(self) -> Tensor:
        return F.softplus(self.log_threshold)

    def _jump_acts(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        pre_acts = F.linear(residual, self.encoder.weight, self.encoder.bias)
        positive = F.relu(pre_acts)
        gate = torch.sigmoid((positive - self.threshold) / self.cfg.jumprelu_bandwidth)
        return positive * gate, pre_acts

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        acts, pre_acts = self._jump_acts(residual)
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, pre_acts)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        acts, pre_acts = self._jump_acts(residual)
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class WhitenedLowRankResidualSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk with normalized residual preconditioning before top-k coding."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        mix_rank = min(d_in, max(cfg.k * 2, d_in // 8))
        self.preconditioner_down = nn.Linear(
            d_in, mix_rank, bias=False, device=device, dtype=dtype
        )
        self.preconditioner_up = nn.Linear(
            mix_rank, d_in, bias=False, device=device, dtype=dtype
        )

        # Start close to the working low-rank residual recipe while making the
        # support-selection space observably different from the plain residual basis.
        self.preconditioner_down.weight.data.zero_()
        self.preconditioner_up.weight.data.zero_()
        diagonal = min(d_in, mix_rank)
        self.preconditioner_down.weight.data[:diagonal, :diagonal] = torch.eye(
            diagonal, device=device, dtype=dtype
        )
        self.preconditioner_up.weight.data[:diagonal, :diagonal] = 0.05 * torch.eye(
            diagonal, device=device, dtype=dtype
        )

    def _encoder_linear_layers(self):
        return super()._encoder_linear_layers() + [("preconditioner_down", self.preconditioner_down), ("preconditioner_up", self.preconditioner_up)]

    def _precondition_residual(self, residual: Tensor) -> Tensor:
        centered = residual - residual.mean(dim=-1, keepdim=True)
        rms = centered.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        normalized = centered * rms
        mixed = normalized + 0.25 * torch.roll(normalized, shifts=1, dims=-1)
        correction = self.preconditioner_up(self.preconditioner_down(normalized))
        return mixed + correction

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        conditioned = self._precondition_residual(residual)
        return fused_encoder(
            conditioned, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        conditioned = self._precondition_residual(residual)
        top_acts, top_indices, pre_acts = fused_encoder(
            conditioned, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class WhitenedLowRankGatedResidualSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk with whitened residual support selection."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.gate_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )

        mix_rank = min(d_in, max(cfg.k * 2, d_in // 8))
        self.preconditioner_down = nn.Linear(
            d_in, mix_rank, bias=False, device=device, dtype=dtype
        )
        self.preconditioner_up = nn.Linear(
            mix_rank, d_in, bias=False, device=device, dtype=dtype
        )

        # Keep the initial behavior close to a stable normalized identity while
        # ensuring the architecture is observably distinct from plain residual gating.
        self.gate_encoder.weight.data.zero_()
        self.gate_encoder.bias.data.fill_(cfg.gated_init_logit)
        self.preconditioner_down.weight.data.zero_()
        self.preconditioner_up.weight.data.zero_()
        diagonal = min(d_in, mix_rank)
        self.preconditioner_down.weight.data[:diagonal, :diagonal] = torch.eye(
            diagonal, device=device, dtype=dtype
        )
        self.preconditioner_up.weight.data[:diagonal, :diagonal] = 0.05 * torch.eye(
            diagonal, device=device, dtype=dtype
        )

    def _encoder_linear_layers(self):
        return super()._encoder_linear_layers() + [("gate_encoder", self.gate_encoder), ("preconditioner_down", self.preconditioner_down), ("preconditioner_up", self.preconditioner_up)]

    def _precondition_residual(self, residual: Tensor) -> Tensor:
        centered = residual - residual.mean(dim=-1, keepdim=True)
        rms = centered.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        normalized = centered * rms
        mixed = normalized + 0.25 * torch.roll(normalized, shifts=1, dims=-1)
        correction = self.preconditioner_up(self.preconditioner_down(normalized))
        return mixed + correction

    def _compute_acts(self, residual: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        conditioned = self._precondition_residual(residual)
        pre_acts = F.linear(conditioned, self.encoder.weight, self.encoder.bias)
        positive = F.relu(pre_acts)
        gate_logits = self.gate_encoder(conditioned) / self.cfg.gated_temperature
        gate = torch.sigmoid(gate_logits)
        acts = positive * gate
        return acts, conditioned, pre_acts

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        acts, _, _ = self._compute_acts(residual)
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk

        acts, _, _ = self._compute_acts(residual)
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankGroupedResidualSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk with group-local winner selection on the sparse residual."""

    def _group_topk(
        self, acts: Tensor
    ) -> tuple[Tensor, Tensor]:
        group_size = self.cfg.group_topk_size
        if self.num_latents % group_size != 0:
            raise ValueError(
                "lowrank_grouped_residual requires num_latents divisible by "
                f"group_topk_size, got num_latents={self.num_latents} and "
                f"group_topk_size={group_size}"
            )

        num_groups = self.num_latents // group_size
        if self.cfg.k > num_groups:
            raise ValueError(
                "lowrank_grouped_residual requires k <= number of groups, "
                f"got k={self.cfg.k} and num_groups={num_groups}"
            )

        grouped = acts.view(*acts.shape[:-1], num_groups, group_size)
        winner_acts, winner_offsets = grouped.max(dim=-1)
        top_group_acts, top_group_indices = torch.topk(
            winner_acts, self.cfg.k, dim=-1, sorted=False
        )
        top_offsets = winner_offsets.gather(-1, top_group_indices)
        top_indices = top_group_indices * group_size + top_offsets
        return top_group_acts, top_indices

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        acts = F.relu(F.linear(residual, self.encoder.weight, self.encoder.bias))
        top_acts, top_indices = self._group_topk(acts)
        return EncoderOutput(top_acts, top_indices, acts)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        acts = F.relu(F.linear(residual, self.encoder.weight, self.encoder.bias))
        top_acts, top_indices = self._group_topk(acts)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class TwoStageResidualSparseCoder(SparseCoder):
    """Allocate a fixed total K across two sparse residual passes."""

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

        self.stage1_k = max(1, cfg.k // 2)
        self.stage2_k = max(1, cfg.k - self.stage1_k)

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()
        self.residual_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.residual_encoder.bias.data.zero_()

        if decoder:
            decoder_init = 0.5 * (
                self.encoder.weight.data + self.residual_encoder.weight.data
            )
            self.W_dec = nn.Parameter(decoder_init)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        return [("encoder", self.encoder), ("residual_encoder", self.residual_encoder)]

    def _decode_sparse(self, acts: Tensor, indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        return decoder_impl(indices, acts.to(self.dtype), self.W_dec.mT)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec

        stage1_acts, stage1_indices, stage1_pre = fused_encoder(
            x_centered, self.encoder.weight, self.encoder.bias, self.stage1_k
        )
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        residual = x_centered - stage1_out
        stage2_acts, stage2_indices, stage2_pre = fused_encoder(
            residual,
            self.residual_encoder.weight,
            self.residual_encoder.bias,
            self.stage2_k,
        )
        stage2_out = self._decode_sparse(stage2_acts, stage2_indices)

        sae_out = stage1_out + stage2_out + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            aux_stage1 = torch.where(dead_mask[None], stage1_pre, -torch.inf)
            aux_stage2 = torch.where(dead_mask[None], stage2_pre, -torch.inf)

            aux1_k = min(self.stage1_k, k_aux)
            aux2_k = min(self.stage2_k, max(1, k_aux - aux1_k))

            aux1_acts, aux1_indices = aux_stage1.topk(aux1_k, sorted=False)
            aux1_out = self._decode_sparse(aux1_acts, aux1_indices)

            aux2_acts, aux2_indices = aux_stage2.topk(aux2_k, sorted=False)
            aux2_out = self._decode_sparse(aux2_acts, aux2_indices)

            e_hat = aux1_out + aux2_out + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class MultiBranchGatedSparseCoder(SparseCoder):
    """Mix several gated encoder branches before the final top-k selection."""

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
        self.num_branches = 3

        self.branch_encoders = nn.ModuleList(
            [
                nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
                for _ in range(self.num_branches)
            ]
        )
        self.branch_gates = nn.ModuleList(
            [
                nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
                for _ in range(self.num_branches)
            ]
        )
        self.branch_mix_logits = nn.Linear(
            d_in, self.num_branches, device=device, dtype=dtype
        )

        for encoder in self.branch_encoders:
            encoder.bias.data.zero_()
        for gate in self.branch_gates:
            gate.weight.data.zero_()
            gate.bias.data.fill_(cfg.gated_init_logit)

        self.branch_mix_logits.weight.data.zero_()
        self.branch_mix_logits.bias.data.zero_()

        if decoder:
            decoder_init = torch.stack(
                [branch.weight.data for branch in self.branch_encoders], dim=0
            ).mean(dim=0)
            self.W_dec = nn.Parameter(decoder_init)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _encoder_linear_layers(self):
        layers = []
        for i, (enc, gate) in enumerate(zip(self.branch_encoders, self.branch_gates)):
            layers.append((f"branch_encoder_{i}", enc))
            layers.append((f"branch_gate_{i}", gate))
        layers.append(("branch_mix_logits", self.branch_mix_logits))
        return layers

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        branch_weights = torch.softmax(self.branch_mix_logits(x), dim=-1)

        branch_acts = []
        for branch_idx, (encoder, gate_encoder) in enumerate(
            zip(self.branch_encoders, self.branch_gates)
        ):
            pre_acts = F.linear(x, encoder.weight, encoder.bias)
            positive = F.relu(pre_acts)
            gate_logits = gate_encoder(x) / self.cfg.gated_temperature
            gate = torch.sigmoid(gate_logits)
            acts = positive * gate
            branch_weight = branch_weights[..., branch_idx].unsqueeze(-1)
            branch_acts.append(acts * branch_weight)

        acts = torch.stack(branch_acts, dim=0).sum(dim=0)
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)
