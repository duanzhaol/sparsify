"""Low-rank encoder implementation for SAE distillation.

This module contains all components for low-rank encoder training:
- LowRankSparseCoder: SparseCoder subclass with low-rank encoder
- LowRankFusedEncoder: Optimized autograd function with sparse gradients
- from_pretrained_lowrank: Factory function for SVD initialization
- compute_distillation_loss: Distillation loss computation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sparsify.config import SparseCoderConfig
from sparsify.fused_encoder import EncoderOutput

if TYPE_CHECKING:
    from sparsify.sparse_coder import SparseCoder


class LowRankFusedEncoder(torch.autograd.Function):
    """
    Low-rank fused encoder: (x @ B.T) @ A.T + bias → ReLU → TopK

    Reuses sparse gradient optimization from FusedEncoder:
    - grad_A: loop over k, use index_add_
    - grad_B: gather via embedding_bag
    """

    @staticmethod
    def forward(
        ctx, input, A, B, bias, k: int, activation: Literal["groupmax", "topk"]
    ):
        """
        input: (N, D)
        B:     (r, D)  - first projection
        A:     (M, r)  - second projection
        bias:  (M,)

        Computes: pre_acts = ReLU((x @ B.T) @ A.T + bias)
        """
        intermediate = input @ B.T  # [N, r]
        pre_linear = intermediate @ A.T  # [N, M]
        preacts = F.relu(pre_linear + bias)  # [N, M]

        if activation == "topk":
            values, indices = torch.topk(preacts, k, dim=-1, sorted=False)
        elif activation == "groupmax":
            values, indices = preacts.unflatten(-1, (k, -1)).max(dim=-1)
            num_latents = preacts.shape[1]
            offsets = torch.arange(
                0, num_latents, num_latents // k, device=preacts.device
            )
            indices = offsets + indices
        else:
            raise ValueError(f"Unknown activation: {activation}")

        ctx.save_for_backward(input, A, B, bias, intermediate, indices)
        ctx.k = k
        return values, indices, preacts

    @staticmethod
    def backward(ctx, grad_values, grad_indices, grad_preacts):
        input, A, B, bias, intermediate, indices = ctx.saved_tensors
        k = ctx.k
        r = B.shape[0]

        grad_A = grad_B = grad_bias = grad_input = None
        grad_intermediate = None

        # === Part 1: Gradient from grad_values (sparse, through top-k selection) ===

        # 1a. grad_bias from grad_values (sparse)
        if ctx.needs_input_grad[3]:
            grad_bias = torch.zeros_like(bias)
            grad_bias.index_add_(
                0, indices.flatten(), grad_values.flatten().type_as(bias)
            )

        # 1b. grad_A from grad_values (sparse, loop over k to avoid large tensor)
        if ctx.needs_input_grad[1]:
            grad_A = torch.zeros_like(A)
            for i in range(k):
                grad_v = grad_values[..., i]
                idx = indices[..., i]
                contrib = grad_v.unsqueeze(-1) * intermediate
                grad_A.index_add_(
                    0, idx.flatten(), contrib.reshape(-1, r).type_as(A)
                )

        # 1c. grad_B from grad_values (via chain rule)
        if ctx.needs_input_grad[2]:
            grad_intermediate = F.embedding_bag(
                indices, A, mode="sum", per_sample_weights=grad_values.type_as(A)
            )
            grad_B = grad_intermediate.T @ input.type_as(B)

        # 1d. grad_input from grad_values (optional, for end-to-end training)
        if ctx.needs_input_grad[0]:
            if grad_intermediate is None:
                grad_intermediate = F.embedding_bag(
                    indices, A, mode="sum", per_sample_weights=grad_values.type_as(A)
                )
            grad_input = grad_intermediate @ B

        # === Part 2: Gradient from grad_preacts (dense, for distillation acts_loss) ===
        # preacts = ReLU(intermediate @ A.T + bias)
        # grad_preacts flows back through ReLU and linear operations

        if grad_preacts is not None and grad_preacts.abs().sum() > 0:
            # Recompute ReLU mask: preacts > 0
            # pre_linear = intermediate @ A.T + bias
            intermediate_fp = intermediate.to(A.dtype)
            pre_linear = intermediate_fp @ A.T + bias
            relu_mask = (pre_linear > 0).type_as(grad_preacts)
            grad_pre_linear = grad_preacts * relu_mask  # [N, M]
            grad_pre_linear_fp = grad_pre_linear.to(A.dtype)

            # grad_bias += grad_pre_linear.sum(0)
            if ctx.needs_input_grad[3]:
                if grad_bias is None:
                    grad_bias = grad_pre_linear_fp.sum(0).type_as(bias)
                else:
                    grad_bias = grad_bias + grad_pre_linear_fp.sum(0).type_as(bias)

            # grad_A += intermediate.T @ grad_pre_linear -> transpose for weight format
            # A has shape [M, r], grad w.r.t. A: grad_pre_linear.T @ intermediate
            if ctx.needs_input_grad[1]:
                grad_A_dense = grad_pre_linear_fp.T @ intermediate_fp  # [M, r]
                if grad_A is None:
                    grad_A = grad_A_dense
                else:
                    grad_A = grad_A + grad_A_dense

            # grad_intermediate_dense = grad_pre_linear @ A
            # grad_B += grad_intermediate_dense.T @ input
            if ctx.needs_input_grad[2]:
                grad_intermediate_dense = grad_pre_linear_fp @ A  # [N, r]
                grad_B_dense = grad_intermediate_dense.T @ input.type_as(B)  # [r, D]
                if grad_B is None:
                    grad_B = grad_B_dense
                else:
                    grad_B = grad_B + grad_B_dense

            # grad_input from grad_preacts
            if ctx.needs_input_grad[0]:
                grad_intermediate_dense = grad_pre_linear @ A  # [N, r]
                grad_input_dense = grad_intermediate_dense @ B  # [N, D]
                if grad_input is None:
                    grad_input = grad_input_dense
                else:
                    grad_input = grad_input + grad_input_dense

        return grad_input, grad_A, grad_B, grad_bias, None, None


def lowrank_fused_encoder(
    input,
    A,
    B,
    bias,
    k: int,
    activation: Literal["groupmax", "topk"],
) -> EncoderOutput:
    """
    Convenience wrapper for low-rank encoder.
    """
    return EncoderOutput(
        *LowRankFusedEncoder.apply(input, A, B, bias, k, activation)  # type: ignore
    )


class LowRankSparseCoder(nn.Module):
    """SparseCoder variant with low-rank encoder (A @ B instead of W_enc).

    This class has the same interface as SparseCoder but uses a factorized
    encoder: pre_acts = ReLU((x @ B.T) @ A.T + bias)

    This reduces memory bandwidth during inference while maintaining accuracy
    through distillation training.
    """

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
        transcoder: bool = False,
    ):
        super().__init__()

        if cfg.encoder_rank <= 0:
            raise ValueError(
                "LowRankSparseCoder requires encoder_rank > 0. "
                "Use SparseCoder for full-rank encoder."
            )

        self.cfg = cfg
        self.d_in = d_in
        self.transcoder = transcoder
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        # Low-rank encoder: pre_acts = (x @ B.T) @ A.T + bias
        self.encoder = None  # Mark as low-rank mode (for compatibility checks)
        self.encoder_A = nn.Linear(
            cfg.encoder_rank, self.num_latents, device=device, dtype=dtype
        )
        self.encoder_B = nn.Linear(
            d_in, cfg.encoder_rank, bias=False, device=device, dtype=dtype
        )
        self.encoder_A.bias.data.zero_()

        if decoder:
            if transcoder:
                self.W_dec = nn.Parameter(
                    torch.zeros(self.num_latents, d_in, device=device, dtype=dtype)
                )
            else:
                # Initialize W_dec = A @ B
                self.W_dec = nn.Parameter(
                    (self.encoder_A.weight.data @ self.encoder_B.weight.data).clone()
                )
                if self.cfg.normalize_decoder:
                    self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))
        self.W_skip = (
            nn.Parameter(torch.zeros(d_in, d_in, device=device, dtype=dtype))
            if cfg.skip_connection
            else None
        )

    @property
    def device(self):
        return self.encoder_A.weight.device

    @property
    def dtype(self):
        return self.encoder_A.weight.dtype

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input using low-rank encoder and select top-k latents."""
        if not self.transcoder:
            x = x - self.b_dec

        return lowrank_fused_encoder(
            x,
            self.encoder_A.weight,
            self.encoder_B.weight,
            self.encoder_A.bias,
            self.cfg.k,
            self.cfg.activation,
        )

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        """Decode from sparse latent representation."""
        from sparsify.utils import decoder_impl

        assert self.W_dec is not None, "Decoder weight was not initialized."
        y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    @torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_bf16_supported(),
    )
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ):
        """Forward pass with optional AuxK loss computation."""
        from sparsify.sparse_coder import ForwardOutput

        top_acts, top_indices, pre_acts = self.encode(x)

        if y is None:
            y = x

        sae_out = self.decode(top_acts, top_indices)
        if self.W_skip is not None:
            sae_out += x.to(self.dtype) @ self.W_skip.mT

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        # AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        if self.cfg.multi_topk:
            top_acts, top_indices = pre_acts.topk(4 * self.cfg.k, sorted=False)
            sae_out = self.decode(top_acts, top_indices)
            multi_topk_fvu = (sae_out - y).pow(2).sum() / total_variance
        else:
            multi_topk_fvu = sae_out.new_tensor(0.0)

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
            multi_topk_fvu,
        )

    def save_to_disk(self, path):
        """Save model to disk."""
        from pathlib import Path
        import json
        from safetensors.torch import save_model

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

    @staticmethod
    def load_from_disk(
        path,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "LowRankSparseCoder":
        """Load model from disk."""
        from pathlib import Path
        import json
        from safetensors import safe_open
        from safetensors.torch import load_model

        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SparseCoderConfig.from_dict(cfg_dict, drop_extra_fields=True)

        safetensors_path = str(path / "sae.safetensors")

        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            first_key = next(iter(f.keys()))
            reference_dtype = f.get_tensor(first_key).dtype

        sae = LowRankSparseCoder(
            d_in, cfg, device=device, decoder=decoder, dtype=reference_dtype
        )

        load_model(
            model=sae,
            filename=safetensors_path,
            device=str(device),
            strict=decoder,
        )
        return sae

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
            # Decoder may be frozen during distillation; nothing to remove.
            return

        import einops

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


def from_pretrained_lowrank(
    teacher: "SparseCoder",
    rank: int,
    device: str | torch.device | None = None,
) -> LowRankSparseCoder:
    """Create a LowRankSparseCoder from a full-rank teacher using SVD initialization.

    Args:
        teacher: Full-rank SparseCoder to distill from
        rank: Target rank for low-rank encoder
        device: Device to place the model on (defaults to teacher's device)

    Returns:
        LowRankSparseCoder with SVD-initialized encoder
    """
    import copy

    if teacher.encoder is None:
        raise ValueError("Teacher must be a full-rank SparseCoder")

    cfg = copy.deepcopy(teacher.cfg)
    cfg.encoder_rank = rank

    device = device or teacher.device
    student = LowRankSparseCoder(
        teacher.d_in,
        cfg,
        device=device,
        dtype=teacher.dtype,
        decoder=True,
        transcoder=teacher.transcoder,
    )

    # SVD initialization for encoder
    W_enc = teacher.encoder.weight.data.float()
    U, S, Vh = torch.linalg.svd(W_enc, full_matrices=False)
    sqrt_S = torch.sqrt(S[:rank])

    # A @ B ≈ W_enc
    A_init = U[:, :rank] @ torch.diag(sqrt_S)  # [M, r]
    B_init = torch.diag(sqrt_S) @ Vh[:rank, :]  # [r, D]

    student.encoder_A.weight.data = A_init.to(teacher.dtype).to(device)
    student.encoder_B.weight.data = B_init.to(teacher.dtype).to(device)
    student.encoder_A.bias.data = teacher.encoder.bias.data.clone().to(device)

    # Copy decoder from teacher
    if teacher.W_dec is not None:
        student.W_dec.data = teacher.W_dec.data.clone().to(device)
    student.b_dec.data = teacher.b_dec.data.clone().to(device)

    if teacher.W_skip is not None and student.W_skip is not None:
        student.W_skip.data = teacher.W_skip.data.clone().to(device)

    return student


class DistillationLoss(NamedTuple):
    """Container for distillation loss components."""

    decode_loss: Tensor
    """MSE between student and teacher decode outputs."""

    acts_loss: Tensor
    """MSE between student and teacher pre-activations at teacher's top-k indices."""

    total: Tensor
    """Weighted sum of losses."""


def compute_distillation_loss(
    student: LowRankSparseCoder | "SparseCoder",
    teacher: "SparseCoder",
    inputs: Tensor,
    student_output: Tensor,
    lambda_decode: float = 0.5,
    lambda_acts: float = 0.1,
) -> DistillationLoss:
    """Compute distillation loss between student and teacher SAE.

    Args:
        student: Low-rank student (LowRankSparseCoder or SparseCoder)
        teacher: Full-rank teacher SparseCoder (frozen)
        inputs: Input tensor (raw, not centered)
        student_output: Student's decoded output (sae_out)
        lambda_decode: Weight for decode distillation loss
        lambda_acts: Weight for acts distillation loss

    Returns:
        DistillationLoss with decode_loss, acts_loss, and total
    """
    # Teacher forward (no grad)
    with torch.no_grad():
        t_acts, t_indices, t_pre = teacher.encode(inputs)
        t_decode = teacher.decode(t_acts, t_indices)

    # Decode distillation: match student output to teacher output
    decode_loss = (student_output - t_decode.detach()).pow(2).mean()

    # Top-k acts distillation: use teacher's indices
    # Get student's pre_acts by re-encoding
    _, _, s_pre = student.encode(inputs)
    s_pre_at_t_idx = s_pre.gather(-1, t_indices)
    t_pre_at_t_idx = t_pre.gather(-1, t_indices)
    acts_loss = (s_pre_at_t_idx - t_pre_at_t_idx.detach()).pow(2).mean()

    total = lambda_decode * decode_loss + lambda_acts * acts_loss

    return DistillationLoss(decode_loss, acts_loss, total)
