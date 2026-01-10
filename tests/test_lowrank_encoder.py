"""Tests for low-rank encoder implementation."""

import pytest
import torch
import torch.nn.functional as F

from lowrank_encoder import (
    LowRankFusedEncoder,
    LowRankSparseCoder,
    lowrank_fused_encoder,
    compute_distillation_loss,
    from_pretrained_lowrank,
)
from sparsify.fused_encoder import fused_encoder


class TestLowRankFusedEncoder:
    """Test LowRankFusedEncoder gradient correctness."""

    @pytest.mark.parametrize("activation", ["topk", "groupmax"])
    def test_gradcheck_lowrank_encoder(self, activation):
        """Verify gradients using torch.autograd.gradcheck."""
        torch.manual_seed(42)

        # Use float64 for numerical gradient checking
        batch_size = 8
        d_in = 64
        num_latents = 256
        rank = 16
        k = 8

        input = torch.randn(batch_size, d_in, requires_grad=True, dtype=torch.float64)
        A = torch.randn(num_latents, rank, requires_grad=True, dtype=torch.float64)
        B = torch.randn(rank, d_in, requires_grad=True, dtype=torch.float64)
        bias = torch.randn(num_latents, requires_grad=True, dtype=torch.float64)

        # Only check gradients for values (not indices or preacts)
        def func(inp, a, b, bi):
            values, indices, preacts = LowRankFusedEncoder.apply(inp, a, b, bi, k, activation)
            return values

        # Use non-deterministic algorithm for gradcheck
        torch.autograd.gradcheck(func, (input, A, B, bias), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_forward_output_shapes(self):
        """Test output shapes are correct."""
        torch.manual_seed(42)

        batch_size = 16
        d_in = 128
        num_latents = 512
        rank = 32
        k = 16

        input = torch.randn(batch_size, d_in)
        A = torch.randn(num_latents, rank)
        B = torch.randn(rank, d_in)
        bias = torch.randn(num_latents)

        values, indices, preacts = lowrank_fused_encoder(input, A, B, bias, k, "topk")

        assert values.shape == (batch_size, k)
        assert indices.shape == (batch_size, k)
        assert preacts.shape == (batch_size, num_latents)

    def test_lowrank_approximates_fullrank(self):
        """Test that low-rank encoder approximates full-rank when rank is high."""
        torch.manual_seed(42)

        batch_size = 16
        d_in = 64
        num_latents = 256
        k = 8

        # Create full-rank encoder
        W_full = torch.randn(num_latents, d_in)
        bias = torch.randn(num_latents)

        # SVD decomposition with full rank
        U, S, Vh = torch.linalg.svd(W_full, full_matrices=False)
        rank = min(num_latents, d_in)  # Full rank
        sqrt_S = torch.sqrt(S[:rank])
        A = U[:, :rank] @ torch.diag(sqrt_S)
        B = torch.diag(sqrt_S) @ Vh[:rank, :]

        # Verify A @ B ≈ W_full
        W_reconstructed = A @ B
        assert torch.allclose(W_full, W_reconstructed, atol=1e-5)

        # Test forward pass gives same results
        input = torch.randn(batch_size, d_in)

        full_values, full_indices, full_preacts = fused_encoder(input, W_full, bias, k, "topk")
        low_values, low_indices, low_preacts = lowrank_fused_encoder(input, A, B, bias, k, "topk")

        # Pre-activations should be nearly identical
        assert torch.allclose(full_preacts, low_preacts, atol=1e-4)

    def test_backward_sparse_gradient(self):
        """Test that gradients are computed correctly for sparse activations."""
        torch.manual_seed(42)

        batch_size = 8
        d_in = 64
        num_latents = 256
        rank = 16
        k = 8

        input = torch.randn(batch_size, d_in, requires_grad=True)
        A = torch.randn(num_latents, rank, requires_grad=True)
        B = torch.randn(rank, d_in, requires_grad=True)
        bias = torch.randn(num_latents, requires_grad=True)

        values, indices, preacts = lowrank_fused_encoder(input, A, B, bias, k, "topk")

        # Compute some loss and backprop
        loss = values.sum()
        loss.backward()

        # Check gradients exist
        assert input.grad is not None
        assert A.grad is not None
        assert B.grad is not None
        assert bias.grad is not None

        # Check gradient shapes
        assert input.grad.shape == input.shape
        assert A.grad.shape == A.shape
        assert B.grad.shape == B.shape
        assert bias.grad.shape == bias.shape


class TestLowRankSparseCoder:
    """Test LowRankSparseCoder."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_lowrank_encode_decode(self):
        """Test encoding and decoding with low-rank encoder."""
        from sparsify.config import SparseCoderConfig

        device = "cuda"
        d_in = 128
        cfg = SparseCoderConfig(
            expansion_factor=8,
            k=16,
            encoder_rank=32,  # Low-rank
        )

        sae = LowRankSparseCoder(d_in, cfg, device=device, dtype=torch.float32)

        # Verify low-rank structure
        assert sae.encoder is None
        assert sae.encoder_A is not None
        assert sae.encoder_B is not None

        # Test forward pass
        x = torch.randn(8, d_in, device=device)
        out = sae(x)

        assert out.sae_out.shape == x.shape
        assert out.latent_acts.shape == (8, cfg.k)
        assert out.latent_indices.shape == (8, cfg.k)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_from_pretrained_lowrank(self):
        """Test SVD initialization from full-rank teacher."""
        from sparsify.sparse_coder import SparseCoder
        from sparsify.config import SparseCoderConfig

        device = "cuda"
        d_in = 128

        # Create full-rank teacher
        teacher_cfg = SparseCoderConfig(expansion_factor=8, k=16)
        teacher = SparseCoder(d_in, teacher_cfg, device=device, dtype=torch.float32)

        # Create low-rank student via SVD
        rank = 32
        student = from_pretrained_lowrank(teacher, rank, device=device)

        # Verify it's a LowRankSparseCoder
        assert isinstance(student, LowRankSparseCoder)

        # Verify structure
        assert student.encoder is None
        assert student.encoder_A is not None
        assert student.encoder_B is not None
        assert student.encoder_A.weight.shape == (teacher.num_latents, rank)
        assert student.encoder_B.weight.shape == (rank, d_in)

        # Verify decoder is copied
        assert torch.allclose(student.W_dec.data, teacher.W_dec.data)
        assert torch.allclose(student.b_dec.data, teacher.b_dec.data)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_classmethod_wrapper(self):
        """Test SparseCoder.from_pretrained_lowrank classmethod."""
        from sparsify.sparse_coder import SparseCoder
        from sparsify.config import SparseCoderConfig

        device = "cuda"
        d_in = 128

        teacher_cfg = SparseCoderConfig(expansion_factor=8, k=16)
        teacher = SparseCoder(d_in, teacher_cfg, device=device, dtype=torch.float32)

        # Use classmethod
        student = SparseCoder.from_pretrained_lowrank(teacher, rank=32, device=device)

        # Verify it returns a LowRankSparseCoder
        assert isinstance(student, LowRankSparseCoder)


class TestDistillationLoss:
    """Test distillation loss computation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_compute_distillation_loss(self):
        """Test distillation loss returns valid values."""
        from sparsify.sparse_coder import SparseCoder
        from sparsify.config import SparseCoderConfig

        device = "cuda"
        d_in = 128

        # Create teacher (full-rank)
        teacher_cfg = SparseCoderConfig(expansion_factor=8, k=16)
        teacher = SparseCoder(d_in, teacher_cfg, device=device, dtype=torch.float32)
        teacher.requires_grad_(False)

        # Create student (low-rank)
        rank = 32
        student = SparseCoder.from_pretrained_lowrank(teacher, rank, device=device)

        # Forward pass
        x = torch.randn(8, d_in, device=device)
        student_out = student(x)

        # Compute distillation loss
        loss = compute_distillation_loss(
            student=student,
            teacher=teacher,
            inputs=x,
            student_output=student_out.sae_out,
            lambda_decode=0.5,
            lambda_acts=0.1,
        )

        # Verify loss structure
        assert loss.decode_loss >= 0
        assert loss.acts_loss >= 0
        assert loss.total >= 0
        assert loss.total == 0.5 * loss.decode_loss + 0.1 * loss.acts_loss

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_distillation_loss_gradients(self):
        """Test gradients flow through distillation loss."""
        from sparsify.sparse_coder import SparseCoder
        from sparsify.config import SparseCoderConfig

        device = "cuda"
        d_in = 128

        teacher_cfg = SparseCoderConfig(expansion_factor=8, k=16)
        teacher = SparseCoder(d_in, teacher_cfg, device=device, dtype=torch.float32)
        teacher.requires_grad_(False)

        student = SparseCoder.from_pretrained_lowrank(teacher, rank=32, device=device)

        x = torch.randn(8, d_in, device=device)
        student_out = student(x)

        loss = compute_distillation_loss(
            student=student,
            teacher=teacher,
            inputs=x,
            student_output=student_out.sae_out,
        )

        # Backprop
        loss.total.backward()

        # Verify gradients exist
        assert student.encoder_A.weight.grad is not None
        assert student.encoder_B.weight.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_acts_loss_gradient_flows_to_encoder(self):
        """Test that acts_loss gradient (via grad_preacts) flows to encoder params.

        This specifically tests the fix for the grad_preacts bug where acts_loss
        gradients were not propagated to encoder_A and encoder_B.
        """
        from sparsify.sparse_coder import SparseCoder
        from sparsify.config import SparseCoderConfig

        device = "cuda"
        d_in = 128

        teacher_cfg = SparseCoderConfig(expansion_factor=8, k=16)
        teacher = SparseCoder(d_in, teacher_cfg, device=device, dtype=torch.float32)
        teacher.requires_grad_(False)

        student = SparseCoder.from_pretrained_lowrank(teacher, rank=32, device=device)

        x = torch.randn(8, d_in, device=device)

        # Compute ONLY acts_loss (lambda_decode=0)
        loss = compute_distillation_loss(
            student=student,
            teacher=teacher,
            inputs=x,
            student_output=student(x).sae_out,
            lambda_decode=0.0,  # Only acts_loss contributes
            lambda_acts=1.0,
        )

        # Clear any existing gradients
        student.zero_grad()

        # Backprop only acts_loss
        loss.total.backward()

        # Verify gradients exist and are non-zero
        # This tests that grad_preacts is correctly handled in backward
        assert student.encoder_A.weight.grad is not None, "encoder_A.weight.grad is None"
        assert student.encoder_B.weight.grad is not None, "encoder_B.weight.grad is None"
        assert student.encoder_A.bias.grad is not None, "encoder_A.bias.grad is None"

        # Gradients should be non-zero (acts_loss produces meaningful gradients)
        assert student.encoder_A.weight.grad.abs().sum() > 0, "encoder_A.weight.grad is zero"
        assert student.encoder_B.weight.grad.abs().sum() > 0, "encoder_B.weight.grad is zero"
        assert student.encoder_A.bias.grad.abs().sum() > 0, "encoder_A.bias.grad is zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
