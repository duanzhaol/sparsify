"""Batched Conjugate Gradient solver for optimal SAE coefficients.

Solves the normal equation: (B_S B_S^T) α = B_S (x - b_dec)
where B_S ∈ R^{batch × K × h} are the selected basis vectors.

All internal computation is done in float32 for numerical stability,
regardless of input dtype.
"""

import torch
from torch import Tensor


def cg_solve(
    B_S: Tensor,
    rhs: Tensor,
    alpha_init: Tensor | None = None,
    max_iter: int = 10,
    tol: float = 1e-6,
    record_residuals: bool = False,
) -> tuple[Tensor, dict]:
    """Solve (B_S B_S^T) α = B_S x via Conjugate Gradient in float32.

    Args:
        B_S: Selected basis vectors, shape (batch, K, h). Any dtype.
        rhs: Target vector (x - b_dec), shape (batch, h). Any dtype.
        alpha_init: Initial coefficients, shape (batch, K). None for zero init.
        max_iter: Maximum CG iterations.
        tol: Convergence tolerance on relative residual norm.
        record_residuals: If True, record residual norm after each iteration.

    Returns:
        alpha: Optimal coefficients in float32, shape (batch, K).
        info: Dict with convergence info.
    """
    orig_dtype = B_S.dtype
    B_S = B_S.float()
    rhs = rhs.float()
    if alpha_init is not None:
        alpha_init = alpha_init.float()

    batch, K, h = B_S.shape

    # Right-hand side of normal equation: b = B_S @ rhs  -> (batch, K)
    b = torch.bmm(B_S, rhs.unsqueeze(-1)).squeeze(-1)  # (batch, K)

    # Gram matrix-vector product: A @ v = B_S @ (B_S^T @ v)
    def gram_mv(v: Tensor) -> Tensor:
        tmp = torch.bmm(v.unsqueeze(1), B_S).squeeze(1)  # (batch, h)
        return torch.bmm(B_S, tmp.unsqueeze(-1)).squeeze(-1)  # (batch, K)

    # Initialize
    if alpha_init is not None:
        alpha = alpha_init.clone()
        r = b - gram_mv(alpha)
    else:
        alpha = torch.zeros(batch, K, device=B_S.device, dtype=torch.float32)
        r = b.clone()

    p = r.clone()
    rs_old = (r * r).sum(dim=-1)  # (batch,)

    b_norm_sq = (b * b).sum(dim=-1).clamp(min=1e-12)

    residuals = []
    converged = torch.zeros(batch, dtype=torch.bool, device=B_S.device)
    iters_to_converge = torch.full(
        (batch,), max_iter, dtype=torch.float, device=B_S.device
    )

    for i in range(max_iter):
        if record_residuals:
            rel_res = (rs_old / b_norm_sq).sqrt().mean().item()
            residuals.append(rel_res)

        newly_converged = (rs_old / b_norm_sq).sqrt() < tol
        just_converged = newly_converged & ~converged
        iters_to_converge[just_converged] = float(i)
        converged = converged | newly_converged

        if converged.all():
            break

        Ap = gram_mv(p)
        pAp = (p * Ap).sum(dim=-1).clamp(min=1e-12)
        step = rs_old / pAp

        alpha = alpha + step.unsqueeze(-1) * p
        r = r - step.unsqueeze(-1) * Ap

        rs_new = (r * r).sum(dim=-1)
        beta = rs_new / rs_old.clamp(min=1e-12)
        p = r + beta.unsqueeze(-1) * p

        rs_old = rs_new

    if record_residuals:
        rel_res = (rs_old / b_norm_sq).sqrt().mean().item()
        residuals.append(rel_res)

    info = {
        "iters_mean": iters_to_converge.mean().item(),
        "iters_median": iters_to_converge.median().item(),
    }
    if record_residuals:
        info["residuals"] = residuals

    return alpha, info


def exact_solve(
    B_S: Tensor,
    rhs: Tensor,
) -> Tensor:
    """Solve for optimal coefficients via exact least squares in float32.

    Uses torch.linalg.lstsq for maximum numerical accuracy.
    This serves as the upper bound for what any iterative solver can achieve.

    Args:
        B_S: Selected basis vectors, shape (batch, K, h). Any dtype.
        rhs: Target vector (x - b_dec), shape (batch, h). Any dtype.

    Returns:
        alpha: Optimal coefficients in float32, shape (batch, K).
    """
    B_S = B_S.float()
    rhs = rhs.float()

    # lstsq solves: min ||B_S^T @ alpha - rhs||^2
    # i.e. A @ x = b where A = B_S^T (K, h)^T = (h, K), b = rhs (h,)
    # We need to solve for each sample in the batch
    # B_S: (batch, K, h), rhs: (batch, h)
    # Rewrite as: B_S^T @ alpha = rhs  =>  (h, K) @ (K,) = (h,)
    A = B_S.transpose(1, 2)  # (batch, h, K)
    b = rhs.unsqueeze(-1)  # (batch, h, 1)

    result = torch.linalg.lstsq(A, b)
    alpha = result.solution.squeeze(-1)  # (batch, K)
    return alpha
