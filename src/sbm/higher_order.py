"""
Higher-order (polynomial) unconstrained optimisation — reduce to QUBO / Ising.

For problems with cubic or higher-degree terms, we provide helper functions
that map them onto quadratic Ising Hamiltonians suitable for SB solvers.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════
# CubicOptimizer  —  f(x) = x·A·x·x + x·B·x   (cubic + quadratic)
# ═══════════════════════════════════════════════════════════════════════════

class CubicOptimizer:
    """Cubic + quadratic objective over continuous variables.

    Minimises
        f(x) = ∑_{i,j,k} A_{ijk} · x_i · x_j · x_k  +  ∑_{i,j} B_{ij} · x_i · x_j

    where ``A`` is a 3‑tensor and ``B`` is a symmetric matrix.

    Parameters
    ----------
    A : Tensor (N, N, N)
        Cubic coefficients (may be non‑symmetric).
    B : Tensor (N, N)
        Quadratic coefficients.
    dim : int
        Number of variables N.
    """

    def __init__(self, A: torch.Tensor, B: torch.Tensor, dim: int):
        super().__init__()
        self.A = A
        self.B = B
        self.dim = dim

    @property
    def N(self) -> int:
        return self.A.shape[0]

    def objective(self, sol: torch.Tensor) -> torch.Tensor:
        """Evaluate the full objective (cubic + quadratic)."""
        assert sol.numel() == self.N
        _sol = sol.view(-1)
        cubic = torch.einsum("ijk,i,j,k", self.A, _sol, _sol, _sol)
        quad = torch.einsum("ij,i,j", self.B, _sol, _sol)
        return cubic + quad

    def to_state(self, state):
        """Identity placeholder (for compatibility with SB solver pipelines)."""
        return state

    def hessian_at_origin(self) -> torch.Tensor:
        """Hessian matrix at ``x = 0`` — the quadratic part only.

        .. math::
            H_{ij} = \\left.\\frac{\\partial^2 f}{\\partial x_i \\partial x_j}
            \\right\\rvert_{x=0}
        """
        from torch.autograd.functional import hessian
        zeros = torch.zeros(self.dim)
        H = hessian(lambda x: self.objective(x), zeros)
        return H


# ═══════════════════════════════════════════════════════════════════════════
# QPLIB loader  —  general QUBO with linear terms
# ═══════════════════════════════════════════════════════════════════════════

def qplib_to_ising(
    Q: np.ndarray,
    b: np.ndarray,
    num_vars: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, int]:
    """Convert a QPLIB-format (Q, b) problem to an Ising coupling matrix.

    The QPLIB objective is :math:`\\frac{1}{2} x^\\top Q x + b^\\top x + q^0`.

    The conversion uses an extra auxiliary spin to absorb the linear term:

        J_ising = Q / 8
        h_ising = Q·1 / 4 + b / 2
        J[extra_var, :] = J[: extra_var] = 0.7 · h_ising

    Parameters
    ----------
    Q : ndarray (N, N)
        Quadratic coefficient matrix.
    b : ndarray (N,)
        Linear coefficient vector.
    num_vars : int
        Number of variables N.
    device : torch.device or None
        Target device.

    Returns
    -------
    J : Tensor (N+1, N+1)
        Ising coupling matrix (one extra variable added).
    num_vars : int
        New variable count (``N+1``).
    """
    if device is None:
        device = torch.device("cpu")

    Q_t = torch.tensor(Q, dtype=torch.float32, device=device)
    Q_sym = 0.5 * (Q_t + Q_t.T)

    b_t = torch.tensor(b, dtype=torch.float32, device=device)
    ones = torch.ones(num_vars, dtype=torch.float32, device=device)

    J_ising = 0.125 * Q_sym
    h_ising = 0.25 * torch.matmul(Q_sym, ones) + 0.5 * b_t

    J_tensor = torch.zeros((num_vars + 1, num_vars + 1), device=device)
    J_tensor[:num_vars, :num_vars] = J_ising
    J_tensor[:num_vars, num_vars] = 0.7 * h_ising
    J_tensor[num_vars, :num_vars] = 0.7 * h_ising

    return J_tensor, num_vars + 1
