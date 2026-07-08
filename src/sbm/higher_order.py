"""
Higher-order (polynomial) unconstrained optimisation — reduce to QUBO / Ising.

For problems with cubic or higher-degree terms, we provide helper functions
that map them onto quadratic Ising Hamiltonians suitable for SB solvers.

.. note::

   The QPLIB / QUBO converter ``qplib_to_ising`` has moved to
   :mod:`sbm.problems` (it is a problem-type converter, not a higher-order
   tool).  It is re-exported here for backward compatibility.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .problems import qplib_to_ising  # noqa: F401 — re-export for backward compat


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
