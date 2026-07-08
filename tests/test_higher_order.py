"""
Tests for :mod:`sbm.higher_order` — CubicOptimizer.

QPLIB / problem-type tests have moved to :mod:`tests.test_problems`.
"""

import numpy as np
import torch

from sbm.higher_order import CubicOptimizer


# ═══════════════════════════════════════════════════════════════════════════
# 1. CubicOptimizer
# ═══════════════════════════════════════════════════════════════════════════

def test_cubic_optimizer_basic():
    """CubicOptimizer objective evaluates correctly for known values."""
    dim = 3
    A = torch.zeros(dim, dim, dim)
    B = torch.eye(dim)  # B = I  →  x·B·x = Σ x_i²
    opt = CubicOptimizer(A, B, dim)

    sol = torch.tensor([1.0, 2.0, 3.0])
    # quadratic part only:  1²+2²+3² = 14
    expected = 14.0
    assert torch.allclose(opt.objective(sol), torch.tensor(expected)), (
        f"Expected {expected}, got {opt.objective(sol)}"
    )


def test_cubic_optimizer_cubic_term():
    """Cubic term A contributes correctly."""
    dim = 2
    A = torch.zeros(dim, dim, dim)
    A[0, 0, 0] = 2.0   # 2·x₀³
    A[1, 0, 0] = 3.0   # 3·x₁·x₀²
    B = torch.zeros(dim, dim)
    opt = CubicOptimizer(A, B, dim)

    sol = torch.tensor([2.0, 1.0])
    # 2·(2)³ + 3·(1)·(2)² = 16 + 12 = 28
    expected = 28.0
    assert torch.allclose(opt.objective(sol), torch.tensor(expected)), (
        f"Expected {expected}, got {opt.objective(sol)}"
    )


def test_cubic_optimizer_hessian():
    """Hessian at origin equals quadratic coefficient."""
    dim = 4
    rng = np.random.default_rng(42)
    B = torch.tensor(rng.normal(size=(dim, dim)), dtype=torch.float32)
    B = (B + B.T) / 2  # symmetrise
    A = torch.zeros(dim, dim, dim)
    opt = CubicOptimizer(A, B, dim)

    H = opt.hessian_at_origin()
    # At x=0, Hessian should be 2·B (since ∂²/∂xᵢ∂xⱼ of x·B·x is 2·B)
    expected = 2.0 * B
    assert torch.allclose(H, expected, atol=1e-5), (
        f"Hessian mismatch\n{H}\n!=\n{expected}"
    )


def test_cubic_optimizer_property():
    """The N property matches dim."""
    opt = CubicOptimizer(torch.zeros(5, 5, 5), torch.eye(5), 5)
    assert opt.N == 5
