"""
Tests for :mod:`sbm.higher_order` — CubicOptimizer & QPLIB loader.
"""

import numpy as np
import torch

from sbm.higher_order import CubicOptimizer, qplib_to_ising


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


# ═══════════════════════════════════════════════════════════════════════════
# 2. QPLIB to Ising
# ═══════════════════════════════════════════════════════════════════════════

def test_qplib_to_ising_shape():
    """qplib_to_ising returns (N+1)×(N+1) matrix."""
    N = 4
    Q = np.eye(N)
    b = np.ones(N)
    J, new_n = qplib_to_ising(Q, b, N)
    assert J.shape == (N + 1, N + 1), f"Expected {(N+1,N+1)}, got {J.shape}"
    assert new_n == N + 1


def test_qplib_to_ising_symmetric():
    """Output J is symmetric."""
    N = 5
    rng = np.random.default_rng(123)
    Q = rng.uniform(-1, 1, size=(N, N))
    b = rng.uniform(-1, 1, size=N)
    J, _ = qplib_to_ising(Q, b, N)
    assert torch.allclose(J, J.T, atol=1e-6), "J must be symmetric"


def test_qplib_to_ising_device():
    """Device parameter is respected."""
    N = 3
    Q = np.eye(N)
    b = np.zeros(N)
    J, _ = qplib_to_ising(Q, b, N, device=torch.device("cpu"))
    assert J.device.type == "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Integration: higher-order → SB solve
# ═══════════════════════════════════════════════════════════════════════════

def test_solve_with_qplib():
    """qplib_to_ising output can be fed to BaseSolver."""
    from sbm import BaseSolver, BSBStrategy

    N = 6
    # Simple MaxCut-like Q: Q = -adjacency (so Ising coupling is positive)
    adj = torch.eye(N)
    Q_np = -adj.numpy()
    b_np = np.zeros(N)

    J, _ = qplib_to_ising(Q_np, b_np, N)
    solver = BaseSolver(strategy=BSBStrategy(dt=0.5), num_iters=100, num_trials=5)
    sols, engs = solver.solve(J)
    assert sols.shape[0] == 5   # 5 trials
    assert sols.shape[1] == N + 1  # N+1 variables (extra auxiliary)


# ═══════════════════════════════════════════════════════════════════════════
# 4. TSP legalizer (from problems.py)
# ═══════════════════════════════════════════════════════════════════════════

def test_tsp_legalizer_valid():
    """Valid spin configuration passes through legalizer unchanged."""
    from sbm.problems import tsp_extract_with_legalizer

    N = 4
    # A valid one-hot assignment: city i at position i
    spins = np.zeros(N * N)
    for i in range(N):
        spins[i * N + i] = 1.0

    path, was_valid, cost = tsp_extract_with_legalizer(spins, N)
    assert was_valid, "Valid config should be detected as valid"
    assert path == [0, 1, 2, 3], f"Expected [0,1,2,3], got {path}"


def test_tsp_legalizer_invalid():
    """Invalid spin configuration is repaired by legalizer."""
    from sbm.problems import tsp_extract_with_legalizer

    N = 4
    # All spins = 0 (empty solution) → legalizer should fill in something
    spins = np.zeros(N * N)

    path, was_valid, cost = tsp_extract_with_legalizer(spins, N)
    assert not was_valid, "Empty config should be invalid"
    assert len(path) == N
    assert -1 not in path, "No missing positions after legalizer"
    assert len(set(path)) == N, "All cities present after legalizer"


def test_tsp_legalizer_overlap():
    """Overlapping city assignments are resolved by legalizer."""
    from sbm.problems import tsp_extract_with_legalizer

    N = 3
    # All three cities claim position 0
    spins = np.zeros(N * N)
    for i in range(N):
        spins[i * N + 0] = 1.0  # city i → position 0

    path, was_valid, cost = tsp_extract_with_legalizer(spins, N)
    assert not was_valid, "Overlap should be invalid"
    assert len(path) == N
    assert -1 not in path
    assert len(set(path)) == N
