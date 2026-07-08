"""
Tests for :mod:`sbm.problems` — all problem-type converters.

Covers: MaxCut, Balanced MinCut, Max-3SAT, QUBO / QPLIB, TSP legalizer.
"""

import numpy as np
import torch

from sbm import BaseSolver, BSBStrategy
from sbm.problems import (
    maxcut_to_ising, maxcut_cut_value,
    bmincut_to_ising, bmincut_cut_value,
    max3sat_to_ising, maxsat_count_satisfied,
    qubo_to_ising, qplib_to_ising,
    read_tsplib, tsp_coords_to_distance,
    tsp_extract_solution, tsp_tour_distance, tsp_validate,
    tsp_extract_with_legalizer,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. MaxCut
# ═══════════════════════════════════════════════════════════════════════════

def test_maxcut_simple():
    """MaxCut on a two-node edge."""
    J = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    J_ising = maxcut_to_ising(J)
    assert J_ising.shape == (2, 2)
    # J_ising = -J/2 → edge = -0.5
    assert torch.allclose(J_ising[0, 1], torch.tensor(-0.5))


def test_maxcut_cut_value():
    """cut_value for an edge."""
    J = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    spins = torch.tensor([1.0, -1.0])  # on opposite sides → cut
    cut = maxcut_cut_value(J, spins)
    assert torch.allclose(cut, torch.tensor(2.0)), f"Expected 2, got {cut}"


def test_maxcut_solve_small():
    """Solve MaxCut on a small graph via BaseSolver."""
    J = torch.tensor([[0.0, 1.0, 1.0],
                      [1.0, 0.0, 1.0],
                      [1.0, 1.0, 0.0]])
    J_ising = maxcut_to_ising(J)
    solver = BaseSolver(strategy=BSBStrategy(dt=0.5), num_iters=200, num_trials=10)
    _, engs = solver.solve(J_ising)
    # MaxCut value for triangle: 2 (cut all edges)
    assert engs.shape[0] == 10


# ═══════════════════════════════════════════════════════════════════════════
# 2. Balanced MinCut
# ═══════════════════════════════════════════════════════════════════════════

def test_bmincut_simple():
    """bmincut_to_ising adds balance penalty."""
    J = torch.eye(3)
    J_plain = bmincut_to_ising(J, lambda_balance=0.0)
    J_balanced = bmincut_to_ising(J, lambda_balance=2.0)
    # With balance, the off-diagonal should be more negative
    assert J_balanced[0, 1] < J_plain[0, 1], (
        "Balance penalty should make off-diagonal more negative"
    )


def test_bmincut_cut_value():
    """cut_value for a simple graph."""
    J = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    spins = torch.tensor([1.0, -1.0])
    cut, imb = bmincut_cut_value(J, spins)
    assert torch.allclose(cut, torch.tensor(1.0)), f"Expected cut=1, got {cut}"
    assert torch.allclose(imb, torch.tensor(0.0)), f"Expected imbalance 0, got {imb}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Max-3SAT
# ═══════════════════════════════════════════════════════════════════════════

def test_max3sat_simple_clause():
    """A single clause (1, 2, 3)."""
    clauses = [(1, 2, 3)]
    J = max3sat_to_ising(clauses, num_vars=3)
    assert J.shape == (4, 4)  # N+1


def test_max3sat_all_true():
    """All variables true should satisfy all clauses."""
    # (1, 2, 3) — all positive → x1=1, x2=1, x3=1 satisfies it
    clauses = [(1, 2, 3)]
    J = max3sat_to_ising(clauses, num_vars=3)
    spins = torch.tensor([1.0, 1.0, 1.0])  # all true
    n = maxsat_count_satisfied(clauses, spins)
    assert n == 1, f"Expected 1 satisfied, got {n}"


def test_max3sat_all_false():
    """All variables false should leave clause unsatisfied."""
    clauses = [(1, 2, 3)]
    J = max3sat_to_ising(clauses, num_vars=3)
    spins = torch.tensor([-1.0, -1.0, -1.0])
    n = maxsat_count_satisfied(clauses, spins)
    assert n == 0, f"Expected 0 satisfied, got {n}"


def test_max3sat_solve():
    """Solve a small Max-3SAT problem."""
    clauses = [(1, -2, 3), (-1, 2, -3)]
    J = max3sat_to_ising(clauses, num_vars=3)
    solver = BaseSolver(strategy=BSBStrategy(dt=0.5), num_iters=300, num_trials=20)
    sols, _ = solver.solve(J)
    # Just check the solver runs without error
    assert sols.shape[0] == 20


# ═══════════════════════════════════════════════════════════════════════════
# 4. QUBO / QPLIB
# ═══════════════════════════════════════════════════════════════════════════

def test_qubo_to_ising_shape():
    """qubo_to_ising returns (N+1)×(N+1) matrix."""
    N = 4
    Q = torch.eye(N)
    J = qubo_to_ising(Q)
    assert J.shape == (N + 1, N + 1), f"Expected {(N+1,N+1)}, got {J.shape}"


def test_qubo_to_ising_symmetric():
    """Output J is symmetric."""
    Q = torch.randn(5, 5)
    Q = (Q + Q.T) / 2
    J = qubo_to_ising(Q)
    assert torch.allclose(J, J.T, atol=1e-6), "J must be symmetric"


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


def test_solve_with_qplib():
    """qplib_to_ising output can be fed to BaseSolver."""
    N = 6
    adj = torch.eye(N)
    Q_np = -adj.numpy()
    b_np = np.zeros(N)

    J, _ = qplib_to_ising(Q_np, b_np, N)
    solver = BaseSolver(strategy=BSBStrategy(dt=0.5), num_iters=100, num_trials=5)
    sols, engs = solver.solve(J)
    assert sols.shape[0] == 5
    assert sols.shape[1] == N + 1


def test_solve_with_qubo():
    """qubo_to_ising output can be fed to BaseSolver."""
    N = 5
    Q = torch.eye(N)
    J = qubo_to_ising(Q)
    solver = BaseSolver(strategy=BSBStrategy(dt=0.5), num_iters=100, num_trials=5)
    sols, _ = solver.solve(J)
    assert sols.shape[0] == 5
    assert sols.shape[1] == N + 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. TSP legalizer (from problems.py)
# ═══════════════════════════════════════════════════════════════════════════

def test_tsp_legalizer_valid():
    """Valid spin configuration passes through legalizer unchanged."""
    N = 4
    spins = np.zeros(N * N)
    for i in range(N):
        spins[i * N + i] = 1.0

    path, was_valid, cost = tsp_extract_with_legalizer(spins, N)
    assert was_valid, "Valid config should be detected as valid"
    assert path == [0, 1, 2, 3], f"Expected [0,1,2,3], got {path}"


def test_tsp_legalizer_invalid():
    """Invalid spin configuration is repaired by legalizer."""
    N = 4
    spins = np.zeros(N * N)

    path, was_valid, cost = tsp_extract_with_legalizer(spins, N)
    assert not was_valid, "Empty config should be invalid"
    assert len(path) == N
    assert -1 not in path
    assert len(set(path)) == N


def test_tsp_legalizer_overlap():
    """Overlapping city assignments are resolved by legalizer."""
    N = 3
    spins = np.zeros(N * N)
    for i in range(N):
        spins[i * N + 0] = 1.0

    path, was_valid, cost = tsp_extract_with_legalizer(spins, N)
    assert not was_valid, "Overlap should be invalid"
    assert len(path) == N
    assert -1 not in path
    assert len(set(path)) == N


# ═══════════════════════════════════════════════════════════════════════════
# 6. TSP utilities
# ═══════════════════════════════════════════════════════════════════════════

def test_tsp_extract_solution():
    """Extract a valid tour from a valid spin configuration."""
    N = 4
    spins = torch.zeros(N * N)
    for i in range(N):
        spins[i * N + i] = 1.0  # city i at position i
    path = tsp_extract_solution(spins, N)
    assert path == [0, 1, 2, 3]


def test_tsp_validate():
    """Validate a correct spin configuration."""
    N = 4
    spins = torch.zeros(N * N)
    for i in range(N):
        spins[i * N + i] = 1.0
    valid, ov, miss = tsp_validate(spins, N)
    assert valid
    assert ov == 0
    assert miss == 0


def test_tsp_validate_invalid():
    """Detect invalid configuration."""
    N = 3
    spins = torch.zeros(N * N)
    spins[0] = 1.0  # only city 0 at pos 0
    valid, ov, miss = tsp_validate(spins, N)
    assert not valid


def test_tsp_tour_distance():
    """Tour distance is computed correctly."""
    N = 3
    dists = torch.tensor([[0.0, 1.0, 2.0],
                          [1.0, 0.0, 3.0],
                          [2.0, 3.0, 0.0]])
    path = [0, 1, 2]
    d = tsp_tour_distance(path, dists)
    assert torch.allclose(torch.tensor(d), torch.tensor(6.0)), f"Expected 6, got {d}"
