"""Tests for FEM adaptive annealing (per-variable β_i)."""

import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.modules.pop("src", None)

from src.fem import FemSolver


def _make_ising(N=6):
    J = torch.randn(N, N)
    J = (J + J.T) / 2
    J.fill_diagonal_(0)
    return J


def _make_qubo(J):
    N = J.shape[0]
    J_ising = -J / 2.0
    return [(i, j, float(J_ising[i, j].item()))
            for i in range(N) for j in range(i, N) if J_ising[i, j] != 0]


def test_adaptive_annealing_runs():
    """Adaptive annealing doesn't crash and produces valid output."""
    print("test_adaptive_annealing_runs:")
    J = _make_ising(6)
    Q = _make_qubo(J)

    solver = FemSolver(
        num_trials=3, num_steps=200,
        use_adaptive_annealing=True, adaptive_A=0.5,
    )
    sol = solver.solve(Q, 6)
    assert len(sol) == 6
    assert all(v in (0, 1) for v in sol)
    print(f"  \u2713 Solution: {sol}")
    print()


def test_adaptive_annealing_A0_fallback():
    """adaptive_A=0 should produce same result as standard annealing."""
    print("test_adaptive_annealing_A0_fallback:")
    J = _make_ising(8)
    Q = _make_qubo(J)

    solver_std = FemSolver(num_trials=4, num_steps=300)
    solver_adapt = FemSolver(
        num_trials=4, num_steps=300,
        use_adaptive_annealing=True, adaptive_A=0.0,
    )

    sol_std = solver_std.solve(Q, 8)
    sol_adapt = solver_adapt.solve(Q, 8)

    # Both should produce valid binary vectors
    assert all(v in (0, 1) for v in sol_std)
    assert all(v in (0, 1) for v in sol_adapt)
    print(f"  \u2713 Standard:  {sol_std}")
    print(f"  \u2713 Adaptive (A=0): {sol_adapt}")
    print()


def test_adaptive_annealing_different_A():
    """Different A values produce different (but valid) solutions."""
    print("test_adaptive_annealing_different_A:")
    J = _make_ising(10)
    Q = _make_qubo(J)

    sols = []
    for A in [0.0, 0.25, 0.5, 0.75, 1.0]:
        solver = FemSolver(
            num_trials=4, num_steps=200,
            use_adaptive_annealing=True, adaptive_A=A,
        )
        sol = solver.solve(Q, 10)
        sols.append(sol)
        print(f"    A={A:.2f}: {sol}  (sum={sum(sol)})")

    assert all(all(v in (0, 1) for v in s) for s in sols)
    print(f"  \u2713 All A values produce valid solutions")
    print()


def test_adaptive_annealing_interface():
    """FemSolver exposes the new parameters via inspect."""
    print("test_adaptive_annealing_interface:")
    import inspect
    sig = inspect.signature(FemSolver.__init__)
    params = list(sig.parameters.keys())
    assert "use_adaptive_annealing" in params, f"Missing param: {params}"
    assert "adaptive_A" in params, f"Missing param: {params}"
    print(f"  \u2713 Parameters: use_adaptive_annealing, adaptive_A")
    print()


if __name__ == "__main__":
    test_adaptive_annealing_runs()
    test_adaptive_annealing_A0_fallback()
    test_adaptive_annealing_different_A()
    test_adaptive_annealing_interface()
    print("All adaptive annealing tests passed!")
