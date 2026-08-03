"""Tests for the DISTCIM (DistIM) solver — distributed SimCIM.

These tests run standalone (no target repo needed) and verify the core
properties of the distributed algorithm:

- centralized == distributed ``standard`` with K=1 (noiseless, bit-exact)
- ``const`` with K=1 == centralized (noiseless, bit-exact)
- field-math unit checks against the paper equations
- message quantization preserves the solution
- the QUBO-level ``solve`` API returns valid 0/1 assignments
"""

import math

import pytest
import torch

from src.distcim import DistCimSolver, SimCimSolver, solve_ising
from src.distcim.distributed import (
    _EmulatedCoordinator,
    partition_columns,
    quantize_fixed,
)
from src.distcim.engines import SimCIMEngine, random_circle_init


def _random_ising(N, seed=0):
    torch.manual_seed(seed)
    J = torch.randn(N, N)
    J = (J + J.T) / 2
    J.fill_diagonal_(0)
    h = torch.zeros(N, 1)
    return J, h


def test_centralized_solution_reproducible():
    J, h = _random_ising(20, seed=1)
    s1, e1 = solve_ising(J, h, num_iters=300, seed=42)
    s2, e2 = solve_ising(J, h, num_iters=300, seed=42)
    assert e1 == e2
    assert (s1 == s2).all()


def test_distributed_standard_k1_matches_centralized():
    J, h = _random_ising(20, seed=2)
    _, e_central = solve_ising(J, h, nparts=1, num_iters=400, seed=3,
                               noise_scale=0.0)
    for nparts in [2, 4]:
        _, e_dist = solve_ising(J, h, nparts=nparts, scheme="standard",
                                time_intvl=1, num_iters=400, seed=3,
                                noise_scale=0.0)
        assert e_dist == e_central


def test_const_k1_matches_centralized():
    J, h = _random_ising(20, seed=2)
    _, e_central = solve_ising(J, h, nparts=1, num_iters=400, seed=3,
                               noise_scale=0.0)
    _, e_const = solve_ising(J, h, nparts=4, scheme="const", time_intvl=1,
                             num_iters=400, seed=3, noise_scale=0.0)
    assert e_const == e_central


def test_field_math_matches_paper():
    """DistIM field equations (paper Eqs. 14-16)."""
    N, nparts, K = 16, 4, 5
    J, _ = _random_ising(N, seed=4)
    x = torch.randn(N, 1).clamp(-1, 1)
    slices = partition_columns(N, nparts)
    J_parts = [J[:, s:e] for (s, e) in slices]
    xs = [x[s:e] for (s, e) in slices]
    h_parts = [torch.zeros(e - s, 1) for (s, e) in slices]
    full = J @ x

    # standard: exact every step
    std = _EmulatedCoordinator(J_parts, h_parts, slices, "standard", 1, None)
    std.states = xs
    std.prepare_step(True)
    std_field = torch.cat([std.field(m) for m in range(nparts)])
    assert torch.allclose(std_field, full, atol=1e-5)

    # const sync == exact; off-sync == intra + frozen message
    const = _EmulatedCoordinator(J_parts, h_parts, slices, "const", K, None)
    const.states = xs
    const.prepare_step(True)
    sync_field = torch.cat([const.field(m) for m in range(nparts)])
    assert torch.allclose(sync_field, full, atol=1e-5)
    const.prepare_step(False)
    off_field = torch.cat([const.field(m) for m in range(nparts)])
    contribs = [torch.matmul(Jp, xs[m]) for m, Jp in enumerate(J_parts)]
    full_ref = sum(contribs[1:], contribs[0])
    intra_ref = torch.cat([contribs[m][s:e] for m, (s, e) in enumerate(slices)])
    inter_ref = torch.cat(
        [(full_ref - contribs[m])[s:e] for m, (s, e) in enumerate(slices)]
    )
    assert torch.allclose(off_field, intra_ref + inter_ref, atol=1e-5)

    # pulse: off-sync == intra only; second sync == intra + K * old message
    pulse = _EmulatedCoordinator(J_parts, h_parts, slices, "pulse", K, None)
    pulse.states = xs
    pulse.prepare_step(True)   # first sync: no old message
    pulse.prepare_step(False)
    assert torch.allclose(
        torch.cat([pulse.field(m) for m in range(nparts)]), intra_ref, atol=1e-5
    )
    pulse.prepare_step(True)   # second sync: intra + K * old
    assert torch.allclose(
        torch.cat([pulse.field(m) for m in range(nparts)]),
        intra_ref + K * inter_ref, atol=1e-5,
    )


def test_quantization_bounded_error():
    x = torch.linspace(-2, 2, 100).reshape(-1, 1)
    scale = x.abs().max().item()
    q = quantize_fixed(x, 8, scale=scale)
    assert (q - x).abs().max().item() <= scale / 128 / 2 + 1e-6


def test_quantized_distributed_matches_within_tolerance():
    J, h = _random_ising(20, seed=5)
    _, e_central = solve_ising(J, h, nparts=1, num_iters=400, seed=6)
    _, e_q = solve_ising(J, h, nparts=4, scheme="const", time_intvl=5,
                         num_iters=400, seed=6, quantize_bits=8)
    rel = abs(e_q - e_central) / max(abs(e_central), 1e-12)
    assert rel < 0.10


def test_qubo_api_returns_binary():
    Q = [(0, 0, -1), (1, 1, -1), (0, 1, 2)]
    sol = SimCimSolver(num_iters=200, seed=1).solve(Q, 2)
    assert sorted(set(sol)) == [0, 1] or set(sol) <= {0, 1}
    assert len(sol) == 2

    dsol = DistCimSolver(num_iters=200, nparts=2, scheme="const",
                         time_intvl=5, seed=1).solve(Q, 2)
    assert len(dsol) == 2
    assert set(dsol) <= {0, 1}


def test_registry_methods_registered():
    from src.method_registry import register_distcim_methods, registry

    register_distcim_methods()
    for name in ["distcim", "distcim-const", "distcim-pulse"]:
        assert name in registry.list_methods()
