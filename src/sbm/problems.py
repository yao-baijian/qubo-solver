"""Problem type mappings — convert real-world problems to Ising coupling J.

Each function returns the Ising coupling matrix J (symmetric, zero-diagonal)
that can be passed to :class:`BaseSolver` or the legacy ``bsb_torch_batch``.

Supported problems
------------------
- **MaxCut** — partition graph into two sets maximising cut edges.
- **Balanced MinCut** — partition into two equal-weight sets minimising cut edges.
- **TSP** — travelling salesman problem (N² binary variables).
- **QUBO** — general quadratic unconstrained binary optimisation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════
# 1. MaxCut
# ═══════════════════════════════════════════════════════════════════════════

def maxcut_to_ising(J: torch.Tensor) -> torch.Tensor:
    """Convert graph adjacency to Ising coupling for MaxCut.

    Parameters
    ----------
    J : Tensor (N, N)
        Graph adjacency matrix (symmetric, non-negative weights).

    Returns
    -------
    J_ising : Tensor (N, N)
        Ising coupling:  J_ising = -J / 2
    """
    return -J / 2.0


def maxcut_cut_value(J: torch.Tensor, spins: torch.Tensor) -> torch.Tensor:
    """Compute MaxCut cut value.

    ``cut = ¼ (∑J - sᵀJs)`` where spins are ±1.
    """
    return 0.25 * (J.sum() - (spins @ J @ spins))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Balanced MinCut
# ═══════════════════════════════════════════════════════════════════════════

def bmincut_to_ising(J: torch.Tensor, lambda_balance: float = 1.0) -> torch.Tensor:
    """Convert graph adjacency to Ising coupling for balanced min-cut.

    Adds a penalty term ``λ (∑ s_i)²`` to encourage equal partition size::

        J_balanced = -J/2 - 2·λ·𝟙·𝟙ᵀ

    Parameters
    ----------
    J : Tensor (N, N)
        Graph adjacency matrix.
    lambda_balance : float
        Strength of the balance penalty.

    Returns
    -------
    J_ising : Tensor (N, N)
    """
    N = J.shape[0]
    ones = torch.ones(N, device=J.device, dtype=J.dtype)
    return -0.5 * J - 2.0 * lambda_balance * torch.outer(ones, ones)


def bmincut_cut_value(J: torch.Tensor, spins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cut value and imbalance for a spin assignment.

    Returns
    -------
    cut : Tensor — cut value.
    imbalance : Tensor — absolute difference in partition sizes.
    """
    cut = 0.25 * (J.sum() - (spins @ J @ spins))
    imbalance = torch.abs(spins.sum())
    return cut, imbalance


# ═══════════════════════════════════════════════════════════════════════════
# 3. TSP — Travelling Salesman Problem
# ═══════════════════════════════════════════════════════════════════════════

def tsp_to_ising(
    city_distances: torch.Tensor,
    fixed_start_city: Optional[int] = None,
    penalty_weight: Optional[float] = None,
) -> torch.Tensor:
    """Convert TSP distance matrix to Ising coupling (QUBO format).

    Encoding: ``x[i, j]`` = 1 if city ``i`` is visited at position ``j``.
    Variables are laid out as a flat vector of length N²::

        index = i * N + j

    The Hamiltonian (to minimise) uses Luk-Vossen formulation:

        H = A · H_row + A · H_col + w · H_dist

    where H_row/col enforce one-hot constraints and H_dist penalises
    long edges.  A negative diagonal bias ``-γ`` is added to all
    variables to prevent the trivial all-zero solution from being
    the global minimum (QUBO drops constant terms, so ``(Σx-1)²``
    would otherwise give zero energy to ``x=0``).

    Parameters
    ----------
    city_distances : Tensor (N, N)
        Symmetric distance matrix (zero diagonal).
    fixed_start_city : int or None
        If set, this city is pinned to position 0.
    penalty_weight : float or None
        Weight for the distance term (auto-computed if None).

    Returns
    -------
    J : Tensor (N², N²)
        QUBO matrix (upper-triangular convention, diagonal = linear terms).
    """
    N = city_distances.shape[0]
    M = N * N
    device = city_distances.device
    dtype = city_distances.dtype

    J = torch.zeros(M, M, device=device, dtype=dtype)

    # ── One-hot constraints (row + column) ───────────────────────────
    # Modified QUBO: A·(Σx-1)² − γ·Σx   (γ small, to penalise all-zero)
    # Since QUBO drops constant +1:
    #   (Σx-1)² →  Σx² + 2Σx_jx_k − 2Σx  (dropping +1)
    #             = Σx + 2Σx_jx_k − 2Σx   (since x²=x)
    #             = 2Σx_jx_k − Σx
    # Adding −γ·Σx gives:  2Σx_jx_k − (1+γ)·Σx
    # So off-diagonal: 2A, diagonal: −A·(1 + γ/A)
    A = 8.0         # constraint strength
    gamma = 1.0     # bias against all-zero (γ > 0, typically 0.5–2.0)

    for i in range(N):
        for j1 in range(N):
            idx1 = i * N + j1
            for j2 in range(j1 + 1, N):
                idx2 = i * N + j2
                J[idx1, idx2] += 2.0 * A
            J[idx1, idx1] += -A * (1 + gamma / A)

    for j in range(N):
        for i1 in range(N):
            idx1 = i1 * N + j
            for i2 in range(i1 + 1, N):
                idx2 = i2 * N + j
                J[idx1, idx2] += 2.0 * A
            J[idx1, idx1] += -A * (1 + gamma / A)

    # ── Distance cost ────────────────────────────────────────────────
    max_dist = city_distances.max().item()
    if penalty_weight is None:
        w = 1.0 / (max(1.0, N * max_dist + 1e-8))
        w = min(w, 0.1)
    else:
        w = penalty_weight

    for j in range(N):
        nxt = (j + 1) % N
        for i1 in range(N):
            for i2 in range(N):
                if i1 == i2:
                    continue
                dist = city_distances[i1, i2].item()
                idx1 = i1 * N + j
                idx2 = i2 * N + nxt
                val = w * dist
                J[idx1, idx2] += val
                J[idx2, idx1] += val

    # ── Fixed start ──────────────────────────────────────────────────
    if fixed_start_city is not None:
        for i in range(N):
            idx = i * N + 0
            J[idx, idx] += 10.0 if i != fixed_start_city else -10.0

    return J


def tsp_extract_solution(spins: torch.Tensor, N: int) -> List[int]:
    """Extract TSP tour from spin vector of length N².

    Returns a list of city indices in order of visitation.
    """
    mat = spins.reshape(N, N)
    path = [-1] * N
    for j in range(N):
        for i in range(N):
            if mat[i, j] > 0:
                path[j] = i
                break
    return path


def tsp_tour_distance(path: List[int], distances: torch.Tensor) -> float:
    """Compute total tour distance."""
    total = 0.0
    n = len(path)
    for i in range(n):
        total += distances[path[i], path[(i + 1) % n]].item()
    return total


def tsp_validate(spins: torch.Tensor, N: int) -> Tuple[bool, int, int]:
    """Check if a TSP solution is valid.

    Returns (is_valid, overlap_count, missing_count).
    """
    mat = spins.reshape(N, N)
    row_sums = (mat > 0).sum(dim=1)
    col_sums = (mat > 0).sum(dim=0)
    return (
        bool(row_sums.eq(1).all() and col_sums.eq(1).all()),
        int((row_sums > 1).sum().item()),
        int((col_sums == 0).sum().item()),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. dt scanning helper
# ═══════════════════════════════════════════════════════════════════════════

def dt_grid(algorithm: str = "bsb") -> List[float]:
    """Return a recommended dt grid for the given algorithm.

    Parameters
    ----------
    algorithm : str
        One of ``"bsb"``, ``"dsb"``, ``"adiabatic"``, ``"digcim"``.

    Returns
    -------
    list of float
        dt values to scan.
    """
    grids = {
        "bsb":      [round(0.1 + i * 0.05, 2) for i in range(24)],   # 0.10–1.25
        "dsb":      [round(0.1 + i * 0.05, 2) for i in range(24)],
        "adiabatic": [round(0.05 + i * 0.05, 2) for i in range(20)],  # 0.05–1.00
        "digcim":   [round(0.05 + i * 0.05, 2) for i in range(20)],
    }
    return grids.get(algorithm.lower(), grids["bsb"])


def scale_grid() -> List[float]:
    """Return a recommended scale grid for quantised SB."""
    return [round(0.5 + i * 0.5, 1) for i in range(31)]  # 0.5–15.5
