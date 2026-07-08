"""Problem type mappings — convert real-world problems to Ising coupling J.

Each function returns the Ising coupling matrix J (symmetric, zero-diagonal)
that can be passed to :class:`BaseSolver` or the legacy ``bsb_torch_batch``.

Supported problems
------------------
- **MaxCut** — partition graph into two sets maximising cut edges.
- **Balanced MinCut** — partition into two equal-weight sets minimising cut edges.
- **MaxSAT** — maximum satisfiability (MAX-3SAT) on N Boolean variables.
- **TSP** — travelling salesman problem (N² binary variables).
- **QUBO / QPLIB** — general quadratic unconstrained binary optimisation with
  optional linear bias term (via the auxiliary-variable trick).
- **Higher-order** — cubic + quadratic objectives (see :mod:`sbm.higher_order`).
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


# ═══════════════════════════════════════════════════════════════════════════
# 5. TSP legalizer — repair invalid solutions
# ═══════════════════════════════════════════════════════════════════════════

def tsp_extract_with_legalizer(
    spin_config: np.ndarray,
    N: int,
    city_distances: Optional[np.ndarray] = None,
) -> Tuple[List[int], bool, float]:
    """Extract a TSP tour, repairing violations via greedy legalizer.

    Returns
    -------
    path : list of int
        City indices in visitation order.
    was_valid : bool
        Whether the raw spin was already valid.
    cost : float
        Tour distance (0 if distances not given).
    """
    spin_matrix = spin_config.reshape(N, N)
    row_sums = np.sum(spin_matrix > 0, axis=1)
    col_sums = np.sum(spin_matrix > 0, axis=0)

    is_valid = bool(np.all(row_sums == 1) and np.all(col_sums == 1))

    if is_valid:
        path = [-1] * N
        for j in range(N):
            for i in range(N):
                if spin_matrix[i, j] > 0:
                    path[j] = i
                    break
        cost = _tour_distance(path, city_distances) if city_distances is not None else 0.0
        return path, True, cost

    # ── Legaliser ────────────────────────────────────────────────────
    positive_spins = []
    for i in range(N):
        for j in range(N):
            if spin_matrix[i, j] > 0:
                positive_spins.append((i, j, spin_matrix[i, j]))
    positive_spins.sort(key=lambda x: x[2], reverse=True)

    assigned_positions = set()
    assigned_cities = set()
    path = [-1] * N

    for city, pos, _ in positive_spins:
        if pos not in assigned_positions and city not in assigned_cities:
            path[pos] = city
            assigned_positions.add(pos)
            assigned_cities.add(city)

    for city, pos, _ in positive_spins:
        if path[pos] == -1 and city not in assigned_cities:
            path[pos] = city
            assigned_positions.add(pos)
            assigned_cities.add(city)

    missing_cities = set(range(N)) - assigned_cities
    missing_positions = set(range(N)) - assigned_positions

    if missing_cities or missing_positions:
        _complete_missing(path, missing_cities, missing_positions)

    if not _validate_path(path):
        path = _greedy_construct(spin_matrix, N)

    cost = _tour_distance(path, city_distances) if city_distances is not None else 0.0
    return path, False, cost


def _tour_distance(path: List[int], distances: np.ndarray) -> float:
    n = len(path)
    total = 0.0
    for i in range(n):
        total += distances[path[i], path[(i + 1) % n]]
    return total


def _complete_missing(path: List[int], missing_cities, missing_positions):
    mc = list(missing_cities)
    mp = list(missing_positions)
    for pos in mp:
        if mc:
            path[pos] = mc.pop(0)


def _greedy_construct(spin_matrix: np.ndarray, N: int) -> List[int]:
    scores = spin_matrix.copy()
    path = [-1] * N
    used_c = set()
    used_p = set()
    while len(used_p) < N:
        best_score = -np.inf
        best_city = -1
        best_pos = -1
        for i in range(N):
            if i in used_c:
                continue
            for j in range(N):
                if j in used_p:
                    continue
                if scores[i, j] > best_score:
                    best_score = scores[i, j]
                    best_city = i
                    best_pos = j
        if best_city != -1 and best_pos != -1:
            path[best_pos] = best_city
            used_c.add(best_city)
            used_p.add(best_pos)
        else:
            rc = set(range(N)) - used_c
            rp = set(range(N)) - used_p
            for pos in rp:
                if rc:
                    path[pos] = rc.pop()
            break
    return path


def _validate_path(path: List[int]) -> bool:
    if -1 in path:
        return False
    if len(set(path)) != len(path):
        return False
    if min(path) < 0 or max(path) >= len(path):
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 6. TSPLIB reader
# ═══════════════════════════════════════════════════════════════════════════

def read_tsplib(filename: str) -> Tuple[int, np.ndarray, str]:
    """Read a TSPLIB-format file and return (dimension, coordinates, name)."""
    coords = []
    dim = 0
    name = ""
    reading = False
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("NAME"):
                name = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                dim = int(line.split(":")[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                reading = True
            elif line.startswith("EOF"):
                break
            elif reading and line:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append([float(parts[1]), float(parts[2])])
    return dim, np.array(coords), name


def tsp_coords_to_distance(coords: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix from TSPLIB coordinates."""
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                d[i, j] = np.sqrt(dx * dx + dy * dy)
    return d


# ═══════════════════════════════════════════════════════════════════════════
# 7. QUBO / QPLIB — general quadratic with linear bias
# ═══════════════════════════════════════════════════════════════════════════

def qubo_to_ising(Q: torch.Tensor) -> torch.Tensor:
    """Convert a general QUBO matrix to an Ising coupling matrix.

    For a binary vector :math:`x \\in \\{0,1\\}^N`, the QUBO objective is
    :math:`x^\\top Q x`.  The transformation to Ising spins
    :math:`s = 2x - 1 \\in \\{\\pm 1\\}^N` is:

        J_ising = Q / 4
        h_i = Σ_j Q_ij / 4
        J[aux, :] = J[: , aux] = h / 2

    This adds one auxiliary spin to absorb the linear (local-field) term.

    Parameters
    ----------
    Q : Tensor (N, N)
        QUBO matrix (upper-triangular convention is fine; it will be symmetrised).

    Returns
    -------
    J : Tensor (N+1, N+1)
        Ising coupling with an extra auxiliary variable.
    """
    N = Q.shape[0]
    device = Q.device
    dtype = Q.dtype
    Q_sym = 0.5 * (Q + Q.T)
    J_ising = 0.25 * Q_sym
    h = 0.25 * Q_sym.sum(dim=1)
    J = torch.zeros(N + 1, N + 1, device=device, dtype=dtype)
    J[:N, :N] = J_ising
    J[:N, N] = 0.5 * h
    J[N, :N] = 0.5 * h
    return J


def qplib_to_ising(
    Q: np.ndarray,
    b: np.ndarray,
    num_vars: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, int]:
    """Convert a QPLIB-format (Q, b) problem to an Ising coupling matrix.

    The QPLIB objective is :math:`\\frac{1}{2} x^\\top Q x + b^\\top x + q^0`.

    The conversion uses an extra auxiliary spin to absorb the linear term::

        J_ising = Q / 8
        h_ising = Q·1 / 4 + b / 2
        J[extra_var, :] = J[: , extra_var] = 0.7 · h_ising

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


# ═══════════════════════════════════════════════════════════════════════════
# 8. MaxSAT — Maximum Satisfiability
# ═══════════════════════════════════════════════════════════════════════════

def max3sat_to_ising(
    clauses: List[Tuple[int, int, int]],
    num_vars: int,
    penalty: float = 4.0,
) -> torch.Tensor:
    """Convert a MAX-3SAT instance (list of 3-literal clauses) to Ising coupling.

    Each clause is a tuple ``(a, b, c)`` where positive integers represent
    positive literals (variable ``i`` is true) and negative integers represent
    negated literals (variable ``|i|`` is false).  Literals are 1-indexed.

    A clause is unsatisfied when all three literals are false.  We add a
    penalty term for each clause that is ``penalty`` when the clause is
    false and 0 otherwise.

    The resulting Ising coupling is an :math:`(N+1) \\times (N+1)` matrix
    (one auxiliary variable absorbs the constant offset).

    Parameters
    ----------
    clauses : list of (int, int, int)
        Each tuple contains three literals (1-indexed, negative = negated).
    num_vars : int
        Number of Boolean variables.
    penalty : float
        Energy penalty per unsatisfied clause (default 4.0).

    Returns
    -------
    J : Tensor (N+1, N+1)
    """
    N = num_vars
    device = torch.device("cpu")

    # Ising variables s = 2x - 1  →  x = (s+1)/2
    # Clause is false when all literals are false (x=0).
    # Using the Choi (2010) reduction: for clause (l1 ∨ l2 ∨ l3),
    # the penalty Hamiltonian (in QUBO) is:
    #   H_pen = penalty · (1 - x(l1) - x(l2) - x(l3) + x(l1)x(l2) + x(l1)x(l3) + x(l2)x(l3))
    # Convert to Ising: x = (s+1)/2

    Q = torch.zeros(N, N, dtype=torch.float32)
    h = torch.zeros(N, dtype=torch.float32)
    const = 0.0

    for a, b, c in clauses:
        literals = [a, b, c]
        signs = [1 if l > 0 else 0 for l in literals]   # 1 = positive literal
        idxs = [abs(l) - 1 for l in literals]

        # In QUBO: penalty · (1 - Σx_i + Σ_{i<j} x_i x_j)
        # For literal l: if positive, use x; if negative, use (1-x).
        # x(l) = x_i if sign=1 else (1 - x_i)
        # This gives:
        # H = penalty · (1 - Σ(l_i) + Σ(l_i · l_j))
        # where l_i = x_i or (1-x_i)

        # We'll compute the QUBO coefficients directly.
        # Let l_i = sign_i * x_i + (1 - sign_i) * (1 - x_i)
        #           = sign_i * x_i + (1 - sign_i) - (1 - sign_i) * x_i
        #           = (2*sign_i - 1) * x_i + (1 - sign_i)
        #           = sgn_i * x_i + offset_i
        # where sgn_i = 2*sign_i - 1 ∈ {+1, -1}
        #       offset_i = 1 - sign_i ∈ {0, 1}

        sgn = [2 * s - 1 for s in signs]
        off = [1 - s for s in signs]

        # Constant part
        c_part = penalty * (1 - sum(off))
        for i in range(3):
            c_part -= penalty * off[i] * sum(sgn[j] for j in range(3) if j != i)
        const += c_part

        # Linear part
        for i in range(3):
            coeff = -penalty * sgn[i]
            for j in range(3):
                if j != i:
                    coeff -= penalty * off[j] * sgn[i]
            h[idxs[i]] += coeff

        # Quadratic part
        for i in range(3):
            for j in range(i + 1, 3):
                Q[idxs[i], idxs[j]] += penalty * sgn[i] * sgn[j]

    # Convert QUBO to Ising:
    # QUBO: x^T Q x + h^T x + const
    # x = (s+1)/2
    # → 0.25 * s^T Q s + 0.5 * (Q·1 + h)^T s + const'
    J_ising = 0.25 * Q
    h_ising = 0.5 * (Q.sum(dim=1) + h)

    # Build full Ising matrix with auxiliary variable
    J = torch.zeros(N + 1, N + 1, dtype=torch.float32)
    J[:N, :N] = J_ising
    J[:N, N] = 0.5 * h_ising
    J[N, :N] = 0.5 * h_ising

    return J


def maxsat_count_satisfied(
    clauses: List[Tuple[int, int, int]],
    spins: torch.Tensor,
) -> int:
    """Count satisfied clauses for a MAX-3SAT problem (given Ising spins ±1).

    Parameters
    ----------
    clauses : list of (int, int, int)
        Clause list (same format as :func:`max3sat_to_ising`).
    spins : Tensor (N,)
        Ising spin assignment (values ±1).

    Returns
    -------
    int
        Number of satisfied clauses.
    """
    satisfied = 0
    for a, b, c in clauses:
        for l in (a, b, c):
            idx = abs(l) - 1
            true = (spins[idx] > 0) if l > 0 else (spins[idx] < 0)
            if true:
                satisfied += 1
                break
    return satisfied
