"""Shared helpers for the DistIM traffic benchmarks.

Loads (or builds) a traffic-flow Ising instance in three tiers:

1. **cached target-repo instance** — ``benchmark_results/traffic_realistic/
   instance_{cars}c_{routes}r.pt`` (produced by
   :mod:`benchmark_traffic_realistic`).
2. **target-repo generation** — the original ``TrafficGenerator`` +
   ``TrafficFlow`` construction (requires the *simulated-ising-machine* repo).
3. **self-contained synthetic fallback** — a networkx grid with simple
   shortest-path routes and the same congestion+one-hot QUBO, so the
   benchmark scripts run anywhere (CPU/CI) with the identical figure of merit
   (congestion).

The returned instance is always ``(car_routes, J, h)`` where ``J`` (N, N) is
the Ising coupling, ``h`` (N, 1) the external field, and ``car_routes`` maps
car id -> list of routes (node sequences), which ``congestion`` needs.
"""

from __future__ import annotations

import itertools
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import torch  # noqa: E402

try:
    import networkx as nx  # noqa: E402
except Exception:  # pragma: no cover - networkx is only needed for synthetic
    nx = None


# --------------------------------------------------------------------------- #
# congestion (paper Fig. 3 figure of merit; lower is better)
# --------------------------------------------------------------------------- #
def congestion(car_routes, bits) -> int:
    """Total congestion ``sum_e load_e^2`` for a binary route assignment."""
    loads = defaultdict(int)
    idx = 0
    for car, routes in car_routes.items():
        pick = None
        for ri in range(len(routes)):
            if bits[idx + ri] > 0.5:
                pick = ri
        idx += len(routes)
        if pick is not None:
            for u, v in zip(routes[pick], routes[pick][1:]):
                loads[(u, v)] += 1
    return sum(v * v for v in loads.values())


def default_route_bits(car_routes) -> List[int]:
    """Binary assignment where every car takes its first (fastest) route."""
    bits = []
    for car, routes in car_routes.items():
        bits += [1] + [0] * (len(routes) - 1)
    return bits


# --------------------------------------------------------------------------- #
# tier 2: original target-repo generation
# --------------------------------------------------------------------------- #
def _build_grid(rows: int, cols: int, seed: int):
    random.seed(seed)
    G = nx.DiGraph()
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                w = 1.0 + random.random()
                G.add_edge(u, u + 1, length=w)
                G.add_edge(u + 1, u, length=w)
            if r + 1 < rows:
                w = 1.0 + random.random()
                G.add_edge(u, u + cols, length=w)
                G.add_edge(u + cols, u, length=w)
    return G


def generate_via_target_repo(num_cars: int, num_routes: int, seed: int,
                             grid: Tuple[int, int] = (14, 10)):
    """Original TrafficGenerator + TrafficFlow construction (target repo)."""
    import target_repo  # noqa: F401
    target_repo.import_target()
    from sim.datasets.generator.traffic_generator import TrafficGenerator
    from sim.optimizations.traffic import TrafficFlow

    rows, cols = grid
    G = _build_grid(rows, cols, seed)
    t0 = time.time()
    gen = TrafficGenerator(G, weight="length", num_cars=num_cars,
                           num_routes=num_routes, rm_percent=0.5, segs=0.3,
                           tol=math.inf, max_trial=300, seed=seed)
    car_routes = {}
    for car, (ori, dest) in enumerate(gen.ori_dest_pairs):
        car_routes[car] = gen.k_shortest_path_remove_nodes(ori, dest)
    tf = TrafficFlow(car_routes)
    J = tf.J.to_dense() if tf.J.is_sparse else tf.J
    h = tf.h
    print(f"target-repo traffic instance: {len(car_routes)} cars, "
          f"{sum(len(r) for r in car_routes.values())} route vars, "
          f"J {tuple(J.shape)} in {time.time()-t0:.0f}s", flush=True)
    return car_routes, J, h


# --------------------------------------------------------------------------- #
# tier 3: self-contained synthetic fallback (same QUBO structure)
# --------------------------------------------------------------------------- #
def build_synthetic_traffic(num_cars: int = 60, num_routes: int = 3,
                            seed: int = 7,
                            grid: Tuple[int, int] = (6, 6),
                            penalty: Optional[float] = None):
    """Traffic-flow instance built only from networkx (no target repo).

    QUBO (same as the target repo's TrafficFlow)::

        min  sum_e load_e^2 + A * sum_c (sum_r x_{c,r} - 1)^2

    with ``load_e`` the number of cars using edge ``e`` and ``A`` the
    one-hot penalty. Returns ``(car_routes, J, h)``.
    """
    if nx is None:
        raise RuntimeError("networkx is required for the synthetic traffic "
                           "fallback instance")
    rows, cols = grid
    G = _build_grid(rows, cols, seed)
    nodes = list(G.nodes)

    rng = random.Random(seed)
    car_routes: Dict[int, List[List[int]]] = {}
    for car in range(num_cars):
        ori, dest = rng.sample(nodes, 2)
        # take only the first `num_routes` shortest simple paths (Yen's);
        # do NOT materialise all paths (exponential on grids)
        paths = list(itertools.islice(
            nx.shortest_simple_paths(G, ori, dest, weight="length"),
            num_routes))
        routes = [p for p in paths][:num_routes]
        if len(routes) < 1:
            routes = [[ori, dest]]
        car_routes[car] = routes

    # variable indexing
    idx = 0
    var = {}          # (car, route) -> var index
    for car, routes in car_routes.items():
        for ri in range(len(routes)):
            var[(car, ri)] = idx
            idx += 1
    n_vars = idx

    # QUBO (x^T Q x convention: diagonal once, off-diagonal twice)
    Q = torch.zeros(n_vars, n_vars)
    max_route_len = max(len(r) - 1 for routes in car_routes.values() for r in routes)
    A = penalty if penalty is not None else float(4 * max_route_len + 4)

    for car, routes in car_routes.items():
        for ri, r in enumerate(routes):
            i = var[(car, ri)]
            Q[i, i] += (len(r) - 1) - A          # edges in route - one-hot diag

    # quadratic: shared edges (+ A for same car)
    for car, routes in car_routes.items():
        for ri in range(len(routes)):
            i = var[(car, ri)]
            edges_i = set(zip(routes[ri], routes[ri][1:]))
            for rj in range(ri + 1, len(routes)):
                j = var[(car, rj)]
                edges_j = set(zip(routes[rj], routes[rj][1:]))
                shared = len(edges_i & edges_j)
                Q[i, j] += shared + A
            for car2 in range(car + 1, len(car_routes)):
                for rj in range(len(car_routes[car2])):
                    j = var[(car2, rj)]
                    edges_j = set(zip(car_routes[car2][rj],
                                      car_routes[car2][rj][1:]))
                    shared = len(edges_i & edges_j)
                    if shared:
                        Q[i, j] += shared

    Q = Q + Q.triu(1).T           # symmetrize
    J = -Q / 2.0
    h = -(Q.sum(dim=1, keepdim=True)) / 2.0
    return car_routes, J, h


# --------------------------------------------------------------------------- #
# tier 1 + dispatch
# --------------------------------------------------------------------------- #
def load_instance(num_cars: int = 1250, num_routes: int = 5, seed: int = 7,
                  cache_dir: Optional[Path] = None,
                  force_synthetic: bool = False):
    """Return ``(car_routes, J, h, source)`` with source in
    ``cached-target | target-repo | synthetic``."""
    cache_dir = cache_dir or (REPO_ROOT / "benchmark_results" / "traffic_realistic")
    fname = f"instance_{num_cars}c_{num_routes}r.pt"

    if not force_synthetic and (cache_dir / fname).exists():
        data = torch.load(cache_dir / fname)
        print(f"loaded cached instance: {len(data['car_routes'])} cars, "
              f"J {tuple(data['J'].shape)}")
        return data["car_routes"], data["J"], data["h"], "cached-target"

    if not force_synthetic:
        try:
            car_routes, J, h = generate_via_target_repo(num_cars, num_routes, seed)
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"car_routes": car_routes, "J": J, "h": h,
                        "num_cars": num_cars, "num_routes": num_routes},
                       cache_dir / fname)
            return car_routes, J, h, "target-repo"
        except Exception as exc:  # pragma: no cover
            print(f"[traffic_common] target-repo generation unavailable "
                  f"({exc}); falling back to synthetic instance")

    car_routes, J, h = build_synthetic_traffic(num_cars=num_cars,
                                               num_routes=num_routes,
                                               seed=seed)
    n_routes = sum(len(r) for r in car_routes.values())
    print(f"synthetic traffic instance: {len(car_routes)} cars, "
          f"{n_routes} route vars, J {tuple(J.shape)}")
    return car_routes, J, h, "synthetic"


def congestion_of_spins(car_routes, spins) -> int:
    """Congestion for +/-1 Ising spins (same convention as the tools)."""
    bits = ((spins + 1) / 2).round().astype(int).tolist()
    return congestion(car_routes, bits)
