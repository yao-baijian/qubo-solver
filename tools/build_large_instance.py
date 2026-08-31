"""Fast vectorized dense traffic-instance builder for large N (50k / 100k).

The synthetic-traffic QUBO coupling is ``Q[i, j] = |edges(route i) ∩
edges(route j)|`` for cross-car pairs — which is exactly ``(R R^T)[i, j]``
where ``R`` is the ``(N, E)`` route → directed-edge incidence matrix — plus
the same-car one-hot penalty ``A`` and the diagonal.  So instead of the
O(cars²) Python loop in ``traffic_common.build_synthetic_traffic``, the dense
``J`` is built vectorized (one ``R R^T`` + a sparse same-car add), making
N=50k/100k dense instances feasible (rank of the coupling ≤ number of grid
edges).

Usage::

    python tools/build_large_instance.py --cars 16668 --routes 3 --seed 7
    python tools/build_large_instance.py --cars 33332 --routes 3 --seed 7
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from traffic_common import _build_grid  # noqa: E402


def build_routes(num_cars: int, num_routes: int, seed: int,
                 grid=(6, 6)):
    """Same route construction as ``build_synthetic_traffic``."""
    import networkx as nx
    rows, cols = grid
    G = _build_grid(rows, cols, seed)
    nodes = list(G.nodes)
    rng = random.Random(seed)
    car_routes = {}
    for car in range(num_cars):
        ori, dest = rng.sample(nodes, 2)
        paths = list(itertools.islice(
            nx.shortest_simple_paths(G, ori, dest, weight="length"),
            num_routes))
        routes = [p for p in paths][:num_routes]
        if len(routes) < 1:
            routes = [[ori, dest]]
        car_routes[car] = routes
    return car_routes


def build_dense_fast(car_routes, num_routes: int,
                     penalty=None):
    """Vectorized dense ``J, h`` matching ``build_synthetic_traffic`` exactly.

    Q = R R^T  (shared edges) + same-car +A (one-hot) + diag(-1 - A).
    """
    n_vars = sum(len(r) for r in car_routes.values())
    max_route_len = max(len(r) - 1 for routes in car_routes.values()
                        for r in routes)
    A = penalty if penalty is not None else float(4 * max_route_len + 4)

    # directed-edge index
    edge_id = {}
    route_edges = []          # list of (var_idx, [edge ids])
    for car, routes in car_routes.items():
        for r in routes:
            ids = []
            for u, v in zip(r, r[1:]):
                key = (u, v)
                if key not in edge_id:
                    edge_id[key] = len(edge_id)
                ids.append(edge_id[key])
            route_edges.append(ids)

    E = len(edge_id)
    R = np.zeros((n_vars, E), dtype=np.float32)
    for i, ids in enumerate(route_edges):
        R[i, ids] = 1.0

    # Q_cross = R R^T (shared edges, symmetric)
    t0 = time.time()
    Q = (R @ R.T).astype(np.float32)      # (N, N)
    print(f"  R R^T: N={n_vars} E={E} in {time.time()-t0:.1f}s", flush=True)

    # same-car one-hot +A (block-diagonal 3x3 blocks) via COO
    t0 = time.time()
    rows, cols, vals = [], [], []
    idx = 0
    for routes in car_routes.values():
        for ri in range(len(routes)):
            for rj in range(ri + 1, len(routes)):
                rows.append(idx + ri)
                cols.append(idx + rj)
                rows.append(idx + rj)
                cols.append(idx + ri)
                vals.append(A)
                vals.append(A)
        idx += len(routes)
    if rows:
        Q[np.array(rows), np.array(cols)] += np.array(vals, dtype=np.float32)
    # diagonal: Q[i,i] = (len-1) - A ; R R^T gave (len-1) on the diagonal,
    # so add just -A
    diag = np.array([-A] * n_vars, dtype=np.float32)
    Q[np.arange(n_vars), np.arange(n_vars)] += diag
    print(f"  same-car + diag in {time.time()-t0:.1f}s", flush=True)

    J = -Q / 2.0
    h = -(Q.sum(axis=1, keepdims=True)) / 2.0
    return car_routes, torch.from_numpy(J), torch.from_numpy(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=16668)
    ap.add_argument("--routes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default=str(
        REPO_ROOT / "benchmark_results" / "traffic_realistic"))
    args = ap.parse_args()

    t0 = time.time()
    car_routes = build_routes(args.cars, args.routes, args.seed)
    print(f"routes: {args.cars} cars in {time.time()-t0:.1f}s", flush=True)
    car_routes, J, h = build_dense_fast(car_routes, args.routes)
    N = J.shape[0]
    nnz = (J != 0).sum().item()
    print(f"built N={N} J {tuple(J.shape)} nnz={nnz} "
          f"density={100*nnz/(N*N):.3f}%", flush=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f"instance_{args.cars}c_{args.routes}r.pt"
    t0 = time.time()
    torch.save({"car_routes": car_routes, "J": J, "h": h,
                "num_cars": args.cars, "num_routes": args.routes}, fname)
    print(f"saved {fname} ({J.element_size()*J.nelement()/1e9:.1f} GB) in "
          f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
