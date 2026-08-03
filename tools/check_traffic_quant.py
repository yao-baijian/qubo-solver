"""Traffic check: best-dt results for original vs FPGA-quantized DistIM.

Builds the synthetic traffic-flow instance (target-repo TrafficGenerator +
TrafficFlow, the Kowloon/HK construction), then for every configuration
sweeps dt in 0.1..1.3 (step 0.1) x seeds and reports only the BEST result:

    original  : centralized SimCIM, float32 states
    quant8    : centralized SimCIM, x -> int8 (FPGA state quantization)
    dist-K1   : distributed SimCIM (4 modules, standard, K=1) + x-int8
    dist-K5   : distributed SimCIM (4 modules, const, K=5)  + x-int8
    dist-K10  : distributed SimCIM (4 modules, const, K=10) + x-int8

Figure of merit = total congestion (paper Fig. 3), lower is better; the
default (fastest) routing is the baseline.

Usage: python tools/check_traffic_quant.py
"""
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import torch  # noqa: E402

import target_repo  # noqa: E402
target_repo.import_target()

from src.distcim import solve_ising  # noqa: E402

DTS = [round(0.1 * i, 1) for i in range(1, 14)]   # 0.1 .. 1.3
SEEDS = [7, 11, 13]
ITERS = 800


def build_traffic(num_cars=8, num_routes=4, seed=7):
    import networkx as nx
    from sim.datasets.generator.traffic_generator import TrafficGenerator
    from sim.optimizations.traffic import TrafficFlow

    random.seed(seed)
    G = nx.DiGraph()
    rows, cols = 8, 6
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

    gen = TrafficGenerator(G, weight="length", num_cars=num_cars,
                           num_routes=num_routes, rm_percent=0.5, segs=0.3,
                           tol=math.inf, max_trial=500, seed=seed)
    car_routes = {}
    for car, (ori, dest) in enumerate(gen.ori_dest_pairs):
        car_routes[car] = gen.k_shortest_path_remove_nodes(ori, dest)
    tf = TrafficFlow(car_routes)
    J = tf.J.to_dense() if tf.J.is_sparse else tf.J
    h = tf.h
    return car_routes, J, h


def congestion(car_routes, bits):
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


def best_config(car_routes, J, h, **kw):
    """Best (energy, congestion) over dt x seeds for a config."""
    best_e, best_c = None, None
    for dt in DTS:
        for s in SEEDS:
            spins, e = solve_ising(J, h, dt=dt, num_iters=ITERS, seed=s, **kw)
            c = congestion(car_routes, ((spins + 1) / 2).round().astype(int).tolist())
            if best_c is None or c < best_c:
                best_e, best_c = e, c
    return best_e, best_c


def main():
    car_routes, J, h = build_traffic()
    n_routes = sum(len(r) for r in car_routes.values())
    print(f"traffic instance: {len(car_routes)} cars, {n_routes} route vars")
    print(f"dt sweep: {DTS[0]}..{DTS[-1]} step 0.1, seeds {SEEDS}, iters {ITERS}\n")

    bits_default = []
    for car, routes in car_routes.items():
        bits_default += [1] + [0] * (len(routes) - 1)
    c_default = congestion(car_routes, bits_default)
    print(f"default-route congestion (baseline): {c_default}\n")

    configs = [
        ("original float32 central", dict(nparts=1)),
        ("quant8 central", dict(nparts=1, x_bits=8)),
        ("dist quant8 K=1 (standard)", dict(nparts=4, scheme="standard",
                                            time_intvl=1, x_bits=8)),
        ("dist quant8 K=5 (const)", dict(nparts=4, scheme="const",
                                         time_intvl=5, x_bits=8)),
        ("dist quant8 K=10 (const)", dict(nparts=4, scheme="const",
                                          time_intvl=10, x_bits=8)),
    ]

    print(f"{'config':<30} {'energy':>12} {'cong':>6} {'vs default':>10}")
    for tag, kw in configs:
        e, c = best_config(car_routes, J, h, model="SimplifiedSimCIM",
                           xi=1.0, A_init=1e-3, As=70.0,
                           pump="constant", pmax=1.1, **kw)
        print(f"{tag:<30} {e:>12.6f} {c:>6} {c - c_default:>+10}")


if __name__ == "__main__":
    main()
