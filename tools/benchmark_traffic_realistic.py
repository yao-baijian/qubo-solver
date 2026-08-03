"""Realistic traffic benchmark: original vs FPGA-quantized DistIM.

Builds (or loads) a ~5000-variable traffic-flow instance through the ORIGINAL
repo's ``TrafficGenerator`` + ``TrafficFlow`` construction on a large synthetic
road network (14x10 grid, 1250 cars, up to 5 routes each -> ~5000 route-choice
spins, dense 5000x5000 coupling matrix). The instance is cached on disk.

Then, for each configuration, sweeps dt in 0.1..1.3 (step 0.1) x seeds and
reports only the BEST congestion / energy:

    original : centralized SimCIM, float32 states
    quant8   : centralized SimCIM, x -> int8 (FPGA state quantization)
    dist-K1  : distributed SimCIM (4 modules, standard, K=1) + x-int8
    dist-K5  : distributed SimCIM (4 modules, const, K=5)  + x-int8
    dist-K10 : distributed SimCIM (4 modules, const, K=10) + x-int8

Figure of merit = total congestion (paper Fig. 3); the default (fastest)
routing is the baseline.

Usage: python tools/benchmark_traffic_realistic.py [--cars 1250]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
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
ITERS = 1000
GRID = (14, 10)

# worker-global instance (set once per process via Pool initializer)
_W = {}


def _init_worker(J, h, car_routes):
    _W["J"], _W["h"], _W["cr"] = J, h, car_routes
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // 6))


def _worker_task(task):
    tag, kw, dt, seed, iters = task
    J, h, cr = _W["J"], _W["h"], _W["cr"]
    spins, e = solve_ising(J, h, dt=dt, num_iters=iters, seed=seed,
                           model="SimplifiedSimCIM", xi=1.0, A_init=1e-3,
                           As=70.0, pump="constant", pmax=1.1, **kw)
    c = congestion(cr, ((spins + 1) / 2).round().astype(int).tolist())
    return tag, e, c


def build_grid(rows, cols, seed):
    import networkx as nx
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


def generate_instance(num_cars, num_routes, seed, cache_dir: Path):
    from sim.datasets.generator.traffic_generator import TrafficGenerator
    from sim.optimizations.traffic import TrafficFlow

    rows, cols = GRID
    G = build_grid(rows, cols, seed)
    print(f"road network: {rows}x{cols} grid = {rows*cols} nodes, "
          f"{G.number_of_edges()} directed edges", flush=True)

    t0 = time.time()
    gen = TrafficGenerator(G, weight="length", num_cars=num_cars,
                           num_routes=num_routes, rm_percent=0.5, segs=0.3,
                           tol=math.inf, max_trial=300, seed=seed)
    car_routes = {}
    for car, (ori, dest) in enumerate(gen.ori_dest_pairs):
        car_routes[car] = gen.k_shortest_path_remove_nodes(ori, dest)
    n_routes = sum(len(r) for r in car_routes.values())
    print(f"generated {num_cars} cars, {n_routes} route vars "
          f"(avg {n_routes/num_cars:.2f}/car) in {time.time()-t0:.0f}s",
          flush=True)

    t0 = time.time()
    tf = TrafficFlow(car_routes)
    J = tf.J.to_dense() if tf.J.is_sparse else tf.J
    h = tf.h
    print(f"TrafficFlow QUBO built: J {tuple(J.shape)}, h {tuple(h.shape)} "
          f"in {time.time()-t0:.0f}s", flush=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"instance_{num_cars}c_{num_routes}r.pt"
    torch.save({"car_routes": car_routes, "J": J, "h": h,
                "num_cars": num_cars, "num_routes": num_routes},
               cache_dir / fname)
    return car_routes, J, h


def load_instance(num_cars, num_routes, seed, cache_dir: Path):
    fname = f"instance_{num_cars}c_{num_routes}r.pt"
    if not (cache_dir / fname).exists():
        return generate_instance(num_cars, num_routes, seed, cache_dir)
    data = torch.load(cache_dir / fname)
    print(f"loaded cached instance: {len(data['car_routes'])} cars, "
          f"J {tuple(data['J'].shape)}")
    return data["car_routes"], data["J"], data["h"]


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


def best_config(car_routes, J, h, iters, dts, seeds, **kw):
    best_e, best_c = None, None
    for dt in dts:
        for s in seeds:
            spins, e = solve_ising(J, h, dt=dt, num_iters=iters, seed=s, **kw)
            c = congestion(car_routes, ((spins + 1) / 2).round().astype(int).tolist())
            if best_c is None or c < best_c:
                best_e, best_c = e, c
    return best_e, best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=1250)
    ap.add_argument("--routes", type=int, default=5)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--dts", nargs="+", type=float, default=DTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    iters = args.iters

    cache_dir = REPO_ROOT / "benchmark_results" / "traffic_realistic"
    car_routes, J, h = load_instance(args.cars, args.routes, 7, cache_dir)
    N = J.shape[0]
    print(f"problem size: {N} spin vars, dense coupling {N}x{N} "
          f"({N*N*4/1e6:.0f} MB float32)")
    print(f"dt sweep {args.dts[0]}..{args.dts[-1]} step 0.1, "
          f"seeds {args.seeds}, iters {iters}, workers {args.workers}\n")

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

    tasks = [(tag, kw, dt, s, iters)
             for tag, kw in configs for dt in args.dts for s in args.seeds]
    print(f"{len(tasks)} solves across {args.workers} workers...", flush=True)
    t0 = time.time()

    results = {tag: (None, None) for tag, _ in configs}
    with Pool(processes=args.workers,
              initializer=_init_worker,
              initargs=(J, h, car_routes)) as pool:
        for tag, e, c in pool.imap_unordered(_worker_task, tasks):
            _, best_c = results[tag]
            if best_c is None or c < best_c:
                results[tag] = (e, c)

    print(f"\n{'config':<30} {'energy':>14} {'cong':>7} {'vs default':>11}")
    for tag, _ in configs:
        e, c = results[tag]
        print(f"{tag:<30} {e:>14.6f} {c:>7} {c - c_default:>+11}")
    print(f"(total {time.time()-t0:.0f}s)", flush=True)

    # persist a markdown-ready table
    out = cache_dir / "results.md"
    lines = ["# Realistic traffic benchmark (DistIM)",
             "",
             f"- road network: {GRID[0]}x{GRID[1]} grid, {args.cars} cars, "
             f"{args.routes} route options/car",
             f"- problem size: {N} spin variables, dense {N}x{N} coupling",
             f"- dt sweep {args.dts[0]}..{args.dts[-1]} (step 0.1) x "
             f"seeds {args.seeds}; best result reported; iters={iters}",
             f"- default-route congestion baseline: {c_default}",
             "",
             "| config | energy | congestion | vs default |",
             "|---|---|---|---|"]
    for tag, _ in configs:
        e, c = results[tag]
        lines.append(f"| {tag} | {e:.6f} | {c} | {c - c_default:+d} |")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
