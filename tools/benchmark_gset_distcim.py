"""Gset MaxCut benchmark: float32 vs FPGA-quantized DistIM SimCIM.

Runs every Gset instance (ground-state MaxCut) with, per algorithm, a
**best-dt sweep** and multiple iteration budgets (same K for all algorithms):

    iters in {1000, 3000, 5000},  dt in 0.1..1.3 step 0.1,  seeds {0,1,2}

Algorithms compared (all run with the SAME iteration budgets K; only the
best-dt result is reported for each):
    float32 : centralized SimCIM, float32 states (the original machine)
    quant8  : centralized SimCIM, FPGA state quantization (x -> int8)
    distq8  : distributed SimCIM (4 modules, const, K=5) + x-int8

NOTE: the ``const`` scheme with a large sync period (K>=10) can collapse to a
uniform spin state on some instances -- this is genuine behaviour of the
reference implementation (verified on the target repo with 4 real ranks).
K<=7 stays stable, so the distributed column uses K=5.

For each (instance, iters, algorithm) the BEST cut over the dt x seed grid is
kept, then compared against the Gset best-known values and between algorithms.
Results are written to ``benchmark_results/gset_distcim_compare.csv``.
Instances are processed in parallel across CPU cores.

Usage::

    python tools/benchmark_gset_distcim.py                # all Gset instances
    python tools/benchmark_gset_distcim.py --instances G1 G6 G11
    python tools/benchmark_gset_distcim.py --no-dist       # skip distributed
    python tools/benchmark_gset_distcim.py --workers 6     # parallel procs
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.distcim import solve_ising                        # noqa: E402
from benchmarks.best_known.gset_maxcut import BEST_KNOWN   # noqa: E402

GSET_DIR = REPO_ROOT / "benchmarks" / "instances" / "maxcut" / "Gset"
OUT_CSV = REPO_ROOT / "benchmark_results" / "gset_distcim_compare.csv"

# SimCIM recipe for MaxCut (paper: pmax=1.0, linear pump, inverse-RMS gain).
MODEL = "SimplifiedSimCIM"
PUMP = "linear"
PMAX = 1.0
XI = "inverse_interaction_rms"
DIST_K = 5          # sync period for the distributed column (stable; K>=10 unstable)


def load_gset(path: Path):
    """Gset format: first line `N M`, then `u v w` (1-indexed) edges."""
    with open(path) as f:
        N, _ = [int(x) for x in f.readline().split()]
    rows = open(path).read().strip().split("\n")[1:]
    data = torch.tensor([list(map(int, l.split())) for l in rows], dtype=torch.long)
    u, v = data[:, 0] - 1, data[:, 1] - 1
    w = data[:, 2].float() if data.shape[1] > 2 else torch.ones(data.shape[0])
    J = torch.zeros(N, N)
    J[u, v] = w
    J[v, u] = w
    return J, N


def cut_value(J: torch.Tensor, spins) -> float:
    spins = torch.as_tensor(spins, dtype=torch.float32)
    return 0.25 * (J.sum() - (spins @ J @ spins)).item()


def run_best(alg, J_ising, J_adj, h, iters, dts, seeds):
    """Return the best MaxCut value over the dt x seed grid for one (alg, iters).

    ``J_ising`` is fed to the solver; ``J_adj`` (the adjacency matrix) is used
    to compute the actual cut value.
    """
    best = float("-inf")
    for dt in dts:
        for s in seeds:
            kw = dict(dt=dt, num_iters=iters, seed=s, model=MODEL,
                      xi=XI, pump=PUMP, pmax=PMAX, device="cpu")
            if alg == "float32":
                kw.update(nparts=1)
            elif alg == "quant8":
                kw.update(nparts=1, x_bits=8)
            elif alg == "distq8":
                kw.update(nparts=4, scheme="const", time_intvl=DIST_K, x_bits=8)
            else:
                raise ValueError(alg)
            spins, _ = solve_ising(J_ising, h, **kw)
            best = max(best, cut_value(J_adj, spins))
    return best


def process_instance(args):
    """Compute all rows for one instance (called in a worker process)."""
    name, iters_list, dts, seeds, algs, use_dist = args
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(1, _WORKERS)))
    J, N = load_gset(GSET_DIR / name)
    h = torch.zeros(N, 1)
    J_ising = -J / 2.0
    bk = BEST_KNOWN.get(name)
    rows = []
    for iters in iters_list:
        best = {a: run_best(a, J_ising, J, h, iters, dts, seeds) for a in algs}
        ratio = {a: (best[a] / bk if bk else float("nan")) for a in algs}
        qv_f = (best["quant8"] / best["float32"] - 1.0
                if best["float32"] else float("nan"))
        dv_f = (best.get("distq8", best["float32"]) / best["float32"] - 1.0
                if best["float32"] else float("nan"))
        rows.append([name, N, iters, bk,
                     *[f"{best[a]:.1f}" for a in algs],
                     *[f"{ratio[a]:.4f}" for a in algs],
                     f"{qv_f:+.4f}", f"{dv_f:+.4f}"])
    return name, rows


_WORKERS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", nargs="+", type=int, default=[1000, 3000, 5000])
    ap.add_argument("--dts", nargs="+", type=float,
                    default=[round(0.1 * i, 1) for i in range(1, 14)])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--instances", nargs="+", default=None,
                    help="Gset names (default: all files in the Gset dir)")
    ap.add_argument("--no-dist", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    global _WORKERS
    _WORKERS = args.workers

    algs = ["float32", "quant8"] + ([] if args.no_dist else ["distq8"])
    names = args.instances or sorted(p.name for p in GSET_DIR.glob("G*"))
    names = [n for n in names if (GSET_DIR / n).exists()]
    print(f"{len(names)} instances, iters={args.iters}, dts={args.dts}, "
          f"seeds={args.seeds}, algs={algs}, workers={args.workers}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = (["instance", "N", "iters", "best_known",
               *[f"{a}_cut" for a in algs],
               *[f"{a}_ratio" for a in algs],
               "quant8_vs_float32", "distq8_vs_float32"])

    tasks = [(name, args.iters, args.dts, args.seeds, algs, not args.no_dist)
             for name in names]
    t0 = time.time()
    results = []
    with Pool(processes=args.workers) as pool:
        for name, rows in pool.imap_unordered(process_instance, tasks):
            results.extend(rows)
            print(f"{name}: {len(rows)} rows  [{time.time()-t0:.0f}s]",
                  flush=True)

    results.sort(key=lambda r: (r[0], r[2]))
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    print(f"\n{len(results)} rows written to {OUT_CSV}")


if __name__ == "__main__":
    main()
