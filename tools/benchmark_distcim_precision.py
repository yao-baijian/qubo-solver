"""Precision benchmark for DistIM (distcim) on the traffic problem.

Sweeps the sync period ``K`` (time_intvl, paper set {1,2,3,5,7,10}) and the
Euler step ``dt`` (0.1 .. 2.0) for every arithmetic precision
(fp32/fp16/int8/int4/fp8/fp4) at fixed step counts (1000, 3000, 5000, 10000)
on the GPU, reporting the **best** congestion / energy over ``dt`` x seeds.

Hyper-parameters follow the paper (doc/03bis-methods.tex): Simplified SimCIM,
pump ramped linearly 0 -> pmax (=1.1 for traffic), coupling gain
``xi = inverse_interaction_rms``, ``As = 70``, ``A_init = 1e-3``, constant
compensation scheme.

Configurations (``--configs``):

    central : centralized SimCIM (nparts=1, exact every-step field)
    dist    : distributed DistIM broadcast-frame freeze-field (nparts=4, const)

Figure of merit = total congestion (paper, lower is better); the default
(fastest) routing is the baseline.

Usage::

    python tools/benchmark_distcim_precision.py --device cuda
    python tools/benchmark_distcim_precision.py --device cpu --force-synthetic
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import torch  # noqa: E402

import traffic_common  # noqa: E402
from src.distcim import solve_ising  # noqa: E402

DTS = [round(0.1 * i, 1) for i in range(1, 21)]       # 0.1 .. 2.0
PAPER_KS = [1, 2, 3, 5, 7, 10]                         # paper sync periods K
SEEDS = [7, 11, 13]
DEFAULT_STEPS = [1000, 3000, 5000, 10000]
DEFAULT_PRECISIONS = ["fp32", "fp16", "int8", "int4", "fp8", "fp4"]

# paper-matching SimCIM hyper-parameters (doc/03bis-methods.tex)
PAPER_KW = dict(
    pump="linear",                        # p(t) ramps 0 -> pmax over the run
    xi="inverse_interaction_rms",         # 1/2 * (sum J^2 / (n-1))^{-1/2}
    pmax=1.1,                             # traffic p_max
    As=70.0,                              # inverse noise scale
    A_init=1e-3,                          # initial amplitude
)

# distributed broadcast-frame freeze-field configuration
CONFIGS = {
    "central": dict(nparts=1, scheme="standard"),
    "dist": dict(nparts=4, scheme="const"),
}

_W = {}


def _init_worker(J, h, car_routes, kw, device, backend):
    if backend == "cupy":
        # the cupy engine converts to cupy internally; keep CPU tensors
        _W["J"], _W["h"] = J, h
    else:
        # move to the worker's device once so every solve reuses it
        _W["J"], _W["h"] = J.to(device), (h.to(device) if h is not None else None)
    _W["cr"] = car_routes
    _W["kw"], _W["device"], _W["backend"] = kw, device, backend


def _worker_task(task):
    cfg, precision, K, steps, dt, seed = task
    J, h, cr = _W["J"], _W["h"], _W["cr"]
    kw = dict(_W["kw"])
    kw["time_intvl"] = K
    t0 = time.perf_counter()
    if _W["backend"] == "cupy":
        from src.distcim.cupy_engine import solve_ising_cupy
        spins, e = solve_ising_cupy(J, h, dt=dt, num_iters=steps, seed=seed,
                                    precision=precision, **PAPER_KW, **kw)
    else:
        spins, e = solve_ising(J, h, dt=dt, num_iters=steps, seed=seed,
                               precision=precision, device=_W["device"],
                               **PAPER_KW, **kw)
    elapsed = time.perf_counter() - t0
    c = traffic_common.congestion_of_spins(cr, spins)
    return cfg, precision, K, steps, dt, seed, e, c, elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cars", type=int, default=250)
    ap.add_argument("--routes", type=int, default=3)     # paper: 3 routes/car
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--precisions", nargs="+", default=DEFAULT_PRECISIONS)
    ap.add_argument("--dts", nargs="+", type=float, default=DTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--ks", nargs="+", type=int, default=PAPER_KS,
                    help="sync periods K to sweep (paper: 1 2 3 5 7 10)")
    ap.add_argument("--configs", nargs="+", default=["dist"],
                    choices=list(CONFIGS))
    ap.add_argument("--backend", default="torch", choices=["torch", "cupy"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force-synthetic", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    car_routes, J, h, source = traffic_common.load_instance(
        args.cars, args.routes, args.seed,
        force_synthetic=args.force_synthetic)
    N = J.shape[0]
    c_default = traffic_common.congestion(
        car_routes, traffic_common.default_route_bits(car_routes))
    print(f"problem size: {N} spin vars, dense {N}x{N} coupling, device "
          f"{args.device}, backend={args.backend}, source={source}")
    print(f"baseline (default-route) congestion: {c_default}")
    print(f"precisions {args.precisions} x K {args.ks} x steps {args.steps} "
          f"x dt {args.dts[0]}..{args.dts[-1]} x seeds {args.seeds}\n",
          flush=True)

    results = []          # dicts with all fields
    for cfg in args.configs:
        kw = CONFIGS[cfg]
        ks = args.ks if cfg == "dist" else [1]     # K irrelevant for central
        tasks = [(cfg, p, k, s, dt, seed)
                 for p in args.precisions
                 for k in ks
                 for s in args.steps
                 for dt in args.dts
                 for seed in args.seeds]
        print(f"[{cfg}/{args.backend}] {len(tasks)} solves across "
              f"{args.workers} workers...", flush=True)
        t0 = time.time()
        if args.workers <= 1:
            # sequential (single CUDA context; recommended on a shared GPU)
            _init_worker(J, h, car_routes, kw, args.device, args.backend)
            for task in tasks:
                cfg_, p, k, s, dt, seed, e, c, elapsed = _worker_task(task)
                results.append(dict(config=cfg_, backend=args.backend,
                                    precision=p, K=k, steps=s, dt=dt,
                                    seed=seed, energy=e, congestion=c,
                                    time_s=elapsed))
        else:
            with Pool(processes=args.workers,
                      initializer=_init_worker,
                      initargs=(J, h, car_routes, kw, args.device,
                                args.backend)) as pool:
                for cfg_, p, k, s, dt, seed, e, c, elapsed in \
                        pool.imap_unordered(_worker_task, tasks):
                    results.append(dict(config=cfg_, backend=args.backend,
                                        precision=p, K=k, steps=s, dt=dt,
                                        seed=seed, energy=e, congestion=c,
                                        time_s=elapsed))
        print(f"  done in {time.time()-t0:.0f}s\n", flush=True)

    # ---- aggregate: best (congestion, energy) per config/K/precision/steps --
    out_dir = REPO_ROOT / "benchmark_results" / "traffic_realistic"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out or "precision"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"

    rows = {}
    for r in results:
        key = (r["config"], r["backend"], r["precision"], r["K"], r["steps"])
        cur = rows.get(key)
        if cur is None or r["congestion"] < cur["congestion"]:
            rows[key] = r

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "backend", "precision", "K", "steps", "best_dt",
                    "best_seed", "best_congestion", "best_energy",
                    "vs_baseline", "avg_time_s"])
        for (cfg, be, p, k, s), r in sorted(rows.items()):
            w.writerow([cfg, be, p, k, s, r["dt"], r["seed"], r["congestion"],
                        f"{r['energy']:.6f}", r["congestion"] - c_default,
                        f"{r['time_s']:.4f}"])

    lines = [
        "# DistIM precision benchmark (traffic)",
        "",
        f"- instance: {N} spin vars, dense {N}x{N} coupling (source={source})",
        f"- device: {args.device}; backend: {args.backend}",
        f"- paper params: pump linear, xi=inverse_interaction_rms, "
        f"pmax=1.1, As=70, A_init=1e-3",
        f"- precisions {args.precisions}; K {args.ks}; steps {args.steps}; "
        f"best over dt {args.dts} x seeds {args.seeds}",
        f"- baseline (default-route) congestion: {c_default}",
        "",
    ]
    for cfg in args.configs:
        ks_used = args.ks if cfg == "dist" else [1]
        for k in ks_used:
            lines += [f"## {cfg} (K={k}, {args.backend})", "",
                      "| precision | steps | best cong | vs baseline | best dt "
                      "| best seed | best energy | avg time (s) |",
                      "|---|---|---|---|---|---|---|---|"]
            for (key_cfg, be, p, kk, s), r in sorted(rows.items()):
                if key_cfg != cfg or be != args.backend or kk != k:
                    continue
                lines.append(
                    f"| {p} | {s} | {r['congestion']} "
                    f"| {r['congestion']-c_default:+d} | {r['dt']} "
                    f"| {r['seed']} | {r['energy']:.6f} | {r['time_s']:.4f} |")
            lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # console summary: best congestion over (dt x seeds) per config/K/precision
    print(f"\n{'config':<9}{'K':>3}{'prec':<6}{'cong':>7}{'vs base':>9}"
          f"{'best dt':>8}")
    for (cfg, be, p, k, s), r in sorted(rows.items()):
        if s != args.steps[0]:
            continue
        print(f"{cfg:<9}{k:>3}{p:<6}{r['congestion']:>7}"
              f"{r['congestion']-c_default:>+9}{r['dt']:>8}")
    print(f"\nresults written to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
