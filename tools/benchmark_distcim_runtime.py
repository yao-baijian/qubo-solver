"""Runtime benchmark: distcim (freeze-field) vs central CIM vs SBM.

Measures per-solve wall time on the GPU at fixed step counts
(1000, 3000, 5000, 10000). For each method we sweep ``dt`` x seeds to find
the best ``dt`` (and best congestion), then time a solve at the best ``dt``
(median of ``--repeats`` runs, with device synchronize).

Backends (``--backends``): ``torch`` (default) and ``cupy``. The CIM family
(``cim`` / ``distcim`` / ``distcim-int8``) runs on either backend; ``sbm`` is
a torch-only simulated bifurcation machine (BSB). The ``distcim`` methods use
the broadcast-frame freeze-field config (nparts=4, const, K=10).

Usage::

    python tools/benchmark_distcim_runtime.py --device cuda
    python tools/benchmark_distcim_runtime.py --device cuda --backends torch cupy
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import torch  # noqa: E402

import traffic_common  # noqa: E402
from src.distcim import solve_ising  # noqa: E402
from src.sbm import BaseSolver, BSBStrategy  # noqa: E402

SEEDS = [7, 11, 13]
DEFAULT_STEPS = [1000, 3000, 5000, 10000]
CIM_DTS = [round(0.1 * i, 1) for i in range(1, 14)]   # 0.1 .. 1.3
SBM_DTS = [0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

DIST_KW = dict(nparts=4, scheme="const", time_intvl=10)


def _sync(backend: str):
    if backend == "cupy":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def _timeit(fn, repeats: int, backend: str):
    """Run ``fn`` ``repeats`` times, return (result, median seconds)."""
    times = []
    result = None
    for _ in range(repeats):
        _sync(backend)
        t0 = time.perf_counter()
        result = fn()
        _sync(backend)
        times.append(time.perf_counter() - t0)
    return result, sorted(times)[len(times) // 2]


# ---- torch solvers -------------------------------------------------------- #
def solve_cim_torch(J, h, steps, dt, seed, device, precision=None):
    return solve_ising(J, h, nparts=1, dt=dt, num_iters=steps, seed=seed,
                       precision=precision, device=device)


def solve_dist_torch(J, h, steps, dt, seed, device, precision=None):
    return solve_ising(J, h, dt=dt, num_iters=steps, seed=seed,
                       precision=precision, device=device, **DIST_KW)


def solve_sbm(J, steps, dt, seed, device):
    torch.manual_seed(seed)
    solver = BaseSolver(strategy=BSBStrategy(dt=dt), num_iters=steps,
                        num_trials=1, device=device)
    solutions, _ = solver.solve(J)
    return solutions[0].cpu().numpy(), None


# ---- cupy solvers --------------------------------------------------------- #
def solve_cim_cupy(J, h, steps, dt, seed, device, precision=None):
    from src.distcim.cupy_engine import solve_ising_cupy
    return solve_ising_cupy(J, h, nparts=1, dt=dt, num_iters=steps, seed=seed,
                            precision=precision)


def solve_dist_cupy(J, h, steps, dt, seed, device, precision=None):
    from src.distcim.cupy_engine import solve_ising_cupy
    return solve_ising_cupy(J, h, dt=dt, num_iters=steps, seed=seed,
                            precision=precision, **DIST_KW)


def build_methods(backend: str, J, h, device):
    """method -> (solve_fn, dt sweep grid) for a backend."""
    if backend == "cupy":
        return {
            "cim": (lambda s, dt, seed: solve_cim_cupy(J, h, s, dt, seed, device),
                    CIM_DTS),
            "distcim": (lambda s, dt, seed: solve_dist_cupy(J, h, s, dt, seed,
                                                            device),
                        CIM_DTS),
            "distcim-int8": (lambda s, dt, seed: solve_dist_cupy(
                J, h, s, dt, seed, device, precision="int8"),
                CIM_DTS),
        }
    return {
        "cim": (lambda s, dt, seed: solve_cim_torch(J, h, s, dt, seed, device),
                CIM_DTS),
        "distcim": (lambda s, dt, seed: solve_dist_torch(J, h, s, dt, seed,
                                                         device),
                    CIM_DTS),
        "distcim-int8": (lambda s, dt, seed: solve_dist_torch(
            J, h, s, dt, seed, device, precision="int8"),
            CIM_DTS),
        "sbm": (lambda s, dt, seed: solve_sbm(J, s, dt, seed, device),
                SBM_DTS),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cars", type=int, default=1250)
    ap.add_argument("--routes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--methods", nargs="+",
                    default=["cim", "distcim", "distcim-int8", "sbm"])
    ap.add_argument("--backends", nargs="+", default=["torch"],
                    choices=["torch", "cupy"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the dt sweep; use fixed dt (--dt)")
    ap.add_argument("--dt", type=float, default=None,
                    help="fixed dt for all methods when --no-sweep")
    ap.add_argument("--force-synthetic", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    car_routes, J, h, source = traffic_common.load_instance(
        args.cars, args.routes, args.seed,
        force_synthetic=args.force_synthetic)
    # keep a CUDA torch copy: torch solvers use it directly, cupy converts
    # zero-copy via the cuda array interface (J is only transferred once).
    J_gpu = J.to(args.device)
    h_gpu = h.to(args.device) if h is not None else None
    N = J.shape[0]
    c_default = traffic_common.congestion(
        car_routes, traffic_common.default_route_bits(car_routes))
    print(f"problem size: {N} spin vars, dense {N}x{N} coupling, device "
          f"{args.device}, backends {args.backends}, source={source}")
    print(f"baseline (default-route) congestion: {c_default}")
    print(f"steps {args.steps}, seeds {args.seeds}, repeats {args.repeats}\n",
          flush=True)

    rows = []  # dicts
    for backend in args.backends:
        methods = build_methods(backend, J_gpu, h_gpu, args.device)
        for name in args.methods:
            if name not in methods:
                continue
            fn, dts = methods[name]
            print(f"== {name} [{backend}] ==", flush=True)
            for steps in args.steps:
                if args.no_sweep:
                    dts = [args.dt if args.dt is not None else 0.1]
                # ---- sweep dt x seeds -> best dt / best congestion ----
                best = None
                for dt in dts:
                    for seed in args.seeds:
                        spins, _e = fn(steps, dt, seed)
                        c = traffic_common.congestion_of_spins(car_routes, spins)
                        if best is None or c < best[1]:
                            best = (dt, c)
                best_dt, best_c = best

                # ---- time a solve at the best dt ----
                _, t_med = _timeit(lambda: fn(steps, best_dt, args.seeds[0]),
                                   args.repeats, backend)
                rows.append(dict(backend=backend, method=name, steps=steps,
                                 best_dt=best_dt, best_congestion=best_c,
                                 time_s=t_med))
                print(f"  steps={steps:>6} best_dt={best_dt:<5} "
                      f"cong={best_c:>7} (vs base {best_c - c_default:+d}) "
                      f"time={t_med:.4f}s", flush=True)
            print(flush=True)

    # ---- persist ----
    out_dir = REPO_ROOT / "benchmark_results" / "traffic_realistic"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out or "runtime"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "method", "steps", "best_dt",
                    "best_congestion", "vs_baseline", "time_s"])
        for r in rows:
            w.writerow([r["backend"], r["method"], r["steps"], r["best_dt"],
                        r["best_congestion"],
                        r["best_congestion"] - c_default, f"{r['time_s']:.4f}"])

    lines = [
        "# DistIM runtime benchmark (traffic): distcim vs cim vs sbm",
        "",
        f"- instance: {N} spin vars, dense {N}x{N} coupling (source={source})",
        f"- device: {args.device}; backends {args.backends}; steps {args.steps}; "
        f"best dt over dt-sweep x seeds {args.seeds}; median of "
        f"{args.repeats} timed runs",
        f"- baseline (default-route) congestion: {c_default}",
        "",
        "| backend | method | steps | best dt | best cong | vs baseline "
        "| time/solve (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['backend']} | {r['method']} | {r['steps']} "
                     f"| {r['best_dt']} | {r['best_congestion']} | "
                     f"{r['best_congestion']-c_default:+d} | {r['time_s']:.4f} |")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"results written to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
