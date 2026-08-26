"""Runtime K-sweep benchmark: distCIM (freeze-field) vs central CIM vs SBM.

For each arithmetic precision and sync period K we time a distCIM solve
(broadcast-frame freeze-field, nparts=4, scheme=const, time_intvl=K) at the
**best dt** found by the dt x K precision sweep
(``benchmark_results/traffic_realistic/precision_ksweep_full.csv``), and
compare the wall time against:
  - central CIM (nparts=1) at the same precision / dt,
  - SBM (torch BSB, reference software solver).

"Compare under the same K, same precision": each distCIM row is paired with
the central CIM of the same precision timed at the same dt, so the only
difference is the distributed freeze-field structure.

The reported ``flop_ratio`` is the theoretical coupling-FLOP saving of the
freeze field: 1 / ((K*(1/n) + (n-1)/n)/K) for nparts=n.

Usage::

    python tools/benchmark_distcim_runtime_ksweep.py --device cuda          # cupy
    python tools/benchmark_distcim_runtime_ksweep.py --device cuda --backends cupy torch
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
PAPER_KS = [1, 2, 3, 5, 7, 10]
PRECISIONS = ["fp32", "fp16", "int8", "int4", "fp8", "fp4"]
SBM_DT = 0.5          # best SBM dt from the earlier runtime benchmark

# paper-matching SimCIM hyper-parameters (doc/03bis-methods.tex)
PAPER_KW = dict(
    pump="linear",
    xi="inverse_interaction_rms",
    pmax=1.1,
    As=70.0,
    A_init=1e-3,
)
DIST_KW = dict(nparts=4, scheme="const")


def _sync(backend: str):
    if backend == "cupy":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def _timeit(fn, repeats: int, backend: str):
    """Run ``fn`` ``repeats`` times, return (result, median seconds)."""
    times, results = [], []
    for _ in range(repeats):
        _sync(backend)
        t0 = time.perf_counter()
        results.append(fn())
        _sync(backend)
        times.append(time.perf_counter() - t0)
    mid = sorted(range(len(times)), key=lambda i: times[i])[len(times) // 2]
    return results[mid], sorted(times)[len(times) // 2]


def load_best_dt(csv_path: Path, ks, precisions):
    """(precision, K) -> (best_dt, best_congestion) from the dt x K sweep."""
    best = {}
    if not csv_path.exists():
        return best
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["config"] != "dist" or r["backend"] != "torch":
                continue
            p, k = r["precision"], int(r["K"])
            if k not in ks or p not in precisions:
                continue
            c = int(r["best_congestion"])
            if (p, k) not in best or c < best[(p, k)][1]:
                best[(p, k)] = (float(r["best_dt"]), c)
    return best


def flop_ratio(nparts: int, K: int) -> float:
    """Central MACs / distCIM MACs per coupling step for nparts=n, sync K."""
    n = nparts
    return 1.0 / ((K * (1.0 / n) + (n - 1) / n) / K)


def build_solvers(backend: str, J_gpu, h_gpu, device):
    """Return (dist_fn, cim_fn) bound to the backend."""
    if backend == "cupy":
        from src.distcim.cupy_engine import solve_ising_cupy

        def dist_fn(J, h, precision, K, steps, dt, seed):
            return solve_ising_cupy(
                J, h, dt=dt, num_iters=steps, seed=seed, precision=precision,
                time_intvl=K, **DIST_KW, **PAPER_KW)

        def cim_fn(J, h, precision, steps, dt, seed):
            return solve_ising_cupy(
                J, h, nparts=1, dt=dt, num_iters=steps, seed=seed,
                precision=precision, **PAPER_KW)

        return dist_fn, cim_fn
    # torch
    def dist_fn(J, h, precision, K, steps, dt, seed):
        return solve_ising(
            J, h, dt=dt, num_iters=steps, seed=seed, precision=precision,
            device=device, time_intvl=K, **DIST_KW, **PAPER_KW)

    def cim_fn(J, h, precision, steps, dt, seed):
        return solve_ising(
            J, h, nparts=1, dt=dt, num_iters=steps, seed=seed,
            precision=precision, device=device, **PAPER_KW)

    return dist_fn, cim_fn


def solve_sbm(J_gpu, steps, dt, seed):
    torch.manual_seed(seed)
    solver = BaseSolver(strategy=BSBStrategy(dt=dt), num_iters=steps,
                        num_trials=1, device="cuda" if torch.cuda.is_available()
                        else "cpu")
    solutions, _ = solver.solve(J_gpu)
    return solutions[0].cpu().numpy(), None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cars", type=int, default=250)
    ap.add_argument("--routes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--precisions", nargs="+", default=PRECISIONS)
    ap.add_argument("--ks", nargs="+", type=int, default=PAPER_KS)
    ap.add_argument("--backends", nargs="+", default=["cupy"],
                    choices=["torch", "cupy"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--force-synthetic", action="store_true")
    ap.add_argument("--sweep-csv", default="precision_ksweep_full.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    car_routes, J, h, source = traffic_common.load_instance(
        args.cars, args.routes, args.seed,
        force_synthetic=args.force_synthetic)
    J_cpu = J.detach().to("cpu") if J.is_cuda else J
    h_cpu = h.detach().to("cpu") if h is not None and h.is_cuda else h
    N = J.shape[0]
    c_default = traffic_common.congestion(
        car_routes, traffic_common.default_route_bits(car_routes))

    sweep_csv = REPO_ROOT / "benchmark_results" / "traffic_realistic" / \
        args.sweep_csv
    best_dt = load_best_dt(sweep_csv, args.ks, args.precisions)
    n_missing = sum(1 for p in args.precisions for k in args.ks
                    if (p, k) not in best_dt)
    if n_missing:
        print(f"WARNING: {n_missing} (precision, K) combos missing from "
              f"{args.sweep_csv}; will use dt=0.01 fallback")

    print(f"problem size: {N} spin vars, dense {N}x{N} coupling, device "
          f"{args.device}, backends {args.backends}, source={source}")
    print(f"baseline (default-route) congestion: {c_default}")
    print(f"precisions {args.precisions} x K {args.ks} x steps {args.steps}; "
          f"best dt per (precision, K) from {args.sweep_csv}; median of "
          f"{args.repeats} timed runs\n", flush=True)

    rows = []            # one row per (backend, precision, K, steps)
    # The RTX 4060 Laptop throttles hard under sustained load (SM clock drops
    # from ~3100 to ~1400 MHz), so every measurement is INTERLEAVED: distCIM,
    # central CIM and SBM for one (precision, K, steps) cell are timed back to
    # back so they share the same thermal state. No cross-cell caching of
    # timings — a cached cim number measured on a cool GPU would unfairly
    # inflate the later distCIM speedup.
    sbm_cache = {}       # steps -> (time_s, cong)  [values only, re-timed]

    for backend in args.backends:
        dist_fn, cim_fn = build_solvers(backend, None, None, args.device)
        J_solve = J_cpu if backend == "cupy" else J.to(args.device)
        h_solve = h_cpu if backend == "cupy" else \
            (h.to(args.device) if h is not None else None)
        J_torch = J.to("cuda") if torch.cuda.is_available() else J
        cim_warmed = set()     # (precision, steps) warmed up
        sbm_warmed = set()

        for p in args.precisions:
            print(f"== {p} [{backend}] ==", flush=True)
            for K in args.ks:
                dt = best_dt.get((p, K), (0.01, None))[0]
                dt_c = dt
                for steps in args.steps:
                    # ---- distCIM at (precision, K, steps, best dt) ----
                    _ = dist_fn(J_solve, h_solve, p, K, steps, dt,
                                args.seed)          # warmup (JIT kernels)
                    (spins, _e), t_dist = _timeit(
                        lambda: dist_fn(J_solve, h_solve, p, K, steps, dt,
                                        args.seed),
                        args.repeats, backend)
                    c_dist = traffic_common.congestion_of_spins(car_routes,
                                                                spins)

                    # ---- central CIM, re-timed NOW (same thermal state) ----
                    if (p, steps) not in cim_warmed:
                        _ = cim_fn(J_solve, h_solve, p, steps, dt_c,
                                   args.seed)       # warmup
                        cim_warmed.add((p, steps))
                    (spins_c, _e), t_c = _timeit(
                        lambda: cim_fn(J_solve, h_solve, p, steps, dt_c,
                                       args.seed),
                        args.repeats, backend)
                    c_c = traffic_common.congestion_of_spins(car_routes,
                                                             spins_c)

                    # ---- SBM reference, re-timed NOW ----
                    if steps not in sbm_warmed:
                        _ = solve_sbm(J_torch, steps, SBM_DT, args.seed)
                        sbm_warmed.add(steps)
                    (spins_s, _e), t_s = _timeit(
                        lambda: solve_sbm(J_torch, steps, SBM_DT, args.seed),
                        args.repeats, "torch")
                    c_s = traffic_common.congestion_of_spins(car_routes,
                                                             spins_s)
                    sbm_cache[steps] = (t_s, c_s)

                    rows.append(dict(backend=backend, precision=p, K=K,
                                     steps=steps, best_dt=dt, dist_time=t_dist,
                                     dist_cong=c_dist, cim_time=t_c,
                                     cim_cong=c_c))
                    print(f"  K={K:<2} steps={steps:>6} dt={dt:<5} "
                          f"dist={t_dist:6.4f}s (cong {c_dist:>5}) "
                          f"cim={t_c:6.4f}s (cong {c_c:>5}) "
                          f"sbm={t_s:6.4f}s "
                          f"speedup_vs_cim={t_c / t_dist:5.2f}x "
                          f"vs_sbm={t_s / t_dist:5.2f}x", flush=True)
        print(flush=True)

    # ---- persist ----
    out_dir = REPO_ROOT / "benchmark_results" / "traffic_realistic"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out or "runtime_ksweep"
    csv_path = out_dir / f"{stem}.csv"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "precision", "K", "steps", "best_dt",
                    "dist_time_s", "dist_cong", "cim_time_s", "cim_cong",
                    "sbm_time_s", "sbm_cong", "speedup_vs_cim",
                    "speedup_vs_sbm", "flop_ratio"])
        for r in rows:
            t_s, c_s = sbm_cache.get(r["steps"], (float("nan"), float("nan")))
            w.writerow([r["backend"], r["precision"], r["K"], r["steps"],
                        r["best_dt"], f"{r['dist_time']:.4f}",
                        r["dist_cong"], f"{r['cim_time']:.4f}",
                        r["cim_cong"], f"{t_s:.4f}", c_s,
                        f"{r['cim_time'] / r['dist_time']:.3f}",
                        f"{t_s / r['dist_time']:.3f}",
                        f"{flop_ratio(4, r['K']):.3f}"])

    # ---- markdown report ----
    lines = [
        "# DistIM runtime K-sweep (traffic): distCIM vs central CIM vs SBM",
        "",
        f"- instance: {N} spin vars, dense {N}x{N} coupling (source={source})",
        f"- device: {args.device}; backends {args.backends}; steps {args.steps}",
        f"- best dt per (precision, K) from `{args.sweep_csv}`; median of "
        f"{args.repeats} timed runs (device sync), seed {args.seed}",
        f"- paper params: pump linear, xi=inverse_interaction_rms, pmax=1.1, "
        f"As=70, A_init=1e-3, nparts=4, scheme=const",
        f"- baseline (default-route) congestion: {c_default}",
        "",
        "> `flop_ratio` = theoretical coupling-FLOP saving of the freeze "
        "field for nparts=4: central N² / distCIM (K·(N/4)² + 3N²/4)/K.",
        "",
    ]
    for be in args.backends:
        for p in args.precisions:
            pr = [r for r in rows if r["backend"] == be and r["precision"] == p]
            if not pr:
                continue
            lines.append(f"## {p} [{be}]")
            lines.append("")
            lines.append("| K | steps | dist dt | distCIM time (s) | central "
                         "CIM (s) | SBM (s) | speedup vs CIM | speedup vs "
                         "SBM | flop ratio | dist cong | cim cong | sbm cong |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in sorted(pr, key=lambda x: (x["K"], x["steps"])):
                t_s, c_s = sbm_cache.get(r["steps"], (float("nan"),
                                                      float("nan")))
                lines.append(
                    f"| {r['K']} | {r['steps']} | {r['best_dt']} "
                    f"| {r['dist_time']:.4f} | {r['cim_time']:.4f} "
                    f"| {t_s:.4f} "
                    f"| {r['cim_time'] / r['dist_time']:.2f}× "
                    f"| {t_s / r['dist_time']:.2f}× "
                    f"| {flop_ratio(4, r['K']):.2f}× "
                    f"| {r['dist_cong']} | {r['cim_cong']} | {c_s:.0f} |")
            lines.append("")

    # summary: best (max) speedup vs cim over steps, per (backend, precision, K)
    lines.append("## Summary — distCIM speedup vs central CIM (best over steps)")
    lines.append("")
    lines.append("| backend | precision | K | best speedup vs CIM | "
                 "theoretical flop ratio |")
    lines.append("|---|---|---|---|---|")
    for be in args.backends:
        for p in args.precisions:
            for K in args.ks:
                pr = [r for r in rows if r["backend"] == be
                      and r["precision"] == p and r["K"] == K]
                if not pr:
                    continue
                best = max(r["cim_time"] / r["dist_time"] for r in pr)
                lines.append(f"| {be} | {p} | {K} | {best:.2f}× | "
                             f"{flop_ratio(4, K):.2f}× |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The distCIM solver now uses the **padded-batched** "
                 "single-GPU broadcast frame: every node's local block runs as "
                 "one batched bmm per step, and the sync is one dense GEMM "
                 "(fp32/fp16) or one `(n, n·B, B)` bmm (quantized) computing "
                 "`c_remote = J·x − J_diag·x`. This is numerically identical to "
                 "the per-block broadcast frame (verified by the test suite) "
                 "but lets the one GPU run all nparts=4 nodes concurrently, so "
                 "the freeze-field FLOP saving becomes a real wall-clock "
                 "speedup even on a single GPU.")
    lines.append("- **Fair timing under thermal throttling**: the RTX 4060 "
                 "Laptop drops its SM clock from ~3100 to ~1400 MHz under "
                 "sustained load, so distCIM / central CIM / SBM for each "
                 "(precision, K, steps) cell are timed back-to-back "
                 "(interleaved) to share the same thermal state.")
    lines.append("- The real acceleration still grows on **multi-node "
                 "hardware** (e.g. the future 4×4090 cluster): each node then "
                 "computes only its local N/nparts block and exchanges the "
                 "frozen field every K steps — same math, but no shared-GPU "
                 "contention; `flop_ratio` (up to ~3.1× at nparts=4, K=10) is "
                 "the ceiling.")
    lines.append("- SBM (torch BSB) is a different algorithm (software "
                 "baseline); distCIM / central CIM solve the *same* SimCIM "
                 "dynamics, so their comparison isolates the freeze-field "
                 "structure.")
    lines.append("")
    md_path = out_dir / f"{stem}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"results written to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
