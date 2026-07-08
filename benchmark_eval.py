#!/usr/bin/env python
"""
Evaluate a solver configuration against best-known values.

Reads the best config from ``benchmark_results/best.csv`` (produced by
:mod:`tests.test_benchmark_solvers`) and re-runs with **many trials**
(typically 100) to compute the **probability of success (PS)** — the
fraction of trials that reach 99 % of the best-known value.

Usage::

    # 1. Run the grid benchmark to find the best config
    python -m tests.test_benchmark_solvers --quick

    # 2. Evaluate best config with 100 trials
    python benchmark_eval.py --trials 100

    # 3. Or evaluate a specific method manually
    python benchmark_eval.py --method BSB --A 0.5 --dt 0.25 --trials 100
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
sys.modules.pop("src", None)

from src.sbm import (
    BaseSolver, BSBStrategy, DSBStrategy,
    AdiabaticStrategy, DigCIMStrategy,
    GSBMixin, GGSBMixin, QuantizationMixin,
)
from src.sbm.problems import dt_grid
from benchmarks.best_known.gset_maxcut import BEST_KNOWN as GSET_BEST


# ═══════════════════════════════════════════════════════════════════════════
# 1. Problem loader (minimal, same as test_benchmark_solvers)
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK_INSTANCES = _script_dir / "benchmarks" / "instances"


def load_gset(path: Path) -> Tuple[torch.Tensor, int]:
    with open(path) as f:
        N, _ = [int(x) for x in f.readline().split()]
    data = torch.tensor(
        [list(map(int, l.split())) for l in open(path).read().strip().split("\n")[1:]],
        dtype=torch.long,
    )
    u, v = data[:, 0] - 1, data[:, 1] - 1
    w = data[:, 2].float() if data.shape[1] > 2 else torch.ones(data.shape[0])
    J = torch.zeros(N, N)
    J[u, v] = w
    J[v, u] = w
    return J, N


# ═══════════════════════════════════════════════════════════════════════════
# 2. Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

def cut_value(J: torch.Tensor, spins: torch.Tensor) -> float:
    return 0.25 * (J.sum() - (spins @ J @ spins)).item()


def make_solver(method: str, dt: float,
                A: Optional[float] = None,
                k: Optional[int] = None,
                strength: Optional[float] = None,
                num_bits: Optional[int] = None,
                iters: int = 500, trials: int = 10) -> BaseSolver:
    """Build a solver from a method name + optional enhancements."""
    strategy_map = {
        "BSB": BSBStrategy(dt=dt),
        "DSB": DSBStrategy(dt=dt),
        "Adiabatic": AdiabaticStrategy(dt=dt),
        "DigCIM": DigCIMStrategy(dt=dt),
    }
    strategy = strategy_map.get(method, BSBStrategy(dt=dt))
    enh = []
    if A is not None:
        enh.append(GSBMixin(A=A))
    if k is not None and strength is not None:
        enh.append(GGSBMixin(k=k, strength=strength))
    if num_bits is not None:
        enh.append(QuantizationMixin(num_bits=num_bits))
    return BaseSolver(strategy=strategy, enhancements=enh,
                      num_iters=iters, num_trials=trials)


def eval_ps(
    method: str, J: torch.Tensor, best_known: float,
    dt: float, A: Optional[float] = None,
    k: Optional[int] = None, strength: Optional[float] = None,
    num_bits: Optional[int] = None,
    iters: int = 500, trials: int = 100,
    target_ratio: float = 0.99,
) -> Tuple[float, float, float]:
    """Run *trials* solves and compute PS (fraction reaching *target_ratio* of best).

    Returns
    -------
    ps : float
        Probability of success (fraction reaching target).
    best_cut : float
        Best cut value found.
    avg_time : float
        Average runtime per solve (seconds).
    """
    solver = make_solver(method, dt, A, k, strength, num_bits, iters, trials)
    J_ising = -J / 2.0

    t0 = time.perf_counter()
    sols, engs = solver.solve(J_ising)
    total_time = time.perf_counter() - t0

    target = best_known * target_ratio
    successes = 0
    best_cut = -1e9

    for i in range(trials):
        spins = sols[i]
        c = cut_value(J, spins)
        if c > best_cut:
            best_cut = c
        if c >= target:
            successes += 1

    ps = successes / trials
    avg_time = total_time / trials
    return ps, best_cut, avg_time


# ═══════════════════════════════════════════════════════════════════════════
# 3. Main
# ═══════════════════════════════════════════════════════════════════════════

def _parse_best_csv(csv_path: Path) -> List[Dict]:
    """Parse best.csv into a list of config dicts."""
    if not csv_path.exists():
        print(f"[!] {csv_path} not found. Run test_benchmark_solvers first.")
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Evaluate best config with PS")
    parser.add_argument("--method", default=None, help="Solver method override")
    parser.add_argument("--A", type=float, default=None, help="GSB A value")
    parser.add_argument("--dt", type=float, default=None, help="dt value")
    parser.add_argument("--k", type=int, default=None, help="GGSB interval")
    parser.add_argument("--strength", type=float, default=None, help="GGSB strength")
    parser.add_argument("--num_bits", type=int, default=None, help="Quantisation bits")
    parser.add_argument("--iters", type=int, default=500, help="Iterations per trial")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--target", type=float, default=0.99, help="Fraction of best-known for success")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to best.csv (default: benchmark_results/best.csv)")
    parser.add_argument("--instances", type=str, nargs="*", default=None,
                        help="Instance names to evaluate (default: all)")
    args = parser.parse_args()

    # ── Load best config from CSV if not specified ────────────────────
    configs: List[Dict] = []
    if args.method is None or args.dt is None:
        csv_path = Path(args.csv) if args.csv else (
                _script_dir / "benchmark_results" / "best.csv")
        configs = _parse_best_csv(csv_path)

    # If no CSV, use manual config
    if not configs and args.method is not None:
        configs = [{
            "method": args.method, "dt": str(args.dt or 0.1),
            "A": str(args.A or ""), "k": str(args.k or ""),
            "strength": str(args.strength or ""), "num_bits": str(args.num_bits or ""),
        }]

    if not configs:
        print("No config found. Specify --method/--dt, or run test_benchmark_solvers first.")
        sys.exit(1)

    # ── Resolve instances ─────────────────────────────────────────────
    gset_dir = BENCHMARK_INSTANCES / "maxcut" / "Gset"
    instance_names = args.instances or sorted(GSET_BEST.keys())

    results: List[Dict] = []

    for cfg in configs:
        method = cfg.get("method", "BSB")
        dt_val = float(cfg.get("dt", 0.1))
        A_val = float(cfg["A"]) if cfg.get("A") else None
        k_val = int(cfg["k"]) if cfg.get("k") else None
        s_val = float(cfg["strength"]) if cfg.get("strength") else None
        bits_val = int(cfg["num_bits"]) if cfg.get("num_bits") else None

        print(f"\n{'=' * 70}")
        print(f"  Method: {method}  dt={dt_val}  A={A_val}  k={k_val}  "
              f"strength={s_val}  bits={bits_val}")
        print(f"  Trials: {args.trials}  Iters: {args.iters}")
        print(f"{'=' * 70}")

        for name in instance_names:
            best = GSET_BEST.get(name)
            if best is None:
                print(f"  [skip] {name}: no best-known value")
                continue

            inst_path = gset_dir / name
            if not inst_path.exists():
                print(f"  [skip] {name}: instance file not found")
                continue

            J, N = load_gset(inst_path)
            ps, best_cut, avg_t = eval_ps(
                method, J, best,
                dt=dt_val, A=A_val, k=k_val, strength=s_val, num_bits=bits_val,
                iters=args.iters, trials=args.trials, target_ratio=args.target,
            )

            print(f"  {name:6s}  N={N:5d}  best={best:>7.0f}  "
                  f"PS={ps:5.1%}  best_cut={best_cut:>7.1f}  time={avg_t:.4f}s")
            results.append({
                "instance": name, "N": N, "best_known": best,
                "method": method, "dt": dt_val,
                "A": A_val if A_val is not None else "",
                "k": k_val if k_val is not None else "",
                "strength": s_val if s_val is not None else "",
                "num_bits": bits_val if bits_val is not None else "",
                "PS": ps, "best_cut": best_cut, "avg_time_s": avg_t,
            })

    # ── Write summary CSV ─────────────────────────────────────────────
    out_dir = _script_dir / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_ps.csv"
    fields = ["instance", "N", "best_known", "method", "dt",
              "A", "k", "strength", "num_bits", "PS", "best_cut", "avg_time_s"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n  ✓ Results written to {out_path}")


if __name__ == "__main__":
    main()
