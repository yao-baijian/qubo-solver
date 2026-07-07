"""Comprehensive benchmark: all SB methods + FEM on standard Ising problems.

Runs every strategy / mixin combination on Gset (maxcut), real-world
bmincut, and erdos-renyi bmincut graphs, then writes the best result and
configuration for each (problem, method) pair to CSV.

Usage::

    python -m tests.test_benchmark_solvers                     # small quick run
    python -m tests.test_benchmark_solvers --quick              # 1 problem each
    python -m tests.test_benchmark_solvers --full               # all + more iters
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# ── path setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.sbm import (  # noqa: E402
    BaseSolver, BSBStrategy, DSBStrategy,
    AdiabaticStrategy, DigCIMStrategy,
    GSBMixin, GGSBMixin, QuantizationMixin,
    bsb_torch_batch,
)
from src.fem import FemSolver  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# 1. Problem Loaders
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK_ROOT = Path(r"C:\project\Hybird-Ising-Partition\benchmarks")


@dataclass
class Problem:
    name: str
    kind: str              # "maxcut" | "bmincut"
    J: torch.Tensor        # Ising coupling (symmetric, zero diag, N×N)
    N: int
    best_known: Optional[float] = None


def load_gset(gset_path: Path) -> Problem:
    """Load a Gset file (maxcut).  Format: first line ``N M``,
    subsequent lines ``u v w`` (1-indexed)."""
    with open(gset_path) as f:
        first = f.readline()
        N, _ = [int(x) for x in first.split()]
    # Use pandas-free loader
    data = torch.tensor(
        [list(map(int, line.split())) for line in open(gset_path).read().strip().split("\n")[1:]],
        dtype=torch.long,
    )
    u = data[:, 0] - 1  # 1 → 0 index
    v = data[:, 1] - 1
    w = data[:, 2].float() if data.shape[1] > 2 else torch.ones(data.shape[0])
    J = torch.zeros(N, N)
    J[u, v] = w
    J[v, u] = w
    name = gset_path.name
    return Problem(name=name, kind="maxcut", J=J, N=N)


def load_bmincut_txt(txt_path: Path) -> Problem:
    """Load a bmincut .txt file.  Format: first line ``N M``,
    subsequent lines ``u v [w]`` (1-indexed, weight defaults to 1)."""
    with open(txt_path) as f:
        first = f.readline()
        parts = first.split()
        N = int(parts[0])
    raw = [line.split() for line in open(txt_path).read().strip().split("\n")[1:]]
    u, v, w = [], [], []
    for row in raw:
        u.append(int(row[0]) - 1)
        v.append(int(row[1]) - 1)
        w.append(float(row[2]) if len(row) > 2 else 1.0)
    u = torch.tensor(u, dtype=torch.long)
    v = torch.tensor(v, dtype=torch.long)
    w = torch.tensor(w, dtype=torch.float)
    J = torch.zeros(N, N)
    J[u, v] = w
    J[v, u] = w
    name = txt_path.stem
    return Problem(name=name, kind="bmincut", J=J, N=N)


def load_erdos_renyi(txt_path: Path) -> Problem:
    """Erdos-Renyi bmincut graphs: same format as bmincut."""
    return load_bmincut_txt(txt_path)


def _first_int(path: Path) -> int:
    """Read first integer from first line of a file (fast, no alloc)."""
    with open(path) as f:
        return int(f.readline().split()[0])


def discover_problems(quick: bool = False, max_nodes: int = 5000) -> List[Problem]:
    """Discover all benchmark problems, skipping those with N > max_nodes."""
    problems: List[Problem] = []

    def _safe_load(loader_fn, path, label=""):
        # Quick pre-check: skip huge files before allocating dense J
        try:
            n = _first_int(path)
            if n > max_nodes:
                print(f"  [skip] {label} {path.name} (N={n} > {max_nodes})")
                return None
        except Exception:
            pass
        try:
            prob = loader_fn(path)
            return prob
        except Exception as e:
            print(f"  [skip] {label} {path.name}: {e}")
            return None

    # Gset (maxcut)
    gset_dir = BENCHMARK_ROOT / "maxcut" / "Gset"
    if gset_dir.is_dir():
        gset_files = sorted(gset_dir.glob("G[0-9]*"))
        if quick:
            gset_files = gset_files[:2]
        for f in gset_files:
            p = _safe_load(load_gset, f, "Gset")
            if p:
                problems.append(p)

    # real-world bmincut
    rw_dir = BENCHMARK_ROOT / "bmincut" / "real_world_graphs"
    if rw_dir.is_dir():
        rw_files = sorted(rw_dir.glob("*.txt"))
        if quick:
            rw_files = rw_files[:1]
        for f in rw_files:
            p = _safe_load(load_bmincut_txt, f, "bmincut")
            if p:
                problems.append(p)

    # erdos-renyi bmincut
    er_dir = BENCHMARK_ROOT / "bmincut" / "erdos_renyi_graphs"
    if er_dir.is_dir():
        er_files = sorted(er_dir.glob("*.txt"))
        if quick:
            er_files = er_files[:1]
        for f in er_files:
            p = _safe_load(load_erdos_renyi, f, "erdos-renyi")
            if p:
                problems.append(p)
    return problems


# ═══════════════════════════════════════════════════════════════════════════
# 2. Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

def cut_value(J: torch.Tensor, spins: torch.Tensor) -> float:
    """Cut value for a spin configuration (±1).  For maxcut: ¼ (∑J - sᵀJs)."""
    return 0.25 * (J.sum() - (spins @ J @ spins)).item()


def energy_ising(J: torch.Tensor, spins: torch.Tensor) -> float:
    """Ising energy: -½ sᵀJs."""
    return -0.5 * (spins @ J @ spins).item()


@dataclass
class Result:
    problem: str
    kind: str
    N: int
    method: str
    config: str
    cut: float
    energy: float
    runtime_s: float
    spins: List[int] = field(repr=False)


# ═══════════════════════════════════════════════════════════════════════════
# 3. SB Methods
# ═══════════════════════════════════════════════════════════════════════════

SB_CONFIGS: List[Tuple[str, Callable[[torch.Tensor, int], BaseSolver]]] = []


def _register(name: str):
    def dec(fn):
        SB_CONFIGS.append((name, fn))
        return fn
    return dec


@_register("BSB")
def _bsb(J, N, device, iters, trials):
    return BaseSolver(strategy=BSBStrategy(dt=0.1), num_iters=iters, num_trials=trials, device=device)


@_register("BSB+GSB(A=0.5)")
def _bsb_gsb05(J, N, device, iters, trials):
    return BaseSolver(strategy=BSBStrategy(dt=0.1),
                      enhancements=[GSBMixin(A=0.5)],
                      num_iters=iters, num_trials=trials, device=device)


@_register("BSB+GSB(A=1.0)")
def _bsb_gsb10(J, N, device, iters, trials):
    return BaseSolver(strategy=BSBStrategy(dt=0.1),
                      enhancements=[GSBMixin(A=1.0)],
                      num_iters=iters, num_trials=trials, device=device)


@_register("BSB+GGSB")
def _bsb_ggsb(J, N, device, iters, trials):
    return BaseSolver(strategy=BSBStrategy(dt=0.1),
                      enhancements=[GGSBMixin(k=20, strength=0.05)],
                      num_iters=iters, num_trials=trials, device=device)


@_register("BSB+Quant4")
def _bsb_quant4(J, N, device, iters, trials):
    return BaseSolver(strategy=BSBStrategy(dt=0.1),
                      enhancements=[QuantizationMixin(num_bits=4)],
                      num_iters=iters, num_trials=trials, device=device)


@_register("BSB+Quant8")
def _bsb_quant8(J, N, device, iters, trials):
    return BaseSolver(strategy=BSBStrategy(dt=0.1),
                      enhancements=[QuantizationMixin(num_bits=8)],
                      num_iters=iters, num_trials=trials, device=device)


@_register("DSB")
def _dsb(J, N, device, iters, trials):
    return BaseSolver(strategy=DSBStrategy(dt=0.1),
                      num_iters=iters, num_trials=trials, device=device)


@_register("Adiabatic")
def _adiabatic(J, N, device, iters, trials):
    return BaseSolver(strategy=AdiabaticStrategy(dt=0.1),
                      num_iters=iters, num_trials=trials, device=device)


@_register("DigCIM")
def _digcim(J, N, device, iters, trials):
    return BaseSolver(strategy=DigCIMStrategy(dt=0.1),
                      num_iters=iters, num_trials=trials, device=device)


@_register("BSB+GSB(A=1.0)+GGSB")
def _bsb_gsb_ggsb(J, N, device, iters, trials):
    return BaseSolver(strategy=BSBStrategy(dt=0.1),
                      enhancements=[GSBMixin(A=1.0), GGSBMixin(k=20, strength=0.05)],
                      num_iters=iters, num_trials=trials, device=device)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Main Benchmark
# ═══════════════════════════════════════════════════════════════════════════


def run_sb(problem: Problem, device: str = "cpu",
           iters: int = 500, trials: int = 10) -> List[Result]:
    """Run all SB configurations on a single problem."""
    J = problem.J.to(device)
    J_ising = -J / 2.0
    results: List[Result] = []
    print(f"  SB benchmark on {problem.name} (N={problem.N}) — {len(SB_CONFIGS)} configs")
    for name, builder in SB_CONFIGS:
        try:
            base = builder(J, problem.N, device, iters, trials)
            t0 = time.perf_counter()
            sols, engs = base.solve(J_ising)
            dt = time.perf_counter() - t0
            best_idx = int(torch.argmin(engs).item())
            best_spins = sols[best_idx]
            cut = cut_value(J, best_spins)
            eng = engs[best_idx].item()
            results.append(Result(
                problem=problem.name, kind=problem.kind, N=problem.N,
                method="SB", config=name,
                cut=cut, energy=eng, runtime_s=dt,
                spins=best_spins.tolist(),
            ))
            print(f"    {name:20s}  cut={cut:.2f}  energy={eng:.4f}  time={dt:.3f}s")
        except Exception as e:
            print(f"    {name:20s}  FAILED: {e}")
    return results


def run_fem(problem: Problem, device: str = "cpu",
            num_steps: int = 200, num_trials: int = 5) -> List[Result]:
    """Run FEM on a single problem."""
    N = problem.N
    J = problem.J.to(device)
    print(f"  FEM on {problem.name} (N={N})")
    results: List[Result] = []
    # FEM operates on QUBO format: build Q from J_ising
    J_ising = -J / 2.0
    Q = []
    for i in range(N):
        for j in range(i, N):
            v = float(J_ising[i, j].item())
            if v != 0.0:
                Q.append((i, j, v))
    try:
        solver = FemSolver(num_steps=num_steps, num_trials=num_trials, dev=device)
        t0 = time.perf_counter()
        sol_list = solver.solve(Q, N)
        dt = time.perf_counter() - t0
        spins = torch.tensor(sol_list, device=device, dtype=torch.float) * 2 - 1  # 0/1 → -1/+1
        cut = cut_value(J, spins)
        eng = energy_ising(J_ising, spins)
        results.append(Result(
            problem=problem.name, kind=problem.kind, N=N,
            method="FEM", config="default",
            cut=cut, energy=eng, runtime_s=dt,
            spins=spins.tolist(),
        ))
        print(f"    FEM   cut={cut:.2f}  energy={eng:.4f}  time={dt:.3f}s")
    except Exception as e:
        print(f"    FEM   FAILED: {e}")
    return results


def run_bsb_legacy(problem: Problem, device: str = "cpu",
                   iters: int = 500, trials: int = 10) -> List[Result]:
    """Run legacy bsb_torch_batch for comparison."""
    J = problem.J.to(device)
    J_ising = -J / 2.0
    results: List[Result] = []
    print(f"  Legacy bSB on {problem.name} (N={problem.N})")
    try:
        init_x = 2 * torch.rand(trials, problem.N, device=device) - 1
        init_y = torch.zeros(trials, problem.N, device=device)
        t0 = time.perf_counter()
        energies, solutions, _ = bsb_torch_batch(J_ising, init_x, init_y, iters, 0.1)
        dt = time.perf_counter() - t0
        best_idx = int(torch.argmin(energies[:, -1]).item())
        best_spins = solutions[best_idx]
        cut = cut_value(J, best_spins)
        eng = energies[best_idx, -1].item()
        results.append(Result(
            problem=problem.name, kind=problem.kind, N=problem.N,
            method="SB_legacy", config="bsb_torch_batch",
            cut=cut, energy=eng, runtime_s=dt,
            spins=best_spins.tolist(),
        ))
        print(f"    bsb_legacy  cut={cut:.2f}  energy={eng:.4f}  time={dt:.3f}s")
    except Exception as e:
        print(f"    bsb_legacy  FAILED: {e}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. CSV Output
# ═══════════════════════════════════════════════════════════════════════════

def write_csv(results: List[Result], path: Path):
    """Write results to CSV, one row per (problem, method, config)."""
    fieldnames = ["problem", "kind", "N", "method", "config",
                  "cut", "energy", "runtime_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "problem": r.problem,
                "kind": r.kind,
                "N": r.N,
                "method": r.method,
                "config": r.config,
                "cut": f"{r.cut:.6f}",
                "energy": f"{r.energy:.6f}",
                "runtime_s": f"{r.runtime_s:.6f}",
            })
    print(f"\nResults written to {path}")


def print_summary(results: List[Result]):
    """Print best config per (problem, method)."""
    by_key: Dict[Tuple[str, str], Result] = {}
    for r in results:
        key = (r.problem, r.method)
        if key not in by_key or r.cut > by_key[key].cut:
            by_key[key] = r
    print("\n" + "=" * 70)
    print(f"{'Problem':20s} {'Method':15s} {'Config':20s} {'Cut':>10s} {'Time':>8s}")
    print("-" * 70)
    for (prob, method), r in sorted(by_key.items()):
        print(f"{prob:20s} {method:15s} {r.config:20s} {r.cut:10.2f} {r.runtime_s:8.3f}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# 6. CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="QUBO solver benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Quick run (1 prob each)")
    parser.add_argument("--full", action="store_true", help="Full run (more iters)")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--output", default=None, help="CSV output path")
    args = parser.parse_args()

    device = args.device
    if args.full:
        iters, trials, fem_steps = 1000, 20, 500
    elif args.quick:
        iters, trials, fem_steps = 100, 4, 100
    else:
        iters, trials, fem_steps = 300, 8, 300

    problems = discover_problems(quick=args.quick)
    print(f"Discovered {len(problems)} problems")
    for p in problems:
        print(f"  {p.name:20s}  N={p.N:5d}  kind={p.kind}")

    all_results: List[Result] = []
    for prob in problems:
        if prob.N > 5000 and device == "cpu":
            print(f"\n  Skipping {prob.name} (N={prob.N} too large for CPU)")
            continue
        print(f"\n{'=' * 60}")
        print(f"  Problem: {prob.name}  (N={prob.N}, {prob.kind})")
        print(f"{'=' * 60}")
        all_results.extend(run_sb(prob, device, iters, trials))
        all_results.extend(run_bsb_legacy(prob, device, iters, trials))
        if prob.N <= 2000:  # FEM is slower on large dense
            all_results.extend(run_fem(prob, device, fem_steps, max(3, trials // 2)))

    output_path = args.output or (ROOT / "benchmark_results.csv")
    write_csv(all_results, output_path)
    print_summary(all_results)

    # Regression: ensure no errors
    n_errors = sum(1 for r in all_results if r.cut == 0 and r.energy == 0)
    if n_errors > 0:
        print(f"\n  ⚠ {n_errors} results have zero cut (check for errors)")
    print(f"\n  ✓ {len(all_results)} results total")


if __name__ == "__main__":
    main()
