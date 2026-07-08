"""
Comprehensive benchmark: all SB methods on standard Ising problems.

For each **main method** (BSB, DSB, Adiabatic, DigCIM), runs a grid over:

- **dt** — time-step values from :func:`sbm.problems.dt_grid`
- **A** — GSB strength (0.0, 0.25, 0.5, 0.75, 1.0)
- **GGSB** — global guidance strength (0.01, 0.05, 0.1) with k=20
- **Quantisation** — fixed-point bits (4, 8)

Writes a per-method CSV and a summary ``best.csv``.  Problem instances are
loaded from ``benchmarks/instances/`` (centralised in this repo).

Usage::

    python -m tests.test_benchmark_solvers            # default
    python -m tests.test_benchmark_solvers --quick     # small, fast
    python -m tests.test_benchmark_solvers --dense     # dense dt grid
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

_script_dir = Path(__file__).resolve().parent
ROOT = _script_dir.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.modules.pop("src", None)

from src.sbm import (  # noqa: E402
    BaseSolver, BSBStrategy, DSBStrategy,
    AdiabaticStrategy, DigCIMStrategy,
    GSBMixin, GGSBMixin, QuantizationMixin,
    bsb_torch_batch,
)
from src.sbm.problems import dt_grid  # noqa: E402
from src.fem import FemSolver  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# 1. Problem loaders  (instances live under benchmarks/instances/)
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK_ROOT = ROOT / "benchmarks" / "instances"


@dataclass
class Problem:
    name: str
    kind: str
    J: torch.Tensor
    N: int


def load_gset(path: Path) -> Problem:
    with open(path) as f:
        N, _ = [int(x) for x in f.readline().split()]
    data = torch.tensor(
        [list(map(int, l.split())) for l in open(path).read().strip().split("\n")[1:]],
        dtype=torch.long,
    )
    u, v = data[:, 0] - 1, data[:, 1] - 1
    w = data[:, 2].float() if data.shape[1] > 2 else torch.ones(data.shape[0])
    J = torch.zeros(N, N); J[u, v] = w; J[v, u] = w
    return Problem(name=path.name, kind="maxcut", J=J, N=N)


def load_bmincut_txt(path: Path) -> Problem:
    with open(path) as f:
        N = int(f.readline().split()[0])
    raw = [l.split() for l in open(path).read().strip().split("\n")[1:]]
    u, v, w = [], [], []
    for row in raw:
        u.append(int(row[0]) - 1); v.append(int(row[1]) - 1)
        w.append(float(row[2]) if len(row) > 2 else 1.0)
    u, v = torch.tensor(u, dtype=torch.long), torch.tensor(v, dtype=torch.long)
    J = torch.zeros(N, N); J[u, v] = w; J[v, u] = w
    return Problem(name=path.stem, kind="bmincut", J=J, N=N)


def load_erdos_renyi(path: Path) -> Problem:
    return load_bmincut_txt(path)


def _first_int(path: Path) -> int:
    with open(path) as f:
        return int(f.readline().split()[0])


def discover_problems(quick: bool = False, max_nodes: int = 5000) -> List[Problem]:
    problems: List[Problem] = []

    def _safe(loader, path, label=""):
        try:
            if _first_int(path) > max_nodes:
                print(f"  [skip] {label} {path.name} (too large)")
                return None
            return loader(path)
        except Exception as e:
            print(f"  [skip] {label} {path.name}: {e}")
            return None

    for sub, loader, label in [
        (BENCHMARK_ROOT / "maxcut" / "Gset", load_gset, "Gset"),
        (BENCHMARK_ROOT / "bmincut" / "real_world_graphs", load_bmincut_txt, "bmincut"),
        (BENCHMARK_ROOT / "bmincut" / "erdos_renyi_graphs", load_erdos_renyi, "erdos-renyi"),
    ]:
        if not sub.is_dir():
            continue
        files = sorted(sub.glob("*"))
        if quick:
            files = files[:2]
        for f in files:
            if f.name.startswith(".") or f.suffix in (".py", ".ipynb"):
                continue
            p = _safe(loader, f, label)
            if p:
                problems.append(p)
    return problems


# ═══════════════════════════════════════════════════════════════════════════
# 2. Evaluation / CSV helpers
# ═══════════════════════════════════════════════════════════════════════════

def cut_value(J: torch.Tensor, spins: torch.Tensor) -> float:
    return 0.25 * (J.sum() - (spins @ J @ spins)).item()


def energy_ising(J: torch.Tensor, spins: torch.Tensor) -> float:
    return -0.5 * (spins @ J @ spins).item()


@dataclass
class Result:
    problem: str = ""
    kind: str = ""
    N: int = 0
    method: str = ""
    cut: float = 0.0
    energy: float = 0.0
    runtime_s: float = 0.0
    A: str = ""
    k: str = ""
    strength: str = ""
    num_bits: str = ""
    spins: str = ""  # serialised as comma-separated ints


RESULT_FIELDS = [
    "problem", "kind", "N", "method",
    "cut", "energy", "runtime_s",
    "A", "k", "strength", "num_bits",
]

OUT_DIR = ROOT / "benchmark_results"


def write_per_method(results: List[Result], method: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{method.lower()}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k, "") for k in RESULT_FIELDS})
    print(f"  -> {path}")


def write_best_summary(all_results: List[Result]):
    best: Dict[str, Result] = {}
    for r in all_results:
        if r.problem not in best or r.cut > best[r.problem].cut:
            best[r.problem] = r
    path = OUT_DIR / "best.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for prob in sorted(best):
            w.writerow({k: getattr(best[prob], k, "") for k in RESULT_FIELDS})
    print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Method runners with grid search over sub-options
# ═══════════════════════════════════════════════════════════════════════════

def _run_base(prob: Problem, method_name: str, base: BaseSolver, **kw) -> List[Result]:
    J = prob.J.to(base.device)
    J_ising = -J / 2.0
    t0 = time.perf_counter()
    sols, engs = base.solve(J_ising)
    dt = time.perf_counter() - t0
    best_idx = int(torch.argmin(engs).item())
    spins = sols[best_idx]
    r = Result(
        problem=prob.name, kind=prob.kind, N=prob.N,
        method=method_name,
        cut=cut_value(J, spins), energy=engs[best_idx].item(), runtime_s=dt,
        spins=",".join(str(int(s)) for s in spins.tolist()),
    )
    for k, v in kw.items():
        setattr(r, k, str(v))
    return [r]


def _build_enhancements(A=None, k=None, strength=None, num_bits=None):
    """Build enhancement list from sub-option values (None = not used)."""
    enh = []
    if A is not None:
        enh.append(GSBMixin(A=A))
    if k is not None and strength is not None:
        enh.append(GGSBMixin(k=k, strength=strength))
    if num_bits is not None:
        enh.append(QuantizationMixin(num_bits=num_bits))
    return enh


def _run_grid(method_name: str, strategy_cls, prob: Problem,
              device: str, iters: int, trials: int,
              dt_values: Optional[List[float]] = None,
              gsb_grid: Optional[List[float]] = None,
              ggsb_strengths: Optional[List[float]] = None,
              quant_bits: Optional[List[int]] = None,
              max_combos: int = 30) -> List[Result]:
    """Cartesian-product grid search over dt × sub-options.

    ``dt_values`` are taken from :func:`dt_grid` by default.  Sub-options
    (GSB ``A``, GGSB, Quantisation) are orthogonal — any combination can
    coexist.  ``None`` is always included for each dimension (meaning "not
    used"), giving the baseline as one row.
    """
    if dt_values is None:
        dt_values = dt_grid(method_name.lower())

    # Build lists with None as "not used" for sub-options
    A_values = [None] + (gsb_grid or [])
    ggsb_values = [None] + ([{"k": 20, "s": s} for s in (ggsb_strengths or [])])
    quant_values = [None] + (quant_bits or [])

    # Cartesian product
    combos = []
    for dt in dt_values:
        for A in A_values:
            for gg in ggsb_values:
                for bits in quant_values:
                    combos.append((dt, A, gg, bits))

    # Limit combos for quick runs
    if len(combos) > max_combos:
        import random
        random.seed(0)
        # always keep at least one baseline per dt value
        baselines = [(dt, None, None, None) for dt in dt_values]
        rest = [c for c in combos if c not in baselines]
        combos = baselines + random.sample(rest, max_combos - len(baselines))

    results: List[Result] = []
    for dt, A, gg, bits in combos:
        k = gg["k"] if gg else None
        s = gg["s"] if gg else None
        enh = _build_enhancements(A=A, k=k, strength=s, num_bits=bits)
        base = BaseSolver(strategy=strategy_cls(dt=dt),
                          enhancements=enh,
                          num_iters=iters, num_trials=trials, device=device)
        results.extend(_run_base(prob, method_name, base,
                                 A=A if A is not None else "",
                                 k=k if k is not None else "",
                                 strength=s if s is not None else "",
                                 num_bits=bits if bits is not None else ""))
    return results


def run_bsb(prob: Problem, device: str, iters: int, trials: int) -> List[Result]:
    return _run_grid("BSB", BSBStrategy, prob, device, iters, trials,
                     dt_values=dt_grid("bsb"),
                     gsb_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
                     ggsb_strengths=[0.01, 0.05, 0.1],
                     quant_bits=[4, 8])


def run_dsb(prob: Problem, device: str, iters: int, trials: int) -> List[Result]:
    return _run_grid("DSB", DSBStrategy, prob, device, iters, trials,
                     dt_values=dt_grid("dsb"),
                     gsb_grid=[0.0, 0.5, 1.0],
                     ggsb_strengths=[0.01, 0.05, 0.1],
                     quant_bits=[4, 8],
                     max_combos=20)


def run_adiabatic(prob: Problem, device: str, iters: int, trials: int) -> List[Result]:
    return _run_grid("Adiabatic", AdiabaticStrategy, prob, device, iters, trials,
                     dt_values=dt_grid("adiabatic"),
                     gsb_grid=[0.0, 0.5, 1.0],
                     ggsb_strengths=[0.01, 0.05, 0.1],
                     max_combos=15)


def run_digcim(prob: Problem, device: str, iters: int, trials: int) -> List[Result]:
    return _run_grid("DigCIM", DigCIMStrategy, prob, device, iters, trials,
                     dt_values=dt_grid("digcim"),
                     gsb_grid=[0.0, 0.5, 1.0],
                     ggsb_strengths=[0.01, 0.05, 0.1],
                     max_combos=15)


def run_legacy(prob: Problem, device: str, iters: int, trials: int) -> List[Result]:
    J = prob.J.to(device)
    J_ising = -J / 2.0
    init_x = 2 * torch.rand(trials, prob.N, device=device) - 1
    init_y = torch.zeros(trials, prob.N, device=device)
    t0 = time.perf_counter()
    energies, solutions, _ = bsb_torch_batch(J_ising, init_x, init_y, iters, 0.1)
    dt = time.perf_counter() - t0
    best_idx = int(torch.argmin(energies[:, -1]).item())
    spins = solutions[best_idx]
    return [Result(
        problem=prob.name, kind=prob.kind, N=prob.N,
        method="BSB_legacy",
        cut=cut_value(J, spins), energy=energies[best_idx, -1].item(), runtime_s=dt,
        spins=spins.tolist(),
    )]


def run_fem(prob: Problem, device: str, iters: int, trials: int) -> List[Result]:
    N = prob.N
    J_ising = -prob.J.to(device) / 2.0
    Q = [(i, j, float(J_ising[i, j].item()))
         for i in range(N) for j in range(i, N) if J_ising[i, j] != 0]
    solver = FemSolver(num_steps=iters, num_trials=trials, dev=device)
    t0 = time.perf_counter()
    sol_list = solver.solve(Q, N)
    dt = time.perf_counter() - t0
    spins = torch.tensor(sol_list, device=device, dtype=torch.float) * 2 - 1
    return [Result(
        problem=prob.name, kind=prob.kind, N=N, method="FEM",
        cut=cut_value(prob.J.to(device), spins),
        energy=energy_ising(J_ising, spins), runtime_s=dt,
        spins=spins.tolist(),
    )]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="QUBO solver benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Quick run (2 instances, fewer combos)")
    parser.add_argument("--dense", action="store_true",
                        help="Dense dt grid (smaller step for more precision)")
    parser.add_argument("--device", default="cpu", help="Torch device")
    args = parser.parse_args()

    # Override dt_grid if --dense
    if args.dense:
        import src.sbm.problems as _probs
        # store the original, replace with a finer grid
        _orig_grid = _probs.dt_grid
        def _dense_grid(algo):
            base = _orig_grid(algo)
            step = base[1] - base[0] if len(base) > 1 else 0.05
            return [round(base[0] + i * step / 2, 3) for i in range(len(base) * 2 - 1)]
        _probs.dt_grid = _dense_grid

    iters, trials = (100, 4) if args.quick else (300, 8)

    problems = discover_problems(quick=args.quick)
    print(f"Discovered {len(problems)} problems\n")

    method_runners = [
        ("BSB", run_bsb),
        ("DSB", run_dsb),
        ("Adiabatic", run_adiabatic),
        ("DigCIM", run_digcim),
    ]

    all_results: List[Result] = []

    for prob in problems:
        print(f"\n{'=' * 60}")
        print(f"  {prob.name:20s}  N={prob.N:5d}  {prob.kind}")
        print(f"{'=' * 60}")

        for name, runner in method_runners:
            if prob.N > 3000 and name == "DigCIM":
                print(f"  {name}: skip (N={prob.N} too large)")
                continue
            res = runner(prob, args.device, iters, trials)
            all_results.extend(res)
            for r in res:
                parts = [f"cut={r.cut:.1f}", f"time={r.runtime_s:.3f}s"]
                if r.A: parts.append(f"A={r.A}")
                if r.k: parts.append(f"k={r.k}")
                if r.strength: parts.append(f"s={r.strength}")
                if r.num_bits: parts.append(f"bits={r.num_bits}")
                print(f"  {r.method:10s}  {' '.join(parts)}")

        # legacy + FEM
        all_results.extend(run_legacy(prob, args.device, iters, trials))
        if prob.N <= 2000:
            all_results.extend(run_fem(prob, args.device, iters // 2, max(3, trials // 2)))

    # ── Write CSVs ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Writing results ...")

    for name, _ in method_runners:
        write_per_method([r for r in all_results if r.method == name], name)

    for m in ("BSB_legacy", "FEM"):
        write_per_method([r for r in all_results if r.method == m], m)

    write_best_summary(all_results)

    # ── Print best table ──────────────────────────────────────────────
    best: Dict[str, Result] = {}
    for r in all_results:
        if r.problem not in best or r.cut > best[r.problem].cut:
            best[r.problem] = r

    print(f"\n{'=' * 70}")
    print(f"{'Problem':20s} {'Method':12s} {'Cut':>10s} {'Time':>8s}  Sub-options")
    print("-" * 70)
    for prob in sorted(best):
        r = best[prob]
        opts = " ".join(f"{k}={v}" for k, v in
                        [("A", r.A), ("k", r.k), ("bits", r.num_bits)] if v)
        print(f"{r.problem:20s} {r.method:12s} {r.cut:10.1f} {r.runtime_s:8.3f}  {opts}")
    print(f"{'=' * 70}")
    print(f"\n  \u2713 {len(all_results)} total results")


if __name__ == "__main__":
    main()
