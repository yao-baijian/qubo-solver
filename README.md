# qubo-solver

Standalone QUBO solver library with a **unified, composable architecture**.

## Solvers

| Solver | Description |
|---|---|
| **FEM** | Free Energy Minimisation — mean-field annealing |
| **SBM** | Unified SB engine with strategy pattern + mixins |

### SBM Architecture

The SBM package uses a **strategy + mixin** pattern for maximum flexibility:

- **Update strategies** — pluggable position/momentum dynamics:
  `BSBStrategy` (ballistic), `DSBStrategy` (discrete), `AdiabaticStrategy`, `DigCIMStrategy`
- **Enhancement mixins** — orthogonal features composed into the loop:
  `GSBMixin` (individual ``p_i`` with nonlinear control),
  `GGSBMixin` (global guidance across batch replicas),
  `QuantizationMixin` (fixed-point simulation)

## Installation

```bash
pip install -e .
```

## Usage

```python
from src.sbm import SbmSolver, BaseSolver, BSBStrategy, GSBMixin

# ── Classic API (backward compatible) ────────────────────────────────
solver = SbmSolver(num_iters=1000, dt=0.1, num_trials=10)
solution = solver.solve(Q, num_vars=2)

# ── New composable API ───────────────────────────────────────────────
import torch
J = torch.randn(N, N); J = (J + J.T) / 2; J.fill_diagonal_(0)

base = BaseSolver(
    strategy=BSBStrategy(dt=0.1),
    enhancements=[GSBMixin(A=0.5)],
    num_iters=500, num_trials=10,
)
solutions, energies = base.solve(-J / 2.0)

# ── FEM ──────────────────────────────────────────────────────────────
from src.fem import FemSolver
solver = FemSolver(num_steps=1000, num_trials=10)
solution = solver.solve(Q, num_vars=2)
```

## Git Submodule Usage

```bash
git submodule add https://github.com/yao-baijian/qubo-solver.git lib/qubo-solver
```

No ``pip install`` needed — add ``lib/qubo-solver/src`` to ``sys.path``, then
import directly: ``from fem import FemSolver``.

## Project Structure

```
qubo-solver/
├── src/               ── package root (add this to sys.path)
│   ├── __init__.py    ── top-level exports
│   ├── fem/           ── mean-field annealing solver
│   │   ├── __init__.py, interface.py, problem.py, solver_fem.py
│   │   └── customized_problem/
│   ├── sbm/           ── unified SB engine
│   │   ├── __init__.py, sbm.py      ── BaseSolver, strategies, mixins
│   │   ├── _legacy.py, _legacy_gsb.py ── backward compat
│   │   └── utils.py
│   ├── digcim/        ── DIGCIM wrapper (backward compat)
│   ├── solver_base.py, method_registry.py
├── tests/             ── test suite
│   ├── test_unified_solver.py
│   └── test_benchmark_solvers.py   ── grid-search benchmark on Gset/bmincut
├── config/            ── default solver configs (FEM, SBM, QIS3)
└── doc/               ── solver documentation (fem.md, sbm.md, qis3.md)
```

## Latest Updates

- **Unified SB architecture**: strategy pattern (BSB, DSB, Adiabatic, DigCIM)
  + enhancement mixins (GSB, GGSB, Quantization)
- **FEM solver**: mean-field annealing with configurable schedule
- **Benchmark suite**: Cartesian-product grid search over sub-option
  combinations on Gset (maxcut) and bmincut problems
- **Backward compatible**: legacy ``bsb_torch_batch`` and ``gsb_batch`` retained

## License

MIT
