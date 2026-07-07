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

No ``pip install`` needed — add ``lib/qubo-solver/src`` to ``sys.path``.

## Project Structure

```
src/
├── fem/          ── self-contained FEM solver package
├── sbm/          ── unified SB: strategies, mixins, legacy solvers
│   ├── sbm.py        ── BaseSolver, Solver, strategies, mixins
│   ├── _legacy.py    ── bsb_torch_batch (backward compat)
│   └── _legacy_gsb.py─ gsb_batch (backward compat)
└── __init__.py   ── top-level exports
```

## License

MIT
