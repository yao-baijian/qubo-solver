# qubo-solver

Standalone solver library for **Quadratic Unconstrained Binary Optimisation (QUBO)**
and **Ising** problems, with a unified, composable architecture.

## What problems can it solve?

| Problem | Description | Solver |
|---------|-------------|--------|
| **MaxCut** | Partition graph vertices into two sets maximising the total weight of cut edges. | SBM (all strategies), FEM |
| **Balanced MinCut** | Partition graph into two equal-size blocks minimising cut edges. | SBM (``BSBStrategy``, ``DSBStrategy``), FEM |
| **TSP** | Find the shortest Hamiltonian cycle visiting every city exactly once. | SBM (all strategies) |
| **General QUBO** | Minimise xᵀQx for binary x ∈ {0,1}ⁿ. | SBM, FEM |

Each problem type has a dedicated converter in ``src/sbm/problems.py``:
``maxcut_to_ising``, ``bmincut_to_ising``, ``tsp_to_ising``.

## Solvers

| Solver | Method | Best for |
|--------|--------|----------|
| **SBM** (SB engine) | Strategy pattern + enhancement mixins | MaxCut, MinCut, TSP, large QUBO |
| **FEM** | Mean-field annealing with configurable β schedule | Small-to-medium QUBO, multi-trial |

### SBM — Strategies

| Strategy | Update rule | Typical dt range |
|----------|-------------|-----------------|
| `BSBStrategy` | Ballistic SB (standard) | 0.10 – 1.25 |
| `DSBStrategy` | Discrete SB (sign coupling) | 0.10 – 1.25 |
| `AdiabaticStrategy` | Scheduled p(t) | 0.05 – 1.00 |
| `DigCIMStrategy` | Digital Chaotic Ising Machine | 0.05 – 1.00 |

### SBM — Enhancement mixins (orthogonal, combinable)

| Mixin | Effect | Key parameter |
|-------|--------|---------------|
| `GSBMixin` | Per-oscillator p_i with nonlinear control | ``A`` (typical 0.2–0.4 for best PS) |
| `GGSBMixin` | Global guidance across batch replicas | ``k``, ``strength`` |
| `QuantizationMixin` | Fixed-point simulation | ``num_bits`` (4 or 8) |

> **Tip**: When using GSB, scan ``A ∈ [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]``.
> The best PS rate typically occurs in the 0.2–0.4 range (Goto et al. 2025).

## Installation

```bash
pip install -e .
```

## Usage

```python
from src.sbm import (
    BaseSolver, BSBStrategy, GSBMixin, dt_grid,
    maxcut_to_ising, bmincut_to_ising, tsp_to_ising,
)
from src.fem import FemSolver
import torch

# ═══ MaxCut ══════════════════════════════════════════════════════════
J_graph = torch.randn(N, N); J_graph = (J_graph + J_graph.T) / 2; J_graph.fill_diagonal_(0)
J_ising = maxcut_to_ising(J_graph)

base = BaseSolver(strategy=BSBStrategy(dt=0.5),
                  enhancements=[GSBMixin(A=0.3)],
                  num_iters=500, num_trials=10)
solutions, energies = base.solve(J_ising)

# ═══ TSP ═════════════════════════════════════════════════════════════
cities = torch.rand(N, 2)
dists = torch.cdist(cities, cities)
J_tsp = tsp_to_ising(dists, fixed_start_city=0)
solutions, energies = BaseSolver(strategy=BSBStrategy(dt=0.3),
                                 num_iters=1000, num_trials=10).solve(J_tsp)

# ═══ FEM (general QUBO) ═════════════════════════════════════════════
Q = [(0, 0, -1), (1, 1, -1), (0, 1, 2)]
solver = FemSolver(num_steps=1000, num_trials=10)
solution = solver.solve(Q, num_vars=2)

# ═══ dt scanning ═════════════════════════════════════════════════════
for dt in dt_grid("bsb"):                # 0.10 to 1.25 step 0.05
    base = BaseSolver(strategy=BSBStrategy(dt=dt), ...)
```

## Git Submodule Usage

```bash
git submodule add https://github.com/yao-baijian/qubo-solver.git lib/qubo-solver
```

No ``pip install`` needed — add ``lib/qubo-solver/src`` to ``sys.path``.

## Project Structure

```
qubo-solver/
├── src/
│   ├── __init__.py
│   ├── fem/           ── mean-field annealing
│   └── sbm/
│       ├── sbm.py          ── BaseSolver, strategies, mixins, Solver
│       ├── problems.py     ── maxcut_to_ising, bmincut_to_ising, tsp_to_ising, dt_grid
│       ├── _legacy.py      ── bsb_torch_batch (backward compat)
│       └── _legacy_gsb.py  ── gsb_batch (backward compat)
├── tests/
│   ├── test_unified_solver.py
│   └── test_benchmark_solvers.py
├── config/
└── doc/
```

## Latest Updates

- **Problem types**: dedicated converters for MaxCut, Balanced MinCut, TSP (``problems.py``).
- **dt scanning**: ``dt_grid("bsb")`` returns recommended dt ranges per strategy.
- **GSB**: typical best ``A`` is 0.2–0.4 (Goto et al. 2025).
- **Adaptive annealing (FEM)**: per-variable β_i modulated by certainty.
- **Unified SB**: strategy pattern + GSB/GGSB/Quantization mixins.

## License

MIT
