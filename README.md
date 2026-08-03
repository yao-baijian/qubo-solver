# qubo-solver

Standalone solver library for **Quadratic Unconstrained Binary Optimisation (QUBO)**
and **Ising** problems, with a unified, composable architecture.

## What problems can it solve?

| Problem | Description | Converter | Solver |
|---------|-------------|-----------|--------|
| **MaxCut** | Partition graph vertices into two sets maximising the total weight of cut edges. | ``maxcut_to_ising`` | SBM (all strategies), FEM |
| **Balanced MinCut** | Partition graph into two equal-size blocks minimising cut edges. | ``bmincut_to_ising`` | SBM (BSB, DSB), FEM |
| **Max-3SAT** | Maximum satisfiability of 3-CNF clauses. | ``max3sat_to_ising`` | SBM (all strategies) |
| **TSP** | Find the shortest Hamiltonian cycle visiting every city exactly once. | ``tsp_to_ising`` | SBM (all strategies) |
| **QUBO** | Minimise xᵀQx for binary x ∈ {0,1}ⁿ. | ``qubo_to_ising`` | SBM, FEM |
| **QPLIB** | General QUBO with linear bias: ½xᵀQx + bᵀx + q⁰. | ``qplib_to_ising`` | SBM (all strategies) |
| **Higher-order** | Cubic + quadratic: ∑Aᵢⱼₖxᵢxⱼxₖ + xᵀBx. | ``CubicOptimizer`` | Hessian analysis only |

All converters live in :mod:`src.sbm.problems` (except ``CubicOptimizer`` which
is in :mod:`src.sbm.higher_order`).

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

| Solver | Description |
|---|---|
| **DISTCIM** | DistIM — distributed Simulated Coherent Ising Machine with sparse synchronization |

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

## DISTCIM — distributed SimCIM (DistIM)

Implements the sparse-synchronization distributed Ising dynamics from the
paper *"Distributed Ising dynamics for real-time large-scale combinatorial
optimization"* (DistIM) on top of a faithful SimCIM engine — an exact port of
the `simulated-ising-machine` reference repo, verified bit-for-bit against it
(see *Verification* below).

### Problem and dynamics

We minimise the Ising Hamiltonian over spins σ ∈ {−1, +1}^N

$$
H(\sigma) = -\tfrac{1}{2}\,\sigma^T J\,\sigma - h^T\sigma,
$$

with symmetric coupling `J` and bias `h`, using the SimCIM dynamics
(paper Methods 3.2, Eqs. 8–11):

$$
dx = \big[(p-1)\,x + \xi\,(Jx + h)\big]\,dt
     + \tfrac{1}{A_s}\sqrt{x^2 + \tfrac12}\;dW,
\qquad x \leftarrow \operatorname{clip}(x,\,-1,\,+1),
$$

read out at the end as σ = sign(x). Here `p` is the pump (ramped 0 → pmax), ξ
the coupling gain, `A_s` the inverse noise scale, `dt` the Euler–Maruyama step.

### Distributed computation — the math

The `N` variables are split by **columns** into `nparts` modules. Module `m`
owns the variable block `S_m` (size `n_m`), the column slice
`J_part = J[:, S_m]` (N×n_m) and the bias slice `h_part = h[S_m]`. Splitting
`J = J_local + J_cross`, the coupling field on module `m`'s rows is
(paper Eqs. 12–13):

$$
(Jx)_{S_m} \;=\; \underbrace{J_m\,x_m}_{\text{local (exact, every step)}}
\;+\;
\underbrace{\sum_{m'\neq m} J_{mm'}\,x_{m'}}_{\,c_m \;=\; \text{inter-module message}},
\qquad J_m = J[S_m,S_m],\quad J_{mm'} = J[S_m,S_{m'}].
$$

The **full local field** fed into the dynamics is therefore

$$
\text{field}_m \;=\; \underbrace{J_m x_m + c_m}_{\text{coupling}} \;+\; h_m,
$$

and the **only quantity exchanged between modules** is the message
`c_m = Σ_{m'≠m} J_{mm'} x_{m'}` — formed by all-reducing each module's
contribution `J[:, S_{m'}] x_{m'}` and taking the local rows. The bias `h_m` is
**purely local** (partitioned by rows) and is never exchanged.

Messages are refreshed every `time_intvl` (`K`) steps, using one of three
schemes (paper Eqs. 14–16):

| Scheme | at a sync step | between syncs |
|---|---|---|
| `standard` (K=1) | `field = J_m x_m + c_m + h_m` (all-reduce every step, exact) | — |
| `const` | `field = J_m x_m + c_m + h_m` (message refreshed) | `field = J_m x_m + c_m^frozen + h_m` |
| `pulse` | `field = J_m x_m + K·c_m^prev + h_m` (impulse) | `field = J_m x_m + h_m` |

`standard` with `K=1` is exactly the centralized machine; `const` holds the
last message between syncs; `pulse` drops the message off-sync and applies the
previous one as a single impulse scaled by `K`.

### How to run

**Centralized** SimCIM (equivalent to the reference repo's `SimplifiedSimCIM`):

```python
from src import SimCimSolver

solver = SimCimSolver(num_iters=1000, dt=0.1)
solution = solver.solve(Q, num_vars=1000)     # list of 0/1
```

**Distributed** DistIM — single process, all modules emulated in one object
(identical math to a real multi-process run, deterministic, runs on CPU):

```python
from src import DistCimSolver

solver = DistCimSolver(nparts=4, scheme="const", time_intvl=10,
                       quantize_bits=8, num_iters=1000)
solution = solver.solve(Q, num_vars=1000)
```

**Low-level Ising interface** (used by the verification harness):

```python
from src import solve_ising

spins, energy = solve_ising(J, h, nparts=4, scheme="const", time_intvl=10,
                            num_iters=1000, seed=0)   # spins in {-1, +1}
```

Real multi-GPU (`backend="torch"`) launches one process per module via
`torch.distributed` (each rank loads its own `J_part`/`h_part` column slice,
as in the reference repo's `tools/partition.py` + `torchrun` workflow). The
emulated backend above is the same math without the process group.

### Quantization

Two orthogonal quantization points, both implemented:

- **State quantization (FPGA scheme)** — the dynamical state variables are
  reduced to fixed point at every step: `x` (position / c-component) to
  **int8** and `y` (momentum / s-component, two-component models) to
  **int16 or int32**. All control parameters — pump `p`/`α`, coupling gain
  `ξ`, time step `dt`, noise scale `As` — remain in **fixed-point or float**
  precision. Enable with `x_bits` / `y_bits`:

  ```python
  from src import DistCimSolver

  solver = DistCimSolver(nparts=4, scheme="const", time_intvl=10,
                         x_bits=8, y_bits=16, num_iters=1000)
  solution = solver.solve(Q, num_vars=1000)
  ```

  Quantization changes the trajectory slightly, so the quantized run may need
  a **tuned `dt`** (the quantized dynamics behave like a slightly different
  machine). On the traffic check, `x`-int8 with `dt=0.2–0.5` matches or beats
  the float32 run; see `tools/check_traffic_quant.py`.

- **Message quantization (communication link)** — `quantize_bits` quantizes
  only the inter-module message `c_m` that crosses the network (fixed-point,
  dynamic range), modelling a low-precision communication link. It is
  orthogonal to state quantization and both can be combined.

Verification (Levels 3 & 5) shows neither quantization fundamentally changes
the solution: message-quantized and state-quantized runs stay within a few %
of the float32 centralized machine, and distributed + quantized reproduces
the centralized + quantized result.

### Verification against the target repo

`tools/verify_distim.py` verifies that this package reproduces the
*simulated-ising-machine* repo (the DistIM reference implementation) and that
distributed + quantization behaves like the original code:

- **Level 1** — bit-exact match with the target repo's `SimplifiedSimCIM`,
  `StandardCIM`, `SimCIM` (same seed / op order).
- **Level 2** — DistIM field math matches the paper equations; noiseless
  distributed `standard`/`const` with `K=1` are bit-exact vs an independent
  partitioned-centralized reference.
- **Level 3** — `const`/`pulse` with `K>1` stay within a few % of centralized
  while exchanging `K`× fewer messages; 8/16-bit message quantization does not
  degrade the solution.
- **Level 4** — a small synthetic traffic-flow instance built through the
  target repo's `TrafficGenerator`/`TrafficFlow` (the Kowloon/HK construction),
  solved identically by both packages and improving congestion over default
  routing.
- **Level 5** — FPGA state quantization: `x`→int8 (and `y`→int16 for the
  two-component `SimCIM`) stays within a few % of the float32 machine, and
  distributed + state-quantized reproduces the centralized + quantized result.

```bash
python tools/verify_distim.py            # full run
python tools/verify_distim.py --quick    # smaller instances
```

The target repo is imported in-process with stubs for its Ascend-NPU / C++
extensions (only needed for SA/HT models); set `SIM_TARGET_REPO` to point at
another checkout. Unit tests (no target repo needed):

```bash
python -m pytest tests/test_distcim.py
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
│   ├── distcim/       ── DistIM distributed SimCIM (standard/const/pulse + quantization)
│   └── sbm/
│       ├── sbm.py          ── BaseSolver, strategies, mixins, Solver
│       ├── problems.py     ── maxcut_to_ising, bmincut_to_ising, max3sat_to_ising,
│       │                       tsp_to_ising, qubo_to_ising, qplib_to_ising, dt_grid
│       ├── higher_order.py ── CubicOptimizer (cubic + quadratic objective)
│       ├── _legacy.py      ── bsb_torch_batch (backward compat)
│       └── _legacy_gsb.py  ── gsb_batch (backward compat)
├── tests/
│   ├── test_unified_solver.py
│   ├── test_benchmark_solvers.py
│   ├── test_adaptive_annealing.py
│   ├── test_distcim.py
│   ├── test_problems.py        ── all problem-type converters
│   └── test_higher_order.py    ── CubicOptimizer
├── config/
└── doc/
```

## Latest Updates

- **More problem types**: Max-3SAT, QUBO, QPLIB + higher-order (``problems.py``, ``higher_order.py``).
- **Problem tests**: new ``test_problems.py`` (23 tests) covering all converters + solvers.
- **TSP legalizer**: ``tsp_extract_with_legalizer`` repairs invalid tours via greedy search.
- **dt scanning**: ``dt_grid("bsb")`` returns recommended dt ranges per strategy.
- **GSB**: typical best ``A`` is 0.2–0.4 (Goto et al. 2025).
- **Adaptive annealing (FEM)**: per-variable β_i modulated by certainty.
- **Unified SB**: strategy pattern + GSB/GGSB/Quantization mixins.

## License

MIT
