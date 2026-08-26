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

## benckmarks

| Benchmark | Best-known |
|--------|--------|
| [Gset](https://web.stanford.edu/~yyye/yyye/Gset/) | [max-cut](https://huggingface.co/datasets/Yuma-Ichikawa/qqa4co-bench) |
| [COLOR](http://mat.gsia.cmu.edu/COLOR/instances.html) | [Vertex-coloring](https://cedric.cnam.fr/~porumbed/graphs/) |
| [SATLIB](http://www.satlib.org/) | [SAT](https://www.cril.univ-artois.fr/) |
| [qplib](https://qplib.zib.de/) | [qplib-bestknown](https://arxiv.org/pdf/2508.01299) |
| [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) | [TSP](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) |
| [Transportation Networks]()  | |
| [SuiteSparse Matrix Collection](https://suitesparse-collection-website.herokuapp.com/Gset)  | |


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

### Distributed computation — the math (broadcast frame)

The `N` variables are split by **columns** into `nparts` modules (a
`nparts × nparts` **block partition** of `J`). Module `m` owns the variable
block `S_m` (size `n_m`) and the bias slice `h_m = h[S_m]`. Splitting
`J = J_local + J_cross` into blocks `J_{ij} = J[S_i, S_j]`, the coupling field
on module `m`'s rows is (paper Eqs. 12–13):

$$
(Jx)_{S_m} \;=\; \underbrace{J_{mm}\,x_m}_{\text{local (exact, every step)}}
\;+\;
\underbrace{\sum_{m'\neq m} J_{mm'}\,x_{m'}}_{\,c_m \;=\; \text{inter-module message}},
$$

and the **only quantity exchanged between modules** is the message
`c_m = Σ_{m'≠m} J_{mm'} x_{m'}`. The bias `h_m` is **purely local** and is
never exchanged.

**Broadcast frame (this implementation).** At a synchronization step every
node `m` computes its **off-diagonal block contributions**
`c_{i,m} = J_{i,m} x_m` for every `i ≠ m` and broadcasts them to node `i`
(`dist.all_to_all` in the real backend); every node `i` receives `c_{i,j}`
from all `j ≠ i` and **combines them once into a single frozen remote field**
`c_remote_i`. Between syncs the field is only

$$
\text{field}_m \;=\; \underbrace{J_{mm} x_m}_{\text{local block, every step}}
\;+\; c_{remote_m} \;+\; h_m,
$$

so the per-node contributions are **never re-added step after step** — no
over-computation. In the emulated backend the block matmuls are batched with
`torch.bmm` (numerically identical to the per-block broadcast frame).

Messages are refreshed every `time_intvl` (`K`) steps, using one of three
schemes (paper Eqs. 14–16):

| Scheme | at a sync step | between syncs |
|---|---|---|
| `standard` (K=1) | `field = J_{mm} x_m + c_remote_m + h_m` (exact, refresh every step) | — |
| `const` | `field = J_{mm} x_m + c_remote_m + h_m` (message refreshed) | `field = J_{mm} x_m + c_remote_m^{frozen} + h_m` |
| `pulse` | `field = J_{mm} x_m + K·c_remote_m^{prev} + h_m` (impulse) | `field = J_{mm} x_m + h_m` |

`standard` with `K=1` is exactly the centralized machine; `const` holds the
combined message between syncs; `pulse` drops the message off-sync and applies
the previous one as a single impulse scaled by `K`.

**Per-step coupling FLOPs** (the freeze-field acceleration): the central
machine does `N²` MACs/step; the broadcast frame does the `N²/nparts` diagonal
blocks every step plus the `(nparts−1)·N²/nparts` off-diagonal blocks once per
sync, i.e. `(K·(N/nparts)² + (nparts−1)·N²/nparts)/K` MACs/step on average —
a `~nparts·K/(K + nparts − 1)` reduction that grows with `nparts` and the sync
period `K`.

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
`torch.distributed` (each rank owns its column block and exchanges the
off-diagonal contributions with `dist.all_to_all`, the real broadcast frame).
The emulated backend above is the same math without the process group.

### CuPy backend

A pure-CuPy implementation of the same broadcast frame lives in
`src/distcim/cupy_engine.py` — same block partition, same "compute
`c_{i,m} = J_{i,m} x_m` and combine into one frozen `c_remote`" exchange, with
all low-precision modes implemented as grid-quantized fp32 emulations
(native fp16 kernels are not available in every CuPy build, so the numbers
match the torch emulated paths):

```python
from src.distcim.cupy_engine import solve_ising_cupy, CupyDistCimSolver

spins, energy = solve_ising_cupy(J, h, nparts=4, scheme="const", time_intvl=10,
                                 precision="int8", num_iters=1000, seed=0)
solver = CupyDistCimSolver(nparts=4, scheme="const", time_intvl=10,
                           precision="fp16", num_iters=1000)
solution = solver.solve(Q, num_vars=1000)
```

`J`/`h` may be torch tensors, cupy arrays or numpy arrays. On Windows the
module adds `$CUDA_PATH/bin` to the DLL search path (`os.add_dll_directory`)
so CuPy can find cuBLAS; CuPy needs a CUDA toolkit installed (a `cupy-cuda12x`
wheel matching your toolkit). Both the precision and the runtime benchmark
accept `--backend torch|cupy` / `--backends torch cupy` to compare them.

### Multi-GPU (4× RTX 4090) — CuPy + NCCL

The single-GPU paths above **batch the nodes** (one batched bmm per step +
one dense GEMM / batched bmm per sync on the padded uniform partition) so the
freeze-field FLOP saving is visible on one GPU. For a **multi-GPU cluster**
the framework ships a real distributed path over **NCCL** — CuPy ≥ 12 bundles
NCCL (`cupyx.distributed`), so **no CUDA C is required**:

- `src/distcim/cupy_engine.py` — `CupyDistNCCLFieldCoupler` (real broadcast
  frame: each rank computes `c_{i,m} = J_{i,m} x_m` and exchanges them with
  `dist.all_to_all`, combining the received messages into one frozen
  `c_remote`) and `CupyDistCIMNCCL` / `solve_ising_cupy_nccl` (per-rank
  engine; all-gathers the spins and all-reduces the Ising energy).
- `src/distcim/distributed.py` — the `TorchDistFieldCoupler` /
  `backend="torch"` path is the same broadcast frame over
  `torch.distributed` (use when a torch build with NCCL is installed).
- `tools/run_distcim_multigpu.py` — launcher that spawns one process per GPU.

```bash
# single GPU smoke test (this machine)
python tools/run_distcim_multigpu.py --nproc 1 --cars 250 --routes 3 --force-synthetic

# 4x RTX 4090, single node
python tools/run_distcim_multigpu.py --nproc 4 --cars 1250 --routes 3 --force-synthetic \
    --steps 10000 --dt 0.01 --precision fp32 --K 10
```

Each rank owns one column block of `J` and runs its SimCIM dynamics on its
own GPU; only the `(N/4)²` local block and the frozen `c_remote` are touched
between syncs, so the wall-clock benefit of the freeze field (up to `flop_ratio`
≈ 3.1× at nparts=4, K=10) finally materialises on real hardware.

**Multi-GPU scaling benchmark** — wall time vs the number of GPUs (1→2→…→N)
at fixed steps / K / precision, to measure the real speedup of the broadcast
frame on the cluster:

```bash
# 4x RTX 4090, single node (Linux, cupy-cuda12x with NCCL)
python tools/benchmark_distcim_multigpu.py --cars 1250 --routes 3 --force-synthetic \
    --nproc 1 2 4 --steps 1000 10000 --dt 0.01 --precision fp32 --K 10 --repeats 3
# options: --cars --routes --seed --nproc --steps --dt --K --precision --pump
#          --pmax --As --A_init --xi --repeats --host --port --out
```

Output: `benchmark_results/traffic_realistic/multigpu_scaling.csv` (per `nproc`
× `steps` median wall time + speedup vs 1 GPU). Requires cupy built with NCCL
(Linux `cupy-cuda12x` bundles it; Windows wheels do not — the script prints a
clear error).

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

- **Arithmetic precision (hardware arithmetic)** — `precision` lowers the
  arithmetic precision of the dominant coupling product `J @ c` (the hardware
  the machine would run on) while the outer dynamics accumulate in float32:
  `fp32` (default), `fp16`, `bf16`, `int8`, `int4`, `fp8` (e4m3fn), `fp4`
  (e2m1). `fp16`/`bf16` use native tensor-core matmuls; `int8` uses the int8
  tensor-core matmul (`torch._int_mm`) on CUDA (emulated elsewhere); `fp8`/
  `fp4` use the exact hardware value grids with an emulated matmul. All modes
  share a per-row scale scheme, so the dequantised field stays in float32:

  ```python
  from src import DistCimSolver, solve_ising

  # QUBO-level
  solver = DistCimSolver(nparts=4, scheme="const", time_intvl=10,
                         precision="int8", num_iters=1000)
  solution = solver.solve(Q, num_vars=1000)

  # Ising-level
  spins, energy = solve_ising(J, h, nparts=4, scheme="const", time_intvl=10,
                              precision="fp16", num_iters=1000)
  ```

  `precision` is orthogonal to `x_bits`/`y_bits` (state quantization) and
  `quantize_bits` (message quantization); all can be combined.

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

### Benchmarks — how to run

All scripts run on CPU with dense `J` (keep `N` ≤ ~5000 per run). Dependencies:
`torch`, `numpy`, `scipy`, `yacs`, `numba`, `networkx`. The
`simulated-ising-machine` reference repo is imported in-process with stubs
(see *Verification* above).

#### Traffic flow

Small synthetic instance (fast, ~1 min) — original vs quantized vs distributed
(K=1/5/10), best-dt sweep over dt in 0.1..1.3 (step 0.1):

```bash
python tools/check_traffic_quant.py
```

Realistic instance (target-repo `TrafficGenerator`/`TrafficFlow` construction;
~5000 vars / dense 5000×5000). The instance is generated once and cached to
disk; the full best-dt sweep takes ~15–30 min (parallel workers):

```bash
python tools/benchmark_traffic_realistic.py                                # 1250 cars -> ~5100 vars
python tools/benchmark_traffic_realistic.py --cars 400 --iters 200 --workers 4   # quick check
# options: --cars --routes --iters --dts --seeds --workers
```

Output: per-config best congestion/energy and `benchmark_results/traffic_realistic/results.md`.

**Precision benchmark (dt × K sweep)** — solution quality of DistIM across
arithmetic precisions (`fp32`/`fp16`/`int8`/`int4`/`fp8`/`fp4`) at fixed steps
(1000/3000/5000/10000), sweeping the Euler step `dt` and the freeze-field sync
period `K`, on GPU (or CPU with `--force-synthetic` for a quick check), for the
`central` and `dist` (broadcast-frame freeze-field) configs and either backend.
The `dist` sweep uses the **paper Methods** hyper-parameters: Simplified
SimCIM, pump ramped linearly 0 → p_max (p_max = 1.1 for traffic),
ξ = ½(ΣJ²/(n−1))⁻¹/² (`inverse_interaction_rms`), A_s = 70, A_init = 10⁻³,
constant compensation, nparts = 4, K ∈ {1, 2, 3, 5, 7, 10}, 3 routes/car:

```bash
python tools/benchmark_distcim_precision.py --device cuda --dts 0.1 0.2 0.4 0.6 0.8 1.0 1.5 2.0 --ks 1 2 3 5 7 10   # torch (paper sweep)
python tools/benchmark_distcim_precision.py --device cuda --backend cupy                                            # cupy
python tools/benchmark_distcim_precision.py --device cpu --force-synthetic --cars 60 --workers 2                    # quick check
# options: --cars --routes --seed --device --steps --precisions --dts --seeds --ks --configs central|dist --backend torch|cupy --workers
```

Output: `benchmark_results/traffic_realistic/precision_ksweep.csv` + `.md`.
Build the markdown report (best congestion vs K per precision, summary over the
whole dt × K × steps sweep, freeze-field FLOP analysis):

```bash
python tools/build_distcim_report.py --csv precision_ksweep.csv --baseline 11904
```

On the 750-variable traffic instance the sweep found **int4 the best precision**
(best congestion 894 vs baseline 11904), with quality essentially preserved
across K = 1…10 — confirming the paper's freeze-field claim.

**Runtime benchmark** — per-solve wall time of `cim` (central SimCIM), `distcim`
(freeze-field `const` K=10), `distcim-int8` and `sbm` at fixed steps
(1000/3000/5000/10000), best-dt sweep then median of `--repeats` timed runs
(device synchronize), on the torch and/or cupy backend:

```bash
python tools/benchmark_distcim_runtime.py --device cuda                                  # torch
python tools/benchmark_distcim_runtime.py --device cuda --backends torch cupy           # both
python tools/benchmark_distcim_runtime.py --device cpu --force-synthetic --cars 60 --repeats 2   # quick check
# options: --cars --routes --seed --device --steps --seeds --methods --backends torch|cupy --repeats --no-sweep --dt
```

Output: `benchmark_results/traffic_realistic/runtime.csv` + `runtime.md`.

Both scripts load the cached target-repo instance when present, else build it
via the target repo, else fall back to a self-contained synthetic traffic
instance (networkx) so they run anywhere. The freeze-field `distcim` config
computes only the local `N/nparts` block between syncs — `(K·(N/nparts)² +
N²)/K` MACs/step vs `N²` for the central machine — which is the measured
acceleration on GPU.

#### Gset MaxCut (ground state)

All 71 Gset instances, best-dt sweep (0.1..1.3 step 0.1), iters 1000/3000/5000,
for float32 / quant8 / distributed-quant8. **Heavy on CPU** — the fine grid ×
71 instances can take hours, so spot-check a few instances first:

```bash
python tools/benchmark_gset_distcim.py --instances G1 G6 G11 --workers 4   # spot check
python tools/benchmark_gset_distcim.py --workers 8                         # full sweep
# options: --instances --dts --seeds --iters --no-dist --workers
```

Note: the `const` scheme with K≥10 can collapse to a uniform spin state on
some instances (genuine reference-repo behaviour, verified on 4 real ranks
with `tools/run_target_dist_g1.py`); the distributed column therefore uses K=5.

Output: `benchmark_results/gset_distcim_compare.csv`.

#### Reference-repo distributed check (4 real ranks)

```bash
python tools/run_target_dist_g1.py    # target repo Standard/ConstApprox with 4 torch.distributed ranks on G1
```

#### Collect the results record

```bash
python tools/collect_results.py       # writes benchmark_results/RESULTS.md
```

`benchmark_results/` is git-ignored (run-specific CSVs); `RESULTS.md` is the
curated record and is force-added (`git add -f`) when committing.

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
│   │   ├── distributed.py  ── DistIMEngine, emulated/torch backends (broadcast frame)
│   │   ├── engines.py      ── SimCIMEngine, CentralFieldCoupler
│   │   ├── precision.py    ── arithmetic precision modes (fp32/fp16/bf16/int8/int4/fp8/fp4)
│   │   └── cupy_engine.py  ── pure-CuPy broadcast-frame backend
│   └── sbm/
│       ├── sbm.py          ── BaseSolver, strategies, mixins, Solver
│       ├── problems.py     ── maxcut_to_ising, bmincut_to_ising, max3sat_to_ising,
│       │                       tsp_to_ising, qubo_to_ising, qplib_to_ising, dt_grid
│       ├── higher_order.py ── CubicOptimizer (cubic + quadratic objective)
│       ├── _legacy.py      ── bsb_torch_batch (backward compat)
│       └── _legacy_gsb.py  ── gsb_batch (backward compat)
├── tools/
│   ├── verify_distim.py             ── 5-level verification vs the reference repo
│   ├── check_traffic_quant.py       ── small traffic: original vs quant vs dist (K)
│   ├── benchmark_traffic_realistic.py ── ~5000-var realistic traffic benchmark
│   ├── benchmark_distcim_precision.py ── precision x dt x K sweep (GPU, best-dt, torch/cupy)
│   ├── build_distcim_report.py      ── REPORT_distcim_gpu.md from the sweep CSV
│   ├── benchmark_distcim_runtime_ksweep.py ── runtime K-sweep: distCIM vs cim vs sbm (GPU)
│   ├── benchmark_distcim_runtime.py  ── runtime: cim vs distcim vs sbm (GPU, torch/cupy)
│   ├── run_distcim_multigpu.py      ── multi-GPU distcim launcher (CuPy+NCCL, per-GPU rank)
│   ├── benchmark_distcim_multigpu.py ── multi-GPU scaling benchmark (1..N GPUs)
│   ├── traffic_common.py            ── shared traffic instance loader + congestion
│   ├── benchmark_gset_distcim.py    ── full Gset MaxCut benchmark
│   ├── run_target_dist_g1.py        ── reference repo distributed check (4 ranks)
│   ├── target_repo.py               ── in-process import of the reference repo (stubs)
│   └── collect_results.py           ── writes benchmark_results/RESULTS.md
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
- **DistCIM precision modes**: arithmetic precision for the coupling matmul
  (`fp32`/`fp16`/`bf16`/`int8`/`int4`/`fp8`/`fp4`) + precision/runtime
  traffic benchmarks on GPU (`tools/benchmark_distcim_precision.py`,
  `tools/benchmark_distcim_runtime_ksweep.py`).
- **Single-GPU acceleration**: the emulated broadcast frame batches the nodes
  (padded uniform partition: one batched bmm/step + dense-GEMM sync,
  `c_remote = J·x − J_diag·x`) so distCIM beats central CIM on one GPU at K≥2
  (up to ~1.5–2.8× across precisions; `runtime.md`).
- **Multi-GPU (4× RTX 4090)**: real distributed path over CuPy-NCCL
  (`CupyDistNCCLFieldCoupler` / `solve_ising_cupy_nccl`) + launcher
  (`run_distcim_multigpu.py`) + scaling benchmark
  (`benchmark_distcim_multigpu.py`).

## License

MIT
