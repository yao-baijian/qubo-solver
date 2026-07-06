# qubo-solver

Standalone QUBO solver library extracted from the TPU full-stack optimization
project.  Provides four solver families:

| Solver | Description |
|---|---|
| **FEM** | Free Energy Minimisation — mean-field annealing |
| **SBM** | Simulated Bifurcation Machine — bSB / GSB |
| **QIS3** | Quantum-Inspired Solver v3 — SB + Branch & Bound |
| **DIGCIM** | Digital Chaotic Ising Machine |

## Installation

```bash
pip install -e .
```

Or via conda:

```bash
conda env create -f environment.yml
conda activate qubo-solver
pip install -e .
```

## Usage

```python
from qubo_solver import FemSolver, SbmSolver, Qis3Solver

# Build a QUBO matrix (list of (i, j, value) tuples)
Q = [(0, 0, -1), (1, 1, -1), (0, 1, 2)]

# Solve with any solver
solver = SbmSolver(num_iters=1000, dt=0.1, num_trials=10)
solution = solver.solve(Q, num_vars=2)
```

## Git Submodule Usage

Add as a submodule in your project:

```bash
git submodule add https://github.com/yao-baijian/qubo-solver.git lib/qubo-solver
pip install -e lib/qubo-solver
```

## License

MIT
