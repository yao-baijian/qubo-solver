"""qubo-solver — standalone QUBO solver library.

Solvers
-------
- FEM (Free Energy Minimisation / mean-field annealing)
- SBM (Simulated Bifurcation Machine)
- QIS3 (Quantum-Inspired Solver v3 — SB + Branch & Bound)
- DIGCIM (Digital Chaotic Ising Machine)
"""

from .fem import FemSolver
from .sbm import SbmSolver
from .qis3 import Qis3Solver

__all__ = ["FemSolver", "SbmSolver", "Qis3Solver"]
