"""qubo-solver — standalone QUBO solver library.

Architecture
------------
- **FEM** — mean-field annealing (Free Energy Minimisation).
- **SBM** — unified simulated bifurcation with strategy pattern + mixins:
  ``BaseSolver``, ``UpdateStrategy`` subclasses (BSB, dSB, adiabatic, Dig-CIM),
  enhancement mixins (GSB, GGSB, Quantisation).

Usage::

    from src.sbm import BaseSolver, BSBStrategy, GSBMixin, SbmSolver

    # New composable API
    base = BaseSolver(strategy=BSBStrategy(dt=0.1),
                      enhancements=[GSBMixin(A=0.5)],
                      num_iters=500, num_trials=10)
    solutions, energies = base.solve(J_ising)

    # Classic API
    solver = SbmSolver(num_iters=1000, dt=0.1)
    solution = solver.solve(Q, num_vars)
"""

from .fem import FemSolver
from .sbm import (
    SbmSolver,
    BaseSolver, Solver,
    UpdateStrategy, BSBStrategy, DSBStrategy,
    AdiabaticStrategy, DigCIMStrategy,
    GSBMixin, GGSBMixin, QuantizationMixin,
)

__all__ = [
    "FemSolver", "SbmSolver",
    "BaseSolver", "Solver",
    "UpdateStrategy", "BSBStrategy", "DSBStrategy",
    "AdiabaticStrategy", "DigCIMStrategy",
    "GSBMixin", "GGSBMixin", "QuantizationMixin",
]
