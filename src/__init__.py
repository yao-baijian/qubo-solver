"""qubo-solver — standalone QUBO solver library.

Architecture
------------
- **Base + Strategy** — :class:`BaseSolver` runs the core loop; pluggable
  :class:`UpdateStrategy` subclasses define position/momentum dynamics
  (BSB, dSB, adiabatic, Dig-CIM).
- **Enhancement mixins** — orthogonal features composed into the loop:
  GSB (individual ``p_i``), GGSB (global guidance), Quantisation.
- **FEM** — mean-field annealing (Free Energy Minimisation).
- **Classic SBM / QIS3** — retained for backward compatibility.

Usage::

    from src.solver import Solver, BSBStrategy, GSBMixin

    solver = Solver(strategy=BSBStrategy(dt=0.1),
                    enhancements=[GSBMixin(A=0.5)],
                    num_iters=500, num_trials=10)
    solution = solver.solve(Q, num_vars)
"""

from .fem import FemSolver
from .sbm import SbmSolver
from .qis3 import Qis3Solver
from .solver import (
    BaseSolver, Solver,
    UpdateStrategy, BSBStrategy, DSBStrategy,
    AdiabaticStrategy, DigCIMStrategy,
    GSBMixin, GGSBMixin, QuantizationMixin,
)

__all__ = [
    "FemSolver", "SbmSolver", "Qis3Solver",
    "BaseSolver", "Solver",
    "UpdateStrategy", "BSBStrategy", "DSBStrategy",
    "AdiabaticStrategy", "DigCIMStrategy",
    "GSBMixin", "GGSBMixin", "QuantizationMixin",
]
