"""SBM (Simulated Bifurcation Machine) solver package.

Unified architecture::

    from src.sbm import BaseSolver, BSBStrategy, GSBMixin, SbmSolver

    # New API — compose strategy + enhancements
    base = BaseSolver(strategy=BSBStrategy(dt=0.1),
                       enhancements=[GSBMixin(A=0.5)],
                       num_iters=500, num_trials=10)
    solutions, energies = base.solve(J_ising)

    # Classic API (backward compatible)
    solver = SbmSolver(num_iters=1000, dt=0.1)
    solution = solver.solve(Q, num_vars)
"""

from .sbm import (
    BaseSolver, Solver, UpdateStrategy,
    BSBStrategy, DSBStrategy, AdiabaticStrategy, DigCIMStrategy,
    GSBMixin, GGSBMixin, QuantizationMixin,
)
from . import problems
from . import higher_order
from ._legacy import bsb_torch_batch, bsb_bmincut_batch
from ._legacy_gsb import gsb_batch
from .problems import (
    maxcut_to_ising, bmincut_to_ising, tsp_to_ising, dt_grid, scale_grid,
    read_tsplib, tsp_coords_to_distance,
    tsp_extract_with_legalizer,
)
from .higher_order import CubicOptimizer, qplib_to_ising

import numpy as np


class SbmSolver:
    """SBM-based QUBO solver (classic API, backward compatible).

    Uses ``bsb_torch_batch`` internally.  For the new composable API
    see :class:`BaseSolver`.
    """

    def __init__(self, num_iters: int = 1000, dt: float = 0.1,
                 num_trials: int = 10, device: str = 'cpu',
                 lambda_balance: float = 1.0, use_compile: bool = False):
        self.num_iters = num_iters
        self.dt = dt
        self.num_trials = num_trials
        self.device = device
        self.lambda_balance = lambda_balance
        self.use_compile = use_compile

    def solve(self, Q, num_vars):
        """Solve a QUBO problem via SBM (classic bSB)."""
        import torch

        Q_mat = torch.zeros(num_vars, num_vars)
        for i, j, val in Q:
            Q_mat[i, j] = val
            if i != j:
                Q_mat[j, i] = val

        J_ising = -Q_mat / 2.0

        init_x = 2 * torch.rand(self.num_trials, num_vars, device=self.device) - 1
        init_y = torch.zeros(self.num_trials, num_vars, device=self.device)

        energies, solutions, _ = bsb_torch_batch(
            J_ising, init_x, init_y, self.num_iters, self.dt,
        )

        final_energies = energies[:, -1]
        best_trial = int(torch.argmin(final_energies))
        best_spins = solutions[best_trial].cpu().numpy()

        solution = ((best_spins + 1) / 2).astype(int).tolist()
        return solution
