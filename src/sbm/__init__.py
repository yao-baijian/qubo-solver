"""SBM (Simulated Bifurcation Machine) solver package."""

from .sbm import bsb_torch_batch, bsb_bmincut_batch
from .gsb import gsb_batch
import numpy as np


class SbmSolver:
    """SBM-based QUBO solver with standard solve(Q, num_vars) interface.

    Converts QUBO to Ising model internally and uses the SBM solver.

    Usage::

        from qubo_solver.sbm import SbmSolver

        solver = SbmSolver(num_iters=1000, dt=0.1)
        solution = solver.solve(Q, num_vars)
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
        """Solve a QUBO problem via SBM.

        Parameters
        ----------
        Q : list of (int, int, float)
            Sparse upper-triangular QUBO matrix.
        num_vars : int
            Number of binary variables.

        Returns
        -------
        list of int
            Binary solution vector of length num_vars (values 0 or 1).
        """
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
