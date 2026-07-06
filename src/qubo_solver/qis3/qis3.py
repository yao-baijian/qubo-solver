"""QIS3 — Quantum-Inspired Solver v3: SB + Branch & Bound + adaptive perturbation."""

import torch
import math
import numpy as np
from qubo_solver.sbm.sbm import bsb_torch_batch


class QIS3:
    """Three-phase quantum-inspired solver.

    Phase 1 — Population SB (multiple bSB trials)
    Phase 2 — Branch & Bound on unresolved variables
    Phase 3 — Adaptive perturbation (random flips + re-run)
    """

    def __init__(self, J, sb_type='bsb', num_iters=1000, dt=0.1,
                 branch_depth=2, popsize=10, adaptive=True,
                 device='cpu'):
        self.J = J
        self.sb_type = sb_type
        self.num_iters = num_iters
        self.dt = dt
        self.branch_depth = branch_depth
        self.popsize = popsize
        self.adaptive = adaptive
        self.device = device
        self.N = J.shape[0]

    def solve(self):
        """Run all three phases and return best solution."""
        # Phase 1: Population SB
        batch = self.popsize
        init_x = 2 * torch.rand(batch, self.N, device=self.device) - 1
        init_y = torch.zeros(batch, self.N, device=self.device)

        energies, solutions, _ = bsb_torch_batch(
            self.J, init_x, init_y, self.num_iters, self.dt,
        )

        final_energies = energies[:, -1]
        best_idx = int(torch.argmin(final_energies))
        best_solution = solutions[best_idx]
        best_energy = final_energies[best_idx].item()

        # Phase 2: Branch & Bound (simplified)
        if self.branch_depth > 0:
            best_solution, best_energy = self._branch_and_bound(
                best_solution, best_energy
            )

        # Phase 3: Adaptive perturbation
        if self.adaptive:
            best_solution, best_energy = self._adaptive_perturb(
                best_solution, best_energy
            )

        return best_solution.cpu().numpy(), best_energy

    def _branch_and_bound(self, solution, energy):
        """Simple branch & bound: flip subsets of variables."""
        N = self.N
        best_sol = solution.clone()
        best_eng = energy

        for depth in range(1, self.branch_depth + 1):
            for start in range(0, N, max(1, N // (depth * 2))):
                candidate = best_sol.clone()
                end = min(start + depth, N)
                candidate[start:end] *= -1

                Jc = candidate @ self.J.T
                e = -0.5 * torch.sum(candidate * Jc)
                eng = -0.25 * torch.sum(self.J) - 0.5 * e

                if eng < best_eng:
                    best_eng = eng.item()
                    best_sol = candidate.clone()

        return best_sol, best_eng

    def _adaptive_perturb(self, solution, energy, max_flips=3):
        """Adaptive perturbation: random flips followed by re-optimization."""
        N = self.N
        best_sol = solution.clone()
        best_eng = energy

        for flip_count in range(1, max_flips + 1):
            for _ in range(5):
                perturbed = best_sol.clone()
                flip_indices = torch.randperm(N)[:flip_count]
                perturbed[flip_indices] *= -1

                init_x = perturbed.unsqueeze(0)
                init_y = torch.zeros(1, N, device=self.device)

                energies, new_solutions, _ = bsb_torch_batch(
                    self.J, init_x, init_y, max(100, self.num_iters // 4), self.dt,
                )
                new_energy = energies[0, -1].item()
                new_sol = new_solutions[0]

                if new_energy < best_eng:
                    best_eng = new_energy
                    best_sol = new_sol.clone()

        return best_sol, best_eng
