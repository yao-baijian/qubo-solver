"""FEM core solver — mean-field annealing via gradient descent on free energy."""

import torch
import math
import numpy as np
from typing import Optional, Callable


class Solver:
    """
    Mean-field annealing solver.
    Optimises variational parameters h via gradient descent on the free energy.
    """

    def __init__(
        self,
        problem,
        num_trials: int = 5,
        num_steps: int = 500,
        betamin: float = 0.01,
        betamax: float = 0.5,
        anneal: str = 'lin',
        optimizer: str = 'adam',
        learning_rate: float = 0.1,
        dev: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        seed: int = 1,
        q: int = 2,
        manual_grad: bool = False,
        h_factor: float = 0.01,
        sparse: bool = False,
        drawer: Optional[Callable] = None,
        use_compile: bool = False,
    ):
        self.problem = problem
        self.num_trials = num_trials
        self.num_steps = num_steps
        self.betamin = betamin
        self.betamax = betamax
        self.anneal = anneal
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.dev = dev
        self.dtype = dtype
        self.seed = seed
        self.q = q
        self.manual_grad = manual_grad
        self.h_factor = h_factor
        self.sparse = sparse
        self.drawer = drawer
        self.use_compile = use_compile

        self._setup()

    def _setup(self):
        torch.manual_seed(self.seed)
        self.N = self.problem.num_nodes

        if self.anneal == 'inv':
            self.beta_arr = 1.0 / (
                self.betamin
                + (self.betamax - self.betamin)
                * torch.linspace(0, 1, self.num_steps)
            )
        elif self.anneal == 'lin':
            self.beta_arr = torch.linspace(self.betamin, self.betamax, self.num_steps)
        elif self.anneal == 'exp':
            self.beta_arr = torch.logspace(
                math.log10(self.betamin),
                math.log10(self.betamax),
                self.num_steps,
            )
        else:
            raise ValueError(f"Unknown anneal type: {self.anneal}")

        self.p = self._init_p()

    def _init_p(self):
        p = torch.softmax(
            torch.randn(self.num_trials, self.N, self.q, device=self.dev)
            * self.h_factor,
            dim=-1,
        )
        return p

    def _s(self, p):
        """Entropy of Bernoulli."""
        p1 = p[..., 1]
        p0 = p[..., 0]
        return -(p0 * torch.log(p0 + 1e-12) + p1 * torch.log(p1 + 1e-12))

    @staticmethod
    def _free_energy(p, expected, beta, entropy):
        return beta * expected - entropy

    def iterate(self, steps, p, beta_arr, lr, manual_grad):
        """Run mean-field iterations.

        Parameters
        ----------
        steps : int
        p : Tensor (num_trials, N, q)
        beta_arr : Tensor (steps,)
        lr : float
        manual_grad : bool

        Returns
        -------
        p_opt : Tensor (num_trials, N, q)
        fe_record : Tensor (steps, num_trials)
        """
        p_opt = p.clone()
        fe_record = torch.zeros(steps, self.num_trials, device=self.dev)

        for i in range(steps):
            beta = beta_arr[i]
            expected = self.problem.generate_expected(p_opt)
            if manual_grad:
                grad = self.problem.generate_grad(p_opt)
            else:
                grad = None

            entropy = self._s(p_opt).sum(dim=1)
            fe = self._free_energy(p_opt, expected, beta, entropy)
            fe_record[i] = fe

            if manual_grad:
                p_opt = p_opt - lr * grad
            else:
                # Automatic differentiation for the free energy
                # (placeholder — user should implement autograd version)
                p_opt = p_opt - lr * grad if grad is not None else p_opt

            # Project to simplex
            p_opt = torch.softmax(p_opt, dim=-1)

        return p_opt, fe_record

    def solve(self):
        """Run the full solver and return (configs, energies).

        Returns
        -------
        configs : Tensor (num_trials, N) — binary assignments (0 or 1)
        energies : Tensor (num_steps,) — free energy trace
        """
        p_opt = self.p.clone()
        fe_record = torch.zeros(self.num_steps, self.num_trials, device=self.dev)

        for step in range(self.num_steps):
            beta = self.beta_arr[step]
            expected = self.problem.generate_expected(p_opt)

            if self.manual_grad:
                grad = self.problem.generate_grad(p_opt)

            entropy = self._s(p_opt).sum(dim=1)
            fe = self._free_energy(p_opt, expected, beta, entropy)
            fe_record[step] = fe

            if self.manual_grad:
                p_opt = p_opt - self.learning_rate * grad

            p_opt = torch.softmax(p_opt, dim=-1)

        configs = self.problem.generate_infer(p_opt)  # (num_trials, N)
        return configs, fe_record[-1]  # return final energies per trial
