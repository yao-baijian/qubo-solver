"""Unified Ising/QUBO solver with strategy-based update rules and mixin enhancements.

Architecture
------------
- :class:`BaseSolver` — core loop (initialise x/y/p, iterate, wall constraint,
  energy tracking, batched on CUDA).
- :class:`UpdateStrategy` subclasses — pluggable position/momentum updates.
- Enhancement **mixins** — orthogonal features that wrap or augment the loop:
  GSB (individual p_i), GGSB (global guidance), Quantisation.
- :class:`Solver` — high-level convenience class that composes a strategy,
  optional enhancements, and provides the standard ``.solve(Q, num_vars)`` API.

Usage::

    from src.solver import Solver, BSBStrategy, GSBMixin

    solver = Solver(
        strategy=BSBStrategy(dt=0.1),
        enhancements=[GSBMixin(A=0.5)],
        num_iters=500,
        num_trials=10,
    )
    solution = solver.solve(Q, num_vars)
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional

import torch


# ═══════════════════════════════════════════════════════════════════════════
# 1. Update Strategies
# ═══════════════════════════════════════════════════════════════════════════


class UpdateStrategy:
    """Base class for a position/momentum update rule.

    Subclasses override ``__call__`` which receives the current state and
    returns updated ``(x, y, p)``.
    """

    def __call__(
        self,
        x: torch.Tensor,       # (batch, N)
        y: torch.Tensor,       # (batch, N)
        p: torch.Tensor,       # (batch, N) or scalar
        J: torch.Tensor,       # (N, N)
        device: torch.device,
        step: int,
        M: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class BSBStrategy(UpdateStrategy):
    """Ballistic SB (standard bSB).  Global linear ``p(t) = step / M``."""

    def __init__(self, dt: float = 0.1):
        self.dt = dt

    def __call__(self, x, y, p, J, device, step, M):
        alpha = step / M
        Jx = x @ J.T
        y = y + ((-1 + alpha) * x + p * Jx) * self.dt
        x = x + y * self.dt
        return x, y, p


class AdiabaticStrategy(UpdateStrategy):
    """Adiabatic SB — ``p(t)`` follows a schedule; ``c`` is the coupling scale."""

    def __init__(self, dt: float = 0.1, c_schedule: Callable = None):
        self.dt = dt
        self.c_schedule = c_schedule or (lambda step, M: 4.0 / math.sqrt(M))

    def __call__(self, x, y, p, J, device, step, M):
        alpha = step / M
        c = self.c_schedule(step, M)
        Jx = x @ J.T
        y = y + ((-1 + alpha) * x + c * Jx) * self.dt
        x = x + y * self.dt
        return x, y, p


class DSBStrategy(UpdateStrategy):
    """Discrete SB — uses ``sign(x)`` for the coupling term."""

    def __init__(self, dt: float = 0.1):
        self.dt = dt

    def __call__(self, x, y, p, J, device, step, M):
        alpha = step / M
        Jx = torch.sign(x) @ J.T
        y = y + ((-1 + alpha) * x + p * Jx) * self.dt
        x = x + y * self.dt
        return x, y, p


class DigCIMStrategy(UpdateStrategy):
    """Digital Chaotic Ising Machine — uses individual ``p_i``."""

    def __init__(self, dt: float = 0.1, xi: Optional[float] = None):
        self.dt = dt
        self.xi = xi

    def __call__(self, x, y, p, J, device, step, M):
        N = J.shape[0]
        xi = self.xi if self.xi is not None else 0.7 / math.sqrt(N)
        Jx = x @ J.T
        y = y + (-p * x + xi * Jx) * self.dt
        x = x + y * self.dt
        return x, y, p


# ═══════════════════════════════════════════════════════════════════════════
# 2. Enhancement Mixins
# ═══════════════════════════════════════════════════════════════════════════


class GSBMixin:
    """Generalised SB — individual ``p_i`` with nonlinear control.

    Replaces the global bifurcation parameter with per-oscillator ``p_i``
    updated via::

        mod = 1.0 - A * x**2
        p  -= mod * p / (M - step)
    """

    def __init__(self, A: float = 1.0):
        self.A = A

    def enhance_p(self, x: torch.Tensor, p: torch.Tensor, step: int, M: int) -> torch.Tensor:
        if self.A == 0.0:
            return p
        mod = 1.0 - self.A * (x ** 2)
        p = p - mod * p / max(M - step, 1)
        return p


class GGSBMixin:
    """Global Guidance SB — shares information across batch replicas.

    Every ``k`` steps, the average spin orientation across the batch is
    computed and weakly injected into each replica's momentum.
    """

    def __init__(self, k: int = 10, strength: float = 0.05):
        self.k = k
        self.strength = strength

    def enhance_momentum(self, x: torch.Tensor, y: torch.Tensor, step: int) -> torch.Tensor:
        if step % self.k != 0:
            return y
        avg_spin = torch.sign(x).mean(dim=0, keepdim=True)  # (1, N)
        y = y + self.strength * avg_spin
        return y


class QuantizationMixin:
    """Simulates fixed-point / low-precision arithmetic.

    Quantises ``x`` and ``y`` to ``num_bits`` at each step.
    """

    def __init__(self, num_bits: int = 8, scale: float = 1.0):
        self.num_bits = num_bits
        self.scale = scale

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        max_val = self.scale
        step = 2.0 * max_val / (2 ** self.num_bits - 1)
        x_q = torch.round(x / step) * step
        return torch.clamp(x_q, -max_val, max_val)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Base Solver
# ═══════════════════════════════════════════════════════════════════════════


class BaseSolver:
    """Core SB loop with pluggable strategy and enhancement mixins.

    Parameters
    ----------
    strategy : UpdateStrategy
        The position/momentum update rule.
    enhancements : list of enhancement mixins, optional
        Mixins applied during the loop (e.g. ``GSBMixin``, ``GGSBMixin``).
    num_iters : int
        Number of integration steps (M).
    num_trials : int
        Number of parallel trials (batch).
    device : str
        Torch device.
    use_compile : bool
        If True, compile the step function with ``torch.compile``.
    """

    def __init__(
        self,
        strategy: UpdateStrategy,
        enhancements: Optional[List] = None,
        num_iters: int = 500,
        num_trials: int = 10,
        device: str = "cpu",
        use_compile: bool = False,
    ):
        self.strategy = strategy
        self.enhancements = enhancements or []
        self.num_iters = num_iters
        self.num_trials = num_trials
        self.device = torch.device(device)
        self.use_compile = use_compile

    def solve(self, J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the solver.

        Parameters
        ----------
        J : Tensor (N, N)
            Ising coupling matrix.

        Returns
        -------
        solutions : Tensor (num_trials, N)
            Final spin configurations (values -1 or +1).
        energies : Tensor (num_trials,)
            Final Ising energy for each trial.
        """
        N = J.shape[0]
        batch = self.num_trials
        device = self.device

        J = J.to(device)
        x = 2 * torch.rand(batch, N, device=device) - 1
        y = torch.zeros(batch, N, device=device)
        p = torch.ones(batch, N, device=device)

        # Separate mixins by type
        gsb_mixins = [m for m in self.enhancements if isinstance(m, GSBMixin)]
        ggsb_mixins = [m for m in self.enhancements if isinstance(m, GGSBMixin)]
        quant_mixins = [m for m in self.enhancements if isinstance(m, QuantizationMixin)]

        M = self.num_iters

        def _step(x, y, p, step):
            # 1. GSB: update p_i
            for gsb in gsb_mixins:
                p = gsb.enhance_p(x, p, step, M)

            # 2. Momentum / position update (strategy)
            x, y, p = self.strategy(x, y, p, J, device, step, M)

            # 3. Wall constraint (perfectly inelastic)
            boundary_mask = torch.abs(x) > 1
            y = torch.where(boundary_mask, torch.zeros_like(y), y)
            x = torch.clamp(x, -1.0, 1.0)

            # 4. GGSB: global guidance
            for ggsb in ggsb_mixins:
                y = ggsb.enhance_momentum(x, y, step)

            # 5. Quantization
            for qm in quant_mixins:
                x = qm.quantize(x)
                y = qm.quantize(y)

            return x, y, p

        step_fn = torch.compile(_step, dynamic=True) if self.use_compile else _step

        for step in range(M):
            x, y, p = step_fn(x, y, p, step)

        solutions = torch.sign(x)
        Js = solutions @ J.T
        energies = -0.25 * torch.sum(J) - 0.5 * (-0.5 * torch.sum(solutions * Js, dim=1))

        return solutions, energies


# ═══════════════════════════════════════════════════════════════════════════
# 4. High-Level Solver (QUBO API)
# ═══════════════════════════════════════════════════════════════════════════


class Solver:
    """High-level QUBO solver with strategy + enhancements.

    Provides the standard ``.solve(Q, num_vars) -> list[int]`` interface
    used by the TPU benchmark pipeline.

    Parameters
    ----------
    strategy : UpdateStrategy or str
        Strategy instance, or one of ``"bsb"``, ``"dsb"``, ``"adiabatic"``,
        ``"digcim"``.
    enhancements : list, optional
        Enhancement mixin instances.
    num_iters : int
        Number of integration steps.
    num_trials : int
        Number of parallel trials.
    device : str
        Torch device.
    use_compile : bool
        If True, compile the step function.
    dt : float
        Time step (used when *strategy* is a string).
    """

    STRATEGY_MAP = {
        "bsb": BSBStrategy,
        "dsb": DSBStrategy,
        "adiabatic": AdiabaticStrategy,
        "digcim": DigCIMStrategy,
    }

    def __init__(
        self,
        strategy: UpdateStrategy | str = "bsb",
        enhancements: Optional[List] = None,
        num_iters: int = 500,
        num_trials: int = 10,
        device: str = "cpu",
        use_compile: bool = False,
        dt: float = 0.1,
    ):
        if isinstance(strategy, str):
            cls = self.STRATEGY_MAP.get(strategy.lower())
            if cls is None:
                raise ValueError(f"Unknown strategy: {strategy}. "
                                 f"Choose from {list(self.STRATEGY_MAP.keys())}")
            strategy = cls(dt=dt)
        self._base = BaseSolver(
            strategy=strategy,
            enhancements=enhancements,
            num_iters=num_iters,
            num_trials=num_trials,
            device=device,
            use_compile=use_compile,
        )

    def solve(self, Q, num_vars) -> list[int]:
        """Solve a QUBO problem.

        Parameters
        ----------
        Q : list of (int, int, float)
            Sparse upper-triangular QUBO matrix.
        num_vars : int
            Number of binary variables.

        Returns
        -------
        list of int
            Binary solution vector (values 0 or 1).
        """
        J = torch.zeros(num_vars, num_vars, device=self._base.device)
        for i, j, val in Q:
            J[i, j] = val
            if i != j:
                J[j, i] = val
        J_ising = -J / 2.0

        solutions, energies = self._base.solve(J_ising)

        best_idx = int(torch.argmin(energies).item())
        best_spins = solutions[best_idx].cpu().numpy()

        return ((best_spins + 1) / 2).astype(int).tolist()
