"""DISTCIM — DistIM: distributed Simulated Coherent Ising Machine.

Implements the sparse-synchronization distributed Ising dynamics from the
paper *"Distributed Ising dynamics for real-time large-scale combinatorial
optimization"* (DistIM) on top of a faithful SimCIM engine.

Public API
----------
- ``SimCimSolver``  : centralized SimCIM QUBO solver (``solve(Q, num_vars)``).
- ``DistCimSolver`` : distributed DistIM QUBO solver
                      (``solve(Q, num_vars)`` with ``nparts``/``scheme``/...).
- ``solve_ising``   : low-level Ising interface returning ``(spins, energy)``,
                      used by the verification harness.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .distributed import DistIMEngine
from .engines import (
    CentralFieldCoupler,
    SimCIMEngine,
    ising_energy,
)
from .precision import PRECISIONS, PrecisionMatmul


def qubo_to_ising(Q: torch.Tensor):
    """Map a QUBO ``min s^T Q s`` to the Ising form
    ``min -1/2 sigma^T J sigma - h^T sigma`` (paper Methods 3.1).

    ``sigma = 2s - 1``  =>  ``J = -Q/2``, ``h = -(Q @ 1)/2``.
    """
    Q = Q.float()
    J = -Q / 2.0
    h = -(Q.sum(dim=1, keepdim=True)) / 2.0
    return J, h


def _build_matrix(Q, num_vars):
    """Convert the sparse ``(i, j, value)`` list into a dense QUBO matrix."""
    Q_mat = torch.zeros(num_vars, num_vars, dtype=torch.float32)
    for i, j, val in Q:
        Q_mat[i, j] = val
        if i != j:
            Q_mat[j, i] = val
    return Q_mat


def solve_ising(
    J: torch.Tensor,
    h: Optional[torch.Tensor] = None,
    nparts: int = 1,
    scheme: str = "standard",
    time_intvl: int = 10,
    model: str = "SimplifiedSimCIM",
    xi: float = 1.0,
    A_init: float = 1.0e-3,
    As: float = 70.0,
    dt: float = 0.1,
    pump: str = "constant",
    pmax: float = 1.1,
    num_iters: int = 1000,
    seed: int = 0,
    noise_scale: float = 1.0,
    device: str = "cpu",
    quantize_bits: Optional[int] = None,
    x_bits: Optional[int] = None,
    y_bits: Optional[int] = None,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    precision: Optional[str] = None,
    backend: str = "emulated",
) -> Tuple[np.ndarray, float]:
    """Solve an Ising problem with (optionally distributed) SimCIM.

    Parameters
    ----------
    J : (N, N) torch.Tensor
        Ising coupling matrix (symmetric).
    h : (N, 1) torch.Tensor or None
        External field (default zeros).
    nparts, scheme, time_intvl : see :class:`DistIMEngine`.
        ``nparts=1`` runs the centralized machine (exactly the target repo's
        ``SimplifiedSimCIM``); ``nparts>1`` uses the DistIM partition.
    quantize_bits : int or None
        Fixed-point width of the exchanged inter-module message.
    x_bits, y_bits : int or None
        FPGA state quantization at every step: ``x`` (c-component) to
        ``x_bits`` (hardware: 8) and ``y`` (s-component) to ``y_bits``
        (hardware: 16 or 32); control params stay float. ``None`` disables.
    x_scale, y_scale : float
        Full-scale range of the fixed-point grid for ``x`` / ``y``.
    precision : str or None
        Arithmetic precision of the coupling matmul ``J @ c`` (see
        :mod:`src.distcim.precision`): ``fp32`` (default), ``fp16``, ``bf16``,
        ``int8``, ``int4``, ``fp8``, ``fp4``.

    Returns
    -------
    (spins, energy) : (N,) binary +/-1 spins and the Ising energy.
    """
    if h is None:
        h = torch.zeros(J.size(0), 1, dtype=J.dtype)
    engine = DistIMEngine(
        J=J,
        h=h,
        nparts=nparts,
        scheme=scheme,
        time_intvl=time_intvl,
        model=model,
        xi=xi,
        A_init=A_init,
        As=As,
        dt=dt,
        pump=pump,
        pmax=pmax,
        num_iters=num_iters,
        seed=seed,
        noise_scale=noise_scale,
        device=device,
        quantize_bits=quantize_bits,
        x_bits=x_bits,
        y_bits=y_bits,
        x_scale=x_scale,
        y_scale=y_scale,
        precision=precision,
        backend=backend,
    )
    spins, energy = engine.run()
    return spins.cpu().numpy(), energy


class SimCimSolver:
    """Centralized SimCIM QUBO solver (DistIM with ``nparts=1``).

    Usage (same interface as the other solver families)::

        from src.distcim import SimCimSolver
        solver = SimCimSolver(num_iters=1000, dt=0.1)
        solution = solver.solve(Q, num_vars)   # list of 0/1
    """

    def __init__(
        self,
        num_iters: int = 1000,
        dt: float = 0.1,
        As: float = 70.0,
        A_init: float = 1.0e-3,
        xi: float = 1.0,
        model: str = "SimplifiedSimCIM",
        pump: str = "constant",
        pmax: float = 1.1,
        seed: int = 0,
        device: str = "cpu",
        x_bits: Optional[int] = None,
        y_bits: Optional[int] = None,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        precision: Optional[str] = None,
    ):
        self.num_iters = num_iters
        self.dt = dt
        self.As = As
        self.A_init = A_init
        self.xi = xi
        self.model = model
        self.pump = pump
        self.pmax = pmax
        self.seed = seed
        self.device = device
        self.x_bits = x_bits
        self.y_bits = y_bits
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.precision = precision

    def solve(self, Q, num_vars) -> List[int]:
        Q_mat = _build_matrix(Q, num_vars)
        J, h = qubo_to_ising(Q_mat)
        spins, _ = solve_ising(
            J, h, nparts=1, scheme="standard", model=self.model,
            xi=self.xi, A_init=self.A_init, As=self.As, dt=self.dt,
            pump=self.pump, pmax=self.pmax, num_iters=self.num_iters,
            seed=self.seed, device=self.device,
            x_bits=self.x_bits, y_bits=self.y_bits,
            x_scale=self.x_scale, y_scale=self.y_scale,
            precision=self.precision,
        )
        return (spins > 0).astype(int).tolist()


class DistCimSolver:
    """Distributed DistIM QUBO solver (multi-module sparse synchronization).

    Additional options: ``nparts``, ``scheme`` (standard/const/pulse),
    ``time_intvl`` (sync period K), ``quantize_bits`` (message precision) and
    ``x_bits``/``y_bits`` (FPGA state quantization of x/y at every step).
    """

    def __init__(
        self,
        num_iters: int = 1000,
        dt: float = 0.1,
        As: float = 70.0,
        A_init: float = 1.0e-3,
        xi: float = 1.0,
        model: str = "SimplifiedSimCIM",
        pump: str = "constant",
        pmax: float = 1.1,
        nparts: int = 2,
        scheme: str = "const",
        time_intvl: int = 10,
        quantize_bits: Optional[int] = None,
        x_bits: Optional[int] = None,
        y_bits: Optional[int] = None,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        precision: Optional[str] = None,
        seed: int = 0,
        device: str = "cpu",
        backend: str = "emulated",
    ):
        self.num_iters = num_iters
        self.dt = dt
        self.As = As
        self.A_init = A_init
        self.xi = xi
        self.model = model
        self.pump = pump
        self.pmax = pmax
        self.nparts = nparts
        self.scheme = scheme
        self.time_intvl = time_intvl
        self.quantize_bits = quantize_bits
        self.x_bits = x_bits
        self.y_bits = y_bits
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.precision = precision
        self.seed = seed
        self.device = device
        self.backend = backend

    def solve(self, Q, num_vars) -> List[int]:
        Q_mat = _build_matrix(Q, num_vars)
        J, h = qubo_to_ising(Q_mat)
        spins, _ = solve_ising(
            J, h, nparts=self.nparts, scheme=self.scheme,
            time_intvl=self.time_intvl, model=self.model, xi=self.xi,
            A_init=self.A_init, As=self.As, dt=self.dt, pump=self.pump,
            pmax=self.pmax, num_iters=self.num_iters, seed=self.seed,
            device=self.device, quantize_bits=self.quantize_bits,
            x_bits=self.x_bits, y_bits=self.y_bits,
            x_scale=self.x_scale, y_scale=self.y_scale,
            precision=self.precision,
            backend=self.backend,
        )
        return (spins > 0).astype(int).tolist()


__all__ = [
    "SimCimSolver",
    "DistCimSolver",
    "solve_ising",
    "DistIMEngine",
    "SimCIMEngine",
    "CentralFieldCoupler",
    "qubo_to_ising",
    "ising_energy",
    "PRECISIONS",
    "PrecisionMatmul",
]
