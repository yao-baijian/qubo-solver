"""DIGCIM — Digital Chaotic Ising Machine (backward-compat wrapper).

Wraps the ``DigCIMStrategy`` from the unified SBM solver.
"""
import numpy as np
import torch
from sbm import BaseSolver, DigCIMStrategy


def digsim(J, num_iters=1000, dt=0.1):
    """Digital Chaotic Ising Machine.

    Parameters
    ----------
    J : ndarray (N, N) — Ising coupling matrix.
    num_iters : int
    dt : float

    Returns
    -------
    solution : ndarray (N,) — spin configuration (±1).
    energy : float — Ising energy.
    """
    J_t = torch.as_tensor(J, dtype=torch.float32)
    base = BaseSolver(
        strategy=DigCIMStrategy(dt=dt),
        num_iters=num_iters, num_trials=1,
    )
    sols, _ = base.solve(-J_t / 2.0)
    spins = sols[0].cpu().numpy()
    energy = -0.5 * spins @ np.array(J) @ spins
    return spins, energy

