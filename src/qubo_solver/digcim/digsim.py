"""DIGCIM — Digital Chaotic Ising Machine simulator."""

import numpy as np


def digsim(J, num_iters=1000, dt=0.1):
    """Digital Chaotic Ising Machine simulator.

    Parameters
    ----------
    J : ndarray (N, N)
        Ising coupling matrix.
    num_iters : int
        Number of iterations.
    dt : float
        Time step.

    Returns
    -------
    solution : ndarray (N,)
        Spin configuration (±1).
    energy : float
        Ising energy.
    """
    N = J.shape[0]
    x = np.random.randn(N) * 0.1
    y = np.zeros(N)

    for i in range(num_iters):
        alpha = i / num_iters
        y += ((-1 + alpha) * x + 0.7 / np.sqrt(N) * (J @ x)) * dt
        x += y * dt
        x = np.clip(x, -1, 1)

    solution = np.sign(x)
    energy = -0.5 * solution @ J @ solution
    return solution, energy
