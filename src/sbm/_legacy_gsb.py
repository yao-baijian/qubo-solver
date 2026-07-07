"""Generalized Simulated Bifurcation (GSB) — Goto et al. 2025.

Implements individual bifurcation parameters ``p_i`` per oscillator with
nonlinear control, rather than a single global ``p(t)`` as in standard SB.

When ``A=0`` the update reduces to the global linear schedule, making GSB
identical to bSB.
"""

import torch


def _gsb_step(x_comp, y_comp, p, J, c, dt, A, M, step):
    """Single GSB iteration step.

    Parameters
    ----------
    x_comp : Tensor (batch, N) — position
    y_comp : Tensor (batch, N) — momentum
    p : Tensor (batch, N) — individual bifurcation parameters
    J : Tensor (N, N) — Ising coupling matrix
    c : float — coupling scaling (typically 1 / lambda_max)
    dt : float — time step
    A : float — nonlinear control strength
    M : int — total number of steps
    step : int — current iteration index (0-based)

    Returns
    -------
    x_comp, y_comp, p : updated tensors
    """
    mod = 1.0 - A * (x_comp ** 2)
    p = p - mod * p / (M - step)

    Jx = torch.matmul(x_comp, J.T)
    y_comp = y_comp + (-p * x_comp + c * Jx) * dt

    x_comp = x_comp + y_comp * dt

    boundary_mask = torch.abs(x_comp) > 1
    y_comp = torch.where(boundary_mask, torch.zeros_like(y_comp), y_comp)
    x_comp = torch.clamp(x_comp, -1, 1)

    return x_comp, y_comp, p


def gsb_batch(J, init_x, init_y, num_iters, dt, A, c=None, use_compile=False):
    """Generalized Simulated Bifurcation with batched trials.

    Parameters
    ----------
    J : Tensor (N, N)
        Ising coupling matrix.
    init_x : Tensor (batch, N)
        Initial positions in [-1, 1].
    init_y : Tensor (batch, N)
        Initial momenta.
    num_iters : int
        Number of integration steps (M).
    dt : float
        Time step.
    A : float
        Nonlinear control strength.  When ``A=0`` the algorithm reduces to
        standard bSB.
    c : float or None
        Coupling scaling (typically ``1 / lambda_max``).  If ``None``,
        defaults to ``4.0 / sqrt(N)`` (matching bSB's xi heuristic).
    use_compile : bool
        If True, wrap the step function with ``torch.compile``.

    Returns
    -------
    energies : Tensor (batch, num_iters)
        Energy trace for each trial.
    solutions : Tensor (batch, N)
        Final spin configuration (-1 or +1) for each trial.
    """
    N = J.shape[0]
    batch_size = init_x.shape[0]
    device = J.device

    x_comp = init_x.clone()
    y_comp = init_y.clone()
    p = torch.ones(batch_size, N, device=device)

    if c is None:
        c = 4.0 / (torch.sqrt(torch.tensor(N, device=device, dtype=torch.float32)))

    energies = torch.zeros(batch_size, num_iters, device=device)
    step_fn = torch.compile(_gsb_step, dynamic=True) if use_compile else _gsb_step

    M = num_iters
    for step in range(M):
        x_comp, y_comp, p = step_fn(x_comp, y_comp, p, J, c, dt, A, M, step)

        if step == M - 1:
            sol = torch.sign(x_comp)
            J_sol = torch.matmul(sol, J.T)
            e = -0.5 * torch.sum(sol * J_sol, dim=1)
            energy = -0.25 * torch.sum(J) - 0.5 * e
            energies[:, step] = energy

    final_solutions = torch.sign(x_comp)
    return energies, final_solutions
