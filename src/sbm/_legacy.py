"""Simulated Bifurcation (SB) solvers — bSB, dSB, and batched variants."""

import torch
import math
import numpy as np


def sb(sb_type, J, init_x, init_y, num_iters, dt):
    N = J.shape[0]
    x_comp = init_x.copy()
    y_comp = init_y.copy()
    xi = 0.7 / math.sqrt(N)
    sol = np.sign(x_comp)
    energies = []
    e = -1 / 2 * sol.T @ J @ sol
    alpha = np.linspace(0, 1, num_iters)
    for i in range(num_iters):
        if sb_type == "bsb":
            y_comp += ((-1 + alpha[i]) * x_comp + xi * (J @ x_comp)) * dt
            x_comp += y_comp * dt
            y_comp[np.abs(x_comp) > 1] = 0.
            x_comp = np.clip(x_comp, -1, 1)
        elif sb_type == "dsb":
            y_comp += ((-1 + alpha[i]) * x_comp + xi * (J @ x_comp.sign())) * dt
            x_comp += y_comp * dt
            y_comp[np.abs(x_comp) > 1] = 0.
            x_comp = np.clip(x_comp, -1, 1)
        elif sb_type == "sb":
            y_comp += xi * (J @ x_comp) * dt

        sol = np.sign(x_comp)
        e = -1 / 2 * sol.T @ J @ sol
        energy = -1 / 4 * J.sum() - 1 / 2 * e
        energies.append(energy)

    return energies


def bsb_torch(J, init_x, init_y, num_iters, dt):
    x_comp = init_x.clone()
    y_comp = init_y.clone()
    N = J.shape[0]
    xi = 0.035 / (math.sqrt(N) * (torch.std(J) + 1e-8))
    energies = []
    alpha = torch.linspace(0, 1, num_iters)

    for i in range(num_iters):
        y_comp += ((-1 + alpha[i]) * x_comp + xi * torch.matmul(J, x_comp)) * dt
        x_comp += y_comp * dt
        boundary_mask = torch.abs(x_comp) > 1
        y_comp[boundary_mask] = 0.
        x_comp = torch.clamp(x_comp, -1, 1)

        if i == num_iters - 1:
            sol = torch.sign(x_comp)
            e = -0.5 * torch.matmul(sol, torch.matmul(J, sol))
            energy = -0.25 * torch.sum(J) - 0.5 * e
            energies.append(energy.item())

    return energies, sol


def _bsb_step(x_comp, y_comp, J, alpha_i, xi, dt):
    """Single SB iteration step."""
    Jx = torch.matmul(x_comp, J.T)
    y_comp = y_comp + ((-1 + alpha_i) * x_comp + xi * Jx) * dt
    x_comp = x_comp + y_comp * dt
    boundary_mask = torch.abs(x_comp) > 1
    y_comp = torch.where(boundary_mask, torch.zeros_like(y_comp), y_comp)
    x_comp = torch.clamp(x_comp, -1, 1)
    return x_comp, y_comp


def bsb_torch_batch(J, init_x, init_y, num_iters, dt, best_known=None, max_iters=5000, use_compile=False):
    N = J.shape[0]
    batch_size = init_x.shape[0]
    x_comp = init_x.clone()
    y_comp = init_y.clone()
    xi = 4.0 / (torch.sqrt(torch.tensor(N, device=J.device, dtype=torch.float32)))
    use_convergence_mode = best_known is not None

    if use_convergence_mode:
        total_iterations = max_iters
        target_energy = best_known * 0.99
        steps_to_converge = torch.full((batch_size,), max_iters, dtype=torch.int32, device=J.device)
    else:
        total_iterations = num_iters

    alpha = torch.linspace(0, 1, total_iterations, device=J.device)
    energies = torch.zeros(batch_size, total_iterations, device=J.device)
    es = torch.zeros(batch_size, total_iterations, device=J.device)

    step_fn = torch.compile(_bsb_step, dynamic=True) if use_compile else _bsb_step

    for i in range(total_iterations):
        x_comp, y_comp = step_fn(x_comp, y_comp, J, alpha[i], xi, dt)

        if use_convergence_mode:
            sol = torch.sign(x_comp)
            J_sol = torch.matmul(sol, J.T)
            e = -0.5 * torch.sum(sol * J_sol, dim=1)
            energy = -0.25 * torch.sum(J) - 0.5 * e
            reached_target = (energy >= target_energy)
            if reached_target.any():
                return i + 1, False

        elif i == total_iterations - 1:
            sol = torch.sign(x_comp)
            J_sol = torch.matmul(sol, J.T)
            e = -0.5 * torch.sum(sol * J_sol, dim=1)
            es[:, i] = e
            energy = -0.25 * torch.sum(J) - 0.5 * e
            energies[:, i] = energy

    final_solutions = torch.sign(x_comp)

    if use_convergence_mode:
        return max_iters, True
    else:
        return energies, final_solutions, es


def bsb_bmincut_batch(J, init_x, init_y, num_iters, dt, lambda_balance=1.0, use_compile=False):
    N = J.shape[0]
    batch_size = init_x.shape[0]

    x_comp = init_x.clone()
    y_comp = init_y.clone()
    xi = 0.7 / torch.sqrt(torch.tensor(N, device=J.device, dtype=torch.float32))

    alpha = torch.linspace(0, 1, num_iters, device=J.device)
    energies = torch.zeros(batch_size, num_iters, device=J.device)
    ones = torch.ones(N, device=J.device)
    J_balanced = -0.5 * J - 2.0 * lambda_balance * torch.outer(ones, ones)

    step_fn = torch.compile(_bsb_step, dynamic=True) if use_compile else _bsb_step

    for i in range(num_iters):
        x_comp, y_comp = step_fn(x_comp, y_comp, J_balanced, alpha[i], xi, dt)

    orig_J = -J
    sol = torch.sign(x_comp)
    xJx = torch.einsum('bi,ij,bj->b', sol, orig_J, sol)
    cut_value = 0.25 * (torch.sum(orig_J) - xJx)
    sum_x = torch.sum(sol, dim=1)
    balance_term = lambda_balance * sum_x**2
    energy = cut_value + balance_term
    energies[:, i] = energy

    return energies, sol, cut_value, sum_x


def qsb(J, init_x, init_y, num_iters, dbg_iter, best_known=0, factor=[6, 4, 4], qtz_type='scaleup', dbg_option='OFF'):
    energies = []
    scl1 = 2 ** factor[0] - 1
    scl2 = 2 ** 7 - 1

    N = J.shape[0]
    xi = 0.75 / (math.sqrt(N) * torch.std(J) + 1e-8)
    JX_dbg = []
    alpha = np.linspace(0, 1, num_iters)
    step = num_iters
    acc_reach = 0
    x_comp = scale_up(np.array(init_x.copy()), scl1)
    y_comp = scale_up(np.array(init_y.copy()), scl1)

    x_comp_init = x_comp.copy()
    y_comp_init = y_comp.copy()

    x_comp_dbg = []
    y_comp_dbg = []

    for i in range(num_iters):
        if i == dbg_iter:
            pass
    return energies


def scale_up(x, scl):
    return x * scl


def scale_down(x, scl):
    return x / scl
