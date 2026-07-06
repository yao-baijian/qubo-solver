import torch
import numpy as np
from typing import Union, Callable, Optional

class OptimizationProblem:
    def __init__(self, num_nodes, num_interactions, coupling_matrix,
                 problem_type, imbalance_weight, hyperedges, node_weights,
                 discretization, customize_expected_func, customize_grad_func,
                 customize_infer_func):
        self.num_nodes = num_nodes
        self.num_interactions = num_interactions
        self.coupling_matrix = coupling_matrix
        self.problem_type = problem_type
        self.imbalance_weight = imbalance_weight
        self.hyperedges = hyperedges
        self.node_weights = node_weights
        self.discretization = discretization
        self.customize_expected_func = customize_expected_func
        self.customize_grad_func = customize_grad_func
        self.customize_infer_func = customize_infer_func

    def generate_expected(self, p):
        if self.problem_type == 'customize':
            return self.customize_expected_func(self.coupling_matrix, p)
        return expected_qubo(self.coupling_matrix, p)

    def generate_grad(self, p):
        if self.problem_type == 'customize':
            return self.customize_grad_func(self.coupling_matrix, p)
        return manual_grad_qubo(self.coupling_matrix, p)

    def generate_infer(self, p):
        if self.problem_type == 'customize':
            return self.customize_infer_func(self.coupling_matrix, p)
        return infer_qubo(self.coupling_matrix, p)


def expected_qubo(J, p):
    """Default expectation for QUBO."""
    p1 = p[..., 1]
    return torch.bmm(
        (p1 @ J).reshape(-1, 1, J.shape[0]),
        p1.reshape(-1, J.shape[0], 1),
    ).reshape(-1)


def manual_grad_qubo(J, p):
    """Default gradient for QUBO."""
    p1 = p[..., 1]
    grad = 2.0 * (p1 @ J) * p[..., 0]
    return grad.reshape(-1, 2)


def infer_qubo(J, p):
    """Default inference — threshold at 0.5."""
    p1 = p[..., 1]
    return (p1 > 0.5).float()
