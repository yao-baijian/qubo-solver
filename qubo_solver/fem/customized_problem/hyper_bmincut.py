"""Hyper-graph balanced minimum cut problem for FEM solver."""
import torch


class HyperBmincut:
    """Hyper-graph balanced minimum cut problem definition."""

    def __init__(self, coupling_matrix, imbalance_weight=5.0):
        self.J = coupling_matrix
        self.imbalance_weight = imbalance_weight

    def expected(self, p):
        """Compute expected energy under variational distribution p."""
        p1 = p[..., 1]
        cut = torch.bmm(
            (p1 @ self.J).reshape(-1, 1, self.J.shape[0]),
            p1.reshape(-1, self.J.shape[0], 1),
        ).reshape(-1)
        balance = p1.sum(dim=1) ** 2
        return cut + self.imbalance_weight * balance

    def grad(self, p):
        """Gradient of expected energy w.r.t. variational parameters."""
        p1 = p[..., 1]
        grad_cut = 2.0 * (p1 @ self.J) * p[..., 0]
        grad_balance = 2.0 * self.imbalance_weight * p1.sum(dim=1, keepdim=True) * p[..., 0]
        return (grad_cut + grad_balance).reshape(-1, 2)

    def infer(self, p):
        """Inference — threshold at 0.5."""
        return (p[..., 1] > 0.5).float()
