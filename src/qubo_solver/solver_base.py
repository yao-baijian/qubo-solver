"""Solver base classes for the qubo-solver package.

Each solver type becomes a class that loads its own config once on
initialisation and exposes phase methods (coarsen, initial_partition,
refine).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch


class SolverBase:
    type: str = "base"

    def __init__(self, config_dir: Optional[Path] = None):
        self._config: Dict[str, Any] = {}
        self._config_dir = config_dir or Path.cwd() / "config"

    def get_param(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set_param(self, key: str, value: Any) -> None:
        self._config[key] = value

    def update_params(self, **kwargs) -> None:
        self._config.update(kwargs)

    def initial_partition(self, J, q) -> np.ndarray:
        raise NotImplementedError

    def refine(self, J, q, partition: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def solve_direct(self, J, q) -> np.ndarray:
        return self.refine(J, q, self.initial_partition(J, q))


class FemSolver(SolverBase):
    type = "fem"

    def __init__(self, config_dir: Optional[Path] = None):
        super().__init__(config_dir)
        self._config = {"num_trials": 5, "num_steps": 500, "learning_rate": 0.1}

    def initial_partition(self, J, q) -> np.ndarray:
        N = J.shape[0]
        return np.random.choice(q, size=N)

    def refine(self, J, q, partition: np.ndarray) -> np.ndarray:
        return partition

    def solve_direct(self, J, q) -> np.ndarray:
        N = J.shape[0]
        p = torch.softmax(torch.randn(1, N, q) * 0.01, dim=-1)
        for _ in range(self._config.get("num_steps", 500)):
            p1 = p[..., 1]
            expected = torch.bmm((p1 @ J).reshape(-1, 1, N), p1.reshape(-1, N, 1)).reshape(-1)
            grad = 2.0 * (p1 @ J) * p[..., 0]
            p = p - self._config.get("learning_rate", 0.1) * grad
            p = torch.softmax(p, dim=-1)
        return torch.argmax(p[0], dim=-1).cpu().numpy()


class SbmSolver(SolverBase):
    type = "sbm"

    def __init__(self, config_dir: Optional[Path] = None):
        super().__init__(config_dir)
        self._config = {"num_iters": 500, "dt": 0.1}

    def initial_partition(self, J, q) -> np.ndarray:
        N = J.shape[0]
        return np.random.choice(q, size=N)

    def refine(self, J, q, partition: np.ndarray) -> np.ndarray:
        return partition
