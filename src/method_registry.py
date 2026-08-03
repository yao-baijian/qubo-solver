"""Method registry — central registry for pipeline methods.

Each method is a combination of a solver and a set of default parameters.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class MethodName:
    """Two-level method name: family (pipeline) + algorithm (solver)."""
    def __init__(self, family: str, algorithm: str):
        self.family = family
        self.algorithm = algorithm

    def __str__(self):
        return f'{self.family}: {self.algorithm}'


class PartitionMethod:
    """Descriptor for a partition pipeline method."""

    def __init__(self, name: str, method_name: MethodName,
                 description: str = "", defaults: Optional[dict] = None):
        self.name = name
        self.method_name = method_name
        self.description = description
        self.defaults = defaults or {}
        self._run_fn: Optional[Callable] = None

    def bind(self, run_fn: Callable) -> None:
        self._run_fn = run_fn

    def run(self, J, q, **overrides) -> Any:
        if self._run_fn is None:
            raise RuntimeError(f"Method '{self.name}' has no run function bound.")
        params = {**self.defaults, **overrides}
        return self._run_fn(J, q, **params)


class _Registry:
    def __init__(self):
        self._methods: Dict[str, PartitionMethod] = {}

    def register(self, method: PartitionMethod) -> None:
        self._methods[method.name] = method

    def get(self, name: str) -> PartitionMethod:
        if name not in self._methods:
            raise KeyError(f"Unknown method: {name}")
        return self._methods[name]

    def list_methods(self) -> list:
        return list(self._methods.keys())


registry = _Registry()


def register_distcim_methods() -> None:
    """Register the DISTCIM (DistIM) methods into the global registry.

    Methods
    -------
    - ``distcim``            : centralized SimCIM (single module)
    - ``distcim-const``      : DistIM with constant compensation (sparse sync)
    - ``distcim-pulse``      : DistIM with pulse compensation (sparse sync)

    Each method runs on an Ising problem ``(J, q)`` where ``q`` is the external
    field ``h``; it returns a ``(spins, energy)`` tuple.
    """
    from src.distcim import solve_ising

    def _make_run(nparts, scheme):
        def _run(J, q=None, **params):
            if q is None:
                import torch
                q = torch.zeros(J.size(0), 1, dtype=torch.float32)
            return solve_ising(
                J, q, nparts=nparts, scheme=scheme,
                num_iters=params.pop("num_iters", 1000),
                dt=params.pop("dt", 0.1),
                model=params.pop("model", "SimplifiedSimCIM"),
                xi=params.pop("xi", 1.0),
                A_init=params.pop("A_init", 1.0e-3),
                As=params.pop("As", 70.0),
                pump=params.pop("pump", "constant"),
                pmax=params.pop("pmax", 1.1),
                seed=params.pop("seed", 0),
                device=params.pop("device", "cpu"),
                quantize_bits=params.pop("quantize_bits", None),
                **params,
            )
        return _run

    central = PartitionMethod(
        name="distcim",
        method_name=MethodName("distcim", "simcim"),
        description="Centralized Simulated Coherent Ising Machine (DistIM, 1 module)",
        defaults=dict(nparts=1, scheme="standard"),
    )
    central.bind(_make_run(1, "standard"))
    registry.register(central)

    const = PartitionMethod(
        name="distcim-const",
        method_name=MethodName("distcim", "const"),
        description="DistIM with constant compensation (sparse synchronization)",
        defaults=dict(nparts=4, scheme="const", time_intvl=10),
    )
    const.bind(_make_run(4, "const"))
    registry.register(const)

    pulse = PartitionMethod(
        name="distcim-pulse",
        method_name=MethodName("distcim", "pulse"),
        description="DistIM with pulse compensation (sparse synchronization)",
        defaults=dict(nparts=4, scheme="pulse", time_intvl=10),
    )
    pulse.bind(_make_run(4, "pulse"))
    registry.register(pulse)


def load_config(solver_name: str, config_dir: Optional[Path] = None) -> dict:
    """Load solver config from JSON file.

    Searches ``config_dir`` (default: ``./config``) for
    ``{solver_name}.json``.
    """
    if config_dir is None:
        config_dir = Path.cwd() / "config"
    path = config_dir / f"{solver_name}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
