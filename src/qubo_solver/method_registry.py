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
