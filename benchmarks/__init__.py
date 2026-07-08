"""
qubo-solver benchmarks.

Directory structure::

    benchmarks/
    ├── __init__.py
    ├── best_known/          ← best-known values per instance
    │   ├── __init__.py
    │   ├── gset_maxcut.py
    │   └── ...
    └── instances/           ← problem instance files
        ├── maxcut/
        │   └── Gset/        ← Gset graphs
        ├── bmincut/
        └── maxsat/
"""

from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent
"""Root of the benchmarks directory."""
