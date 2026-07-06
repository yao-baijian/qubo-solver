"""Utility functions for QIS3 solver."""
from qubo_solver.qis3.qis3 import QIS3
import numpy as np


def run_qis3(J, **kwargs):
    """Convenience wrapper for running QIS3."""
    qis3 = QIS3(J=J, **kwargs)
    return qis3.solve()
