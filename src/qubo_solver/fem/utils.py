"""Utility functions for the FEM solver."""
import numpy as np
import torch
from typing import Union


def parse_file(problem_type, filename, index_start=0, map_type='normal'):
    """Parse problem files into coupling matrices."""
    # Placeholder — actual implementation depends on file format
    raise NotImplementedError("File parsing not implemented in standalone package.")


def read_graph(*args, **kwargs):
    """Read graph from file — delegating to parse_file."""
    raise NotImplementedError("read_graph not implemented in standalone package.")


def scale_up(x, scl):
    """Scale up a tensor (for quantized solvers)."""
    return x * scl


def scale_down(x, scl):
    """Scale down a tensor (for quantized solvers)."""
    return x / scl
