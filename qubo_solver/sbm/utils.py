import numpy as np
from typing import Tuple
import torch
from scipy.sparse import lil_matrix, csr_matrix


def load_data(name='data/Gset/G30'):
    file = open(name, 'r')
    for (idx, line) in enumerate(file):
        if idx == 0:
            N = int(line.split(' ')[0])
            J = np.zeros([N, N])
        else:
            J[int(line.split(' ')[0]) - 1][int(line.split(' ')[1]) - 1] = (line.split(' ')[2])
    file.close()
    tor_arr = -J
    return tor_arr


def load_dimacs10_data(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        while first_line == '':
            first_line = file.readline().strip()
        N = int(first_line.split()[0])
        J = np.zeros((N, N), dtype=float)

        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u = int(parts[0]) - 1
            for v_str in parts[1:]:
                v = int(v_str) - 1
                J[u][v] = 1.0
                J[v][u] = 1.0

    tor_arr = -J
    return tor_arr


def load_qplib_data(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    num_vars = int(lines[3].split('#')[0].strip())
    objective_sense = lines[2].lower()

    Q = np.zeros((num_vars, num_vars))
    b = np.zeros(num_vars)

    for idx, line in enumerate(lines):
        if 'number of quadratic terms in objective' in line:
            num_terms = int(line.split()[0])

    return Q, b
