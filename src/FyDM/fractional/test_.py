"""Created on Jun 05 09:45:04 2024"""
from src.FyDM.fractional.GrunwaldLetnikov import GrunwaldLetnikov


def printer(i_max, j_max, k_max, kplus1=False):
    k_range = range(1, k_max) if kplus1 else range(0, k_max - 1)
    signs = ['-', '+', '-'] if kplus1 else ['+', '-', '+']
    for k in k_range:
        for j in range(1, j_max):
            for i in range(1, i_max):
                print(f'{signs[0]} S1(u_[{i + 1},{j}]^{k + 1} + u_[{i - 1},{j}]^{k + 1}) {signs[1]} '
                      f'(1 + 2S1 + 2S2)u_[{i},{j}]^{k} {signs[2]} '
                      f'S2(u_[{i},{j + 1}]^{k + 1} + u_[{i},{j - 1}]^{k + 1})')


# printer(4, 4, 4, False)

import numpy as np


def generate_fdm_matrix(n, S1, S2, stencil_side='lhs'):
    N = n * n  # Total number of grid points
    A = np.zeros((N, N))

    diag_val = 1 + 2 * S1 + 2 * S2
    offsets = [-n, -1, 0, 1, n]  # Corresponding to u_{i,j-1}, u_{i-1,j}, u_{i,j}, u_{i+1,j}, u_{i,j+1}
    values = [-S2, -S1, diag_val, -S2, -S2] if stencil_side == 'lhs' else [S2, S2, -diag_val, S2, S1]

    for i in range(N):
        for offset, value in zip(offsets, values):
            j = i + offset
            if 0 <= j < N:
                if (offset == -1 and i % n == 0) or (offset == 1 and (i + 1) % n == 0):
                    continue
                A[i, j] = value

    return A


gl = GrunwaldLetnikov(0.8,
                      1 / 2,
                      1 / 2,
                      1 / 4,
                      1,
                      1,
                      50,
                      0,
                      [
                          [0, lambda x, y, t: t**2 * np.sin(1) * np.sin(y)],
                          [0, lambda x, y, t: t**2 * np.sin(x) * np.sin(1)]
                      ])

# Parameters
n = 4
S1 = gl.s_factor()[0]  # Example value
S2 = gl.s_factor()[1]  # Example value

# Generate matrix
lhs = generate_fdm_matrix(n, S1, S2)
rhs = generate_fdm_matrix(n, S1, S2, 'rhs')

print(gl.omega_factor(2, False))

# print(lhs)
