"""Created on Mar 20 03:46:01 2024"""

from src.FyDM.__backend.fdm_ import tri_diagonal_matrix
from src.FyDM.__backend.fdm_matrix_solver_ import MatrixEq


class D2t:

    def __init__(self, n_steps, dt, dx):
        self.n_steps = n_steps
        self.dt = dt
        self.dx = dx

    def matrix(self):
        return (self.dt / self.dx)**2 * tri_diagonal_matrix(self.n_steps)


class D2x:

    def __init__(self, n_steps, dt, dx):
        self.n_steps = n_steps
        self.dt = dt
        self.dx = dx

    def matrix(self):
        return (self.dt / self.dx)**2 * tri_diagonal_matrix(self.n_steps)


n_steps, dt, dx = 1000, 0.1, 1 / 100

c = MatrixEq(D2t(n_steps, dt, dx),
             D2x(n_steps, dt, dx))

c.solve()
