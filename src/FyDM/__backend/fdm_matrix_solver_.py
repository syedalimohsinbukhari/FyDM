"""Created on Mar 20 03:45:02 2024"""
import numpy as np
from matplotlib import pyplot as plt

from src.FyDM.__backend.fdm_ import enforce_boundary_condition, identity_matrix, initial_condition_matrix

x_ = np.linspace(0, 1, 1000)


class MatrixEq:

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def solve(self):
        identity = identity_matrix(self.lhs.n_steps)
        g_matrix = 0.5 * (2 * identity - self.rhs.matrix())
        ic = initial_condition_matrix(self.lhs.n_steps, lambda x: -x**2, x_)
        ic2 = initial_condition_matrix(self.lhs.n_steps, lambda x: 2 * x, x_) * self.lhs.dt

        sol = [ic]

        for i in range(1, 101):
            sol.append(np.linalg.inv(g_matrix) @ (sol[i - 1] + ic2))
            enforce_boundary_condition(sol[i], [0, 0])

        #     if i % 5 == 0:
        #         plt.plot(x_, sol[i], label=f'dt = {i * self.lhs.dt}')
        #
        # # enforce_boundary_condition(sol[-1], [0, 0])
        #
        # plt.legend(loc='best')
        # plt.show()
