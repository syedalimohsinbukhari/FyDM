"""Created on Jan 11 13:59:35 2024"""
import inspect

import numpy as np

from src.FyDM.__backend.fdm_ import DirichletBCs, identity_matrix, OneDimensionalFDM, OneDimensionalPDESolver

L, c = 1, 1

n_size, t_size = 4, 4
dx, dt = 1 / n_size, 1 / t_size

x_ = np.linspace(0, L, (n_size * L) + 1)

n = np.arange(1, (L * n_size) + 1, 1)

pde_ = OneDimensionalFDM([0, L],
                         dx,
                         dt,
                         t_size)

fdm_properties = pde_.pde_properties


class D2IC:

    def __init__(self, ic1=None, ic2=None):
        self.IC1 = 0 if ic1 is None else ic1
        self.IC2 = 0 if ic2 is None else ic2

    def c1(self, x=None):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        return self.IC1(x) if inspect.isfunction(self.IC1) else self.IC1

    def c2(self, x=None):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        return self.IC2(x) if inspect.isfunction(self.IC2) else self.IC2

    def __list(self):
        return [self.IC1, self.IC2]


solver_ = OneDimensionalPDESolver(fdm_properties,
                                  [pde_.d2_central(), -c**2 * pde_.d2_central()],
                                  D2IC(lambda x: x, 0),
                                  DirichletBCs())

solution_ = solver_.solve()

print(solution_)

m = [
    [4, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 4, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-2, 0, 0, 4, -1, 0, 0, 0, 0, 0, 0, 0],
    [0, -2, 0, -1, 4, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, -2, 0, -1, 4, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -2, 0, 0, 4, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -2, 0, -1, 4, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, -2, 0, -1, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -2, 0, 0, 4, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 4, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 4, -1]
]

b = [[1 / 2, 1, 3 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

m = np.matrix(m)
b = np.matrix(b).transpose()

solve = np.linalg.inv(m) @ b

# print(np.reshape(solve, (4, 3)))

cp = pde_.d2_central() * dt
x1 = 2 * identity_matrix(5) - cp
# print(x1)

x2 = np.array([x_]).transpose()

# print(np.linalg.inv(x1) @ x2)

# def exact_solution(n, x_, k, L, i_dt):
#     t_n = (16 * (1 - (-1)**n)) / (n * np.pi)**3
#     t_n *= np.exp(-(n * np.pi)**2 * 0.25 * i_dt)
#
#     f1_ = t_n * np.sin(n * np.pi * np.array([x_]).transpose() * 0.5)
#
#     return np.sum(f1_, axis=1)


# def exact_solution(n, x_, k, L, i_dt):
#     q_n = 8 / (n * np.pi)**2
#     q_n *= np.sin(n * np.pi * x_ * 0.5)
#
#     c_n = 16 / (n * np.pi)**3
#     c_n *= (1 - np.cos(n * np.pi))
#
#     f1_ = 4 / (n * np.pi)**2
#     f1_ *= q_n
#
#     f2_ = c_n - f1_
#     f2_ *= np.exp(-(n * np.pi * 0.5)**2 * i_dt)
#
#     f3_ = np.sin(n * np.pi * np.array([x_]).transpose() * 0.5)
#
#     return np.sum((f1_ + f2_) * f3_, axis=1)
#
#
# for i in range(len(solution_)):
#     plt.plot(x_, exact_solution(n, x_, k, L, dt * i), '-.')
#     plt.plot(x_, solution_[i])
# plt.show()
