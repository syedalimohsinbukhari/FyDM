"""Created on Apr 06 00:30:51 2024"""
import numpy as np

from src.FyDM.__backend.fdm_ import DirichletBCs, OneDimensionalFDM, OneDimensionalPDESolver

L, c = 1, 1

n_size, t_size = 4, 4
dx, dt = 1 / n_size, 1 / t_size

x_ = np.linspace(0, L, (n_size * L) + 1)

n = np.arange(1, (L * n_size) + 1, 1)

pde_ = OneDimensionalFDM([0, L],
                         dx,
                         dt,
                         t_size)

solver_ = OneDimensionalPDESolver(pde_.pde_properties,
                                  [pde_.d1_backward(), -c * pde_.d2_central()],
                                  lambda x: x,
                                  DirichletBCs())

p = solver_.solve()

m = [
    [9, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-4, 9, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -4, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 9, -4, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, -4, 9, -4, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, -4, 9, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 9, -4, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, -4, 9, -4, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, -4, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 0, 0, 9, -4, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, 0, -4, 9, -4],
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 9, -4]
]

b = [[0.25, 0.5, 0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

m = np.matrix(m)
b = np.matrix(b).transpose()

solve = np.linalg.inv(m) @ b

q = np.reshape(solve, (4, 3))

print(q)
print(p[1:])
# print(solver_.rhs())

# solution_ = solver_.solve()
#
# print(solution_)
