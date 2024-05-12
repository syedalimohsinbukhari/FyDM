"""Created on May 09 10:19:40 2024"""

import numpy as np

from src.FyDM.__backend.fdm_ import OneDimensionalFDM

k, L = 1, 1
x_size, t_size = 4, 4
dx, dt = 1 / x_size, 1 / t_size

x_ = np.linspace(0, L, ((x_size * L) + 1))
n = np.arange(1, 4001, 1)

p = OneDimensionalFDM([0, L],
                      dx,
                      dt,
                      t_size)

print(p.d2_central())

# solver = OneDimensionalPDESolver(p.pde_properties,
#                                  [p.d1_backward(), -k * p.lax_wendroff_convection()],
#                                  lambda x: x * (1 - x),
#                                  [0, 0])
#
# solution = solver.solve()
#
#
# # def exact_solution(n, x, k, L, i_dt):
# #     f1 = 40 * (1 - (-1)**n) * (n * np.pi)**-1
# #
# #     f2_ = n * np.pi * np.array([x]).transpose()
# #     f2_ /= L
# #     f2 = np.sin(f2_)
# #
# #     f3_ = (n * np.pi) / L
# #     f3 = np.exp(-k * f3_**2 * i_dt)
# #
# #     return np.sum(f1 * f2 * f3, axis=1)
#
#
# def exact_solution_x_1_minus_x(n, x, k, L, i_dt):
#     f1 = 2 * (1 - (-1)**n)
#     f1 = f1 / (n * np.pi)**3
#
#     f2 = np.sin(n * np.pi * np.array([x]).transpose())
#
#     f3 = np.exp(-(n * np.pi)**2 * i_dt)
#
#     return np.sum(f1 * f2 * f3, axis=1)
#
#
# print('done')
# f, ax = plt.subplots(1, 2)
# for i, v in enumerate(solution[0:10]):
#     ax[0].plot(x_, v)
#     ax[1].plot(x_, exact_solution_x_1_minus_x(n, x_, k, L, dt * i))
#
# plt.tight_layout()
# plt.show()
