"""Created on Jan 11 13:59:35 2024"""

import numpy as np

from src.FyDM.__backend.fdm_ import DirichletBCs, OneDimensionalFDM, OneDimensionalPDESolver

L, k = 2, 1

n_size, t_size = 500, 20
dx, dt = 1 / n_size, 1 / t_size

x_ = np.linspace(0, L, n_size * L)

n = np.arange(1, (L * n_size) + 1, 1)

pde_ = OneDimensionalFDM([0, L], dx,
                         dt,
                         t_size,
                         forcing_term=lambda x, t: 1 - abs(x - 1))

fdm_properties = pde_.pde_properties

solver_ = OneDimensionalPDESolver(fdm_properties,
                                  [pde_.d1_backward(), -k * pde_.lax_wendroff_convection(), -1 * pde_.forcing_term()],
                                  lambda x: 2 * x - x ** 2,
                                  DirichletBCs())
solution_ = solver_.solve()

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
