"""Created on May 12 06:18:20 2024"""

import numpy as np
from matplotlib import pyplot as plt

from src.FyDM.__backend.fdm_ import (identity_matrix, initial_condition_matrix,
                                     OneDimensionalFDM,
                                     OneDimensionalPDESolver)
from src.FyDM.specials.heat_equation import heat_equation


def one_d_crank_nicolson_solver(x_range, delta_x, delta_t, time_steps, initial_condition, boundary_conditions=None,
                                forcing_term=0, wrap_boundaries=False, n_steps=None, has_single_term=True):
    fdm_ = OneDimensionalFDM(x_range, delta_x, delta_t, time_steps, forcing_term,
                             wrap_boundaries, n_steps)

    if forcing_term != 0:
        fdm_matrices = [fdm_.d1_backward(), -k * fdm_.d2_cn(), -fdm_.forcing_term()]
    else:
        fdm_matrices = [fdm_.d1_backward(), -k * fdm_.d2_cn()]

    ic_values = np.linspace(*fdm_.pde_properties[0], int(fdm_.n_steps))
    ic = initial_condition_matrix(fdm_.n_steps, initial_condition, ic_values)
    h_matrix = identity_matrix(fdm_.n_steps) + (k * fdm_.d2_cn())

    h_matrix[0][0] = boundary_conditions[0]
    h_matrix[0][1:] = [0] * (len(h_matrix[0]) - 1)
    h_matrix[-1][0:-1] = [0] * (len(h_matrix[0]) - 1)
    h_matrix[-1][-1] = boundary_conditions[1]

    pde_ = OneDimensionalPDESolver(fdm_.pde_properties,
                                   fdm_matrices,
                                   (h_matrix @ ic).transpose()[0].tolist(),
                                   boundary_conditions,
                                   has_single_term)

    return pde_.solve_cn()


def exact_solution(n, x_, k, L, i_dt):
    # # x
    # f1 = -2 * (-1)**n * (n * np.pi)**-1

    # # x - x**2
    # f1 = 4 * (1 - (-1)**n) * (n * np.pi)**-3

    # # 50 * x * (1 - x)
    f1 = 200 * (1 - (-1)**n) * (np.pi * n)**-3

    # # 50 * x * (1-x**2)
    # f1 = -600 * (-1)**n * (n * np.pi)**-3

    f2 = np.sin(n * np.pi * np.array([x_]).transpose())
    f3 = np.exp(-(n * np.pi)**2 * i_dt)

    return np.sum(f1 * f2 * f3, axis=1)


k, L = 1, 1
x_size, t_size = 2000, 2000
dx, dt = 1 / x_size, 1 / t_size
x_ = np.linspace(*[0, L], t_size + 1)
n = np.arange(1, 10_00, 1)

c = one_d_crank_nicolson_solver([0, L], dx, dt, t_size,
                                lambda x: 50 * x * (1 - x),
                                boundary_conditions=[0, 0])

d = heat_equation([0, L],
                  dx,
                  dt,
                  k,
                  t_size,
                  lambda x: 50 * x * (1 - x),
                  [0, 0],
                  solution_method='lw')

n_ = 1000

f, ax = plt.subplots(1, 3, sharey=True)
pp = exact_solution(n, x_, k, L, dt * n_)
ax[0].plot(x_, pp, 'b-.', label='exact_solution')
ax[1].plot(x_, c[n_], 'r--', label='Crank-Nicolson approximation')
# # ax[1].plot(x_, pp, 'g--', label='Exact')
ax[2].plot(x_, d[n_], 'g-.', label='EulerBackwards approximation')
# # [i.legend(loc='best') for i in ax]
plt.show()
