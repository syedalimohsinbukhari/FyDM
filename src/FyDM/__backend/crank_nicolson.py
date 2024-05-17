"""Created on May 12 06:18:20 2024"""

import numpy as np
from matplotlib import pyplot as plt

from src.FyDM.__backend.fdm_ import (enforce_boundary_condition, identity_matrix, initial_condition_matrix,
                                     OneDimensionalFDM,
                                     OneDimensionalPDESolver, tri_diagonal_matrix)
from src.FyDM.specials.heat_equation import heat_equation


def one_d_crank_nicolson_solver(x_range, delta_x, delta_t, time_steps, initial_condition, boundary_conditions=None,
                                forcing_term=0, wrap_boundaries=False, n_steps=None, has_single_term=True):
    fdm_ = OneDimensionalFDM(x_range, delta_x, delta_t, time_steps, forcing_term,
                             wrap_boundaries, n_steps)

    if forcing_term != 0:
        fdm_matrices = [fdm_.d1_backward(), -k * fdm_.d2_cn(), -fdm_.forcing_term()]
    else:
        fdm_matrices = [fdm_.d1_backward(), -k * fdm_.d2_cn()]

    h_matrix = identity_matrix(fdm_.n_steps) + (k * fdm_.d2_cn())

    pde_ = OneDimensionalPDESolver(fdm_.pde_properties,
                                   fdm_matrices,
                                   initial_condition,
                                   boundary_conditions,
                                   has_single_term)

    return pde_.solve_cn(h_matrix)


def exact_solution(n, x_, k, L, i_dt):
    # # x
    # f1 = -2 * (-1)**n * (n * np.pi)**-1

    # x - x**2
    f1 = 4 * (1 - (-1)**n) * (n * np.pi)**-3

    # # 50 * x * (1 - x)
    # f1 = 200 * (1 - (-1)**n) * (np.pi * n)**-3
    #
    # # 50 * x * (1-x**2)
    # f1 = -600 * (-1)**n * (n * np.pi)**-3

    # # 50 * x * (1-x**2)
    # f1 = -200 * (np.pi * n)**-3
    # f1 -= 400 * (-1)**n * (n * np.pi)**-3

    f2 = np.sin(n * np.pi * np.array([x_]).transpose())
    f3 = np.exp(-(n * np.pi)**2 * i_dt)

    return np.sum(f1 * f2 * f3, axis=1)


# k, L = 1, 1
# x_size, t_size = 500, 500
# dx, dt = 1 / x_size, 1 / t_size
# x_ = np.linspace(*[0, L], t_size + 1)
# n = np.arange(1, 10_00, 1)


# c = one_d_crank_nicolson_solver([0, L], dx, dt, t_size,
#                                 lambda x: 50 * x * (1 - x),
#                                 boundary_conditions=[0, 0])
#


#
# n_ = 1000
#
# f, ax = plt.subplots(1, 3, sharey=True)
# pp = exact_solution(n, x_, k, L, dt * n_)
# ax[0].plot(x_, pp, 'b-.', label='exact_solution')
# ax[1].plot(x_, c[n_], 'r--', label='Crank-Nicolson approximation')


# # ax[1].plot(x_, pp, 'g--', label='Exact')
# # # ax[2].plot(x_, d[n_] - pp, 'g-.', label='EulerBackwards approximation')
# # # [i.legend(loc='best') for i in ax]
# # plt.show()


def CN2(x_range, delta_x, delta_t, diffusivity, time_steps, initial_condition, boundary_conditions=None,
        forcing_term=0):
    factor = diffusivity * delta_t
    factor /= (2 * delta_x**2)

    n_steps = int(1 / delta_x) + 1

    A = tri_diagonal_matrix(n_steps, elements=[-factor, (1 + (2 * factor)), -factor])
    B = tri_diagonal_matrix(n_steps, elements=[factor, (1 - (2 * factor)), factor])

    ic_values = np.linspace(*x_range, n_steps)
    ic = initial_condition_matrix(n_steps,
                                  initial_condition,
                                  ic_values)

    for rows in [0, 1]:
        A[0][rows] = factor
        A[-1][-(rows + 1)] = factor

    lhs = np.linalg.inv(A)
    sol = [ic]

    for t_ in range(0, time_steps):
        enforce_boundary_condition(sol[t_], boundary_conditions)
        sol.append(lhs @ (B @ sol[t_]))
    enforce_boundary_condition(sol[-1], boundary_conditions)

    return np.array([i.transpose()[0] for i in sol])


k, L = 1, 1
x_size, t_size = 500, 500
dx, dt = 1 / x_size, 1 / t_size
x_ = np.linspace(*[0, L], t_size + 1)
n = np.arange(1, 10_000, 1)

c = CN2([0, L], dx, dt, k, t_size, lambda x: x - x**2, boundary_conditions=[0, 0])

d = one_d_crank_nicolson_solver([0, L],
                                dx,
                                dt,
                                t_size,
                                lambda x: x - x**2,
                                [0, 0])

e = heat_equation([0, L],
                  dx,
                  dt,
                  k,
                  t_size,
                  lambda x: x - x**2,
                  [0, 0],
                  solution_method='lw')

f, ax = plt.subplots(1, 3, sharey=True)
for i, v in enumerate(c[:10]):
    pp = exact_solution(n, x_, k, L, dt * i)
    ax[0].plot(x_, v - pp)
    ax[1].plot(x_, d[i] - pp)
    ax[2].plot(x_, e[i] - pp)
plt.show()
