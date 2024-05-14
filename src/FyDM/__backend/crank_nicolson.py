"""Created on May 12 06:18:20 2024"""

import numpy as np
from matplotlib import pyplot as plt

from src.FyDM import TOLERANCE
from src.FyDM.__backend.fdm_ import identity_matrix, initial_condition_matrix, OneDimensionalFDM, \
    OneDimensionalPDESolver


def one_d_crank_nicolson_solver(x_range, delta_x, delta_t, time_steps, initial_condition, forcing_term=0,
                                wrap_boundaries=False, n_steps=None, elements=None, tolerance=TOLERANCE,
                                boundary_conditions=None, has_single_term=True):
    fdm_ = OneDimensionalFDM(x_range, delta_x, delta_t, time_steps, forcing_term,
                             wrap_boundaries, n_steps, elements,
                             tolerance)

    if forcing_term != 0:
        fdm_matrices = [fdm_.d1_backward(), -k**2 * fdm_.d2_cn(), -fdm_.forcing_term()]
    else:
        fdm_matrices = [fdm_.d1_backward(), -k**2 * fdm_.d2_cn()]

    ic_values = np.linspace(*fdm_.pde_properties[0], int(fdm_.n_steps))
    ic = initial_condition_matrix(fdm_.n_steps, initial_condition, ic_values)
    h_matrix = identity_matrix(fdm_.n_steps) + (k**2 * fdm_.d2_cn())

    pde_ = OneDimensionalPDESolver(fdm_.pde_properties,
                                   fdm_matrices,
                                   (h_matrix @ ic).transpose()[0].tolist(),
                                   boundary_conditions,
                                   has_single_term)

    return pde_.solve()


def exact_solution(n, x_, k, L, i_dt):
    # f1 = -2 * (-1)**n * (n * np.pi)**-1
    f1 = 4 * (1 - (-1)**n) * (n * np.pi)**-3
    f2 = np.sin(n * np.pi * np.array([x_]).transpose())
    f3 = np.exp(-(n * np.pi)**2 * i_dt)

    return np.sum(f1 * f2 * f3, axis=1)


k, L = 1, 1
x_size, t_size = 100, 100
dx, dt = 1 / x_size, 1 / t_size
x_ = np.linspace(*[0, L], t_size + 1)
n = np.arange(1, 4001, 1)

c = one_d_crank_nicolson_solver([0, L],
                                dx,
                                dt,
                                t_size,
                                lambda x: x - x**2,
                                boundary_conditions=[0, 0])

n_ = 1

f, ax = plt.subplots(1, 3)
pp = exact_solution(n, x_, k, L, dt * n_)
ax[0].plot(x_, c[n_], 'r--')
ax[0].plot(x_, pp, 'b-.')
ax[1].plot(x_, c[n_] - pp, 'g-.')

plt.show()
