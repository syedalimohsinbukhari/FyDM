"""Created on May 12 06:18:20 2024"""

import numpy as np

from src.FyDM.__backend.fdm_ import (enforce_boundary_condition, identity_matrix, initial_condition_matrix,
                                     OneDimensionalFDM,
                                     OneDimensionalPDESolver, tri_diagonal_matrix)


def one_d_crank_nicolson_solver(x_range, delta_x, delta_t, k, time_steps, initial_condition, boundary_conditions=None,
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
