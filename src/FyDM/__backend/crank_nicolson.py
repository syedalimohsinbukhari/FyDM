"""Created on May 12 06:18:20 2024"""

from .fdm_ import identity_matrix, OneDimensionalFDM, OneDimensionalPDESolver

# TODO:
#     Add docstring
#     Check with forcing term



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
