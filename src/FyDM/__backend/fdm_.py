"""Created on Feb 19 23:52:28 2024"""

__all__ = ['OneDimensionalFDM', 'OneDimensionalPDESolver', 'bi_diagonal_matrix', 'enforce_boundary_condition',
           'initial_condition_matrix', 'tri_diagonal_matrix']

from math import floor

import numpy as np
from numpy.typing import NDArray

from .. import FList, Func, IFloat, IFloatOrFList, N_DECIMAL, OptIFloat, OptList, TOLERANCE


class OneDimensionalFDM:

    def __init__(self, x_range, delta_x, delta_t, n_steps: OptIFloat = None, wrap_boundaries: bool = False,
                 elements: OptList = None, tolerance: IFloat = TOLERANCE):
        self.x_range = x_range
        self.dx = delta_x
        self.dt = delta_t
        self.wrap_boundaries = wrap_boundaries
        self.elements = elements

        # take the difference of provided x values
        x_diff = x_range[1] - x_range[0]

        # Calculate the number of steps based on the specified step size or the provided x range
        self.n_steps = n_steps if n_steps is not None else floor(x_diff / delta_x)

        # Calculate the actual step size based on the calculated number of steps
        dx2 = x_diff / self.n_steps

        # Adjust the step size if it differs from the specified step size
        if abs(delta_x - dx2) > tolerance:
            print('The recalculated `delta_x` from `n_steps` does not match the provided value.\n'
                  f'The `delta_x` parameter has been modified from {delta_x} to '
                  f'{np.round(dx2, N_DECIMAL)}')
            self.dx = dx2

    def __factors(self, diff_type):
        dx, dt = self.dx, self.dt
        constants = {'fwd': dt / dx,
                     'bkw': dt / dx,
                     'cnt': dt / (2 * dx),
                     'cnt2': dt / dx**2}

        return constants[diff_type]

    @property
    def pde_properties(self):
        return self.x_range, self.dx, self.dt

    def d1_forward(self):
        return self.__factors('fwd') * bi_diagonal_matrix(self.n_steps,
                                                          self.wrap_boundaries,
                                                          'fwd',
                                                          self.elements)

    def d1_backward(self):
        return self.__factors('bkw') * bi_diagonal_matrix(self.n_steps,
                                                          self.wrap_boundaries,
                                                          'bkw',
                                                          self.elements)

    def d1_central(self):
        return self.__factors('cnt') * bi_diagonal_matrix(self.n_steps,
                                                          self.wrap_boundaries,
                                                          'cnt',
                                                          self.elements)

    def d2_central(self):
        return self.__factors('cnt2') * tri_diagonal_matrix(self.n_steps,
                                                            self.wrap_boundaries,
                                                            self.elements)


def identity_matrix(n_steps: int):
    return np.eye(n_steps)


class OneDimensionalPDESolver:

    def __init__(self, fdm_properties, initial_condition, boundary_conditions, fdm_matrices, has_single_term=True):
        self.fdm_p = fdm_properties
        self.ic = initial_condition
        self.bc = boundary_conditions
        self.fdm = fdm_matrices
        self.hST = has_single_term
        self.ic_values = None

        if isinstance(initial_condition, Func):
            n_steps = self.fdm_p[0][1] / self.fdm_p[1]
            self.ic_values = np.linspace(*self.fdm_p[0], int(n_steps))

        self.n_steps = self.fdm[0].shape[1]

    def lhs(self):
        identity_ = identity_matrix(self.n_steps)
        for matrix_ in self.fdm[1:]:
            identity_ += matrix_

        return identity_

    def rhs(self):
        return initial_condition_matrix(self.n_steps, self.ic, self.ic_values)

    def __solver(self):
        return np.linalg.inv(self.lhs()) @ self.rhs()

    def solve(self, time_steps: int = 10):
        lhs = np.linalg.inv(self.lhs())

        solution: list = [self.rhs(), self.__solver()]

        print(f"LHS matrix size = {lhs.shape}")
        print(f"RHS matrix size = {solution[0].shape}")
        print(f"Number of time-iterations = {time_steps}")

        for i in range(1, time_steps):
            solution.append(lhs @ solution[i])
            enforce_boundary_condition(solution[i - 1], self.bc, i > 0)

        return [i.transpose()[0] for i in solution]


def initial_condition_matrix(n_steps: int, initial_condition: IFloatOrFList or Func, values=None):
    if isinstance(initial_condition, (int, float)):
        return [initial_condition] * n_steps

    elif isinstance(initial_condition, list):
        return initial_condition

    elif isinstance(initial_condition, Func):
        if len(values) != n_steps:
            raise ValueError('The length of vector provided does not match with the number of steps provided')
        return np.array([initial_condition(values)]).transpose()


def enforce_boundary_condition(matrix, boundary_conditions: FList, overwrite: bool = False):
    if overwrite:
        matrix[0] = boundary_conditions[0]
        matrix[-1] = boundary_conditions[-1]

    return matrix


def null_matrix(n_rows: int, n_cols: OptIFloat = None) -> NDArray:
    """
    Returns a zero matrix for given `n_rows` and `n_cols`.

    Args:
        n_rows: Number of rows in the resultant matrix.
        n_cols: Number of columns in the resultant matrix.

    Returns:
        NDArray:
            Null matrix.
    """

    return np.zeros((n_rows, n_rows if n_cols is None else n_cols))


def bi_diagonal_matrix(n_steps, wrap_boundaries: bool = False, diff_type: str = 'fwd',
                       elements: OptList = None):

    difference_indices = {'fwd': [0, 1],
                          'bkw': [0, -1],
                          'cnt': [-1, 1]}

    difference_n_steps = {'fwd': [n_steps, n_steps - 1],
                          'bkw': [n_steps, n_steps - 1],
                          'cnt': [n_steps - 1, n_steps - 1]}

    difference_i = difference_indices[diff_type]
    difference_n = difference_n_steps[diff_type]

    elements = elements if elements else [1, -1]

    bi_diagonal_ = (np.diag([elements[1]] * difference_n[0], difference_i[0]) +
                    np.diag([elements[0]] * difference_n[1], difference_i[1]))

    if wrap_boundaries:
        if diff_type == 'bkw':
            bi_diagonal_[0][-1] = elements[0]
        elif diff_type == 'fwd':
            bi_diagonal_[-1][0] = elements[0]
        else:
            bi_diagonal_[0][-1] = elements[1]
            bi_diagonal_[-1][0] = elements[0]

    return bi_diagonal_


def tri_diagonal_matrix(n_steps: int, wrap_boundaries: bool = False, elements: OptList = None):
    elements = elements if elements else [1, -2, 1]
    tri_diagonal_ = null_matrix(n_steps)

    for row in range(tri_diagonal_.shape[1]):
        for col in range(tri_diagonal_.shape[0]):
            if row == col + 1:
                tri_diagonal_[row][col] = elements[0]
            if row == col:
                tri_diagonal_[row][col] = elements[1]
            if row == col - 1:
                tri_diagonal_[row][col] = elements[2]

    if wrap_boundaries:
        tri_diagonal_[0][-1] = elements[-1]
        tri_diagonal_[-1][0] = elements[-1]

    return tri_diagonal_
