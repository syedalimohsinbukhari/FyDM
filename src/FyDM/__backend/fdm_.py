"""Created on Feb 19 23:52:28 2024"""

__all__ = ['OneDimensionalFDM', 'OneDimensionalPDESolver', 'bi_diagonal_matrix', 'enforce_boundary_condition',
           'initial_condition_matrix', 'tri_diagonal_matrix', 'DirichletBCs']

from math import floor
from typing import Callable

import numpy as np
import yaml
from numpy.typing import NDArray

from .. import FList, Func, IFloat, IFloatOrFList, N_DECIMAL, OptIFloat, OptList, TOLERANCE


class DirichletBCs:

    def __init__(self, alpha: IFloat = 0, beta: IFloat = 0):
        self.alpha = alpha
        self.beta = beta

    def bcs(self):
        return [self.alpha, self.beta]


class OneDimensionalFDM:

    def __init__(self,
                 x_range: FList,
                 delta_x: IFloat,
                 delta_t: IFloat,
                 time_steps: IFloat = 1000,
                 forcing_term: IFloat or Func = 0,
                 wrap_boundaries: bool = False,
                 n_steps: OptIFloat = None,
                 elements: OptList = None,
                 tolerance: IFloat = TOLERANCE):
        self.x_range = x_range
        self.dx = delta_x
        self.dt = delta_t
        self.ts = time_steps
        self.ft = forcing_term
        self.wrap_boundaries = wrap_boundaries
        self.n_steps = n_steps
        self.elements = elements
        self.tolerance = tolerance

        self.__expand()

    def __expand(self):
        x_range, n_steps, delta_x, tolerance = self.x_range, self.n_steps, self.dx, self.tolerance

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

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        return cls(data['rod_length'],
                   data['dx'],
                   data['dt'],
                   data.get('time_steps', 1 / data['dt']),
                   data.get('ft', 0),
                   data.get('wrap_boundaries', False),
                   data.get('n_steps', None))

    def __factors(self, diff_type):
        dx, dt = self.dx, self.dt

        dt_dx = dt / dx

        constants = {'fwd': dt_dx,
                     'bkw': dt_dx,
                     'cnt': 0.5 * dt_dx,
                     'cnt2': dx**-1 * dt_dx,
                     'lw': [dt_dx, dt_dx**2]}

        return constants[diff_type]

    @property
    def pde_properties(self):
        return self.x_range, self.dx, self.dt, self.ts, isinstance(self.ft, Func)

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

    def lax_wendroff_advection(self):
        return self.d1_central() + (self.d2_central() * self.dt)

    def lax_wendroff_convection(self):
        return self.d2_central() + (self.d2_central() * self.dt)

    def forcing_term(self):
        x_values = np.linspace(*self.x_range, self.n_steps)
        t_values = np.arange(0, (self.dt * self.ts) + self.dt, self.dt)

        return self.dt * self.ft(x_values, np.array([t_values]).transpose()) if isinstance(self.ft, Func) else self.ft


def identity_matrix(n_steps: int):
    return np.eye(n_steps)


class OneDimensionalPDESolver:

    def __init__(self,
                 fdm_properties,
                 fdm_matrices,
                 initial_condition,
                 boundary_conditions: OptList or Callable = None,
                 has_single_term: bool = True):

        self.fdm_p = fdm_properties
        self.fdm = fdm_matrices
        self.ic = initial_condition
        self.bc = boundary_conditions.bcs() if isinstance(boundary_conditions, DirichletBCs) else boundary_conditions
        self.hST = has_single_term

        self.ic_values = None
        self.flag = 0

        if isinstance(initial_condition, Func):
            n_steps = self.fdm_p[0][1] / self.fdm_p[1]
            self.ic_values = np.linspace(*self.fdm_p[0], int(n_steps) + 1)

        try:
            if self.ic._D2IC__list():
                self.flag = 1
        except AttributeError:
            pass

        self.n_steps = self.fdm[0].shape[1]

        if not self.fdm_p[-1]:
            self.fdm.append(np.array([0] * self.fdm[0].shape[1]))

    def lhs(self):
        identity_ = identity_matrix(self.n_steps)
        if self.flag:
            identity_ *= 2

        for matrix_ in self.fdm[1:-1]:
            if self.flag:
                matrix_ *= self.fdm_p[2]
            identity_ += matrix_

        return identity_

    def rhs(self):
        if self.flag == 0:
            temp_ = initial_condition_matrix(self.n_steps,
                                             self.ic,
                                             self.ic_values)

            # temp_ += boundary_condition_matrix(self.n_steps,
            #                                    self.bc)
        else:
            temp_ = initial_condition_matrix(self.n_steps,
                                             self.ic.c1,
                                             self.ic_values)

        return temp_ - np.array([self.fdm[-1]]).transpose()

    def solve(self):
        x_range, dx, dt, time_steps, _ = self.fdm_p

        p = self.lhs()
        p[0][0] = 1
        p[0][1:] = [0] * (len(p[0]) - 1)
        p[-1][0:-1] = [0] * (len(p[0]) - 1)
        p[-1][-1] = 1

        # print(p)

        lhs = np.linalg.inv(p)
        solution: list = [self.rhs()]

        print(f"LHS matrix size = {lhs.shape}")
        print(f"RHS matrix size = {solution[0].shape}")
        print(f"Number of time-iterations = {time_steps}")
        print(f"dt = {dt} * {time_steps} -> {(time_steps * dt) - x_range[0]}s")

        for i in range(0, time_steps):
            enforce_boundary_condition(solution[i], self.bc)
            solution.append(lhs @ solution[i])

        return np.array([i.transpose()[0] for i in solution])


def initial_condition_matrix(n_steps: int,
                             initial_condition: IFloatOrFList or Func,
                             values=None):
    """
    Generate a matrix representing initial conditions for a given number of time steps.

    Parameters
    ----------
    n_steps:
        The number of time steps.
    initial_condition:
        - The initial condition. If int or float, the matrix will be filled with this value.
        - If a list is provided, the matrix will be created from the list.
        - If Callable, the function will be called with 'values' as input to generate the matrix.
    values:
        Values to be used in the initial condition calculation if 'initial_condition' is a Callable.

    Returns
    -------
    array:
        A numpy array representing the initial condition matrix.
    """

    if isinstance(initial_condition, (int, float)):
        return np.full((n_steps, 1), initial_condition)

    elif isinstance(initial_condition, list):
        return np.array(initial_condition).reshape(-1, 1)

    elif isinstance(initial_condition, Func):
        # val_ = values[1:-1]
        # if len(val_) != n_steps:
        #     raise ValueError('The length of vector provided does not match with the number of steps provided')
        return np.array([initial_condition(values)]).transpose()

    raise ValueError("Invalid initial_condition type")


def boundary_condition_matrix(n_steps: int, boundary_conditions):
    null_ = null_matrix(n_steps, 1)
    null_[[0, -1]] = boundary_conditions

    return null_


def enforce_boundary_condition(matrix: NDArray,
                               boundary_conditions: FList) -> NDArray:
    """
    Enforce boundary conditions on a matrix.

    Parameters
    ----------
    matrix:
        The matrix to which boundary conditions should be applied.
    boundary_conditions:
        List of boundary conditions. The first and last elements are applied to the first and last rows of the matrix,
        respectively.

    Returns
    -------
    array
        The matrix with boundary conditions applied.
    """

    matrix[0] = boundary_conditions[0]
    matrix[-1] = boundary_conditions[-1]

    return matrix


def null_matrix(n_rows: int,
                n_cols: OptIFloat = None) -> NDArray:
    """
    Returns a zero matrix for given `n_rows` and `n_cols`.

    Parameters
    ----------
    n_rows:
        Number of rows in the resultant matrix.
    n_cols:
        Number of columns in the resultant matrix.

    Returns
    -------
    NDArray:
        Null matrix.

    """

    return np.zeros((n_rows, n_rows if n_cols is None else n_cols))


def bi_diagonal_matrix(n_steps, wrap_boundaries: bool = False, diff_type: str = 'fwd', elements: OptList = None):
    """
    Generate a bi-diagonal matrix for a given number of time steps.

    Parameters
    ----------
    n_steps:
        The number of time steps.
    wrap_boundaries:
        If True, wrap the boundaries of the matrix. Default is False.
    diff_type:
        The type of difference scheme to use. Default is 'fwd'. Can be 'fwd' (forward), 'bkw' (backward), or
        'cnt' (central).
    elements:
        The diagonal elements of the matrix. Should be a list of two integers. Default is [1, -1].

    Returns
    -------
    array:
        A bi-diagonal matrix representing the specified difference scheme and boundary conditions.
    """

    n_steps += 1

    elements = elements if elements else [1, -1]

    main_diagonal = np.full(n_steps, elements[1])
    upper_diagonal = np.full(n_steps - 1, elements[0])
    lower_diagonal = np.full(n_steps - 1, elements[0])

    if diff_type == 'bkw':
        matrix = np.diag(main_diagonal) + np.diag(upper_diagonal, k=1)
    elif diff_type == 'fwd':
        matrix = np.diag(main_diagonal) + np.diag(lower_diagonal, k=-1)
    else:
        matrix = np.diag(main_diagonal) + np.diag(upper_diagonal, k=1) + np.diag(lower_diagonal, k=-1)

    if wrap_boundaries:
        matrix[0, -1] = elements[0]
        matrix[-1, 0] = elements[0]

    return matrix


def tri_diagonal_matrix(n_steps: int, wrap_boundaries: bool = False, elements: OptList = None):
    n_steps += 1
    elements = elements if elements else [1, -2, 1]
    diag_main = np.full(n_steps, elements[1])
    diag_upper = np.full(n_steps - 1, elements[0])
    diag_lower = np.full(n_steps - 1, elements[2])

    tri_diagonal_ = np.diag(diag_main) + np.diag(diag_upper, k=1) + np.diag(diag_lower, k=-1)

    if wrap_boundaries:
        tri_diagonal_[0, -1] = elements[-1]
        tri_diagonal_[-1, 0] = elements[-1]

    return tri_diagonal_
