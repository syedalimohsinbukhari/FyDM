"""Created on Feb 19 23:52:28 2024"""

__all__ = ['OneDimensionalFDM', 'OneDimensionalPDESolver', 'bi_diagonal_matrix', 'enforce_boundary_condition',
           'initial_condition_matrix', 'tri_diagonal_matrix', 'identity_matrix']

from math import floor
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .boundary_conditions import DirichletBCs
from .. import FList, Func, IFloat, IFloatOrFList, N_DECIMAL, OptIFloat, OptList, TOLERANCE


class Difference:
    def __init__(self, delta_x, delta_t):
        self.dx = delta_x
        self.dt = delta_t

        self.dt_dx = self.dt / self.dx

    def _dt_dx(self):
        return self.dt / self.dx

    def fwd(self):
        return self._dt_dx()

    def bkw(self):
        return self._dt_dx()

    def cnt(self):
        return 0.5 * self._dt_dx()

    def cnt2(self):
        return self.dx**-1 * self._dt_dx()

    def lw(self):
        return [self._dt_dx(), self._dt_dx()**2]

    def cn(self):
        return 0.5 * self.cnt2()

    def factors(self, factor_type):
        return {'fwd': self.fwd(),
                'bkw': self.bkw(),
                'cnt': self.cnt(),
                'cnt2': self.cnt2(),
                'lw': self.lw(),
                'cn': self.cn()}[factor_type]


class OneDimensionalFDM:
    """A class for performing finite difference method computations on a 1D grid.

    Parameters
    ----------
    x_range : FList
        Range of the spatial grid.
    delta_x : float
        Spatial step size.
    delta_t : float
        Temporal step size.
    time_steps : int
        Number of time steps.
    forcing_term : Union[float, Callable[[np.ndarray, np.ndarray], np.ndarray]], optional
        Forcing term for the PDE, either a constant or a function of x and t.
    wrap_boundaries : bool, optional
        Whether to wrap boundaries (periodic boundaries).
    n_steps : int, optional
        Number of spatial steps.
    tolerance : float, optional
        Tolerance for comparing float values.
    """

    def __init__(self,
                 x_range: FList,
                 delta_x: IFloat,
                 delta_t: IFloat,
                 time_steps: IFloat,
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

        self.n_steps += 1

        # Adjust the step size if it differs from the specified step size
        if abs(delta_x - dx2) > tolerance:
            print('The recalculated `delta_x` from `n_steps` does not match the provided value.\n'
                  f'The `delta_x` parameter has been modified from {delta_x} to '
                  f'{np.round(dx2, N_DECIMAL)}')
            self.dx = dx2

    def _factors(self, factor_type):
        return Difference(self.dx, self.dt).factors(factor_type)

    @property
    def pde_properties(self):
        """
        Get the properties of the PDE.

        Returns
        -------
        Tuple
            x_range, dx, dt, ts, True if forcing term is a function, else False.
        """

        return self.x_range, self.dx, self.dt, self.ts, isinstance(self.ft, Func)

    def d1_forward(self):
        """
        Compute the forward difference operator.

        Returns
        -------
        np.ndarray
            The forward difference operator matrix.
        """

        return self._factors('fwd') * bi_diagonal_matrix(self.n_steps,
                                                         self.wrap_boundaries,
                                                         'fwd',
                                                         self.elements)

    def d1_backward(self):
        """
        Compute the backward difference operator.

        Returns
        -------
        np.ndarray
            The backward difference operator matrix.
        """

        return self._factors('bkw') * bi_diagonal_matrix(self.n_steps,
                                                         self.wrap_boundaries,
                                                         'bkw',
                                                         self.elements)

    def d1_central(self):
        """
         Compute the central difference operator.

         Returns
         -------
         np.ndarray
             The central difference operator matrix.
         """

        return self._factors('cnt') * bi_diagonal_matrix(self.n_steps,
                                                         self.wrap_boundaries,
                                                         'cnt',
                                                         self.elements)

    def d2_central(self):
        """
        Compute the central difference operator for the second derivative.

        Returns
        -------
        np.ndarray
            The central difference operator matrix for the second derivative.
        """
        return self._factors('cnt2') * tri_diagonal_matrix(self.n_steps,
                                                           self.wrap_boundaries,
                                                           self.elements)

    def d2_cn(self):
        """
        Compute the Crank-Nicolson difference operator for the second derivative.

        Returns
        -------
        np.ndarray
            The Crank-Nicolson difference operator matrix for the second derivative.
        """

        return self._factors('cn') * tri_diagonal_matrix(self.n_steps,
                                                         self.wrap_boundaries,
                                                         self.elements)

    def lax_wendroff_advection(self):
        """
        Compute the Lax-Wendroff advection operator.

        Returns
        -------
        np.ndarray
            The Lax-Wendroff advection operator matrix.
        """

        return self.d1_central() + (self.d2_central() * self.dt)

    def lax_wendroff_convection(self):
        """
        Compute the Lax-Wendroff convection operator.

        Returns
        -------
        np.ndarray
            The Lax-Wendroff convection operator matrix.
        """

        return self.d2_central() + (self.d2_central() * self.dt)

    def forcing_term(self):
        """
        Compute the forcing term matrix.

        Returns
        -------
        np.ndarray
            The forcing term matrix.
        """

        x_values = np.linspace(*self.x_range, self.n_steps)
        t_values = np.arange(0, (self.dt * self.ts) + self.dt, self.dt)

        mat_ = self.ft(x_values, np.array([t_values]).transpose())
        if mat_.ndim == 1:
            new_shape = self.n_steps
            mat_ = self.dt * np.array([mat_]).transpose()
            mat_ = np.reshape(mat_.tolist() * new_shape, (new_shape, new_shape))
        else:
            mat_ *= self.dt

        return mat_ if isinstance(self.ft, Func) else self.ft


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
        # self.flag = 0

        if isinstance(initial_condition, Func):
            n_steps = self.fdm_p[0][1] / self.fdm_p[1]
            self.ic_values = np.linspace(*self.fdm_p[0], int(n_steps) + 1)

        # try:
        #     if self.ic._D2IC__list():
        #         self.flag = 1
        # except AttributeError:
        #     pass

        self.n_steps = self.fdm[0].shape[1]

        if not self.fdm_p[-1]:
            self.fdm.append(null_matrix(self.fdm[0].shape[1]))

    def lhs(self):
        identity_ = identity_matrix(self.n_steps)
        # if self.flag:
        #     identity_ *= 2

        for matrix_ in self.fdm[1:-1]:
            # if self.flag:
            #     matrix_ *= self.fdm_p[2]
            identity_ += matrix_

        return identity_

    def rhs(self):
        return initial_condition_matrix(self.n_steps,
                                        self.ic,
                                        self.ic_values)

    def solve(self):
        x_range, dx, dt, time_steps, _ = self.fdm_p

        lhs = np.linalg.inv(self.lhs())
        solution = self.__report(x_range, dt, time_steps, lhs)

        for i in range(0, int(time_steps)):
            forcing_term = self.fdm[-1][:, i:i + 1]
            enforce_boundary_condition(solution[i], self.bc)
            solution.append(lhs @ (solution[i] + forcing_term))

        enforce_boundary_condition(solution[-1], self.bc)

        return np.array([i.transpose()[0] for i in solution])

    def solve_cn(self, h_matrix):
        x_range, dx, dt, time_steps, _ = self.fdm_p

        p = self.lhs()

        for row in [0, 1]:
            _factor = Difference(dx, dt).factors('cn')
            p[0][row] = _factor
            p[-1][-(row + 1)] = _factor

        lhs = np.linalg.inv(p)

        solution = self.__report(x_range, dt, time_steps, lhs)

        for i in range(0, int(time_steps)):
            forcing_term = self.fdm[-1][:, i:i + 1]
            enforce_boundary_condition(solution[i], self.bc)
            cn_step = h_matrix @ (solution[i] + forcing_term)
            solution.append(lhs @ cn_step)

        enforce_boundary_condition(solution[-1], self.bc)

        return np.array([i.transpose()[0] for i in solution])

    def __report(self, x_range, dt, time_steps, lhs):
        solution: list = [self.rhs()]
        print(f"LHS matrix size = {lhs.shape}")
        print(f"RHS matrix size = {solution[0].shape}")
        print(f"Number of time-iterations = {time_steps}")
        print(f"dt = {dt} * {time_steps} -> {(time_steps * dt) - x_range[0]}s")
        return solution


def initial_condition_matrix(n_steps: int, initial_condition: IFloatOrFList or Func, values=None):
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


def enforce_boundary_condition(matrix: NDArray, boundary_conditions: FList) -> NDArray:
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


def null_matrix(n_rows: int, n_cols: OptIFloat = None) -> NDArray:
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
    """
    Generate a tri-diagonal matrix.

    Parameters:
    -----------
    n_steps:
        Number of steps in the matrix.
    wrap_boundaries:
        Whether to wrap boundaries.
    elements:
        List containing the elements of the tri-diagonal matrix in the order
         [upper_diagonal, main_diagonal, lower_diagonal, wrap_element]. Defaults to [1, -2, 1].

    Returns:
    --------
    np.ndarray
        Tri-diagonal matrix.
    """

    elements = elements if elements else [1, -2, 1]
    diagonal_main = np.full(n_steps, elements[1])
    diagonal_upper = np.full(n_steps - 1, elements[0])
    diagonal_lower = np.full(n_steps - 1, elements[2])

    tri_diagonal_ = np.diag(diagonal_main) + np.diag(diagonal_upper, k=1) + np.diag(diagonal_lower, k=-1)

    if wrap_boundaries:
        tri_diagonal_[0, -1] = elements[-1]
        tri_diagonal_[-1, 0] = elements[-1]

    return tri_diagonal_
