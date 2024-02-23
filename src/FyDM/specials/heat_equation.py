"""Created on Feb 23 09:06:48 2024"""

from .. import FList, Func, IFloat, IFloatOrFList
from ..__backend.fdm_ import OneDimensionalFDM, OneDimensionalPDESolver


# TODO: Add capability of handling forcing term
# TODO: Add capability of solving HEq using explicit method

def heat_equation(x_range: FList, delta_x: IFloat, delta_t: IFloat, diffusivity: IFloat,
                  initial_conditions: IFloatOrFList or Func, boundary_conditions: FList, time_steps: int,
                  wrap_boundaries: bool = False, solution_method: str = 'implicit'):
    """
    Solves the one-dimensional heat equation using finite difference methods.


    Parameters
    ----------
    x_range:
        A list containing the start and end points of the spatial domain.
    delta_x:
        Spatial step size.
    delta_t:
        Time step size.
    diffusivity:
        Diffusivity coefficient.
    initial_conditions:
        A list containing the initial temperature distribution.
    boundary_conditions:
        A list containing the boundary conditions (left and right).
    time_steps:
        Number of time steps to solve for.
    wrap_boundaries:
        Whether to wrap the boundaries (default is False).
    solution_method:
        Whether to solve the given heat equation via `explicit` or `implicit` method. Default is `implicit`.

    Returns
    -------
    NdArray:
        Array containing the temperature distribution at each time step.
    """

    pde_ = OneDimensionalFDM(x_range,
                             delta_x,
                             delta_t,
                             wrap_boundaries=wrap_boundaries)

    fdm_properties = pde_.pde_properties
    fdm_matrices = [pde_.d1_backward(), -diffusivity * pde_.d2_central()]

    fdm_ = OneDimensionalPDESolver(fdm_properties,
                                   initial_conditions,
                                   boundary_conditions,
                                   fdm_matrices)

    return fdm_.solve(time_steps)
