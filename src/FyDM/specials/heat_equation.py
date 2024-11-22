"""Created on Feb 23 09:06:48 2024"""

from .. import FList, Func, IFloat, IFloatOrFList, OptIFloat
from ..__backend.fdm_ import OneDimensionalFDM, OneDimensionalPDESolver


# TODO:
#       Add capability of handling forcing term
#       Add capability of solving HEq using explicit method
#       Non-homogeneous BCs don't work

def heat_equation(x_range: FList,
                  delta_x: IFloat,
                  delta_t: IFloat,
                  diffusivity: IFloat,
                  time_steps: OptIFloat,
                  initial_conditions: IFloatOrFList or Func,
                  boundary_conditions: FList,
                  forcing_term: IFloat or Func = 0,
                  wrap_boundaries: bool = False,
                  solution_method: str = 'euler backward'):
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
        Number of time steps to solve for. Default is 10.
    forcing_term:
        Forcing term, if any. Either constant or a function of variables. Default is constant 0.
    wrap_boundaries:
        Whether to wrap the boundaries (default is False).
    solution_method:
        Whether to solve the given heat equation via `explicit` or `implicit` method. Default is `euler backward`.

    Returns
    -------
    NdArray:
        Array containing the temperature distribution at each time step.
    """

    pde_ = OneDimensionalFDM(x_range,
                             delta_x,
                             delta_t,
                             time_steps,
                             forcing_term,
                             wrap_boundaries)

    fdm_matrices = [pde_.d1_backward()]

    if solution_method in ['euler_backward', 'eb']:
        fdm_matrices.extend([-diffusivity * pde_.d2_central()])
    elif solution_method in ['lax_wendroff', 'lw']:
        fdm_matrices.extend([-diffusivity * pde_.lax_wendroff_convection()])

    fdm_ = OneDimensionalPDESolver(pde_.pde_properties,
                                   fdm_matrices,
                                   initial_conditions,
                                   boundary_conditions)

    return fdm_.solve()
