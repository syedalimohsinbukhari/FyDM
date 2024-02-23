"""Created on Feb 23 09:06:48 2024"""

from src.FyDM.__backend.fdm_ import OneDimensionalFDM, OneDimensionalPDESolver


# TODO: Add capability of handling forcing term
class HeatEquation:

    def __init__(self, x_range, delta_x, delta_t, diffusivity, initial_conditions, boundary_conditions, time_steps,
                 wrap_boundaries: bool = False):
        self.x_range = x_range
        self.dx = delta_x
        self.dt = delta_t
        self.k = diffusivity
        self.time_steps = time_steps
        self.ic = initial_conditions
        self.bc = boundary_conditions
        self.wrap_boundaries = wrap_boundaries

    def pde_initializer(self):
        return OneDimensionalFDM(self.x_range,
                                 self.dx,
                                 self.dt,
                                 wrap_boundaries=self.wrap_boundaries)

    def fdm(self):
        pde_ = self.pde_initializer()
        fdm_properties, fdm_matrices = pde_.pde_properties, [pde_.d1_backward(), -self.k * pde_.d2_central()]

        return OneDimensionalPDESolver(fdm_properties,
                                       self.ic,
                                       self.bc,
                                       fdm_matrices)

    def solve(self):
        return self.fdm().solve(self.time_steps)
