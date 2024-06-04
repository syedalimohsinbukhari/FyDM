"""Created on Jun 01 01:35:08 2024"""

import numpy as np

from src.FyDM.__backend.fdm_ import tri_diagonal_matrix


def omega(k, alpha):
    """Omega factor."""
    return 1 - (alpha + 1) / k


class GrunwaldLetnikov:
    """Class for Grunwald-Letnikov type fractional derivative."""

    def __init__(self, alpha, delta_x, delta_y, delta_t, x_limit, y_limit, t_limit,
                 initial_conditions, boundary_conditions):
        self.alpha = alpha

        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_t = delta_t

        self.X = x_limit
        self.Y = y_limit
        self.T = t_limit

        self.ic = initial_conditions
        self.bc = boundary_conditions

    def s_factor(self):
        """S factors for GL fractional order derivative."""
        return self.T**self.alpha * 0.5 * self.delta_x**-2, self.T**self.alpha * 0.5 * self.delta_y**-2

    def omega_factor(self, starting_value=0, summed=True):
        """Calculate the omega factor for GL fractional-order derivative class."""
        k_values = np.arange(1, self.T / self.delta_t)
        omega_values = omega(k_values, self.alpha)

        omega_factor = np.zeros(k_values.size + 1)
        omega_factor[0] = 1
        omega_factor[1:] = np.cumprod(omega_values)

        return np.sum(omega_factor[starting_value:]) if summed else omega_factor[starting_value:]

    def j_matrix(self, n_elements):
        n = n_elements * n_elements
        matrix = np.zeros((n, n))

        for i in range(n):
            if (i + 1) % n_elements != 0:  # Not on the right boundary
                matrix[i, i + 1] = 1  # j+1

            if i % n_elements != 0:  # Not on the left boundary
                matrix[i, i - 1] = 1  # j-1

        return matrix

    def lhs(self, n_elements):
        sf = self.s_factor()
        p1_k1 = tri_diagonal_matrix(n_elements**2, False, elements=[1, 0, 1])
        p2_k1 = np.eye(n_elements**2)
        p3_k1 = self.j_matrix(n_elements)

        return -sf[0] * p1_k1 + (1 + (2 * sf[0]) + (2 * sf[1])) * p2_k1 - sf[1] * p3_k1

    def rhs(self, n_elements):
        sf = self.s_factor()
        p1_k = tri_diagonal_matrix(n_elements**2, False, elements=[1, 0, 1])
        p2_k = np.eye(n_elements**2)
        p3_k = self.j_matrix(n_elements)

        return sf[0] * p1_k - (-self.alpha + (2 * sf[0]) + (2 * sf[1])) * p2_k + sf[1] * p3_k

    def solve(self, n_elements):
        lhs = self.lhs(n_elements)
        rhs = self.rhs(n_elements)

        omega_summation = self.omega_factor(2, False)

        for k in range(self.T + 1):
            if k == 2:
                print(omega_summation[:k])


gl = GrunwaldLetnikov(0.8,
                      1 / 2,
                      1 / 2,
                      1 / 4,
                      1,
                      1,
                      50,
                      0,
                      [
                          [0, lambda x, y, t: t**2 * np.sin(1) * np.sin(y)],
                          [0, lambda x, y, t: t**2 * np.sin(x) * np.sin(1)]
                      ])
gl.solve(4)

# print(np.arange(0, 50 + 1 / 4, 1 / 4))

# p1_k = tri_diagonal_matrix(n_elements**2, False, elements=[1, 0, 1])
# p2_k = np.eye(n_elements**2)
# p3_k = j_matrix(n_elements)
#
# rhs = sf[0] * p1_k1 - (-gl.alpha + (2 * sf[0]) + (2 * sf[1])) * p2_k1 + sf[1] * p3_k1
#
# print(gl.omega_factor())
