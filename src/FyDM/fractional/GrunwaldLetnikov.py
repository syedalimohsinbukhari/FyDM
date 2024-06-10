"""Created on Jun 01 01:35:08 2024"""

import numpy as np


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

    def s_factor(self) -> list:
        """S factors for GL fractional order derivative."""
        return [self.T**self.alpha * 0.5 * self.delta_x**-2, self.T**self.alpha * 0.5 * self.delta_y**-2]

    def omega_factor(self, starting_value=0, summed=True):
        """Calculate the omega factor for GL fractional-order derivative class."""
        k_values = np.arange(1, self.T / self.delta_t)
        omega_values = omega(k_values, self.alpha)

        omega_factor = np.zeros(k_values.size + 1)
        omega_factor[0] = 1
        omega_factor[1:] = np.cumprod(omega_values)

        return np.sum(omega_factor[starting_value:]) if summed else omega_factor[starting_value:]

    @staticmethod
    def j_matrix(n_elements):
        n = n_elements * n_elements
        matrix = np.zeros((n, n))

        for i in range(n):
            if (i + 1) % n_elements != 0:  # Not on the right boundary
                matrix[i, i + 1] = 1  # j+1

            if i % n_elements != 0:  # Not on the left boundary
                matrix[i, i - 1] = 1  # j-1

        return matrix

    def generate_fdm_matrix(self, n, S1, S2, stencil_side='lhs'):
        N = n * n  # Total number of grid points
        A = np.zeros((N, N))

        diag_val = 1 + 2 * S1 + 2 * S2
        offsets = [-n, -1, 0, 1, n]  # Corresponding to u_{i,j-1}, u_{i-1,j}, u_{i,j}, u_{i+1,j}, u_{i,j+1}
        values = [-S2, -S1, diag_val, -S2, -S2] if stencil_side == 'lhs' else [S2, S2, -diag_val, S2, S1]

        for i in range(N):
            for offset, value in zip(offsets, values):
                j = i + offset
                if 0 <= j < N:
                    if (offset == -1 and i % n == 0) or (offset == 1 and (i + 1) % n == 0):
                        continue
                    A[i, j] = value

        return A

    def solve(self, n_elements):
        s_factor, omega_factor = self.s_factor(), self.omega_factor
        lhs = self.generate_fdm_matrix(n_elements, *s_factor, 'lhs')
        rhs = self.generate_fdm_matrix(n_elements, *s_factor, 'rhs')

        lhs = np.linalg.inv(lhs)

        solution = [lhs @ rhs]
        summation_: list = []

        for k in range(1, self.T):
            if 1 <= k < 3:
                s = 2
                factor = k - s + 1
                omega_summation = omega_factor(s, False)
                summation_ = [np.sum(solution[:factor + 1][i] * v, axis=1)
                              for i, v in enumerate(omega_summation[:factor + 1])]

            solution.append(lhs @ (solution[k - 1] + summation_ if summation_ else solution[k - 1]))

        # return solution


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

p = gl.solve(4)

# plt.plot(p[0][5])
# plt.show()

# def exact_solution(x_, y_, i_dt):
#     return i_dt**2 * np.sin(np.array([x_]).transpose()) * np.sin(y_)
#
#
# x = np.linspace(0, 1, 128)
# y = np.linspace(0, 1, 10)
# t = np.linspace(0, 1, 10)
#
# p = exact_solution(x, 0.1, 1)
#
# plt.plot(x, p, 'r-.')
# plt.show()
