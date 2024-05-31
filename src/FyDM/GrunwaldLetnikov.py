"""Created on Jun 01 01:35:08 2024"""

import numpy as np


def omega(k, alpha):
    """Omega factor."""
    return 1 - (alpha + 1) / k


class GrunwaldLetnikov:
    """Class for Grunwald-Letnikov type fractional derivative."""

    def __init__(self, alpha, delta_x, delta_y, delta_t, t_range):
        self.alpha = alpha

        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_t = delta_t

        self.T = t_range

    def omega_factor(self):
        """Calculate the omega factor for GL fractional-order derivative class."""
        k_values = np.arange(1, self.T / self.delta_t)
        omega_values = omega(k_values, self.alpha)

        omega_factor = np.zeros(k_values.size + 1)
        omega_factor[0] = 1
        omega_factor[1:] = np.cumprod(omega_values)

        return np.sum(omega_factor)


gl = GrunwaldLetnikov(0.8, 1 / 2, 1 / 2, 1 / 4, 50).omega_factor()
print(gl)
