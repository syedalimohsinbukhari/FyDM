"""Created on Mar 31 00:06:02 2024"""
import numpy as np
from scipy.integrate import quad


class WaveEquation1D:
    """Wave Equation 1D solver."""

    def __init__(self, x, c2, length, dx, dt, fx, gx):
        self.x = x
        self.c2 = c2
        self.length = length
        self.dx = dx
        self.dt = dt
        self.fx = fx
        self.gx = gx

    def __inner_integral(self, k_factor, x_value):
        f1 = 2 / self.length
        f2 = k_factor * np.pi * x_value * self.length**-1

        return f1, f2

    def alpha_k(self, k_factor, x_value):
        f1, f2 = self.__inner_integral(k_factor, x_value)

        def integral(x_value):
            return self.fx(x_value) * np.sin(f2)

        return f1 * quad(integral, 0, self.length)[0]

    def beta_k(self, k_factor, x_value):
        f1, f2 = self.__inner_integral(k_factor, x_value)
        return f1 * quad(self.gx(x_value) * np.sin(f2), 0, self.length)[0]

    def solve(self):
        k_factor = np.arange(1, 4001, 1)

        sum_ = 0
        for _x in self.x:
            ak = self.alpha_k(k_factor, _x)
            bk = self.beta_k(k_factor, _x)

            f1 = np.sqrt(self.c2) * k_factor * np.pi * self.length**-1 * self.dt

            f2 = ak * np.cos(f1) + bk * np.sin(f1)

            f3 = np.sin(k_factor * _x * np.pi * self.length**-1)

            sum_ += f2 + f3

        return sum_


L, k = 1, 1

n_size, t_size = 5, 5
dx, dt = 1 / n_size, 1 / t_size
x_ = np.array([np.linspace(0, L, n_size * L)]).transpose()

c = WaveEquation1D(x_, k, L, dx, dt, lambda x: x * (1 - x), lambda x: 0)
p = c.solve()

print(p)
