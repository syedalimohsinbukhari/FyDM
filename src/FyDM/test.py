"""Created on May 07 14:27:22 2024"""

import numpy as np
from matplotlib import pyplot as plt

from src.FyDM.__backend.fdm_ import OneDimensionalFDM, OneDimensionalPDESolver

L, k = 1, 1
x_size, t_size = 400, 400
dx, dt = 1 / x_size, 1 / t_size

x_ = np.linspace(0, L, (x_size * L) + 1)
n = np.arange(1, 4001, 1)

pde_ = OneDimensionalFDM([0, L], dx, dt, time_steps=t_size)
solver_ = OneDimensionalPDESolver(pde_.pde_properties,
                                  [pde_.d1_backward(), -k**2 * pde_.lax_wendroff_convection()],
                                  lambda x: x,
                                  [0, 0])

solution = solver_.solve()


def exact_solution(n, x_, k, L, i_dt):
    f1 = (-2 * (-1)**n) / (n * np.pi)
    f2 = np.exp(-(k * n * np.pi)**2 * i_dt)
    f3 = np.sin(n * np.pi * L**-1 * np.array([x_]).transpose())

    return np.sum(f1 * f2 * f3, axis=1)


f, ax = plt.subplots(1, 1, sharey=True, figsize=(10, 6))
for i, v in enumerate(solution):
    p = exact_solution(n, x_, k, L, dt * i)
    ax.plot(x_, v - p, '-.', label=f't={i}')

plt.tight_layout()
plt.show()
