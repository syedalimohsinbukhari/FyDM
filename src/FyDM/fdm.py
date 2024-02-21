"""Created on Jan 11 13:59:35 2024"""

import numpy as np

from src.FyDM import OptList
from src.FyDM.__backend.fdm_ import OneDimensionalFDM, OneDimensionalPDESolver, bi_diagonal_matrix

x_range, dx, dt, k = [0, 1], 0.1, 1, 0.003
x_ = np.linspace(0, 1, 10)

oneD_FDM = OneDimensionalFDM(x_range, dx, dt)

d = OneDimensionalPDESolver(lambda x: 50 * x * (1 - x),
                            [0, 0],
                            [oneD_FDM.d1_backward(), -k * oneD_FDM.d2_central()],
                            ic_values=x_)


def bd(n_steps: int, wrap_boundaries: bool = False, diff_type: str = 'fwd', elements: OptList = None):
    difference_indices = {'fwd': [0, 1],
                          'bkw': [0, -1],
                          'cnt': [-1, 1]}

    difference_n_steps = {'fwd': [n_steps, n_steps - 1],
                          'bkw': [n_steps, n_steps - 1],
                          'cnt': [n_steps - 1, n_steps - 1]}

    difference_i = difference_indices[diff_type]
    difference_n = difference_n_steps[diff_type]

    elements = elements if elements else [1, -1]

    bi_diagonal_ = (np.diag([elements[1]] * difference_n[0], difference_i[0]) +
                    np.diag([elements[0]] * difference_n[1], difference_i[1]))

    if wrap_boundaries:
        if diff_type == 'bkw':
            bi_diagonal_[0][-1] = elements[0]
        elif diff_type == 'fwd':
            bi_diagonal_[-1][0] = elements[0]
        else:
            bi_diagonal_[0][-1] = elements[1]
            bi_diagonal_[-1][0] = elements[0]

    return bi_diagonal_


print(bi_diagonal_matrix(5, True, 'cnt'))
print(bd(5, True, 'cnt'))

# d1 = d.solve()
#
#
# def exact_solution(n, i_dx, i_dt, k):
#     sum_ = 0
#     for i in range(n):
#         if i % 2 == 1:
#             f1_ = 400 / (pi**3 * i**3)
#             f1_ *= sin(i * pi * i_dx) * exp(-i**2 * pi**2 * k * i_dt)
#             sum_ += f1_
#
#     return sum_
#
#
# plt.plot(x_, [i - j for i, j in zip(d1[0], [exact_solution(20, i, 1, k) for i in x_])],
#          label='numerical - original n=20 difference')
# plt.plot(x_, [i - j for i, j in zip(d1[0], [exact_solution(200, i, 1, k) for i in x_])],
#          label='numerical - original n=200 difference')
# plt.plot(x_, [i - j for i, j in zip(d1[0], [exact_solution(2000, i, 1, k) for i in x_])],
#          label='numerical - original n=2000 difference')
#
# plt.plot(x_, [i - j for i, j in zip(d1[1], [exact_solution(20, i, 2, k) for i in x_])],
#          label='numerical - original n=20 difference')
# plt.plot(x_, [i - j for i, j in zip(d1[1], [exact_solution(200, i, 2, k) for i in x_])],
#          label='numerical - original n=200 difference')
# plt.plot(x_, [i - j for i, j in zip(d1[1], [exact_solution(2000, i, 2, k) for i in x_])],
#          label='numerical - original n=2000 difference')
#
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()
