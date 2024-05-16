"""Created on May 15 20:52:15 2024"""

from .. import IFloat


class DirichletBCs:
    """
    Class to define Dirichlet boundary conditions.

    Parameters:
    -----------
    alpha : int or float, optional
        Value of the function at the left boundary. Default is 0.
    beta : int or float, optional
        Value of the function at the right boundary. Default is 0.
    """

    def __init__(self, alpha: IFloat = 0, beta: IFloat = 0):
        self.alpha = alpha
        self.beta = beta

    def bcs(self):
        """
        Get the boundary conditions.

        Returns:
        --------
        List[int or float]
            List containing the values of the function at the left and right boundaries.
        """

        return [self.alpha, self.beta]
