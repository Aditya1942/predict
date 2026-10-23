"""Dixon-Coles scoreline distribution."""

import math

import numpy as np

from predict.params import MAX_GOALS, RHO


def _poisson(lam: float, max_goals: int) -> np.ndarray:
    return np.array([math.exp(-lam) * lam**k / math.factorial(k) for k in range(max_goals + 1)])


def scoreline_matrix(home: float, away: float, *, rho: float = RHO, max_goals: int = MAX_GOALS) -> np.ndarray:
    """`[i, j]` = P(home scores i, away scores j), Dixon-Coles adjusted, normalised to sum 1."""
    matrix = np.outer(_poisson(home, max_goals), _poisson(away, max_goals))
    matrix[0, 0] *= 1 - home * away * rho
    matrix[0, 1] *= 1 + home * rho
    matrix[1, 0] *= 1 + away * rho
    matrix[1, 1] *= 1 - rho
    return matrix / matrix.sum()
