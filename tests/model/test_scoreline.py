import math

import pytest

from predict.model.scoreline import scoreline_matrix


def test_independent_poisson_when_rho_zero():
    m = scoreline_matrix(1.0, 1.0, rho=0.0)
    assert m.shape == (11, 11)
    assert m.sum() == pytest.approx(1.0)
    assert m[0, 0] == pytest.approx(math.exp(-2), abs=1e-6)


def test_negative_rho_moves_probability_into_low_draws():
    dc, ind = scoreline_matrix(1.4, 1.1, rho=-0.05), scoreline_matrix(1.4, 1.1, rho=0.0)
    assert dc[0, 0] > ind[0, 0] and dc[1, 1] > ind[1, 1]
    assert dc[1, 0] < ind[1, 0] and dc[0, 1] < ind[0, 1]
