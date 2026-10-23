import pytest

from predict.domain import MARKET_IDS
from predict.model.markets import market_probabilities, pick
from predict.model.scoreline import scoreline_matrix


def test_markets_from_known_projected_goals():
    p = market_probabilities(scoreline_matrix(1.5, 1.0, rho=0.0))
    assert set(p) == set(MARKET_IDS)
    assert all(sum(a.values()) == pytest.approx(1.0) for a in p.values())
    assert p["btts"]["yes"] == pytest.approx(0.4911, abs=1e-4)
    assert p["total:2.5"]["over"] == pytest.approx(0.4562, abs=1e-4)
    assert p["team:home:0.5"]["over"] == pytest.approx(0.7769, abs=1e-4)
    assert p["team:home:1.5"]["over"] == pytest.approx(0.4422, abs=1e-4)


def test_equal_teams_give_equal_win_chances():
    p = market_probabilities(scoreline_matrix(1.3, 1.3))
    assert p["result"]["home"] == pytest.approx(p["result"]["away"])


def test_pick():
    assert pick({"home": 0.4, "draw": 0.3, "away": 0.3}) == "home"
    assert pick({"over": 0.45, "under": 0.55}) == "under"
