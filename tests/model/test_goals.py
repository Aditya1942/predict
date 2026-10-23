import pytest

from predict.domain import LeagueBase, TeamRating
from predict.model.goals import CL_BASE, projected_goals


def test_league_match():
    g = projected_goals(
        TeamRating(attack=1.2, concede=0.8),
        TeamRating(attack=0.9, concede=1.1),
        LeagueBase(home_goals=1.5, away_goals=1.2),
    )
    assert (g.home, g.away) == (pytest.approx(1.98), pytest.approx(0.864))


def test_neutral_venue_uses_mean_base():
    g = projected_goals(
        TeamRating(attack=1, concede=1),
        TeamRating(attack=1, concede=1),
        LeagueBase(home_goals=1.5, away_goals=1.2),
        neutral=True,
    )
    assert g.home == pytest.approx(1.35) and g.away == pytest.approx(1.35)


def test_cross_league_elo_factor():
    g = projected_goals(TeamRating(attack=1, concede=1, elo=1750), TeamRating(attack=1, concede=1, elo=1600), CL_BASE)
    assert g.home == pytest.approx(1.9300, abs=1e-4) and g.away == pytest.approx(1.0777, abs=1e-4)


def test_no_elo_factor_when_either_elo_missing():
    g = projected_goals(TeamRating(attack=1, concede=1, elo=1750), TeamRating(attack=1, concede=1), CL_BASE)
    assert g.home == pytest.approx(1.60)
