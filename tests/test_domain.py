from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from predict.domain import LEAGUES, MARKET_IDS, Competition, Match, MatchRecord, season_of
from tests.helpers import make_team


def test_season_of():
    assert season_of(datetime(2026, 10, 22, tzinfo=UTC)) == 2026
    assert season_of(datetime(2027, 3, 1, tzinfo=UTC)) == 2026


def test_competition_properties():
    assert [c.display for c in Competition] == [
        "Premier League",
        "La Liga",
        "Bundesliga",
        "Serie A",
        "Ligue 1",
        "Champions League",
    ]
    assert [c.is_league for c in Competition] == [True] * 5 + [False]
    assert [c.understat_league for c in Competition] == ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1", None]
    assert Competition.CL.odds_sport == "soccer_uefa_champs_league"
    assert [c.clubelo_country for c in Competition] == ["ENG", "ESP", "GER", "ITA", "FRA", None]
    assert [c.couk_division for c in Competition] == ["E0", "SP1", "D1", "I1", "F1", None]
    assert LEAGUES == tuple(c for c in Competition if c.is_league)
    assert Competition("PL") is Competition.PL


def test_market_ids_in_display_order():
    assert len(MARKET_IDS) == 11 and MARKET_IDS[:2] == ("result", "btts")
    assert MARKET_IDS[-1] == "team:away:2.5"


def test_kickoffs_must_be_timezone_aware():
    naive = datetime(2026, 8, 1, 15, 0)
    with pytest.raises(ValidationError):
        Match(id=1, competition=Competition.PL, home=make_team(1, "A"), away=make_team(2, "B"), kickoff=naive)
    with pytest.raises(ValidationError):
        MatchRecord(kickoff=naive, competition=Competition.PL, home_name="A", away_name="B", home_goals=1, away_goals=0)
