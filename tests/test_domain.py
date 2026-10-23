from datetime import UTC, datetime

from predict.domain import LEAGUES, MARKET_IDS, Competition, season_of


def test_season_of():
    assert season_of(datetime(2026, 10, 22, tzinfo=UTC)) == 2026
    assert season_of(datetime(2027, 3, 1, tzinfo=UTC)) == 2026


def test_competition_properties():
    assert [c.display for c in Competition] == [
        "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Champions League",
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
