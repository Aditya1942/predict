import pytest

from predict.domain import Competition, TeamRating
from predict.model.strength import combine, form_rating, league_base, performance, season_rating, season_weight
from tests.helpers import rec

CUR = [rec(1, 2, 2, 1, 2.0, 1.0, day=1), rec(2, 1, 1, 2, 1.0, 2.0, day=8)]


def test_performance():
    assert performance(2, 1.0) == pytest.approx(1.3) and performance(2, None) == 2.0


def test_season_weight():
    assert (season_weight(0), season_weight(9), season_weight(25)) == (0.0, 0.5, 1.0)


def test_promoted_team_blends_toward_promoted_defaults():
    r = season_rating(1, CUR, last=[])
    assert r.attack == pytest.approx(0.9037, abs=1e-4) and r.concede == pytest.approx(1.0963, abs=1e-4)


def test_same_numbers_both_seasons_ignore_weight():
    assert season_rating(1, CUR, last=CUR).attack == pytest.approx(4 / 3)


def test_team_with_no_matches_this_season_uses_last_season():
    r = season_rating(1, [], last=CUR)
    assert r.attack == pytest.approx(4 / 3) and r.concede == pytest.approx(2 / 3)


def test_team_in_neither_season_gets_promoted_defaults():
    r = season_rating(7, CUR, last=CUR)
    assert (r.attack, r.concede) == (0.85, 1.15)


def test_form_weights_recent_matches_more():
    recent = [rec(1, 2, 2, 1, 2.0, 1.0, day=1), rec(2, 1, 1, 1, 1.0, 1.0, day=8)]  # newest is the weaker game
    f = form_rating(1, recent, {Competition.PL: 1.5})
    assert f.attack == pytest.approx(0.9697, abs=1e-4)


def test_form_uses_goals_when_no_xg():
    f = form_rating(1, [rec(1, 9, 3, 0, day=1, competition=Competition.CL)], {Competition.CL: 1.5})
    assert f.attack == pytest.approx(2.0)


def test_form_uses_only_latest_six_and_none_without_records():
    recent = [rec(1, 9, 100, 0, day=1)] + [rec(1, 9, 3, 0, day=d) for d in range(2, 8)]
    f = form_rating(1, recent, {Competition.PL: 1.5})
    assert f.attack == pytest.approx(2.0) and f.concede == pytest.approx(0.25)  # floored from 0
    assert form_rating(5, recent, {Competition.PL: 1.5}) is None


def test_form_floors_attack_so_combine_stays_positive():
    f = form_rating(1, [rec(1, 9, 0, 1, day=1)], {Competition.PL: 1.5})
    assert f.attack == 0.25 and f.concede == pytest.approx(1 / 1.5)
    c = combine(TeamRating(attack=1.0, concede=1.0), f)
    assert c.attack > 0 and c.attack == pytest.approx(0.25**0.3)


def test_form_floor_is_a_keyword():
    f = form_rating(1, [rec(1, 9, 0, 1, day=1)], {Competition.PL: 1.5}, floor=0.5)
    assert f.attack == 0.5


def test_combine_geometric():
    c = combine(TeamRating(attack=1.1, concede=1.0), TeamRating(attack=1.4, concede=1.0))
    assert c.attack == pytest.approx(1.18253, abs=1e-5)
    assert combine(TeamRating(attack=1.1, concede=1.0), None).attack == 1.1


def test_combine_keeps_season_elo():
    c = combine(TeamRating(attack=1, concede=1, elo=1700), TeamRating(attack=1.2, concede=0.9, elo=1500))
    assert c.elo == 1700


def test_league_base_blends_home_away_goals():
    b = league_base(CUR, last=CUR)
    assert (b.home_goals, b.away_goals) == (pytest.approx(1.5), pytest.approx(1.5))


def test_league_base_weights_current_by_matches_per_team():
    b = league_base(CUR, last=[rec(1, 2, 4, 0, day=1)], full=4)  # 2 matches per team -> weight 0.5
    assert (b.home_goals, b.away_goals) == (pytest.approx(2.75), pytest.approx(0.75))


def test_league_base_with_one_empty_season_and_with_none():
    assert league_base(CUR, last=[]).home_goals == pytest.approx(1.5)
    assert league_base([], last=CUR).away_goals == pytest.approx(1.5)
    with pytest.raises(ValueError):
        league_base([], last=[])
