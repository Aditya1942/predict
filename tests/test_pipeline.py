from datetime import UTC, datetime, timedelta

import pytest

from predict.domain import MARKET_IDS, Competition, Prediction, ProjectedGoals
from predict.errors import MatchNotFound, PredictionLocked, TeamNotFound
from predict.pipeline import Predictor
from predict.store import Store
from predict.teams import Aliases, TeamResolver
from tests.helpers import FakeFdo, StubRater, make_match, make_team

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)

ARSENAL = make_team(1, "Arsenal")
CHELSEA = make_team(2, "Chelsea")
EVERTON = make_team(3, "Everton")
FULHAM = make_team(4, "Fulham")
BENFICA = make_team(5, "Benfica")
REAL_MADRID = make_team(10, "Real Madrid CF")
REAL_BETIS = make_team(11, "Real Betis Balompié")
REAL_SOCIEDAD = make_team(12, "Real Sociedad de Fútbol")
TEAMS = [ARSENAL, CHELSEA, EVERTON, FULHAM, BENFICA, REAL_MADRID, REAL_BETIS, REAL_SOCIEDAD]


def days(n: float) -> datetime:
    return NOW + timedelta(days=n)


def never(candidates):
    raise AssertionError(f"unexpected ambiguity: {candidates}")


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def predictor(matches, store, *, rater=None, now=NOW) -> Predictor:
    return Predictor(
        FakeFdo(matches),
        rater or StubRater(),
        TeamResolver(TEAMS, Aliases.load()),
        store,
        now=lambda: now,
    )


def test_find_match_either_home_away_order(store):
    soon = make_match(CHELSEA, ARSENAL, kickoff=days(3), id=7)
    later = make_match(ARSENAL, CHELSEA, kickoff=days(30), id=8)
    other = make_match(ARSENAL, EVERTON, kickoff=days(1), id=9)
    found = predictor([later, other, soon], store).find_match("Arsenal v Chelsea", never)
    assert found == soon


@pytest.mark.parametrize("query", ["Arsenal vs Chelsea", "arsenal - chelsea", "Chelsea v. Arsenal"])
def test_find_match_other_separators(store, query):
    match = make_match(CHELSEA, ARSENAL, kickoff=days(3))
    assert predictor([match], store).find_match(query, never) == match


def test_find_match_one_team_returns_earliest(store):
    first = make_match(EVERTON, ARSENAL, kickoff=days(2), id=1)
    second = make_match(ARSENAL, CHELSEA, kickoff=days(9), id=2)
    cl = make_match(BENFICA, ARSENAL, kickoff=days(5), id=3, competition=Competition.CL)
    assert predictor([second, cl, first], store).find_match("arsenal", never) == first


def test_find_match_looks_from_today_for_the_search_window(store):
    fdo = FakeFdo([make_match(ARSENAL, CHELSEA, kickoff=days(3))])
    p = Predictor(fdo, StubRater(), TeamResolver(TEAMS, Aliases.load()), store, now=lambda: NOW)
    p.find_match("arsenal", never)
    assert fdo.upcoming_calls == [(ARSENAL.id, NOW.date(), NOW.date() + timedelta(days=60))]


def test_find_match_outside_window_is_not_found(store):
    far = make_match(ARSENAL, CHELSEA, kickoff=days(61))
    with pytest.raises(MatchNotFound) as caught:
        predictor([far], store).find_match("arsenal", never)
    assert "Arsenal" in str(caught.value)
    assert "60 days" in str(caught.value)


def test_find_match_two_teams_without_a_meeting_is_not_found(store):
    elsewhere = [make_match(ARSENAL, EVERTON, kickoff=days(1)), make_match(CHELSEA, FULHAM, kickoff=days(2))]
    with pytest.raises(MatchNotFound) as caught:
        predictor(elsewhere, store).find_match("Arsenal v Chelsea", never)
    assert "Arsenal" in str(caught.value) and "Chelsea" in str(caught.value)


def test_find_match_unknown_team_propagates(store):
    with pytest.raises(TeamNotFound):
        predictor([], store).find_match("Zzzzzz", never)


def test_find_match_asks_which_team_when_ambiguous(store):
    asked = []

    def choose(candidates):
        asked.append([t.name for t in candidates])
        return candidates[1]

    match = make_match(REAL_MADRID, ARSENAL, kickoff=days(2))
    found = predictor([match], store).find_match("real", choose)
    assert asked == [["Real Betis Balompié", "Real Madrid CF", "Real Sociedad de Fútbol"]]
    assert found == match


def test_find_match_asks_for_each_ambiguous_team(store):
    match = make_match(REAL_MADRID, REAL_SOCIEDAD, kickoff=days(2))
    picks = iter([REAL_MADRID, REAL_SOCIEDAD])
    found = predictor([match], store).find_match("real v real", lambda candidates: next(picks))
    assert found == match


def test_predict_after_kickoff_is_locked(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(1))
    late = NOW + timedelta(days=1, minutes=1)
    with pytest.raises(PredictionLocked) as caught:
        predictor([match], store, now=late).predict(match)
    assert caught.value.match == match
    assert "locked" in str(caught.value)
    assert store.all_predictions() == []


def test_predict_at_kickoff_is_locked(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=NOW)
    with pytest.raises(PredictionLocked):
        predictor([match], store).predict(match)


def test_predict_saves_a_prediction_whose_markets_each_sum_to_one(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3), id=42)
    rater = StubRater(ProjectedGoals(home=1.9, away=0.9))
    saved = predictor([match], store, rater=rater).predict(match)

    assert rater.rated == [match]
    assert saved.id is not None
    assert saved.match == match
    assert saved.created_at == NOW
    assert saved.shared_at is None
    assert saved.projected == saved.projected_base == ProjectedGoals(home=1.9, away=0.9)
    assert set(saved.probabilities) == set(MARKET_IDS)
    for market_id, answers in saved.probabilities.items():
        assert sum(answers.values()) == pytest.approx(1.0), market_id
    assert saved.probabilities["result"]["home"] > saved.probabilities["result"]["away"]

    assert store.predictions_for_match(42) == [saved]
    assert store.all_predictions() == [saved]


def test_each_run_saves_another_prediction(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3), id=42)
    p = predictor([match], store)
    first, second = p.predict(match), p.predict(match)
    assert first.id != second.id
    assert [x.id for x in store.predictions_for_match(42)] == [first.id, second.id]


def test_prediction_is_a_domain_type(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3))
    assert isinstance(predictor([match], store).predict(match), Prediction)
