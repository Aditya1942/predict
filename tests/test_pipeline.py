from datetime import UTC, datetime, timedelta

import pytest

from predict.domain import MARKET_IDS, Competition, Prediction, ProjectedGoals
from predict.errors import JevFailed, MatchNotFound, PredictionLocked, ResearchFailed, SourceError, TeamNotFound
from predict.model.markets import market_probabilities
from predict.model.scoreline import scoreline_matrix
from predict.news.adjust import apply_impacts
from predict.pipeline import Predictor
from predict.sources.odds_api import OddsSnapshot
from predict.store import Store
from predict.teams import Aliases, TeamResolver
from tests.helpers import (
    FakeDossiers,
    FakeFdo,
    FakeOdds,
    StubJev,
    StubRater,
    make_dossier,
    make_factor,
    make_match,
    make_team,
)

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


CLAUDE = "claude-opus-5-5"
JEV = "jev-1.13.0"


def predictor(matches, store, *, rater=None, now=NOW, odds=None, dossiers=None, jev=None) -> Predictor:
    return Predictor(
        FakeFdo(matches),
        rater or StubRater(),
        TeamResolver(TEAMS, Aliases.load()),
        store,
        odds,
        dossiers or FakeDossiers(make_dossier(created_at=NOW, model=CLAUDE)),
        jev,
        CLAUDE,
        JEV,
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
    resolver = TeamResolver(TEAMS, Aliases.load())
    p = Predictor(fdo, StubRater(), resolver, store, None, FakeDossiers(), None, CLAUDE, JEV, now=lambda: NOW)
    p.find_match("arsenal", never)
    assert fdo.upcoming_calls == [(ARSENAL.id, NOW.date(), NOW.date() + timedelta(days=60))]


def test_find_match_skips_matches_that_have_already_kicked_off(store):
    """The upcoming list can be cached from before kickoff, so it may still hold a Match that has started."""
    started = make_match(ARSENAL, CHELSEA, kickoff=NOW - timedelta(minutes=1), id=1)
    starting_now = make_match(ARSENAL, EVERTON, kickoff=NOW, id=2)
    next_up = make_match(ARSENAL, FULHAM, kickoff=days(3), id=3)
    p = predictor([started, starting_now, next_up], store)
    assert p.find_match("arsenal", never) == next_up
    with pytest.raises(MatchNotFound):
        p.find_match("Arsenal v Chelsea", never)
    with pytest.raises(MatchNotFound):
        p.find_match("Arsenal v Everton", never)


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


BOOK = {
    "result": {"home": 0.48, "draw": 0.28, "away": 0.24},
    "btts": {"yes": 0.55, "no": 0.45},
}


def book_snapshot(credits: int | None = 480) -> OddsSnapshot:
    return OddsSnapshot(probabilities=BOOK, credits_remaining=credits, fetched_at=NOW)


def test_predict_adds_bookmaker_probabilities_hot_takes_and_credits(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3), id=42)
    odds = FakeOdds(book_snapshot())
    saved = predictor([match], store, rater=StubRater(ProjectedGoals(home=1.9, away=0.9)), odds=odds).predict(match)

    assert odds.asked == [match]
    assert saved.bookmaker == BOOK
    assert saved.credits_remaining == 480
    assert saved.odds_note is None
    assert [(t.market, t.answer) for t in saved.hot_takes] == [("result", "home")]
    take = saved.hot_takes[0]
    assert take.model_p == saved.probabilities["result"]["home"]
    assert take.book_p == 0.48
    assert store.predictions_for_match(42) == [saved]  # the extras are saved with the Prediction


def test_predict_without_an_odds_source_says_no_key(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3))
    saved = predictor([match], store).predict(match)
    assert saved.bookmaker is None
    assert saved.hot_takes == []
    assert saved.odds_note == "no ODDS_API_KEY set"
    assert saved.credits_remaining is None


def test_odds_failing_still_saves_a_prediction_with_the_reason(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3), id=42)
    odds = FakeOdds(unavailable="odds quota used up or key invalid (HTTP 429)")
    saved = predictor([match], store, odds=odds).predict(match)
    assert saved.id is not None
    assert saved.bookmaker is None
    assert saved.hot_takes == []
    assert saved.odds_note == "odds quota used up or key invalid (HTTP 429)"
    assert set(saved.probabilities) == set(MARKET_IDS)
    assert store.predictions_for_match(42) == [saved]


def test_locked_match_does_not_spend_odds_credits(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=NOW)
    odds = FakeOdds(book_snapshot())
    with pytest.raises(PredictionLocked):
        predictor([match], store, odds=odds).predict(match)
    assert odds.asked == []


# --- Team News --------------------------------------------------------------------------------------------------------


def news_dossier(**changes):
    return make_dossier(created_at=NOW - timedelta(hours=1), model=CLAUDE, **changes)


SAKA_OUT = {"home": (make_factor("absence", "Saka", misses="yes", sources=("https://example.com/saka",)),)}
MAJOR = [0.0, 0.0, 0.0, 1.0]
SAKA_MAJOR = {"home0_attack": MAJOR, "home0_defence": MAJOR}


def test_news_moves_the_projected_goals_and_is_saved(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3), id=42)
    dossiers = FakeDossiers(news_dossier(**SAKA_OUT))
    p = predictor(
        [match], store, rater=StubRater(ProjectedGoals(home=1.9, away=0.9)), dossiers=dossiers, jev=StubJev(SAKA_MAJOR)
    )
    saved = p.predict(match)

    assert saved.news_applied is True
    assert saved.news_note is None
    assert [(i.side, i.factor.player) for i in saved.impacts] == [("home", "Saka")]
    assert saved.projected_base == ProjectedGoals(home=1.9, away=0.9)
    assert saved.projected == apply_impacts(saved.projected_base, saved.impacts)
    assert saved.projected.home < saved.projected_base.home
    assert saved.projected.away > saved.projected_base.away
    adjusted = market_probabilities(scoreline_matrix(saved.projected.home, saved.projected.away))
    assert saved.probabilities == adjusted
    assert (saved.claude_model, saved.jev_model) == (CLAUDE, JEV)
    assert store.predictions_for_match(42) == [saved]  # impacts and the news fields are saved with it


def test_news_with_nothing_to_score_is_still_applied(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3))
    jev = StubJev()
    saved = predictor([match], store, dossiers=FakeDossiers(news_dossier()), jev=jev).predict(match)
    assert saved.news_applied is True
    assert saved.impacts == []
    assert saved.projected == saved.projected_base
    assert jev.states == []  # no factors, so Jev is not asked


def test_the_dossier_is_asked_for_with_fresh_and_the_chosen_model(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3))
    dossiers = FakeDossiers(news_dossier())
    p = predictor([match], store, dossiers=dossiers, jev=StubJev())
    p.predict(match)
    p.predict(match, fresh=True, model="claude-sonnet-5-5")
    assert dossiers.calls == [(match, False, CLAUDE), (match, True, "claude-sonnet-5-5")]


def test_jev_is_told_todays_date_not_the_dossiers(store):
    """A reused dossier can be from before midnight."""
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3))
    dossier = make_dossier(created_at=NOW - timedelta(hours=13), model=CLAUDE, **SAKA_OUT)  # 21 Oct 23:00
    jev = StubJev()
    predictor([match], store, dossiers=FakeDossiers(dossier), jev=jev).predict(match)
    assert jev.states[0]["match"]["today"] == "2026-10-22"


@pytest.mark.parametrize(
    "failure",
    [
        ResearchFailed("Claude research failed: no result message came back"),
        SourceError("fpl", "boom"),
    ],
)
def test_research_failing_falls_back_to_the_stats_model(store, failure):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3), id=42)
    saved = predictor([match], store, dossiers=FakeDossiers(error=failure), jev=StubJev()).predict(match)
    assert saved.id is not None
    assert saved.news_applied is False
    assert saved.news_note == (failure.reason if isinstance(failure, ResearchFailed) else str(failure))
    assert "--model" not in saved.news_note
    assert saved.impacts == []
    assert saved.projected == saved.projected_base
    for market_id, answers in saved.probabilities.items():
        assert sum(answers.values()) == pytest.approx(1.0), market_id
    assert store.predictions_for_match(42) == [saved]


def test_a_usage_limit_suggests_a_cheaper_model(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3))
    dossiers = FakeDossiers(error=ResearchFailed("Claude research failed: usage limit reached"))
    saved = predictor([match], store, dossiers=dossiers, jev=StubJev()).predict(match)
    assert saved.news_applied is False
    assert "usage limit reached" in saved.news_note
    assert "rerun with --model claude-sonnet-5-5" in saved.news_note
    assert set(saved.probabilities) == set(MARKET_IDS)


def test_jev_failing_falls_back_to_the_stats_model(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3), id=42)
    jev = StubJev(error=JevFailed("Jev request failed: HTTP 500"))
    saved = predictor([match], store, dossiers=FakeDossiers(news_dossier(**SAKA_OUT)), jev=jev).predict(match)
    assert saved.news_applied is False
    assert saved.news_note == "Jev request failed: HTTP 500"
    assert saved.impacts == []
    assert saved.projected == saved.projected_base
    assert (saved.claude_model, saved.jev_model) == (None, None)
    assert store.predictions_for_match(42) == [saved]


def test_without_a_jev_key_news_is_not_applied_and_claude_is_not_run(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3))
    dossiers = FakeDossiers(news_dossier(**SAKA_OUT))
    saved = predictor([match], store, dossiers=dossiers, jev=None).predict(match)
    assert saved.news_applied is False
    assert saved.news_note == "no TYPESAFE_API_KEY set"
    assert dossiers.calls == []


def test_locked_match_does_not_research(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=NOW)
    dossiers = FakeDossiers(news_dossier())
    with pytest.raises(PredictionLocked):
        predictor([match], store, dossiers=dossiers, jev=StubJev()).predict(match)
    assert dossiers.calls == []


def test_share_marks_the_prediction_shared_now(store):
    match = make_match(ARSENAL, CHELSEA, kickoff=days(3), id=42)
    p = predictor([match], store)
    assert p.predict(match).shared_at is None
    shared = p.predict(match, share=True)
    assert shared.shared_at == NOW
    assert store.predictions_for_match(42)[1].shared_at == NOW
