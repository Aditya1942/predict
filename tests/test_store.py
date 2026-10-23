from datetime import UTC, datetime, timedelta

import pytest

from predict.domain import Prediction, ProjectedGoals, Result
from predict.store import Store
from tests.helpers import make_match, make_team

T0 = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def test_cache_miss_is_none(store):
    assert store.cache_get("k", timedelta(hours=1), T0) is None
    assert store.cache_latest("k") is None


def test_cache_hit_within_max_age_and_miss_after(store):
    store.cache_put("k", "body", T0)
    assert store.cache_get("k", timedelta(hours=1), T0 + timedelta(minutes=59)) == "body"
    assert store.cache_get("k", timedelta(hours=1), T0 + timedelta(hours=1)) is None


def test_zero_max_age_never_hits(store):
    store.cache_put("k", "body", T0)
    assert store.cache_get("k", timedelta(0), T0) is None


def test_put_overwrites_and_latest_ignores_age(store):
    store.cache_put("k", "old", T0)
    store.cache_put("k", "new", T0 + timedelta(days=1))
    assert store.cache_latest("k") == ("new", T0 + timedelta(days=1))
    assert store.cache_get("k", timedelta(hours=1), T0 + timedelta(days=400)) is None
    assert store.cache_latest("k") == ("new", T0 + timedelta(days=1))


def test_latest_returns_aware_datetime(store):
    store.cache_put("k", "body", T0)
    _, fetched_at = store.cache_latest("k")
    assert fetched_at.tzinfo is not None
    assert fetched_at == T0


def test_file_store_persists_and_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "predict.db"
    first = Store(path)
    first.cache_put("k", "body", T0)
    first.close()
    second = Store(path)
    assert second.cache_latest("k") == ("body", T0)
    second.close()


def prediction(*, match_id: int = 7, created: datetime = T0, shared: datetime | None = None) -> Prediction:
    projected = ProjectedGoals(home=1.6, away=1.1)
    return Prediction(
        match=make_match(make_team(1, "Arsenal"), make_team(2, "Chelsea"), kickoff=T0 + timedelta(days=2), id=match_id),
        created_at=created,
        projected_base=projected,
        projected=projected,
        probabilities={"result": {"home": 0.5, "draw": 0.25, "away": 0.25}},
        shared_at=shared,
    )


def test_save_prediction_assigns_ids_and_round_trips(store):
    first, second = prediction(), prediction(created=T0 + timedelta(hours=1))
    id1, id2 = store.save_prediction(first), store.save_prediction(second)
    assert id1 != id2
    saved = [first.model_copy(update={"id": id1}), second.model_copy(update={"id": id2})]
    assert store.predictions_for_match(7) == saved


def test_predictions_are_grouped_by_match_and_all_listed_in_order(store):
    a = store.save_prediction(prediction(match_id=7))
    b = store.save_prediction(prediction(match_id=8))
    c = store.save_prediction(prediction(match_id=7))
    assert [p.id for p in store.predictions_for_match(7)] == [a, c]
    assert [p.id for p in store.predictions_for_match(8)] == [b]
    assert store.predictions_for_match(99) == []
    assert [p.id for p in store.all_predictions()] == [a, b, c]


def test_update_prediction_rewrites_the_row(store):
    saved = prediction().model_copy(update={"id": store.save_prediction(prediction())})
    shared = saved.model_copy(update={"shared_at": T0 + timedelta(minutes=5)})
    store.update_prediction(shared)
    assert store.all_predictions() == [shared]


def test_update_prediction_needs_a_saved_prediction(store):
    with pytest.raises(ValueError, match="saved"):
        store.update_prediction(prediction())
    unknown = prediction().model_copy(update={"id": 99})
    with pytest.raises(ValueError, match="saved"):
        store.update_prediction(unknown)


def make_dossier(created_at, model="claude-opus-5-5"):
    from predict.domain import MatchDossier, NewsDossier, NewsFactor

    factor = NewsFactor(
        kind="absence",
        player="Saka",
        summary="Out",
        misses_match="yes",
        matches_missed_of_last_6=2,
        sources=["https://example.com"],
    )
    return MatchDossier(
        home=NewsDossier(team="Arsenal", factors=[factor], lineup_confirmed=False),
        away=NewsDossier(team="Chelsea", factors=[], lineup_confirmed=True),
        created_at=created_at,
        claude_model=model,
    )


def test_latest_dossier_is_none_without_one(store):
    assert store.latest_dossier(7) is None


def test_dossier_round_trips_and_the_latest_wins(store):
    older, newer = make_dossier(T0), make_dossier(T0 + timedelta(hours=7), model="other")
    store.save_dossier(7, older)
    assert store.latest_dossier(7) == older
    store.save_dossier(7, newer)
    assert store.latest_dossier(7) == newer
    assert store.latest_dossier(8) is None


# --- results -----------------------------------------------------------------------------------------------------


def test_results_start_empty(store):
    assert store.results() == {}


def test_a_finished_result_round_trips(store):
    store.save_result(7, "finished", Result(home_goals=2, away_goals=1), T0)
    assert store.results() == {7: ("finished", Result(home_goals=2, away_goals=1))}


def test_a_void_result_has_no_score(store):
    store.save_result(7, "void", None, T0)
    assert store.results() == {7: ("void", None)}


def test_a_result_for_a_match_replaces_the_earlier_one(store):
    store.save_result(7, "void", None, T0)
    store.save_result(7, "finished", Result(home_goals=0, away_goals=0), T0 + timedelta(days=1))
    assert store.results() == {7: ("finished", Result(home_goals=0, away_goals=0))}


def test_a_finished_result_needs_a_score_and_a_void_one_must_not_have_one(store):
    with pytest.raises(ValueError, match="finished"):
        store.save_result(7, "finished", None, T0)
    with pytest.raises(ValueError, match="void"):
        store.save_result(7, "void", Result(home_goals=1, away_goals=0), T0)
    assert store.results() == {}


def test_unsettled_match_ids_are_predicted_matches_that_kicked_off_with_no_result(store):
    after = T0 + timedelta(days=3)  # the Matches kick off at T0 + 2 days
    store.save_prediction(prediction(match_id=1))
    store.save_prediction(prediction(match_id=1))  # two Predictions, one Match
    store.save_prediction(prediction(match_id=2))
    store.save_prediction(prediction(match_id=3))
    assert store.unsettled_match_ids(after) == [1, 2, 3]
    store.save_result(2, "void", None, T0)
    assert store.unsettled_match_ids(after) == [1, 3]


def test_unsettled_match_ids_leave_out_matches_that_have_not_kicked_off_before_the_cutoff(store):
    store.save_prediction(prediction(match_id=1))
    kickoff = store.all_predictions()[0].match.kickoff
    assert store.unsettled_match_ids(kickoff) == []  # strictly before
    assert store.unsettled_match_ids(kickoff + timedelta(seconds=1)) == [1]
