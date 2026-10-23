from datetime import UTC, datetime, timedelta

import pytest

from predict.domain import Prediction, ProjectedGoals
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
