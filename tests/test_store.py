from datetime import UTC, datetime, timedelta

import pytest

from predict.store import Store

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
