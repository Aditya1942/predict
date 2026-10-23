from datetime import UTC, datetime, timedelta

import httpx
import pytest

from predict.errors import SourceError
from predict.http import CachedHttp
from predict.store import Store

T0 = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
TTL = timedelta(hours=1)


class FakeClock:
    def __init__(self) -> None:
        self.t = T0

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kwargs) -> None:
        self.t += timedelta(**kwargs)


class Harness:
    def __init__(self, status: int = 200, body: str = '{"ok": true}') -> None:
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []
        self.status = status
        self.body = body
        self.clock = FakeClock()
        self.store = Store(":memory:")
        client = httpx.Client(transport=httpx.MockTransport(self._handle))
        self.http = CachedHttp(self.store, client, now=self.clock, sleep=self.sleeps.append)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, text=self.body)


@pytest.fixture
def h():
    harness = Harness()
    yield harness
    harness.store.close()


def test_cache_hit_within_ttl_makes_no_request(h):
    assert h.http.get_text("https://a.test/x", ttl=TTL) == '{"ok": true}'
    h.clock.advance(minutes=59)
    assert h.http.get_text("https://a.test/x", ttl=TTL) == '{"ok": true}'
    assert len(h.requests) == 1


def test_expired_entry_refetches(h):
    h.http.get_text("https://a.test/x", ttl=TTL)
    h.clock.advance(hours=1, seconds=1)
    h.body = "second"
    assert h.http.get_text("https://a.test/x", ttl=TTL) == "second"
    assert len(h.requests) == 2


def test_zero_ttl_always_refetches_but_still_stores_latest(h):
    h.http.get_text("https://a.test/x", ttl=timedelta(0))
    h.http.get_text("https://a.test/x", ttl=timedelta(0))
    assert len(h.requests) == 2
    assert h.store.cache_latest("https://a.test/x") == ('{"ok": true}', T0)


def test_get_json_parses(h):
    assert h.http.get_json("https://a.test/x", ttl=TTL) == {"ok": True}


def test_get_json_rejects_invalid_json_without_caching():
    h = Harness(body="<html>maintenance</html>")
    with pytest.raises(SourceError) as e:
        h.http.get_json("https://api.test/x", ttl=TTL)
    assert str(e.value) == "api.test: invalid JSON"
    assert h.store.cache_latest("https://api.test/x") is None
    h.body = '{"ok": true}'
    assert h.http.get_json("https://api.test/x", ttl=TTL) == {"ok": True}
    assert len(h.requests) == 2
    h.store.close()


def test_get_json_rejects_invalid_json_already_cached_by_get_text(h):
    h.body = "not json"
    assert h.http.get_text("https://a.test/x", ttl=TTL) == "not json"
    with pytest.raises(SourceError, match="invalid JSON"):
        h.http.get_json("https://a.test/x", ttl=TTL)


def test_non_2xx_raises_source_error_and_is_not_cached():
    h = Harness(status=502, body="bad gateway")
    with pytest.raises(SourceError, match="502") as e:
        h.http.get_text("https://api.test/x", ttl=TTL)
    assert e.value.source == "api.test"
    assert str(e.value) == "api.test: HTTP 502"
    assert h.store.cache_latest("https://api.test/x") is None
    h.store.close()


def test_transport_failure_raises_source_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    store = Store(":memory:")
    http = CachedHttp(store, httpx.Client(transport=httpx.MockTransport(boom)), now=FakeClock(), sleep=lambda s: None)
    with pytest.raises(SourceError, match="api.test: .*no route"):
        http.get_text("https://api.test/x", ttl=TTL)
    store.close()


def test_params_are_sent_and_sorted_into_the_key(h):
    h.http.get_text("https://a.test/x", ttl=TTL, params={"b": "2", "a": "1"})
    h.http.get_text("https://a.test/x", ttl=TTL, params={"a": "1", "b": "2"})
    assert len(h.requests) == 1
    assert h.requests[0].url.params["a"] == "1"
    h.http.get_text("https://a.test/x", ttl=TTL, params={"a": "1", "b": "3"})
    assert len(h.requests) == 2


def test_api_key_is_sent_but_never_stored_in_a_key(h):
    h.http.get_text("https://a.test/x", ttl=TTL, params={"q": "1", "apiKey": "SECRET"})
    assert h.requests[0].url.params["apiKey"] == "SECRET"
    keys = [r[0] for r in h.store._conn.execute("SELECT key FROM http_cache")]
    assert len(keys) == 1
    assert "SECRET" not in keys[0] and "apiKey" not in keys[0]
    h.http.get_text("https://a.test/x", ttl=TTL, params={"q": "1", "apiKey": "OTHER"})
    assert len(h.requests) == 1


def test_header_values_are_sent_but_not_part_of_the_key(h):
    h.http.get_text("https://a.test/x", ttl=TTL, headers={"X-Auth-Token": "TOKEN1"})
    assert h.requests[0].headers["X-Auth-Token"] == "TOKEN1"
    h.http.get_text("https://a.test/x", ttl=TTL, headers={"X-Auth-Token": "TOKEN2"})
    assert len(h.requests) == 1
    keys = [r[0] for r in h.store._conn.execute("SELECT key FROM http_cache")]
    assert all("TOKEN" not in k for k in keys)


def test_min_interval_sleeps_between_real_requests_to_one_host(h):
    h.http.get_text("https://a.test/one", ttl=TTL, min_interval=6.5)
    assert h.sleeps == []
    h.http.get_text("https://a.test/two", ttl=TTL, min_interval=6.5)
    assert h.sleeps == [pytest.approx(6.5)]


def test_min_interval_only_sleeps_for_the_remaining_time(h):
    h.http.get_text("https://a.test/one", ttl=TTL, min_interval=6.5)
    h.clock.advance(seconds=2.5)
    h.http.get_text("https://a.test/two", ttl=TTL, min_interval=6.5)
    assert h.sleeps == [pytest.approx(4.0)]
    h.clock.advance(seconds=10)
    h.http.get_text("https://a.test/three", ttl=TTL, min_interval=6.5)
    assert len(h.sleeps) == 1


def test_min_interval_ignores_cache_hits_and_other_hosts(h):
    h.http.get_text("https://a.test/one", ttl=TTL, min_interval=6.5)
    h.http.get_text("https://a.test/one", ttl=TTL, min_interval=6.5)
    h.http.get_text("https://b.test/one", ttl=TTL, min_interval=6.5)
    assert h.sleeps == []
    assert len(h.requests) == 2


def test_no_min_interval_never_sleeps(h):
    h.http.get_text("https://a.test/one", ttl=TTL)
    h.http.get_text("https://a.test/two", ttl=TTL)
    assert h.sleeps == []
