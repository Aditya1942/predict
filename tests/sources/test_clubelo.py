from datetime import UTC, date, datetime

import httpx
import pytest

from predict.errors import SourceError
from predict.http import CachedHttp
from predict.sources.clubelo import ClubElo, ClubEloRow
from predict.store import Store

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
DAY = date(2026, 10, 20)

CSV = """Rank,Club,Country,Level,Elo,From,To
1,Man City,ENG,1,2050.5,2026-10-15,2026-10-21
2,Arsenal,ENG,1,2000.5,2026-10-16,2026-10-22
3,Real Madrid,ESP,1,1990.0,2026-10-14,2026-10-20
None,Leeds,ENG,2,1700.0,2026-10-12,2026-10-25
None,Brighton,ENG,1,1850.0,2026-10-12,2026-10-25
"""


class Rig:
    def __init__(self, respond) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond
        self.store = Store(":memory:")
        client = httpx.Client(transport=httpx.MockTransport(self._handle))
        self.http = CachedHttp(self.store, client, now=lambda: NOW, sleep=lambda s: None)
        self.clubelo = ClubElo(self.http, self.store, now=lambda: NOW)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond(request)


@pytest.fixture
def make_rig():
    rigs: list[Rig] = []

    def make(respond) -> Rig:
        rig = Rig(respond)
        rigs.append(rig)
        return rig

    yield make
    for rig in rigs:
        rig.store.close()


def ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=CSV)


def down(request: httpx.Request) -> httpx.Response:
    return httpx.Response(502, text="Bad Gateway")


def test_parses_csv_keyed_by_club(make_rig):
    rig = make_rig(ok)
    rows = rig.clubelo.ratings(DAY)
    assert set(rows) == {"Man City", "Arsenal", "Real Madrid", "Leeds", "Brighton"}
    assert rows["Man City"] == ClubEloRow(club="Man City", country="ENG", level=1, elo=2050.5)
    assert isinstance(rows["Man City"].level, int)
    assert rows["Leeds"].level == 2
    assert str(rig.requests[0].url) == "http://api.clubelo.com/2026-10-20"


def test_league_elo_is_the_mean_of_level_one_rows_of_the_country(make_rig):
    rows = make_rig(ok).clubelo.ratings(DAY)
    # Man City, Arsenal and Brighton; Leeds is level 2 and Real Madrid is Spanish
    assert ClubElo.league_elo(rows, "ENG") == pytest.approx((2050.5 + 2000.5 + 1850.0) / 3)
    assert ClubElo.league_elo(rows, "ESP") == pytest.approx(1990.0)


def test_league_elo_without_clubs_raises(make_rig):
    rows = make_rig(ok).clubelo.ratings(DAY)
    with pytest.raises(SourceError, match="GER"):
        ClubElo.league_elo(rows, "GER")


def test_ratings_are_cached_for_a_day(make_rig):
    rig = make_rig(ok)
    rig.clubelo.ratings(DAY)
    rig.clubelo.ratings(DAY)
    assert len(rig.requests) == 1


def test_success_saves_the_body_as_latest(make_rig):
    rig = make_rig(ok)
    rig.clubelo.ratings(DAY)
    assert rig.store.cache_latest("clubelo:latest") == (CSV, NOW)


def test_http_failure_falls_back_to_saved_ratings(make_rig):
    healthy = True

    def respond(request: httpx.Request) -> httpx.Response:
        return ok(request) if healthy else down(request)

    rig = make_rig(respond)
    fresh = rig.clubelo.ratings(DAY)
    healthy = False
    assert rig.clubelo.ratings(date(2026, 10, 21)) == fresh  # a date that is not cached
    assert len(rig.requests) == 2


def test_http_failure_with_nothing_saved_raises(make_rig):
    with pytest.raises(SourceError, match="clubelo: unavailable and no saved ratings") as exc:
        make_rig(down).clubelo.ratings(DAY)
    assert exc.value.source == "clubelo"


def test_a_failed_fetch_does_not_overwrite_the_saved_copy(make_rig):
    rig = make_rig(down)
    rig.store.cache_put("clubelo:latest", CSV, datetime(2026, 10, 1, tzinfo=UTC))
    rig.clubelo.ratings(DAY)
    assert rig.store.cache_latest("clubelo:latest") == (CSV, datetime(2026, 10, 1, tzinfo=UTC))


@pytest.mark.parametrize(
    "body",
    ["<html>maintenance</html>", "", "Rank,Club,Country,Level,Elo,From,To\n1,Arsenal,ENG,one,2000,a,b\n"],
)
def test_unusable_body_counts_as_a_failure_and_is_not_saved(make_rig, body):
    rig = make_rig(lambda request: httpx.Response(200, text=body))
    with pytest.raises(SourceError, match="no saved ratings"):
        rig.clubelo.ratings(DAY)
    assert rig.store.cache_latest("clubelo:latest") is None


def test_unusable_body_falls_back_to_saved_ratings(make_rig):
    rig = make_rig(lambda request: httpx.Response(200, text="<html>maintenance</html>"))
    rig.store.cache_put("clubelo:latest", CSV, NOW)
    assert "Arsenal" in rig.clubelo.ratings(DAY)
