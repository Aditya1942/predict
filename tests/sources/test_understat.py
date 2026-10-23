import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from predict.domain import Competition
from predict.errors import SourceError
from predict.http import CachedHttp
from predict.sources.understat import Understat
from predict.store import Store

FIXTURES = Path(__file__).parent.parent / "fixtures"
NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)


class Rig:
    def __init__(self, payload: object) -> None:
        self.requests: list[httpx.Request] = []
        self._payload = payload
        self.store = Store(":memory:")
        client = httpx.Client(transport=httpx.MockTransport(self._handle))
        self.understat = Understat(CachedHttp(self.store, client, now=lambda: NOW, sleep=lambda s: None))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self._payload)


@pytest.fixture
def make_rig():
    rigs: list[Rig] = []

    def make(payload: object) -> Rig:
        rig = Rig(payload)
        rigs.append(rig)
        return rig

    yield make
    for rig in rigs:
        rig.store.close()


def fixture() -> dict:
    return json.loads((FIXTURES / "understat_epl.json").read_text())


def test_only_played_matches_become_records(make_rig):
    rig = make_rig(fixture())
    records = rig.understat.league_matches(Competition.PL, 2026)
    assert [(r.home_name, r.away_name) for r in records] == [("Arsenal", "Coventry"), ("Hull", "Manchester United")]


def test_goals_are_ints_and_xg_floats(make_rig):
    first = make_rig(fixture()).understat.league_matches(Competition.PL, 2026)[0]
    assert (first.home_goals, first.away_goals) == (3, 0)
    assert isinstance(first.home_goals, int)
    assert first.home_xg == pytest.approx(1.85424)
    assert first.away_xg == pytest.approx(0.558336)
    assert isinstance(first.home_xg, float)
    assert first.competition is Competition.PL
    assert first.home_id is None and first.away_id is None


def test_kickoff_is_aware_utc(make_rig):
    records = make_rig(fixture()).understat.league_matches(Competition.PL, 2026)
    assert records[0].kickoff == datetime(2026, 8, 21, 19, 0, tzinfo=UTC)
    assert records[1].kickoff == datetime(2026, 8, 22, 11, 30, tzinfo=UTC)
    assert records[0].kickoff.utcoffset().total_seconds() == 0


def test_requests_league_endpoint_with_ajax_header(make_rig):
    rig = make_rig(fixture())
    rig.understat.league_matches(Competition.PL, 2026)
    request = rig.requests[0]
    assert str(request.url) == "https://understat.com/getLeagueData/EPL/2026"
    assert request.headers["X-Requested-With"] == "XMLHttpRequest"


def test_second_call_is_served_from_cache(make_rig):
    rig = make_rig(fixture())
    rig.understat.league_matches(Competition.PL, 2026)
    rig.understat.league_matches(Competition.PL, 2026)
    assert len(rig.requests) == 1


def test_champions_league_has_no_understat_data(make_rig):
    rig = make_rig(fixture())
    with pytest.raises(SourceError, match="Champions League"):
        rig.understat.league_matches(Competition.CL, 2026)
    assert rig.requests == []


@pytest.mark.parametrize("payload", [[], {"dates": "nope"}, {"nodates": []}, {"dates": [{"isResult": True}]}])
def test_unexpected_shapes_raise_source_error(make_rig, payload):
    with pytest.raises(SourceError, match="understat"):
        make_rig(payload).understat.league_matches(Competition.PL, 2026)
