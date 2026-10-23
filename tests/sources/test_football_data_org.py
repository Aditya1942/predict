import copy
import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from predict.domain import Competition, Match, MatchRecord, Result
from predict.errors import SourceError
from predict.http import CachedHttp
from predict.sources.football_data_org import FootballDataOrg, MatchStatus, regular_time_score
from predict.store import Store

FIXTURES = Path(__file__).parent.parent / "fixtures"
NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
TOKEN = "test-token"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def serve_by_status(payload: dict) -> Callable[[httpx.Request], dict]:
    """Fake server: honours the `status` query filter like football-data.org does."""

    def respond(request: httpx.Request) -> dict:
        wanted = request.url.params.get("status")
        if wanted is None:
            return payload
        allowed = wanted.split(",")
        return {**payload, "matches": [m for m in payload["matches"] if m["status"] in allowed]}

    return respond


class Rig:
    def __init__(self, respond: Callable[[httpx.Request], dict]) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond
        self.store = Store(":memory:")
        client = httpx.Client(transport=httpx.MockTransport(self._handle))
        http = CachedHttp(self.store, client, now=lambda: NOW, sleep=lambda s: None)
        self.fdo = FootballDataOrg(http, TOKEN)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self._respond(request))


@pytest.fixture
def make_rig():
    rigs: list[Rig] = []

    def make(respond: Callable[[httpx.Request], dict]) -> Rig:
        rig = Rig(respond)
        rigs.append(rig)
        return rig

    yield make
    for rig in rigs:
        rig.store.close()


def test_teams_parses_fields(make_rig):
    rig = make_rig(lambda r: load("fdo_teams_pl.json"))
    teams = rig.fdo.teams(Competition.PL)
    assert [t.id for t in teams] == [57, 58, 61]
    arsenal = teams[0]
    assert (arsenal.name, arsenal.short_name, arsenal.tla, arsenal.country) == (
        "Arsenal FC",
        "Arsenal",
        "ARS",
        "England",
    )
    assert rig.requests[0].url.path == "/v4/competitions/PL/teams"


def test_teams_are_cached_for_a_day(make_rig):
    rig = make_rig(lambda r: load("fdo_teams_pl.json"))
    rig.fdo.teams(Competition.PL)
    rig.fdo.teams(Competition.PL)
    assert len(rig.requests) == 1


def test_team_upcoming_parses_aware_utc_kickoff_and_query(make_rig):
    rig = make_rig(serve_by_status(load("fdo_team_matches.json")))
    matches = rig.fdo.team_upcoming(57, date(2026, 10, 20), date(2026, 11, 20))
    assert [m.id for m in matches] == [538002]
    m = matches[0]
    assert m.kickoff == datetime(2026, 10, 25, 16, 30, tzinfo=UTC)
    assert m.kickoff.utcoffset() == timedelta(0)
    assert (m.competition, m.stage, m.neutral) == (Competition.PL, "REGULAR_SEASON", False)
    assert (m.home.id, m.home.name, m.home.short_name, m.home.tla) == (57, "Arsenal FC", "Arsenal", "ARS")
    assert (m.away.id, m.away.name) == (61, "Chelsea FC")
    request = rig.requests[0]
    assert request.url.path == "/v4/teams/57/matches"
    assert dict(request.url.params) == {
        "status": "SCHEDULED,TIMED",
        "competitions": "PL,PD,BL1,SA,FL1,CL",
        "dateFrom": "2026-10-20",
        "dateTo": "2026-11-20",
    }


def test_team_upcoming_skips_matches_with_an_undecided_opponent(make_rig):
    payload = load("fdo_team_matches.json")
    decided = payload["matches"][1]
    tbd = copy.deepcopy(decided)
    tbd["id"] = 538005
    tbd["awayTeam"] = {"id": None, "name": None, "shortName": None, "tla": None, "crest": None}
    rig = make_rig(lambda r: {"matches": [tbd, decided]})
    assert [m.id for m in rig.fdo.team_upcoming(57, date(2026, 10, 20), date(2026, 11, 20))] == [538002]


def test_team_upcoming_is_sorted_by_kickoff(make_rig):
    payload = load("fdo_team_matches.json")
    later = copy.deepcopy(payload["matches"][1])
    later.update(id=538003, utcDate="2026-11-01T14:00:00Z")
    earlier = copy.deepcopy(payload["matches"][1])
    earlier.update(id=538004, utcDate="2026-10-23T19:00:00Z")
    rig = make_rig(lambda r: {"matches": [later, earlier]})
    matches = rig.fdo.team_upcoming(57, date(2026, 10, 20), date(2026, 11, 20))
    assert [m.id for m in matches] == [538004, 538003]


def test_team_results_uses_finished_status_and_ninety_minute_score(make_rig):
    rig = make_rig(serve_by_status(load("fdo_team_matches.json")))
    records = rig.fdo.team_results(57, date(2026, 9, 1), date(2026, 10, 20))
    assert records == [
        MatchRecord(
            kickoff=datetime(2026, 10, 3, 14, 0, tzinfo=UTC),
            competition=Competition.PL,
            home_name="Arsenal FC",
            away_name="Aston Villa FC",
            home_goals=2,
            away_goals=1,
            home_xg=None,
            away_xg=None,
            home_id=57,
            away_id=58,
        )
    ]
    params = rig.requests[0].url.params
    assert rig.requests[0].url.path == "/v4/teams/57/matches"
    assert params["status"] == "FINISHED"
    assert params["competitions"] == "PL,PD,BL1,SA,FL1,CL"
    assert (params["dateFrom"], params["dateTo"]) == ("2026-09-01", "2026-10-20")


def test_extra_time_match_gives_the_90_minute_score(make_rig):
    rig = make_rig(serve_by_status(load("fdo_extra_time.json")))
    (record,) = rig.fdo.team_results(81, date(2026, 3, 1), date(2026, 3, 20))
    assert record.competition is Competition.CL
    assert (record.home_goals, record.away_goals) == (2, 2)
    assert (record.home_id, record.away_id) == (81, 524)
    assert (record.home_name, record.away_name) == ("FC Barcelona", "Paris Saint-Germain FC")


def test_regular_time_score_regular_uses_full_time():
    score = {"duration": "REGULAR", "fullTime": {"home": 3, "away": 1}, "halfTime": {"home": 1, "away": 0}}
    assert regular_time_score(score) == (3, 1)


def test_regular_time_score_extra_time_uses_regular_time():
    score = load("fdo_extra_time.json")["matches"][0]["score"]
    assert score["fullTime"] == {"home": 3, "away": 2}
    assert regular_time_score(score) == (2, 2)


def test_regular_time_score_penalty_shootout_goalless_is_nil_nil():
    score = {
        "duration": "PENALTY_SHOOTOUT",
        "fullTime": {"home": 5, "away": 4},
        "regularTime": {"home": 0, "away": 0},
        "extraTime": {"home": 0, "away": 0},
        "penalties": {"home": 5, "away": 4},
    }
    assert regular_time_score(score) == (0, 0)


def test_regular_time_score_without_goals_raises():
    with pytest.raises(ValueError, match="score"):
        regular_time_score({"duration": "REGULAR", "fullTime": {"home": None, "away": None}})


@pytest.mark.parametrize(
    ("code", "stage", "neutral"),
    [("CL", "FINAL", True), ("CL", "SEMI_FINALS", False), ("CL", "LEAGUE_STAGE", False), ("PL", "FINAL", False)],
)
def test_neutral_only_for_champions_league_final(make_rig, code, stage, neutral):
    match = copy.deepcopy(load("fdo_team_matches.json")["matches"][1])
    match["competition"]["code"] = code
    match["stage"] = stage
    rig = make_rig(lambda r: {"matches": [match]})
    (parsed,) = rig.fdo.team_upcoming(57, date(2026, 10, 20), date(2026, 11, 20))
    assert parsed.competition == Competition(code)
    assert parsed.stage == stage
    assert parsed.neutral is neutral


def test_every_request_carries_the_auth_token(make_rig):
    rig = make_rig(lambda r: {"matches": [], "teams": []})
    rig.fdo.teams(Competition.PL)
    rig.fdo.team_upcoming(57, date(2026, 10, 20), date(2026, 11, 20))
    rig.fdo.team_results(57, date(2026, 9, 1), date(2026, 10, 20))
    rig.fdo.matches_by_ids([1, 2])
    assert len(rig.requests) == 4
    assert all(r.headers["X-Auth-Token"] == TOKEN for r in rig.requests)
    assert all(r.url.host == "api.football-data.org" for r in rig.requests)


def test_matches_by_ids_maps_postponed_to_no_result_and_finished_to_ninety_minutes(make_rig):
    payload = load("fdo_team_matches.json")
    finished = payload["matches"][0]
    postponed = copy.deepcopy(payload["matches"][1])
    postponed["status"] = "POSTPONED"
    extra_time = load("fdo_extra_time.json")["matches"][0]
    rig = make_rig(lambda r: {"matches": [postponed, finished, extra_time]})
    statuses = rig.fdo.matches_by_ids([538002, 538001, 540001])
    assert rig.requests[0].url.path == "/v4/matches"
    assert rig.requests[0].url.params["ids"] == "538002,538001,540001"
    assert all(isinstance(s, MatchStatus) and isinstance(s.match, Match) for s in statuses)
    by_id = {s.match.id: s for s in statuses}
    assert (by_id[538002].status, by_id[538002].result) == ("POSTPONED", None)
    assert (by_id[538001].status, by_id[538001].result) == ("FINISHED", Result(home_goals=2, away_goals=1))
    assert by_id[540001].result == Result(home_goals=2, away_goals=2)


def test_matches_by_ids_is_never_cached(make_rig):
    rig = make_rig(lambda r: load("fdo_team_matches.json"))
    rig.fdo.matches_by_ids([538001])
    rig.fdo.matches_by_ids([538001])
    assert len(rig.requests) == 2


def test_matches_by_ids_empty_makes_no_request(make_rig):
    rig = make_rig(lambda r: {"matches": []})
    assert rig.fdo.matches_by_ids([]) == []
    assert rig.requests == []


def test_unexpected_payload_raises_source_error(make_rig):
    rig = make_rig(lambda r: {"matches": [{"id": 1}]})
    with pytest.raises(SourceError, match="unexpected response"):
        rig.fdo.team_upcoming(57, date(2026, 10, 20), date(2026, 11, 20))


@pytest.mark.parametrize("payload", [[], "oops", 7, True])
def test_non_object_json_raises_source_error(make_rig, payload):
    rig = make_rig(lambda r: payload)
    with pytest.raises(SourceError, match="unexpected response"):
        rig.fdo.teams(Competition.PL)
    with pytest.raises(SourceError, match="unexpected response"):
        rig.fdo.team_upcoming(57, date(2026, 10, 20), date(2026, 11, 20))
    with pytest.raises(SourceError, match="unexpected response"):
        rig.fdo.team_results(57, date(2026, 9, 1), date(2026, 10, 20))
    with pytest.raises(SourceError, match="unexpected response"):
        rig.fdo.matches_by_ids([1])


def test_matches_with_non_object_teams_raise_source_error(make_rig):
    match = copy.deepcopy(load("fdo_team_matches.json")["matches"][1])
    match["homeTeam"] = "Arsenal"
    rig = make_rig(lambda r: {"matches": [match]})
    with pytest.raises(SourceError, match="unexpected response"):
        rig.fdo.team_upcoming(57, date(2026, 10, 20), date(2026, 11, 20))
