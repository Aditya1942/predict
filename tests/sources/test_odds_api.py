import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from predict.errors import OddsUnavailable
from predict.sources.odds_api import OddsApi
from predict.store import Store
from predict.teams import Aliases, TeamResolver
from tests.helpers import make_match, make_team

FIXTURES = Path(__file__).parent.parent / "fixtures"
NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
KICKOFF = datetime(2026, 10, 25, 16, 30, tzinfo=UTC)

ARSENAL = make_team(1, "Arsenal FC")
CHELSEA = make_team(2, "Chelsea FC")
EVERTON = make_team(3, "Everton FC")
TEAMS = [ARSENAL, CHELSEA, EVERTON]
MATCH = make_match(ARSENAL, CHELSEA, kickoff=KICKOFF, id=42)

EVENT_ODDS = json.loads((FIXTURES / "odds_event.json").read_text())


def listing(home="Arsenal", away="Chelsea", *, event_id="evt-arsenal-chelsea", commence="2026-10-25T16:30:00Z"):
    return {"id": event_id, "sport_key": "soccer_epl", "commence_time": commence, "home_team": home, "away_team": away}


class Server:
    """A fake The Odds API: records requests and answers the events and event-odds endpoints."""

    def __init__(self, events=None, odds=None, *, status=200, remaining="487") -> None:
        self.events = [listing()] if events is None else events
        self.odds = EVENT_ODDS if odds is None else odds
        self.status = status
        self.remaining = remaining
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"message": "nope"})
        body = self.odds if request.url.path.endswith("/odds") else self.events
        return httpx.Response(200, json=body, headers={"x-requests-remaining": self.remaining})


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def odds_api(server, store, *, now=NOW, key="secret") -> OddsApi:
    client = httpx.Client(transport=httpx.MockTransport(server))
    return OddsApi(client, store, key, TeamResolver(TEAMS, Aliases.load()), now=lambda: now)


def test_snapshot_parses_the_fixture(store):
    server = Server()
    snap = odds_api(server, store).snapshot(MATCH)
    result = snap.probabilities["result"]
    assert list(result) == ["home", "draw", "away"]
    assert sum(result.values()) == pytest.approx(1.0)
    assert result["home"] > result["draw"] > 0
    assert set(snap.probabilities["btts"]) == {"yes", "no"}
    assert 0.4 < snap.probabilities["total:2.5"]["over"] < 0.6
    assert set(snap.probabilities["team:home:1.5"]) == {"over", "under"}
    assert set(snap.probabilities["team:away:0.5"]) == {"over", "under"}
    assert snap.credits_remaining == 487
    assert snap.fetched_at == NOW


def test_only_the_configured_lines_are_kept(store):
    probabilities = odds_api(Server(), store).snapshot(MATCH).probabilities
    assert set(probabilities) == {
        "result",
        "btts",
        "total:1.5",
        "total:2.5",
        "total:3.5",
        "team:home:0.5",
        "team:home:1.5",
        "team:home:2.5",
        "team:away:0.5",
        "team:away:1.5",
        # Chelsea's 2.5 line is not in the fixture, its 3.5 line is not a configured Line
    }


def test_team_totals_side_comes_from_the_description(store):
    probabilities = odds_api(Server(), store).snapshot(MATCH).probabilities
    # Arsenal 2.5: over 4.00/3.90, under 1.25/1.27; Chelsea 1.5: over 3.20/3.10, under 1.36/1.38
    assert probabilities["team:home:2.5"]["over"] == pytest.approx(0.24, abs=0.03)
    assert probabilities["team:away:1.5"]["over"] == pytest.approx(0.30, abs=0.02)


def test_requests_hit_the_documented_endpoints_with_the_key(store):
    server = Server()
    odds_api(server, store, key="secret").snapshot(MATCH)
    events, odds = server.requests
    assert events.url.path == "/v4/sports/soccer_epl/events"
    assert events.url.params["apiKey"] == "secret"
    assert odds.url.path == "/v4/sports/soccer_epl/events/evt-arsenal-chelsea/odds"
    assert dict(odds.url.params) == {
        "apiKey": "secret",
        "regions": "eu",
        "markets": "h2h,totals,btts,team_totals",
        "oddsFormat": "decimal",
    }


def bookmaker(key, home, draw, away):
    outcomes = [{"name": "Arsenal", "price": home}, {"name": "Draw", "price": draw}, {"name": "Chelsea", "price": away}]
    return {"key": key, "markets": [{"key": "h2h", "outcomes": outcomes}]}


def test_aggregation_is_the_median_per_answer_then_renormalised(store):
    def fair(key, home, draw, away):  # prices of 1/p carry no margin, so devigging leaves p unchanged
        return bookmaker(key, 1 / home, 1 / draw, 1 / away)

    odds = {
        **EVENT_ODDS,
        "bookmakers": [fair("a", 0.5, 0.3, 0.2), fair("b", 0.2, 0.5, 0.3), fair("c", 0.35, 0.25, 0.4)],
    }
    result = odds_api(Server(odds=odds), store).snapshot(MATCH).probabilities["result"]
    # medians 0.35 / 0.30 / 0.30 sum to 0.95 and are scaled back to 1; the means would be 0.35 / 0.35 / 0.30
    assert result == pytest.approx({"home": 0.35 / 0.95, "draw": 0.30 / 0.95, "away": 0.30 / 0.95})


def test_each_bookmaker_is_devigged_before_the_median(store):
    odds = {**EVENT_ODDS, "bookmakers": [bookmaker("a", 1.8, 3.6, 4.5)]}
    result = odds_api(Server(odds=odds), store).snapshot(MATCH).probabilities["result"]
    raw = {"home": 1 / 1.8, "draw": 1 / 3.6, "away": 1 / 4.5}
    assert result["home"] == pytest.approx(raw["home"] / sum(raw.values()))


def test_a_bookmaker_without_a_market_does_not_block_others(store):
    odds = {**EVENT_ODDS, "bookmakers": [bookmaker("a", 2.0, 4.0, 4.0), {"key": "broken", "markets": "oops"}]}
    snap = odds_api(Server(odds=odds), store).snapshot(MATCH)
    assert snap.probabilities["result"]["home"] == pytest.approx(0.5)
    assert "btts" not in snap.probabilities


def test_no_usable_markets_is_unavailable(store):
    with pytest.raises(OddsUnavailable, match="Arsenal FC v Chelsea FC"):
        odds_api(Server(odds={**EVENT_ODDS, "bookmakers": []}), store).snapshot(MATCH)


@pytest.mark.parametrize("status", [401, 429])
def test_quota_or_key_problems_are_unavailable(store, status):
    with pytest.raises(OddsUnavailable) as err:
        odds_api(Server(status=status), store).snapshot(MATCH)
    assert err.value.reason == f"odds quota used up or key invalid (HTTP {status})"


def test_other_http_failures_are_unavailable_and_never_leak_the_key(store):
    with pytest.raises(OddsUnavailable, match="HTTP 500") as err:
        odds_api(Server(status=500), store, key="secret").snapshot(MATCH)
    assert "secret" not in err.value.reason


def test_network_failure_is_unavailable(store):
    def boom(request):
        raise httpx.ConnectError("no route", request=request)

    client = httpx.Client(transport=httpx.MockTransport(boom))
    api = OddsApi(client, store, "secret", TeamResolver(TEAMS, Aliases.load()), now=lambda: NOW)
    with pytest.raises(OddsUnavailable, match="unreachable") as err:
        api.snapshot(MATCH)
    assert "secret" not in err.value.reason


def test_invalid_json_is_unavailable(store):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html>")))
    api = OddsApi(client, store, "k", TeamResolver(TEAMS, Aliases.load()), now=lambda: NOW)
    with pytest.raises(OddsUnavailable, match="unexpected"):
        api.snapshot(MATCH)


def test_no_matching_event_names_both_teams(store):
    server = Server(events=[listing("Everton", "Chelsea", event_id="other")])
    with pytest.raises(OddsUnavailable, match="bookmakers don't list Arsenal FC v Chelsea FC"):
        odds_api(server, store).snapshot(MATCH)
    assert len(server.requests) == 1  # no odds fetched, so no credits spent


@pytest.mark.parametrize("commence", ["2026-10-25T19:30:00Z", "2026-10-25T13:30:00Z"])  # exactly 3h off
def test_event_within_three_hours_of_kickoff_matches(store, commence):
    assert odds_api(Server(events=[listing(commence=commence)]), store).snapshot(MATCH).probabilities


@pytest.mark.parametrize("commence", ["2026-10-25T19:31:00Z", "2026-10-25T13:29:00Z", "2026-11-01T16:30:00Z"])
def test_event_further_from_kickoff_does_not_match(store, commence):
    with pytest.raises(OddsUnavailable, match="don't list"):
        odds_api(Server(events=[listing(commence=commence)]), store).snapshot(MATCH)


def test_the_right_event_is_picked_among_several(store):
    events = [
        listing("Everton", "Arsenal", event_id="e1", commence="2026-10-25T14:00:00Z"),
        listing("Chelsea", "Arsenal", event_id="e2", commence="2026-10-25T16:30:00Z"),  # reverse fixture
        listing("Arsenal", "Chelsea", event_id="e3", commence="2026-10-25T16:30:00Z"),
    ]
    server = Server(events=events)
    odds_api(server, store).snapshot(MATCH)
    assert server.requests[1].url.path.split("/")[-2] == "e3"


def test_malformed_events_are_skipped(store):
    events = [{"id": "bad"}, listing()]
    odds_api(Server(events=events), store).snapshot(MATCH)


def test_second_call_within_the_ttl_makes_no_http_calls(store):
    server = Server()
    first = odds_api(server, store).snapshot(MATCH)
    again = odds_api(server, store, now=NOW + timedelta(minutes=59)).snapshot(MATCH)
    assert len(server.requests) == 2  # only the first snapshot's events + odds calls
    assert again == first


def test_after_the_ttl_it_fetches_again(store):
    server = Server()
    odds_api(server, store).snapshot(MATCH)
    odds_api(server, store, now=NOW + timedelta(minutes=61)).snapshot(MATCH)
    assert len(server.requests) == 4


def test_failures_are_not_cached(store):
    with pytest.raises(OddsUnavailable):
        odds_api(Server(status=429), store).snapshot(MATCH)
    server = Server()
    assert odds_api(server, store).snapshot(MATCH).probabilities
    assert len(server.requests) == 2


def test_snapshots_are_cached_per_match(store):
    server = Server()
    odds_api(server, store).snapshot(MATCH)
    odds_api(server, store).snapshot(make_match(ARSENAL, CHELSEA, kickoff=KICKOFF, id=43))
    assert len(server.requests) == 4


def test_missing_credits_header_is_none(store):
    server = Server(remaining="n/a")
    assert odds_api(server, store).snapshot(MATCH).credits_remaining is None
