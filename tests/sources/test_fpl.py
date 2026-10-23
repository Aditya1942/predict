import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from predict.errors import SourceError
from predict.http import CachedHttp
from predict.sources.fpl import URL, Fpl
from predict.store import Store
from predict.teams import Aliases, TeamResolver
from tests.helpers import make_team

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
FIXTURE = json.loads((Path(__file__).parent.parent / "fixtures" / "fpl_bootstrap.json").read_text())

ARSENAL = make_team(1, "Arsenal FC")
MAN_CITY = make_team(2, "Manchester City FC")
BENFICA = make_team(3, "SL Benfica")
RESOLVER = TeamResolver([ARSENAL, MAN_CITY, BENFICA], Aliases.load())


def payload(*elements: dict, teams=(("Arsenal", 1),)) -> dict:
    return {
        "teams": [{"id": id, "name": name, "short_name": name[:3].upper()} for name, id in teams],
        "elements": list(elements),
    }


def player(name="Saka", team=1, status="a", chance=None, news="") -> dict:
    return {"web_name": name, "team": team, "status": status, "chance_of_playing_next_round": chance, "news": news}


def fpl(body, store, *, now=NOW, requests=None) -> Fpl:
    def handle(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    return Fpl(CachedHttp(store, client, now=lambda: now))


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def test_injured_player_becomes_a_hint(store):
    body = payload(player("Saka", status="i", chance=25, news="Hamstring injury - 25% chance of playing"))
    assert fpl(body, store).hints(ARSENAL, RESOLVER) == [
        "Saka: injured, 25% chance — Hamstring injury - 25% chance of playing"
    ]


def test_available_players_are_left_out(store):
    body = payload(player("Raya"), player("Saka", status="d", chance=50, news="Knock"))
    assert fpl(body, store).hints(ARSENAL, RESOLVER) == ["Saka: doubtful, 50% chance — Knock"]


@pytest.mark.parametrize(
    ("status", "chance", "word"),
    [
        ("d", 75, "doubtful"),
        ("i", 0, "injured"),
        ("s", 0, "suspended"),
        ("u", 0, "unavailable"),
        ("n", 0, "not available"),
    ],
)
def test_status_words(store, status, chance, word):
    body = payload(player("Saka", status=status, chance=chance, news="n"))
    assert fpl(body, store).hints(ARSENAL, RESOLVER) == [f"Saka: {word}, {chance}% chance — n"]


def test_missing_chance_means_zero_for_a_player_who_is_out_and_is_left_out_for_a_doubt(store):
    body = payload(
        player("Saliba", status="i", chance=None, news="Back injury"),
        player("Saka", status="d", chance=None, news="Late fitness test"),
    )
    assert fpl(body, store).hints(ARSENAL, RESOLVER) == [
        "Saliba: injured, 0% chance — Back injury",
        "Saka: doubtful — Late fitness test",
    ]


def test_no_news_text_drops_the_dash(store):
    body = payload(player("Saka", status="s", chance=0, news=""))
    assert fpl(body, store).hints(ARSENAL, RESOLVER) == ["Saka: suspended, 0% chance"]


def test_only_the_asked_teams_players_are_listed(store):
    body = payload(
        player("Saka", team=1, status="i", chance=0, news="x"),
        player("Haaland", team=2, status="i", chance=0, news="y"),
        teams=(("Arsenal", 1), ("Man City", 2)),
    )
    assert fpl(body, store).hints(MAN_CITY, RESOLVER) == ["Haaland: injured, 0% chance — y"]


def test_team_outside_fpl_has_no_hints(store):
    body = payload(player("Saka", status="i", chance=0, news="x"))
    assert fpl(body, store).hints(BENFICA, RESOLVER) == []


def test_recorded_fixture_maps_fpl_names_to_teams(store):
    hints = fpl(FIXTURE, store).hints(MAN_CITY, RESOLVER)  # FPL calls them "Man City"
    assert hints
    assert all(" — " in h or h.endswith("chance") for h in hints)
    assert not any(h.startswith("Enzo") for h in hints)  # available players are left out
    assert any(h.startswith("Semenyo: injured") for h in fpl(FIXTURE, store).hints(MAN_CITY, RESOLVER))
    assert any(h.startswith("Saka: doubtful, 50% chance") for h in fpl(FIXTURE, store).hints(ARSENAL, RESOLVER))


def test_bootstrap_is_cached_for_an_hour(store):
    requests: list[httpx.Request] = []
    fpl(FIXTURE, store, requests=requests).hints(ARSENAL, RESOLVER)
    fpl(FIXTURE, store, now=NOW + timedelta(minutes=59), requests=requests).hints(MAN_CITY, RESOLVER)
    assert [str(r.url) for r in requests] == [URL]
    fpl(FIXTURE, store, now=NOW + timedelta(minutes=61), requests=requests).hints(MAN_CITY, RESOLVER)
    assert len(requests) == 2


@pytest.mark.parametrize(
    "body",
    [
        {"teams": []},
        {"elements": "x", "teams": [{"id": 1, "name": "Arsenal"}]},
        [],
        {"teams": [{"id": 1}], "elements": []},
    ],
)
def test_unexpected_shape_is_a_source_error(store, body):
    with pytest.raises(SourceError, match="fpl"):
        fpl(body, store).hints(ARSENAL, RESOLVER)
