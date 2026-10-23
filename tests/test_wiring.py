import itertools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from predict import wiring
from predict.config import Settings
from predict.errors import SourceError
from predict.news.scoring import TypeSafeJev
from predict.sources.odds_api import OddsApi
from predict.wiring import build

FIXTURES = Path(__file__).parent / "fixtures"
START = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)


def test_build_wires_the_real_services(monkeypatch, tmp_path):
    requested: list[str] = []
    teams = json.loads((FIXTURES / "fdo_teams_pl.json").read_text())
    real_client = httpx.Client

    def handle(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(200, json=teams)

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))
    ticks = itertools.count()
    now = lambda: START + timedelta(seconds=10 * next(ticks))  # noqa: E731  each call is later than the last

    settings = Settings(_env_file=None, football_data_org_token="token", predict_home=tmp_path / "home")
    services = build(settings, now)
    try:
        assert sorted(requested) == [f"/v4/competitions/{c}/teams" for c in ("BL1", "CL", "FL1", "PD", "PL", "SA")]
        assert services.resolver.resolve_query("arsenal").name == "Arsenal FC"
        assert (tmp_path / "home" / "predict.db").exists()
        assert services.predictor is not None
        assert services.now is now
    finally:
        services.close()


def test_build_wires_odds_only_when_a_key_is_set(monkeypatch, tmp_path):
    teams = json.loads((FIXTURES / "fdo_teams_pl.json").read_text())
    real_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=teams))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs))
    ticks = itertools.count()
    now = lambda: START + timedelta(seconds=10 * next(ticks))  # noqa: E731  later each call, so nothing is throttled

    without = build(Settings(_env_file=None, football_data_org_token="t", predict_home=tmp_path / "a"), now)
    with_key = build(
        Settings(_env_file=None, football_data_org_token="t", odds_api_key="k", predict_home=tmp_path / "b"), now
    )
    try:
        assert without.odds is None
        assert isinstance(with_key.odds, OddsApi)
    finally:
        without.close()
        with_key.close()


def test_build_wires_jev_only_when_a_key_is_set(monkeypatch, tmp_path):
    teams = json.loads((FIXTURES / "fdo_teams_pl.json").read_text())
    real_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=teams))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs))
    ticks = itertools.count()
    now = lambda: START + timedelta(seconds=10 * next(ticks))  # noqa: E731  later each call, so nothing is throttled

    without = build(Settings(_env_file=None, football_data_org_token="t", predict_home=tmp_path / "a"), now)
    with_key = build(
        Settings(
            _env_file=None,
            football_data_org_token="t",
            typesafe_api_key="k",
            jev_model="jev-test",
            predict_home=tmp_path / "b",
        ),
        now,
    )
    try:
        assert without.jev is None
        assert isinstance(with_key.jev, TypeSafeJev)
    finally:
        without.close()
        with_key.close()


def test_build_hands_the_progress_callback_to_the_researcher(monkeypatch, tmp_path):
    teams = json.loads((FIXTURES / "fdo_teams_pl.json").read_text())
    real_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=teams))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs))
    given: dict = {}
    monkeypatch.setattr(wiring, "ClaudeResearcher", lambda **kwargs: given.update(kwargs))
    progress = lambda message: None  # noqa: E731
    settings = Settings(_env_file=None, football_data_org_token="t", predict_home=tmp_path)
    ticks = itertools.count()
    build(settings, lambda: START + timedelta(seconds=10 * next(ticks)), progress=progress).close()
    assert given["progress"] is progress


def test_build_propagates_a_source_failure(monkeypatch, tmp_path):
    real_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs))
    settings = Settings(_env_file=None, football_data_org_token="token", predict_home=tmp_path)
    with pytest.raises(SourceError):
        build(settings, lambda: START)
