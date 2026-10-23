import itertools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from predict.config import Settings
from predict.errors import SourceError
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
    finally:
        services.close()


def test_build_propagates_a_source_failure(monkeypatch, tmp_path):
    real_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs))
    settings = Settings(_env_file=None, football_data_org_token="token", predict_home=tmp_path)
    with pytest.raises(SourceError):
        build(settings, lambda: START)
