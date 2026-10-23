"""Builds the real services: store, HTTP sources, team resolver, ratings and predictor."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import httpx

from predict.config import Settings
from predict.domain import Competition
from predict.http import CachedHttp
from predict.pipeline import Predictor
from predict.ratings import RatingsService
from predict.sources.clubelo import ClubElo
from predict.sources.football_data_org import FootballDataOrg
from predict.sources.odds_api import OddsApi
from predict.sources.understat import Understat
from predict.store import Store
from predict.teams import Aliases, TeamResolver

_TIMEOUT = 30.0


def _nothing() -> None:
    pass


@dataclass(frozen=True)
class Services:
    store: Store
    fdo: FootballDataOrg
    resolver: TeamResolver
    ratings: RatingsService
    predictor: Predictor
    close: Callable[[], None] = _nothing
    odds: OddsApi | None = None


def build(settings: Settings, now: Callable[[], datetime]) -> Services:
    """Wire everything up. Fetches every Competition's team list (cached for a day) for the resolver."""
    store = Store(settings.predict_home / "predict.db")
    client = httpx.Client(timeout=_TIMEOUT, follow_redirects=True)

    def close() -> None:
        client.close()
        store.close()

    try:
        http = CachedHttp(store, client, now=now)
        fdo = FootballDataOrg(http, settings.football_data_org_token)
        teams = [team for comp in Competition for team in fdo.teams(comp)]
        resolver = TeamResolver(teams, Aliases.load())
        ratings = RatingsService(fdo, Understat(http), ClubElo(http, store, now=now), resolver, now=now)
        odds = OddsApi(client, store, settings.odds_api_key, resolver, now=now) if settings.odds_api_key else None
        predictor = Predictor(fdo, ratings, resolver, store, odds, now=now)
    except BaseException:
        close()
        raise
    return Services(
        store=store, fdo=fdo, resolver=resolver, ratings=ratings, predictor=predictor, close=close, odds=odds
    )
