"""The Odds API (v4) client: Bookmaker Probabilities for one Match.

Fetches the Match's event, then its h2h, totals, btts and team_totals odds (about 4 credits). Each bookmaker's
prices are devigged per Market, the answers are combined by median across bookmakers and renormalised.
"""

import statistics
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

from predict.domain import MARKET_IDS, Match, Probabilities
from predict.errors import OddsUnavailable, SourceError
from predict.odds import devig
from predict.params import ODDS_TTL, TEAM_LINES, TOTAL_LINES
from predict.store import Store
from predict.teams import TeamResolver

SOURCE = "odds_api"
BASE_URL = "https://api.the-odds-api.com"
KICKOFF_WINDOW = timedelta(hours=3)  # a listed event may start this far from the Match's kickoff
MARKETS = "h2h,totals,btts,team_totals"
_CREDITS_HEADER = "x-requests-remaining"
_PRICES = dict[str, dict[str, float]]  # market id -> answer -> decimal price


class OddsSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    probabilities: Probabilities
    credits_remaining: int | None
    fetched_at: AwareDatetime


class OddsApi:
    def __init__(
        self,
        client: httpx.Client,
        store: Store,
        api_key: str,
        resolver: TeamResolver,
        *,
        now: Callable[[], datetime],
    ) -> None:
        self._client = client
        self._store = store
        self._api_key = api_key
        self._resolver = resolver
        self._now = now

    def snapshot(self, match: Match) -> OddsSnapshot:
        """Bookmaker Probabilities for the Match, cached per Match for `ODDS_TTL`. Raises OddsUnavailable."""
        key = f"odds:{match.id}"
        cached = self._store.cache_get(key, ODDS_TTL, self._now())
        if cached is not None:
            try:
                return OddsSnapshot.model_validate_json(cached)
            except ValidationError:
                pass  # an unreadable copy is just a miss
        sport = match.competition.odds_sport
        event_id = self._find_event(match, sport)
        payload, headers = self._get(
            f"/v4/sports/{sport}/events/{event_id}/odds",
            {"regions": "eu", "markets": MARKETS, "oddsFormat": "decimal"},
        )
        probabilities = _book_probabilities(payload)
        if not probabilities:
            raise OddsUnavailable(f"bookmakers list no usable odds for {_title(match)}")
        snapshot = OddsSnapshot(
            probabilities=probabilities,
            credits_remaining=_credits(headers.get(_CREDITS_HEADER)),
            fetched_at=self._now(),
        )
        self._store.cache_put(key, snapshot.model_dump_json(), snapshot.fetched_at)
        return snapshot

    def _find_event(self, match: Match, sport: str) -> str:
        events, _ = self._get(f"/v4/sports/{sport}/events", {})
        if not isinstance(events, list):
            raise OddsUnavailable("unexpected response from the odds service")
        for event in events:
            event_id = self._event_id(event, match)
            if event_id is not None:
                return event_id
        raise OddsUnavailable(f"bookmakers don't list {_title(match)}")

    def _event_id(self, event: Any, match: Match) -> str | None:
        """The event's id if it starts within `KICKOFF_WINDOW` of kickoff and its names map to the Match's teams."""
        try:
            commence = datetime.fromisoformat(event["commence_time"])
            home, away, event_id = event["home_team"], event["away_team"], str(event["id"])
            if abs(commence - match.kickoff) > KICKOFF_WINDOW:
                return None
            mapped = self._resolver.map_source_names(SOURCE, [home, away])
        except (KeyError, TypeError, ValueError, SourceError):  # a malformed or ambiguous event is not ours
            return None
        return event_id if mapped[home] == match.home.id and mapped[away] == match.away.id else None

    def _get(self, path: str, params: dict[str, str]) -> tuple[Any, httpx.Headers]:
        try:
            response = self._client.get(BASE_URL + path, params={"apiKey": self._api_key, **params})
        except httpx.HTTPError as exc:
            # the exception text can quote the URL, which carries the key
            raise OddsUnavailable(f"odds service unreachable ({type(exc).__name__})") from exc
        if response.status_code in (401, 429):
            raise OddsUnavailable(f"odds quota used up or key invalid (HTTP {response.status_code})")
        if not response.is_success:
            raise OddsUnavailable(f"odds service error (HTTP {response.status_code})")
        try:
            return response.json(), response.headers
        except ValueError as exc:
            raise OddsUnavailable("unexpected response from the odds service") from exc


def _title(match: Match) -> str:
    return f"{match.home.short_name} v {match.away.short_name}"


def _credits(value: str | None) -> int | None:
    try:
        return int(float(value)) if value is not None else None
    except (ValueError, OverflowError):  # not a number, or "inf"
        return None


def _book_probabilities(event: Any) -> Probabilities:
    """Median of each answer's devigged probability across bookmakers, in Market id order, each Market summing to 1."""
    per_market: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    try:
        home, away, bookmakers = event["home_team"], event["away_team"], event["bookmakers"]
        parsed = [_bookmaker_prices(bookmaker, home, away) for bookmaker in bookmakers]
    except (KeyError, TypeError, AttributeError, ValueError):
        raise OddsUnavailable("unexpected response from the odds service") from None
    for markets in parsed:
        for market_id, prices in markets.items():
            for answer, probability in devig(prices).items():
                per_market[market_id][answer].append(probability)
    result: Probabilities = {}
    for market_id in MARKET_IDS:
        if market_id not in per_market:
            continue
        medians = {answer: statistics.median(values) for answer, values in per_market[market_id].items()}
        total = sum(medians.values())
        result[market_id] = {answer: median / total for answer, median in medians.items()}
    return result


def _bookmaker_prices(bookmaker: Any, home: str, away: str) -> _PRICES:
    """One bookmaker's usable Markets; a Market that is malformed or incomplete is left out."""
    prices: _PRICES = {}
    try:
        markets = list(bookmaker["markets"])
    except (KeyError, TypeError):
        return prices
    parsers = {
        "h2h": lambda outcomes: _h2h(outcomes, home, away),
        "totals": _totals,
        "btts": _btts,
        "team_totals": lambda outcomes: _team_totals(outcomes, home, away),
    }
    for market in markets:
        try:
            parse = parsers.get(market["key"])
            if parse is not None:
                prices.update(parse(market["outcomes"]))
        except (KeyError, TypeError, AttributeError, ValueError):
            continue
    return {market_id: p for market_id, p in prices.items() if _usable(p)}


def _usable(prices: dict[str, float]) -> bool:
    try:
        devig(prices)
    except ValueError:
        return False
    return True


def _same(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def _h2h(outcomes: list[dict], home: str, away: str) -> _PRICES:
    prices: dict[str, float] = {}
    for outcome in outcomes:
        name = outcome["name"]
        for answer, label in (("home", home), ("away", away), ("draw", "Draw")):
            if _same(name, label):
                prices.setdefault(answer, float(outcome["price"]))
                break
    if set(prices) != {"home", "draw", "away"}:
        return {}
    return {"result": {answer: prices[answer] for answer in ("home", "draw", "away")}}


def _btts(outcomes: list[dict]) -> _PRICES:
    prices: dict[str, float] = {}
    for outcome in outcomes:
        answer = outcome["name"].strip().casefold()
        if answer in ("yes", "no"):
            prices.setdefault(answer, float(outcome["price"]))
    if set(prices) != {"yes", "no"}:
        return {}
    return {"btts": {"yes": prices["yes"], "no": prices["no"]}}


def _sides(
    outcomes: list[dict], lines: tuple[float, ...], team_of: Callable[[dict], str | None]
) -> dict[tuple[str, float], dict[str, float]]:
    """Complete Over/Under pairs at the configured Lines, keyed by (team, line); a None `team_of` skips the outcome."""
    found: dict[tuple[str, float], dict[str, float]] = defaultdict(dict)
    for outcome in outcomes:
        side = outcome["name"].strip().casefold()
        line = float(outcome["point"])
        team = team_of(outcome)
        if side in ("over", "under") and line in lines and team is not None:
            found[(team, line)].setdefault(side, float(outcome["price"]))
    return {group: {"over": s["over"], "under": s["under"]} for group, s in found.items() if len(s) == 2}


def _totals(outcomes: list[dict]) -> _PRICES:
    return {f"total:{line}": sides for (_, line), sides in _sides(outcomes, TOTAL_LINES, lambda outcome: "").items()}


def _team_totals(outcomes: list[dict], home: str, away: str) -> _PRICES:
    """The team a line is for comes from the outcome's `description`."""

    def team_of(outcome: dict) -> str | None:
        description = outcome["description"]
        return "home" if _same(description, home) else "away" if _same(description, away) else None

    return {f"team:{team}:{line}": sides for (team, line), sides in _sides(outcomes, TEAM_LINES, team_of).items()}
