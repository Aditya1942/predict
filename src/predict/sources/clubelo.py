"""ClubElo client: club Elo ratings on a date, with the last good copy saved as a fallback."""

import csv
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

from pydantic import BaseModel, ConfigDict

from predict.errors import SourceError
from predict.http import CachedHttp
from predict.store import Store

SOURCE = "clubelo"
BASE_URL = "http://api.clubelo.com"
TTL = timedelta(hours=24)
LATEST_KEY = "clubelo:latest"
_COLUMNS = ("Club", "Country", "Level", "Elo")


class ClubEloRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    club: str
    country: str
    level: int
    elo: float


class ClubElo:
    def __init__(
        self,
        http: CachedHttp,
        store: Store,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._http = http
        self._store = store
        self._now = now

    def ratings(self, on: date) -> dict[str, ClubEloRow]:
        """Every club's rating on `on`, keyed by ClubElo's club name.

        When ClubElo fails or answers with something unusable, the last good copy is used instead.
        """
        try:
            body = self._http.get_text(f"{BASE_URL}/{on.isoformat()}", ttl=TTL)
            rows = _parse(body)
        except SourceError as exc:
            return self._saved(exc)
        self._store.cache_put(LATEST_KEY, body, self._now())
        return rows

    @staticmethod
    def league_elo(rows: dict[str, ClubEloRow], country: str) -> float:
        """Mean Elo of the country's top-flight clubs."""
        elos = [r.elo for r in rows.values() if r.country == country and r.level == 1]
        if not elos:
            raise SourceError(SOURCE, f"no level 1 clubs for {country}")
        return sum(elos) / len(elos)

    def _saved(self, failure: SourceError) -> dict[str, ClubEloRow]:
        saved = self._store.cache_latest(LATEST_KEY)
        if saved is not None:
            try:
                return _parse(saved[0])
            except SourceError:
                pass
        raise SourceError(SOURCE, "unavailable and no saved ratings") from failure


def _parse(body: str) -> dict[str, ClubEloRow]:
    reader = csv.DictReader(body.splitlines())
    if reader.fieldnames is None or not all(c in reader.fieldnames for c in _COLUMNS):
        raise SourceError(SOURCE, "unexpected response: not the ratings CSV")
    try:
        rows = [
            ClubEloRow(club=raw["Club"], country=raw["Country"], level=int(raw["Level"]), elo=float(raw["Elo"]))
            for raw in reader
        ]
    except (TypeError, ValueError) as exc:  # a short row gives None; a bad number gives ValueError
        raise SourceError(SOURCE, f"unexpected response: {exc!r}") from exc
    if not rows:
        raise SourceError(SOURCE, "unexpected response: no ratings")
    return {row.club: row for row in rows}
