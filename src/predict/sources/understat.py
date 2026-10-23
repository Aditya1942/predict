"""Understat client: league Matches with xG (an unofficial JSON endpoint behind the league page)."""

from datetime import UTC, datetime, timedelta

from predict.domain import Competition, MatchRecord
from predict.errors import SourceError
from predict.http import CachedHttp

SOURCE = "understat"
BASE_URL = "https://understat.com/getLeagueData"
TTL = timedelta(hours=6)
_HEADERS = {"X-Requested-With": "XMLHttpRequest"}  # the endpoint refuses requests without it
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class Understat:
    def __init__(self, http: CachedHttp) -> None:
        self._http = http

    def league_matches(self, comp: Competition, season: int) -> list[MatchRecord]:
        """Played Matches of a league season, oldest first, with xG; ids are not set."""
        league = comp.understat_league
        if league is None:
            raise SourceError(SOURCE, f"no data for the {comp.display}")
        payload = self._http.get_json(f"{BASE_URL}/{league}/{season}", ttl=TTL, headers=_HEADERS)
        try:
            raw_matches = payload["dates"]
            records = [_record(raw, comp) for raw in raw_matches if raw["isResult"]]
        except (KeyError, TypeError, ValueError) as exc:  # pydantic ValidationError is a ValueError
            raise SourceError(SOURCE, f"unexpected response: {exc!r}") from exc
        return sorted(records, key=lambda r: r.kickoff)


def _record(raw: dict, comp: Competition) -> MatchRecord:
    return MatchRecord(
        kickoff=datetime.strptime(raw["datetime"], _TIME_FORMAT).replace(tzinfo=UTC),
        competition=comp,
        home_name=raw["h"]["title"],
        away_name=raw["a"]["title"],
        home_goals=int(raw["goals"]["h"]),
        away_goals=int(raw["goals"]["a"]),
        home_xg=float(raw["xG"]["h"]),
        away_xg=float(raw["xG"]["a"]),
    )
