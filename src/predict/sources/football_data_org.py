"""football-data.org v4 client: teams, upcoming Matches and 90-minute Results."""

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict

from predict.domain import Competition, Match, MatchRecord, Result, Team
from predict.errors import SourceError
from predict.http import CachedHttp

BASE_URL = "https://api.football-data.org/v4"
SOURCE = "api.football-data.org"
MIN_INTERVAL = 6.5  # free tier: 10 requests/min
_COMPETITIONS = ",".join(c.value for c in Competition)


class MatchStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    match: Match
    status: str
    result: Result | None = None


def regular_time_score(score: dict) -> tuple[int, int]:
    """Goals after 90 minutes plus stoppage time (extra time and penalties ignored)."""
    block = score["fullTime"] if score["duration"] == "REGULAR" else score["regularTime"]
    home, away = block["home"], block["away"]
    if home is None or away is None:
        raise ValueError("score not available")
    return home, away


class FootballDataOrg:
    def __init__(self, http: CachedHttp, token: str) -> None:
        self._http = http
        self._token = token

    def teams(self, comp: Competition) -> list[Team]:
        payload = self._get(f"/competitions/{comp.value}/teams", ttl=timedelta(hours=24))
        return _parse_all(payload.get("teams"), _team)

    def team_upcoming(self, team_id: int, date_from: date, date_to: date) -> list[Match]:
        matches = self._team_matches(team_id, "SCHEDULED,TIMED", date_from, date_to)
        return sorted(_parse_all(matches, _decided_match), key=lambda m: m.kickoff)

    def team_results(self, team_id: int, date_from: date, date_to: date) -> list[MatchRecord]:
        matches = self._team_matches(team_id, "FINISHED", date_from, date_to)
        return sorted(_parse_all(matches, _record), key=lambda r: r.kickoff)

    def matches_by_ids(self, ids: list[int]) -> list[MatchStatus]:
        if not ids:
            return []
        payload = self._get("/matches", ttl=timedelta(0), params={"ids": ",".join(str(i) for i in ids)})
        return _parse_all(payload.get("matches"), _match_status)

    def _team_matches(self, team_id: int, status: str, date_from: date, date_to: date) -> Any:
        payload = self._get(
            f"/teams/{team_id}/matches",
            ttl=timedelta(hours=1),
            params={
                "status": status,
                "competitions": _COMPETITIONS,
                "dateFrom": date_from.isoformat(),
                "dateTo": date_to.isoformat(),
            },
        )
        return payload.get("matches")

    def _get(self, path: str, *, ttl: timedelta, params: dict | None = None) -> dict:
        payload = self._http.get_json(
            BASE_URL + path,
            ttl=ttl,
            params=params,
            headers={"X-Auth-Token": self._token},
            min_interval=MIN_INTERVAL,
        )
        if not isinstance(payload, dict):
            raise SourceError(SOURCE, f"unexpected response: expected a JSON object, got {type(payload).__name__}")
        return payload


def _parse_all[T](items: Any, parse: Callable[[dict], T | None]) -> list[T]:
    """Parse every item, dropping those `parse` returns None for."""
    try:
        parsed = [parse(item) for item in items]
    except (KeyError, TypeError, AttributeError, ValueError) as exc:  # pydantic ValidationError is a ValueError
        raise SourceError(SOURCE, f"unexpected response: {exc!r}") from exc
    return [p for p in parsed if p is not None]


def _team(raw: dict) -> Team:
    area = raw.get("area") or {}
    return Team(
        id=raw["id"],
        name=raw["name"],
        short_name=raw.get("shortName") or raw["name"],
        tla=raw.get("tla"),
        country=area.get("name"),
    )


def _kickoff(raw: dict) -> datetime:
    return datetime.fromisoformat(raw["utcDate"]).astimezone(UTC)


def _match(raw: dict) -> Match:
    competition = Competition(raw["competition"]["code"])
    stage = raw.get("stage")
    return Match(
        id=raw["id"],
        competition=competition,
        home=_team(raw["homeTeam"]),
        away=_team(raw["awayTeam"]),
        kickoff=_kickoff(raw),
        stage=stage,
        neutral=competition is Competition.CL and stage == "FINAL",
    )


def _decided_match(raw: dict) -> Match | None:
    """None while a knockout tie's opponent is still undecided (team ids are null)."""
    if raw["homeTeam"].get("id") is None or raw["awayTeam"].get("id") is None:
        return None
    return _match(raw)


def _record(raw: dict) -> MatchRecord:
    home_goals, away_goals = regular_time_score(raw["score"])
    return MatchRecord(
        kickoff=_kickoff(raw),
        competition=Competition(raw["competition"]["code"]),
        home_name=raw["homeTeam"]["name"],
        away_name=raw["awayTeam"]["name"],
        home_goals=home_goals,
        away_goals=away_goals,
        home_id=raw["homeTeam"]["id"],
        away_id=raw["awayTeam"]["id"],
    )


def _match_status(raw: dict) -> MatchStatus:
    status = raw["status"]
    result = None
    if status == "FINISHED":
        home_goals, away_goals = regular_time_score(raw["score"])
        result = Result(home_goals=home_goals, away_goals=away_goals)
    return MatchStatus(match=_match(raw), status=status, result=result)
