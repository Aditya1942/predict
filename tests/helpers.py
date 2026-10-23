import os
import time
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta

import pytest

from predict.domain import Competition, LeagueBase, Match, MatchRecord, ProjectedGoals, Team, TeamRating
from predict.errors import OddsUnavailable
from predict.ratings import RatedMatch
from predict.sources.odds_api import OddsSnapshot

_EPOCH = datetime(2026, 8, 1, tzinfo=UTC)


def make_team(id: int, name: str, country: str | None = None) -> Team:
    return Team(id=id, name=name, short_name=name, country=country)


def make_match(
    home: Team,
    away: Team,
    *,
    kickoff: datetime,
    competition: Competition = Competition.PL,
    id: int = 1,
    neutral: bool = False,
) -> Match:
    return Match(id=id, competition=competition, home=home, away=away, kickoff=kickoff, neutral=neutral)


def rec(
    home_id: int,
    away_id: int,
    hg: int,
    ag: int,
    hxg: float | None = None,
    axg: float | None = None,
    *,
    day: int,
    competition: Competition = Competition.PL,
) -> MatchRecord:
    return MatchRecord(
        kickoff=_EPOCH + timedelta(days=day),
        competition=competition,
        home_name=f"T{home_id}",
        away_name=f"T{away_id}",
        home_goals=hg,
        away_goals=ag,
        home_xg=hxg,
        away_xg=axg,
        home_id=home_id,
        away_id=away_id,
    )


class FakeFdo:
    """football-data.org stand-in: upcoming Matches honour the requested date window."""

    def __init__(self, matches: list[Match]) -> None:
        self.matches = matches
        self.upcoming_calls: list[tuple[int, date, date]] = []

    def team_upcoming(self, team_id: int, date_from: date, date_to: date) -> list[Match]:
        self.upcoming_calls.append((team_id, date_from, date_to))
        found = [
            m
            for m in self.matches
            if team_id in (m.home.id, m.away.id) and date_from <= m.kickoff.date() <= date_to
        ]
        return sorted(found, key=lambda m: m.kickoff)


class StubRater:
    """Rates every Match the same way, so a test controls the Projected Goals."""

    def __init__(self, projected: ProjectedGoals | None = None) -> None:
        self.projected = projected or ProjectedGoals(home=1.7, away=1.1)
        self.rated: list[Match] = []

    def rate_match(self, match: Match) -> RatedMatch:
        self.rated.append(match)
        rating = TeamRating(attack=1.0, concede=1.0)
        return RatedMatch(
            home=rating,
            away=rating,
            base=LeagueBase(home_goals=1.5, away_goals=1.2),
            projected=self.projected,
        )


class FakeOdds:
    """The Odds API stand-in: returns `snapshot`, or raises `OddsUnavailable(unavailable)` when that is set."""

    def __init__(self, snapshot: OddsSnapshot | None = None, *, unavailable: str | None = None) -> None:
        self.snapshot_value = snapshot
        self.unavailable = unavailable
        self.asked: list[Match] = []

    def snapshot(self, match: Match) -> OddsSnapshot:
        self.asked.append(match)
        if self.unavailable is not None:
            raise OddsUnavailable(self.unavailable)
        assert self.snapshot_value is not None
        return self.snapshot_value


@contextmanager
def system_zone(name: str):
    """Make the process's local time zone `name` (an IANA name) for the duration of the block."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is not available on this platform")
    before = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = before
        time.tzset()
