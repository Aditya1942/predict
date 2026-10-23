import os
import time
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta

import pytest

from predict.domain import (
    Competition,
    LeagueBase,
    Match,
    MatchDossier,
    MatchRecord,
    NewsDossier,
    NewsFactor,
    ProjectedGoals,
    Team,
    TeamRating,
)
from predict.errors import OddsUnavailable
from predict.news.scoring import JevAnswers
from predict.ratings import RatedMatch
from predict.sources.football_data_org import MatchStatus
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

    def __init__(self, matches: list[Match], statuses: list[MatchStatus] | None = None) -> None:
        self.matches = matches
        self.statuses = statuses or []
        self.upcoming_calls: list[tuple[int, date, date]] = []
        self.status_calls: list[list[int]] = []
        self.status_error: Exception | None = None

    def matches_by_ids(self, ids: list[int]) -> list[MatchStatus]:
        self.status_calls.append(list(ids))
        if self.status_error is not None:
            raise self.status_error
        return [s for s in self.statuses if s.match.id in ids]

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


def make_factor(
    kind: str = "absence",
    player: str | None = "Saka",
    *,
    misses: str = "yes",
    summary: str = "Out for three weeks with a hamstring injury.",
    sources: tuple[str, ...] = (),
) -> NewsFactor:
    return NewsFactor(kind=kind, player=player, summary=summary, misses_match=misses, sources=list(sources))


def make_dossier(
    *, home: tuple[NewsFactor, ...] = (), away: tuple[NewsFactor, ...] = (), created_at: datetime, model: str
) -> MatchDossier:
    return MatchDossier(
        home=NewsDossier(team="Home", factors=list(home)),
        away=NewsDossier(team="Away", factors=list(away)),
        created_at=created_at,
        claude_model=model,
    )


class FakeDossiers:
    """The DossierService stand-in: hands out `dossier`, or raises `error`, noting how it was asked."""

    def __init__(self, dossier: MatchDossier | None = None, *, error: Exception | None = None) -> None:
        self.dossier = dossier
        self.error = error
        self.calls: list[tuple[Match, bool, str]] = []

    def get(self, match: Match, *, fresh: bool, model: str) -> MatchDossier:
        self.calls.append((match, fresh, model))
        if self.error is not None:
            raise self.error
        assert self.dossier is not None
        return self.dossier


class StubJev:
    """Jev stand-in: each question is answered by `scores` (by name) or level 0, "no real change"; or raises `error`."""

    def __init__(self, scores: dict[str, list[float]] | None = None, *, error: Exception | None = None) -> None:
        self.scores = scores or {}
        self.error = error
        self.states: list[dict] = []

    def ask(self, state: dict, nouls: dict[str, str], scores: dict[str, tuple[str, list[str]]]) -> JevAnswers:
        self.states.append(state)
        if self.error is not None:
            raise self.error
        return JevAnswers(
            nouls={name: 0.5 for name in nouls},
            score_probabilities={name: self.scores.get(name, [1.0, 0.0, 0.0, 0.0]) for name in scores},
        )


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
