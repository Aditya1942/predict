from datetime import UTC, datetime, timedelta

from predict.domain import Competition, Match, MatchRecord, Team

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
