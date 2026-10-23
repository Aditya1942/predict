"""Team ratings and Projected Goals for a Match, assembled from football-data.org, Understat and ClubElo."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from predict.domain import (
    LEAGUES,
    Competition,
    LeagueBase,
    Match,
    MatchRecord,
    ProjectedGoals,
    Team,
    TeamRating,
    season_of,
)
from predict.errors import SourceError
from predict.model.goals import CL_BASE, projected_goals
from predict.model.strength import combine, form_rating, league_base, season_rating
from predict.params import FORM_WINDOW_DAYS
from predict.sources.clubelo import ClubEloRow
from predict.teams import TeamResolver

CL_MEAN_GOALS = (CL_BASE.home_goals + CL_BASE.away_goals) / 2
_NEUTRAL = TeamRating(attack=1.0, concede=1.0)

_XgKey = tuple[date, int | None, int | None]


class TeamsSource(Protocol):
    def teams(self, comp: Competition) -> list[Team]: ...

    def team_results(self, team_id: int, date_from: date, date_to: date) -> list[MatchRecord]: ...


class XgSource(Protocol):
    def league_matches(self, comp: Competition, season: int) -> list[MatchRecord]: ...


class EloSource(Protocol):
    def ratings(self, on: date) -> dict[str, ClubEloRow]: ...

    def league_elo(self, rows: dict[str, ClubEloRow], country: str) -> float: ...


class RatedMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    home: TeamRating
    away: TeamRating
    base: LeagueBase
    projected: ProjectedGoals


@dataclass(frozen=True)
class _League:
    """One league's Understat records played before the Match, with football-data.org ids set."""

    comp: Competition
    current: list[MatchRecord]
    last: list[MatchRecord]
    base: LeagueBase
    xg: dict[_XgKey, tuple[float, float]]

    @property
    def mean_goals(self) -> float:
        return (self.base.home_goals + self.base.away_goals) / 2


class RatingsService:
    def __init__(self, fdo: TeamsSource, understat: XgSource, clubelo: EloSource, resolver: TeamResolver) -> None:
        self._fdo = fdo
        self._understat = understat
        self._clubelo = clubelo
        self._resolver = resolver

    def rate_match(self, match: Match) -> RatedMatch:
        if match.competition.is_league:
            league = self._league(match.competition, match.kickoff)
            home = self._league_team(match.home, match, league)
            away = self._league_team(match.away, match, league)
            base = league.base
        else:
            elo_rows = self._clubelo.ratings(match.kickoff.date())
            leagues: dict[Competition, _League] = {}
            home = self._cl_team(match.home, match, elo_rows, leagues)
            away = self._cl_team(match.away, match, elo_rows, leagues)
            base = CL_BASE
        return RatedMatch(
            home=home, away=away, base=base, projected=projected_goals(home, away, base, neutral=match.neutral)
        )

    def _league(self, comp: Competition, kickoff: datetime) -> _League:
        """Understat records for this and last season, only Matches played before `kickoff`."""
        season = season_of(kickoff)
        current = [r for r in self._understat.league_matches(comp, season) if r.kickoff < kickoff]
        last = [r for r in self._understat.league_matches(comp, season - 1) if r.kickoff < kickoff]
        if not current and not last:
            raise SourceError("understat", f"no {comp.display} matches for {season - 1} or {season}")
        names = {name for r in (*current, *last) for name in (r.home_name, r.away_name)}
        ids = self._resolver.map_source_names("understat", names)  # unmapped names stay None
        current, last = _with_ids(current, ids), _with_ids(last, ids)
        xg = {
            (r.kickoff.date(), r.home_id, r.away_id): (r.home_xg, r.away_xg)
            for r in (*last, *current)
            if r.home_id is not None and r.away_id is not None and r.home_xg is not None and r.away_xg is not None
        }
        return _League(comp, current, last, league_base(current, last), xg)

    def _league_team(self, team: Team, match: Match, league: _League) -> TeamRating:
        comp_avg = {league.comp: league.mean_goals, Competition.CL: CL_MEAN_GOALS}
        season = season_rating(team.id, league.current, league.last)
        return combine(season, self._form(team, match, comp_avg, league.xg))

    def _cl_team(
        self, team: Team, match: Match, elo_rows: dict[str, ClubEloRow], leagues: dict[Competition, _League]
    ) -> TeamRating:
        comp = self._league_of(team)
        if comp is not None:  # top-5 team: rated within its league, Elo is its league's mean
            if comp not in leagues:
                leagues[comp] = self._league(comp, match.kickoff)
            rating = self._league_team(team, match, leagues[comp])
            return rating.model_copy(update={"elo": self._clubelo.league_elo(elo_rows, comp.clubelo_country)})
        club = self._resolver.find_in_source("clubelo", team, elo_rows)
        form = self._form(team, match, {Competition.CL: CL_MEAN_GOALS}, {})
        return combine(_NEUTRAL, form).model_copy(update={"elo": elo_rows[club].elo})

    def _league_of(self, team: Team) -> Competition | None:
        """The first top-5 league whose teams include `team`."""
        return next((c for c in LEAGUES if any(t.id == team.id for t in self._fdo.teams(c))), None)

    def _form(
        self, team: Team, match: Match, comp_avg: dict[Competition, float], xg: dict[_XgKey, tuple[float, float]]
    ) -> TeamRating | None:
        date_from = (match.kickoff - timedelta(days=FORM_WINDOW_DAYS)).date()
        date_to = (match.kickoff - timedelta(days=1)).date()
        results = self._fdo.team_results(team.id, date_from, date_to)
        recent = [
            _with_xg(r, xg)
            for r in results
            if r.kickoff < match.kickoff and r.competition in comp_avg  # form_rating needs an average for each
        ]
        return form_rating(team.id, recent, comp_avg)


def _with_ids(records: list[MatchRecord], ids: dict[str, int | None]) -> list[MatchRecord]:
    return [r.model_copy(update={"home_id": ids[r.home_name], "away_id": ids[r.away_name]}) for r in records]


def _with_xg(record: MatchRecord, xg: dict[_XgKey, tuple[float, float]]) -> MatchRecord:
    found = xg.get((record.kickoff.date(), record.home_id, record.away_id))
    if found is None:
        return record
    return record.model_copy(update={"home_xg": found[0], "away_xg": found[1]})
