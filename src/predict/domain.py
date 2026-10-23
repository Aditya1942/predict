"""Domain types shared across the predictor. Names follow GLOSSARY.md."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

_DISPLAY = {
    "PL": "Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "CL": "Champions League",
}
_UNDERSTAT = {
    "PL": "EPL",
    "PD": "La_liga",
    "BL1": "Bundesliga",
    "SA": "Serie_A",
    "FL1": "Ligue_1",
    "CL": None,
}
_ODDS_SPORT = {
    "PL": "soccer_epl",
    "PD": "soccer_spain_la_liga",
    "BL1": "soccer_germany_bundesliga",
    "SA": "soccer_italy_serie_a",
    "FL1": "soccer_france_ligue_one",
    "CL": "soccer_uefa_champs_league",
}
_CLUBELO_COUNTRY = {"PL": "ENG", "PD": "ESP", "BL1": "GER", "SA": "ITA", "FL1": "FRA", "CL": None}
_COUK_DIVISION = {"PL": "E0", "PD": "SP1", "BL1": "D1", "SA": "I1", "FL1": "F1", "CL": None}


class Competition(StrEnum):
    """football-data.org competition codes."""

    PL = "PL"
    PD = "PD"
    BL1 = "BL1"
    SA = "SA"
    FL1 = "FL1"
    CL = "CL"

    @property
    def display(self) -> str:
        return _DISPLAY[self.value]

    @property
    def is_league(self) -> bool:
        return self is not Competition.CL

    @property
    def understat_league(self) -> str | None:
        return _UNDERSTAT[self.value]

    @property
    def odds_sport(self) -> str:
        return _ODDS_SPORT[self.value]

    @property
    def clubelo_country(self) -> str | None:
        return _CLUBELO_COUNTRY[self.value]

    @property
    def couk_division(self) -> str | None:
        return _COUK_DIVISION[self.value]


LEAGUES: tuple[Competition, ...] = (
    Competition.PL,
    Competition.PD,
    Competition.BL1,
    Competition.SA,
    Competition.FL1,
)

Side = Literal["home", "away"]
Probabilities = dict[str, dict[str, float]]

MARKET_IDS: tuple[str, ...] = (
    "result",
    "btts",
    "total:1.5",
    "total:2.5",
    "total:3.5",
    "team:home:0.5",
    "team:home:1.5",
    "team:home:2.5",
    "team:away:0.5",
    "team:away:1.5",
    "team:away:2.5",
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Team(_Frozen):
    id: int
    name: str
    short_name: str
    tla: str | None = None
    country: str | None = None


class Match(_Frozen):
    id: int
    competition: Competition
    home: Team
    away: Team
    kickoff: AwareDatetime
    stage: str | None = None
    neutral: bool = False


class Result(_Frozen):
    home_goals: int
    away_goals: int


class MatchRecord(_Frozen):
    kickoff: AwareDatetime
    competition: Competition
    home_name: str
    away_name: str
    home_goals: int
    away_goals: int
    home_xg: float | None = None
    away_xg: float | None = None
    home_id: int | None = None
    away_id: int | None = None


class ProjectedGoals(_Frozen):
    home: float
    away: float


class TeamRating(_Frozen):
    attack: float
    concede: float
    elo: float | None = None


class LeagueBase(_Frozen):
    home_goals: float
    away_goals: float


class HotTake(_Frozen):
    """An answer where the model's probability and the Bookmaker Probability differ by a lot."""

    market: str
    answer: str
    model_p: float
    book_p: float

    @property
    def diff(self) -> float:
        """Positive when the model rates the answer higher than the bookmakers do."""
        return self.model_p - self.book_p


class Prediction(_Frozen):
    """A saved Prediction. `projected_base` is the stats-only model, `projected` what the Markets use.

    `bookmaker` holds Bookmaker Probabilities for the Markets bookmakers list; when there are none,
    `odds_note` says why.
    """

    id: int | None = None
    match: Match
    created_at: AwareDatetime
    projected_base: ProjectedGoals
    projected: ProjectedGoals
    probabilities: Probabilities
    shared_at: AwareDatetime | None = None
    bookmaker: Probabilities | None = None
    hot_takes: list[HotTake] = Field(default_factory=list)
    odds_note: str | None = None
    credits_remaining: int | None = None


def season_of(dt: datetime) -> int:
    """The season's start year: July onwards belongs to the season starting that year."""
    return dt.year if dt.month >= 7 else dt.year - 1
