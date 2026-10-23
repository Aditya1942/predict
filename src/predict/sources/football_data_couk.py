"""football-data.co.uk client: a league season's results with bookmaker closing odds (backtest only)."""

import csv
import io
from datetime import UTC, datetime, timedelta

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from predict.domain import Competition
from predict.errors import SourceError
from predict.http import CachedHttp

SOURCE = "football-data.co.uk"
BASE_URL = "https://www.football-data.co.uk/mmz4281"
TTL = timedelta(days=7)
_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "AvgCH", "AvgCD", "AvgCA", "AvgC>2.5", "AvgC<2.5")


class ClosingOdds(BaseModel):
    """One played Match with the market-average closing odds. `kickoff` is the match date at 00:00 UTC."""

    model_config = ConfigDict(frozen=True)

    kickoff: AwareDatetime
    home_name: str
    away_name: str
    home_goals: int
    away_goals: int
    odds_home: float = Field(gt=0)
    odds_draw: float = Field(gt=0)
    odds_away: float = Field(gt=0)
    odds_over25: float = Field(gt=0)
    odds_under25: float = Field(gt=0)


class FootballDataCoUk:
    def __init__(self, http: CachedHttp) -> None:
        self._http = http

    def season(self, comp: Competition, season: int) -> list[ClosingOdds]:
        """Played Matches of a league season that have closing odds, in file order. `season` is the start year."""
        division = comp.couk_division
        if division is None:
            raise SourceError(SOURCE, f"no data for the {comp.display}")
        url = f"{BASE_URL}/{season % 100:02d}{(season + 1) % 100:02d}/{division}.csv"
        return _parse(self._http.get_text(url, ttl=TTL))


def _parse(body: str) -> list[ClosingOdds]:
    reader = csv.DictReader(io.StringIO(body.lstrip("﻿")))
    missing = [column for column in _COLUMNS if column not in (reader.fieldnames or [])]
    if missing:
        raise SourceError(SOURCE, f"unexpected response: missing columns {', '.join(missing)}")
    try:
        rows = [_row(raw) for raw in reader]
    except ValueError as exc:  # a bad date or number; pydantic's ValidationError is a ValueError too
        raise SourceError(SOURCE, f"unexpected response: {exc}") from exc
    return [row for row in rows if row is not None]


def _row(raw: dict[str, str | None]) -> ClosingOdds | None:
    """None for a row that was not played yet or has no closing odds."""
    values = {column: (raw.get(column) or "").strip() for column in _COLUMNS}
    if not all(values.values()):
        return None
    return ClosingOdds(
        kickoff=_date(values["Date"]),
        home_name=values["HomeTeam"],
        away_name=values["AwayTeam"],
        home_goals=int(values["FTHG"]),
        away_goals=int(values["FTAG"]),
        odds_home=float(values["AvgCH"]),
        odds_draw=float(values["AvgCD"]),
        odds_away=float(values["AvgCA"]),
        odds_over25=float(values["AvgC>2.5"]),
        odds_under25=float(values["AvgC<2.5"]),
    )


def _date(text: str) -> datetime:
    """dd/mm/yyyy (older files: dd/mm/yy) as midnight UTC."""
    year_format = "%y" if len(text.rsplit("/", 1)[-1]) == 2 else "%Y"
    return datetime.strptime(text, f"%d/%m/{year_format}").replace(tzinfo=UTC)
