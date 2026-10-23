"""Walk-forward backtest of the stats model against bookmaker closing odds.

Every Match is rated only from same-league Understat records played strictly before its date, plus the
season before, so nothing from the day itself leaks in. `python -m predict.backtest --season 2025` runs all
five leagues and prints PASS or FAIL.
"""

import argparse
import sys
from bisect import bisect_left
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict

from predict.config import DEFAULT_HOME
from predict.domain import LEAGUES, Competition, MatchRecord, Team
from predict.errors import SourceError
from predict.http import CachedHttp
from predict.model.goals import projected_goals
from predict.model.markets import market_probabilities, pick
from predict.model.scoreline import scoreline_matrix
from predict.model.strength import combine, form_rating, league_base, season_rating
from predict.odds import devig
from predict.params import FORM_WEIGHT, FORM_WINDOW_DAYS, RHO, SEASON_FULL_WEIGHT_MATCHES
from predict.sources.football_data_couk import ClosingOdds, FootballDataCoUk
from predict.sources.understat import Understat
from predict.store import Store
from predict.teams import Aliases, TeamResolver

PASS_THRESHOLD = 0.02  # the model may trail the bookmakers' Match Result Brier by this much on average
_COUK = "football_data_couk"


class UnderstatSource(Protocol):
    def league_matches(self, comp: Competition, season: int) -> list[MatchRecord]: ...


class CoukSource(Protocol):
    def season(self, comp: Competition, season: int) -> list[ClosingOdds]: ...


class BacktestPrediction(BaseModel):
    """The model's Match Result and Total Goals 2.5 probabilities for one played Match, from earlier dates only."""

    model_config = ConfigDict(frozen=True)

    odds: ClosingOdds
    result: dict[str, float]
    total25: dict[str, float]


class BacktestReport(BaseModel):
    """Mean Brier scores (lower is better) and Match Result hit counts, model against bookmakers."""

    model_config = ConfigDict(frozen=True)

    n: int
    model_brier_result: float
    book_brier_result: float
    model_brier_total25: float
    book_brier_total25: float
    model_hits_result: int
    book_hits_result: int


def brier(probs: dict[str, float], outcome: str) -> float:
    """Sum over the answers of (probability - 1 if it happened else 0) squared."""
    if outcome not in probs:
        raise ValueError(f"outcome {outcome!r} is not one of the answers {list(probs)}")
    return sum((p - (answer == outcome)) ** 2 for answer, p in probs.items())


def walk_forward(
    comp: Competition,
    season: int,
    understat: UnderstatSource,
    couk: CoukSource,
    aliases: Aliases,
    *,
    rho: float = RHO,
    form_weight: float = FORM_WEIGHT,
    full: int = SEASON_FULL_WEIGHT_MATCHES,
) -> list[BacktestPrediction]:
    """Predict every Match in the football-data.co.uk file, in file order.

    A Match with no earlier Understat record in either season is left out.
    """
    odds = couk.season(comp, season)
    last = sorted(understat.league_matches(comp, season - 1), key=lambda r: r.kickoff)
    current = sorted(understat.league_matches(comp, season), key=lambda r: r.kickoff)
    couk_ids = _map_names(odds, last, current, aliases)
    id_of = {name: i for i, name in enumerate(sorted(_names(last, current)), start=1)}
    last, current = _with_ids(last, id_of), _with_ids(current, id_of)
    last_times, current_times = [r.kickoff for r in last], [r.kickoff for r in current]

    predictions = []
    for row in odds:
        before_last = last[: bisect_left(last_times, row.kickoff)]
        before_current = current[: bisect_left(current_times, row.kickoff)]
        if not before_last and not before_current:
            continue
        base = league_base(before_current, before_last, full=full)
        comp_avg = {comp: (base.home_goals + base.away_goals) / 2}
        window_start = row.kickoff - timedelta(days=FORM_WINDOW_DAYS)
        recent = [r for r in (*before_last, *before_current) if r.kickoff >= window_start]
        ratings = [
            combine(
                season_rating(couk_ids[name], before_current, before_last, full=full),
                form_rating(couk_ids[name], recent, comp_avg),
                weight=form_weight,
            )
            for name in (row.home_name, row.away_name)
        ]
        projected = projected_goals(ratings[0], ratings[1], base)
        probabilities = market_probabilities(scoreline_matrix(projected.home, projected.away, rho=rho))
        predictions.append(
            BacktestPrediction(odds=row, result=probabilities["result"], total25=probabilities["total:2.5"])
        )
    return predictions


def run_backtest(
    comp: Competition,
    season: int,
    understat: UnderstatSource,
    couk: CoukSource,
    aliases: Aliases,
    *,
    rho: float = RHO,
    form_weight: float = FORM_WEIGHT,
    full: int = SEASON_FULL_WEIGHT_MATCHES,
) -> BacktestReport:
    predictions = walk_forward(comp, season, understat, couk, aliases, rho=rho, form_weight=form_weight, full=full)
    if not predictions:
        raise SourceError("backtest", f"no matches to evaluate for the {comp.display} {season}/{season + 1} season")
    model_result = book_result = model_total = book_total = 0.0
    model_hits = book_hits = 0
    for f in predictions:
        row = f.odds
        outcome = "home" if row.home_goals > row.away_goals else "away" if row.home_goals < row.away_goals else "draw"
        total = "over" if row.home_goals + row.away_goals > 2.5 else "under"
        book_probs = devig({"home": row.odds_home, "draw": row.odds_draw, "away": row.odds_away})
        book_ou = devig({"over": row.odds_over25, "under": row.odds_under25})
        model_result += brier(f.result, outcome)
        book_result += brier(book_probs, outcome)
        model_total += brier(f.total25, total)
        book_total += brier(book_ou, total)
        model_hits += pick(f.result) == outcome
        book_hits += pick(book_probs) == outcome
    n = len(predictions)
    return BacktestReport(
        n=n,
        model_brier_result=model_result / n,
        book_brier_result=book_result / n,
        model_brier_total25=model_total / n,
        book_brier_total25=book_total / n,
        model_hits_result=model_hits,
        book_hits_result=book_hits,
    )


def format_reports(season: int, reports: dict[Competition, BacktestReport]) -> tuple[str, bool]:
    """The printout (a row per league, the mean, then PASS or FAIL) and whether the check passed."""
    lines = [
        f"Backtest {season}/{(season + 1) % 100:02d} against bookmaker closing odds (Brier score, lower is better)",
        f"{'League':<18}{'Matches':>8} | {'Result: model':>13}{'book':>8}{'gap':>9} |"
        f" {'Total 2.5: model':>16}{'book':>8}{'gap':>9} | {'Result hits: model':>18}{'book':>6}",
    ]
    for comp, r in reports.items():
        lines.append(
            f"{comp.display:<18}{r.n:>8} | {r.model_brier_result:>13.4f}{r.book_brier_result:>8.4f}"
            f"{r.model_brier_result - r.book_brier_result:>+9.4f} |"
            f" {r.model_brier_total25:>16.4f}{r.book_brier_total25:>8.4f}"
            f"{r.model_brier_total25 - r.book_brier_total25:>+9.4f} |"
            f" {r.model_hits_result:>18}{r.book_hits_result:>6}"
        )
    gap = _mean(r.model_brier_result - r.book_brier_result for r in reports.values())
    total_gap = _mean(r.model_brier_total25 - r.book_brier_total25 for r in reports.values())
    lines.append(f"{'mean':<18}{'':>8} | {'':>13}{'':>8}{gap:>+9.4f} | {'':>16}{'':>8}{total_gap:>+9.4f} |")
    passed = gap <= PASS_THRESHOLD + 1e-9  # allow for float rounding right at the limit
    lines.append(f"Mean Match Result Brier gap {gap:+.4f}, limit {PASS_THRESHOLD}: {'PASS' if passed else 'FAIL'}")
    return "\n".join(lines) + "\n", passed


def main(argv: Sequence[str] | None = None) -> int:
    """Exit 0 on PASS, 1 on FAIL, 2 when a source fails."""
    parser = argparse.ArgumentParser(prog="python -m predict.backtest", description=__doc__)
    parser.add_argument("--season", type=int, required=True, help="start year of the season, e.g. 2025 for 2025/26")
    parser.add_argument("--home", type=_path, default=DEFAULT_HOME, help="folder of the cache database")
    args = parser.parse_args(argv)

    store = Store(args.home / "predict.db")
    client = httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        http = CachedHttp(store, client, now=lambda: datetime.now(UTC))
        understat, couk, aliases = Understat(http), FootballDataCoUk(http), Aliases.load()
        reports = {}
        for comp in LEAGUES:
            print(f"{comp.display} ...", file=sys.stderr, flush=True)
            reports[comp] = run_backtest(comp, args.season, understat, couk, aliases)
    except SourceError as exc:
        print(exc, file=sys.stderr)
        return 2
    finally:
        client.close()
        store.close()
    text, passed = format_reports(args.season, reports)
    print(text, end="")
    return 0 if passed else 1


def _path(text: str) -> Path:
    return Path(text).expanduser()


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _names(*seasons: list[MatchRecord]) -> set[str]:
    return {name for records in seasons for r in records for name in (r.home_name, r.away_name)}


def _with_ids(records: list[MatchRecord], id_of: dict[str, int]) -> list[MatchRecord]:
    return [r.model_copy(update={"home_id": id_of[r.home_name], "away_id": id_of[r.away_name]}) for r in records]


def _map_names(
    odds: list[ClosingOdds], last: list[MatchRecord], current: list[MatchRecord], aliases: Aliases
) -> dict[str, int]:
    """football-data.co.uk team name -> id (the same pseudo-ids `_with_ids` gives the Understat names)."""
    understat_names = sorted(_names(last, current))
    teams = [Team(id=i, name=name, short_name=name) for i, name in enumerate(understat_names, start=1)]
    couk_names = {name for row in odds for name in (row.home_name, row.away_name)}
    mapped = TeamResolver(teams, aliases).map_source_names(_COUK, sorted(couk_names))
    unmapped = sorted(name for name, team_id in mapped.items() if team_id is None)
    if unmapped:
        raise SourceError(
            "football-data.co.uk",
            f"no Understat team for {', '.join(map(repr, unmapped))}; add them to [source.{_COUK}] in aliases.toml",
        )
    return {name: team_id for name, team_id in mapped.items() if team_id is not None}


if __name__ == "__main__":
    raise SystemExit(main())
