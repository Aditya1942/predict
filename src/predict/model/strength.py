"""Season Strength and Form ratings."""

import math
from collections.abc import Mapping

from predict.domain import Competition, LeagueBase, MatchRecord, TeamRating
from predict.params import (
    FORM_FLOOR,
    FORM_MATCHES,
    FORM_WEIGHT,
    FORM_WEIGHTS,
    PROMOTED_ATTACK,
    PROMOTED_CONCEDE,
    SEASON_FULL_WEIGHT_MATCHES,
    XG_WEIGHT,
)


def performance(goals: int, xg: float | None, *, xg_weight: float = XG_WEIGHT) -> float:
    """Blend of xG and actual goals; just the goals when there is no xG."""
    if xg is None:
        return float(goals)
    return xg_weight * xg + (1 - xg_weight) * goals


def season_weight(matches_played: float, *, full: int = SEASON_FULL_WEIGHT_MATCHES) -> float:
    """How much a season's numbers count: grows linearly to 1 at `full` matches."""
    return min(1.0, matches_played / full)


def _involves(record: MatchRecord, team_id: int) -> bool:
    return team_id in (record.home_id, record.away_id)


def _performances(record: MatchRecord, team_id: int) -> tuple[float, float]:
    """(performance for, performance against) from the team's point of view."""
    home = performance(record.home_goals, record.home_xg)
    away = performance(record.away_goals, record.away_xg)
    return (home, away) if record.home_id == team_id else (away, home)


def _mean_goals(records: list[MatchRecord]) -> tuple[float, float]:
    n = len(records)
    return sum(r.home_goals for r in records) / n, sum(r.away_goals for r in records) / n


def league_base(
    current: list[MatchRecord], last: list[MatchRecord], *, full: int = SEASON_FULL_WEIGHT_MATCHES
) -> LeagueBase:
    """Mean home and away goals per match, blending the current season with the last one."""
    if not current and not last:
        raise ValueError("league_base needs at least one season of records")
    if not last:
        home, away = _mean_goals(current)
        return LeagueBase(home_goals=home, away_goals=away)
    last_home, last_away = _mean_goals(last)
    if not current:
        return LeagueBase(home_goals=last_home, away_goals=last_away)
    cur_home, cur_away = _mean_goals(current)
    teams = {r.home_id if r.home_id is not None else r.home_name for r in current}
    teams |= {r.away_id if r.away_id is not None else r.away_name for r in current}
    w = season_weight(2 * len(current) / len(teams), full=full)
    return LeagueBase(
        home_goals=w * cur_home + (1 - w) * last_home,
        away_goals=w * cur_away + (1 - w) * last_away,
    )


def _season_ratio(team_id: int, records: list[MatchRecord]) -> tuple[float, float, int] | None:
    """(attack, concede, matches played) relative to the league mean, or None if the team is absent."""
    mine = [r for r in records if _involves(r, team_id)]
    if not mine:
        return None
    league_mean = sum(
        performance(r.home_goals, r.home_xg) + performance(r.away_goals, r.away_xg) for r in records
    ) / (2 * len(records))
    if league_mean <= 0:
        return 1.0, 1.0, len(mine)
    pairs = [_performances(r, team_id) for r in mine]
    attack = sum(p[0] for p in pairs) / len(pairs) / league_mean
    concede = sum(p[1] for p in pairs) / len(pairs) / league_mean
    return attack, concede, len(mine)


def season_rating(
    team_id: int,
    current: list[MatchRecord],
    last: list[MatchRecord],
    *,
    full: int = SEASON_FULL_WEIGHT_MATCHES,
) -> TeamRating:
    """Attack and concede relative to the league, blending this season with last season."""
    last_ratio = _season_ratio(team_id, last)
    last_attack, last_concede = (last_ratio[0], last_ratio[1]) if last_ratio else (PROMOTED_ATTACK, PROMOTED_CONCEDE)
    current_ratio = _season_ratio(team_id, current)
    if current_ratio is None:
        return TeamRating(attack=last_attack, concede=last_concede)
    cur_attack, cur_concede, played = current_ratio
    w = season_weight(played, full=full)
    return TeamRating(
        attack=w * cur_attack + (1 - w) * last_attack,
        concede=w * cur_concede + (1 - w) * last_concede,
    )


def form_rating(
    team_id: int,
    recent: list[MatchRecord],
    comp_avg: Mapping[Competition, float],
    *,
    floor: float = FORM_FLOOR,
) -> TeamRating | None:
    """Recency-weighted attack and concede over the team's latest matches, or None without any.

    Both are floored so a run of clean sheets or blanks cannot zero out a Projected Goals side.
    """
    mine = sorted((r for r in recent if _involves(r, team_id)), key=lambda r: r.kickoff, reverse=True)
    mine = mine[:FORM_MATCHES]
    if not mine:
        return None
    weights = FORM_WEIGHTS[: len(mine)]
    total = sum(weights)
    attack = concede = 0.0
    for weight, record in zip(weights, mine, strict=True):
        perf_for, perf_against = _performances(record, team_id)
        avg = comp_avg[record.competition]
        attack += weight * perf_for / avg
        concede += weight * perf_against / avg
    return TeamRating(attack=max(floor, attack / total), concede=max(floor, concede / total))


def combine(season: TeamRating, form: TeamRating | None, *, weight: float = FORM_WEIGHT) -> TeamRating:
    """Geometric blend of season and form ratings; the season Elo is kept."""
    if form is None:
        return season
    return TeamRating(
        attack=math.pow(season.attack, 1 - weight) * math.pow(form.attack, weight),
        concede=math.pow(season.concede, 1 - weight) * math.pow(form.concede, weight),
        elo=season.elo,
    )
