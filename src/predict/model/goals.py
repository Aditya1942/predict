"""Projected Goals from team ratings."""

import math

from predict.domain import LeagueBase, ProjectedGoals, TeamRating
from predict.params import CL_AWAY_GOALS, CL_HOME_GOALS, ELO_KAPPA

CL_BASE = LeagueBase(home_goals=CL_HOME_GOALS, away_goals=CL_AWAY_GOALS)


def projected_goals(
    home: TeamRating,
    away: TeamRating,
    base: LeagueBase,
    *,
    neutral: bool = False,
    kappa: float = ELO_KAPPA,
) -> ProjectedGoals:
    """Goals each side is projected to score.

    On a neutral venue both sides use the mean of the base home and away goals. When both teams
    have an Elo, the gap shifts goals towards the stronger side.
    """
    if neutral:
        base_home = base_away = (base.home_goals + base.away_goals) / 2
    else:
        base_home, base_away = base.home_goals, base.away_goals
    goals_home = base_home * home.attack * away.concede
    goals_away = base_away * away.attack * home.concede
    if home.elo is not None and away.elo is not None:
        factor = math.exp(kappa * (home.elo - away.elo) / 2)
        goals_home *= factor
        goals_away /= factor
    return ProjectedGoals(home=goals_home, away=goals_away)
