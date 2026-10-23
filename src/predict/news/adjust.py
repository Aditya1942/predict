"""Applying scored News Factors to Projected Goals, capped."""

from math import prod

from predict.domain import FactorImpact, ProjectedGoals, Side
from predict.params import NEWS_CAP


def apply_impacts(
    base: ProjectedGoals, impacts: list[FactorImpact], *, cap: tuple[float, float] = NEWS_CAP
) -> ProjectedGoals:
    """Each side's goals times its own attack multipliers and the opponent's concede multipliers, within `cap`."""
    low, high = cap

    def factor(attacking: Side, defending: Side) -> float:
        product = prod(i.attack_mult for i in impacts if i.side == attacking) * prod(
            i.concede_mult for i in impacts if i.side == defending
        )
        return min(max(product, low), high)

    return ProjectedGoals(home=base.home * factor("home", "away"), away=base.away * factor("away", "home"))


def top_reasons(impacts: list[FactorImpact], side: Side, n: int = 3) -> list[FactorImpact]:
    """The side's `n` News Factors that move a goal projection the most, largest first."""
    ours = [i for i in impacts if i.side == side]
    return sorted(ours, key=lambda i: max(abs(i.attack_mult - 1), abs(i.concede_mult - 1)), reverse=True)[:n]
