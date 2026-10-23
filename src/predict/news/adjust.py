"""Applying scored News Factors to Projected Goals, capped."""

from math import prod

from predict.domain import FactorImpact, ProjectedGoals, Side
from predict.params import NEWS_CAP

_NEGLIGIBLE = 0.005  # a multiplier closer to 1 than this rounds to a "+0%" change, so it is not worth listing


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


def _effect(impact: FactorImpact) -> float:
    return max(abs(impact.attack_mult - 1), abs(impact.concede_mult - 1))


def top_reasons(impacts: list[FactorImpact], side: Side, n: int = 3) -> list[FactorImpact]:
    """The side's `n` News Factors that move a goal projection the most, largest first; negligible ones are left out."""
    ours = [i for i in impacts if i.side == side and round(_effect(i), 6) >= _NEGLIGIBLE]
    return sorted(ours, key=_effect, reverse=True)[:n]
