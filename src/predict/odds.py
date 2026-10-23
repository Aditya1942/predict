"""Bookmaker odds helpers."""

from predict.domain import HotTake, Probabilities
from predict.params import HOT_TAKE_THRESHOLD

_EPSILON = 1e-9  # so a gap of exactly 10 points counts despite float noise (0.60 - 0.50 is 0.0999...)


def devig(prices: dict[str, float]) -> dict[str, float]:
    """Implied probabilities from decimal odds: 1/price for each answer, scaled to sum to 1.

    Dividing by the sum removes the bookmaker's margin proportionally across the answers.
    """
    if not prices or any(price <= 0 for price in prices.values()):
        raise ValueError(f"devig needs decimal prices above 0, got {prices}")
    implied = {answer: 1 / price for answer, price in prices.items()}
    total = sum(implied.values())
    return {answer: p / total for answer, p in implied.items()}


def hot_takes(model: Probabilities, book: Probabilities, *, threshold: float = HOT_TAKE_THRESHOLD) -> list[HotTake]:
    """Answers where the model and the bookmakers are `threshold` or more apart, largest gap first.

    In the three-way Match Result every answer that differs either way counts. In a two-way Market only the
    answer the model rates higher is reported, as the other answer is the same gap seen from the other side.
    Markets and answers missing from `book` are skipped.
    """
    found: list[HotTake] = []
    for market_id, answers in model.items():
        listed = book.get(market_id)
        if listed is None:
            continue
        for answer, model_p in answers.items():
            if answer not in listed:
                continue
            take = HotTake(market=market_id, answer=answer, model_p=model_p, book_p=listed[answer])
            gap = abs(take.diff) if market_id == "result" else take.diff
            if gap >= threshold - _EPSILON:
                found.append(take)
    return sorted(found, key=lambda take: abs(take.diff), reverse=True)
