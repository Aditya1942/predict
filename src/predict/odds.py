"""Bookmaker odds helpers."""


def devig(prices: dict[str, float]) -> dict[str, float]:
    """Implied probabilities from decimal odds: 1/price for each answer, scaled to sum to 1.

    Dividing by the sum removes the bookmaker's margin proportionally across the answers.
    """
    if not prices or any(price <= 0 for price in prices.values()):
        raise ValueError(f"devig needs decimal prices above 0, got {prices}")
    implied = {answer: 1 / price for answer, price in prices.items()}
    total = sum(implied.values())
    return {answer: p / total for answer, p in implied.items()}
