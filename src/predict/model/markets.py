"""Market probabilities from a scoreline matrix."""

import numpy as np

from predict.domain import Probabilities
from predict.params import TEAM_LINES, TOTAL_LINES


def _over_under(mask_over: np.ndarray, matrix: np.ndarray) -> dict[str, float]:
    return {"over": float(matrix[mask_over].sum()), "under": float(matrix[~mask_over].sum())}


def market_probabilities(matrix: np.ndarray) -> Probabilities:
    """Probabilities for every Market id, keyed by answer."""
    home = np.arange(matrix.shape[0])[:, None]
    away = np.arange(matrix.shape[1])[None, :]
    probs: Probabilities = {
        "result": {
            "home": float(np.tril(matrix, -1).sum()),
            "draw": float(np.trace(matrix)),
            "away": float(np.triu(matrix, 1).sum()),
        },
        "btts": {
            "yes": float(matrix[1:, 1:].sum()),
            "no": float(matrix[0, :].sum() + matrix[1:, 0].sum()),
        },
    }
    for line in TOTAL_LINES:
        probs[f"total:{line}"] = _over_under(home + away > line, matrix)
    for line in TEAM_LINES:
        probs[f"team:home:{line}"] = _over_under(np.broadcast_to(home > line, matrix.shape), matrix)
    for line in TEAM_LINES:
        probs[f"team:away:{line}"] = _over_under(np.broadcast_to(away > line, matrix.shape), matrix)
    return probs


def pick(answers: dict[str, float]) -> str:
    """The highest-probability answer; ties go to the first in dict order."""
    return max(answers, key=answers.__getitem__)
