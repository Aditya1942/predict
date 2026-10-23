"""Settling Matches with their Results, and the Prediction Log built from the Counted Predictions."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from predict.domain import MARKET_IDS, HotTake, Prediction, Result
from predict.model.markets import pick
from predict.params import SETTLE_AFTER, VOID_KICKOFF_SHIFT
from predict.sources.football_data_org import MatchStatus
from predict.store import Store

VOID_STATUSES = frozenset({"POSTPONED", "CANCELLED", "SUSPENDED", "AWARDED"})
RECENT = 10
_BATCH = 20  # Matches asked about per request


class StatusSource(Protocol):
    def matches_by_ids(self, ids: list[int]) -> list[MatchStatus]: ...


def settle_pending(
    store: Store,
    fdo: StatusSource,
    now: datetime,
    *,
    settle_after: timedelta = SETTLE_AFTER,
    void_shift: timedelta = VOID_KICKOFF_SHIFT,
) -> int:
    """Settle every predicted Match that kicked off at least `settle_after` ago and has no result yet.

    A Match is void if it was postponed, cancelled, suspended or awarded, or if its kickoff has moved more than
    `void_shift` from the one it was predicted for (even if it has since been played, as the Prediction was made
    for another time). Otherwise a FINISHED Match is settled with its 90-minute Result; any other status is left
    for a later run. Returns how many Matches were settled.
    """
    ids = store.unsettled_match_ids(now - settle_after)
    predicted = {i: max(p.match.kickoff for p in store.predictions_for_match(i)) for i in ids}
    settled = 0
    for start in range(0, len(ids), _BATCH):
        for found in fdo.matches_by_ids(ids[start : start + _BATCH]):
            kickoff = predicted.get(found.match.id)
            if kickoff is None:
                continue
            if found.status in VOID_STATUSES or abs(found.match.kickoff - kickoff) > void_shift:
                store.save_result(found.match.id, "void", None, now)
            elif found.status == "FINISHED" and found.result is not None:
                store.save_result(found.match.id, "finished", found.result, now)
            else:
                continue
            settled += 1
    return settled


def counted_prediction(preds: list[Prediction]) -> Prediction | None:
    """The one Prediction of a Match the Log scores: the latest shared one, otherwise the latest created.

    Only Predictions created before kickoff are eligible.
    """
    eligible = [p for p in preds if p.created_at < p.match.kickoff]
    shared = [p for p in eligible if p.shared_at is not None]
    if shared:
        return max(shared, key=lambda p: (p.shared_at, p.created_at, p.id or 0))
    return max(eligible, key=lambda p: (p.created_at, p.id or 0), default=None)


def outcome(market_id: str, result: Result) -> str:
    """The answer of the Market that happened."""
    home, away = result.home_goals, result.away_goals
    kind, *rest = market_id.split(":")
    if kind == "result" and not rest:
        return "home" if home > away else "away" if away > home else "draw"
    if kind == "btts" and not rest:
        return "yes" if home > 0 and away > 0 else "no"
    if kind == "total" and len(rest) == 1:
        return "over" if home + away > float(rest[0]) else "under"
    if kind == "team" and len(rest) == 2 and rest[0] in ("home", "away"):
        return "over" if (home if rest[0] == "home" else away) > float(rest[1]) else "under"
    raise ValueError(f"unknown market {market_id!r}")


def _group(market_id: str) -> str:
    """The Hit Rate group a Market belongs to; both teams' Team Goals at a Line are one group."""
    kind, *rest = market_id.split(":")
    if kind == "result":
        return "Match Result"
    if kind == "btts":
        return "BTTS"
    if kind == "total":
        return f"Total Goals {rest[0]}"
    return f"Team Goals {rest[1]}"


def _hot_take_hit(take: HotTake, happened: str) -> bool:
    """Rated higher than the bookmakers and it happened, or rated lower and it did not."""
    return (take.diff > 0 and happened == take.answer) or (take.diff < 0 and happened != take.answer)


@dataclass(frozen=True)
class LogSummary:
    """`hit_rates` and `hot_takes` are (hits, scored). `recent` is the last Counted Predictions, newest first."""

    hit_rates: dict[str, tuple[int, int]]
    hot_takes: tuple[int, int]
    recent: list[tuple[Prediction, Result]]


def summarize(store: Store) -> LogSummary:
    """Score the Counted Prediction of every finished Match; void and unsettled Matches never count."""
    results = store.results()
    by_match: dict[int, list[Prediction]] = {}
    for prediction in store.all_predictions():
        by_match.setdefault(prediction.match.id, []).append(prediction)
    counted: list[tuple[Prediction, Result]] = []
    for match_id, preds in by_match.items():
        status, result = results.get(match_id, ("", None))
        chosen = counted_prediction(preds)
        if status == "finished" and result is not None and chosen is not None:
            counted.append((chosen, result))
    counted.sort(key=lambda pair: (pair[0].match.kickoff, pair[0].match.id), reverse=True)

    hits: dict[str, list[int]] = {_group(market_id): [0, 0] for market_id in MARKET_IDS}
    hot_hits = hot_scored = 0
    for prediction, result in counted:
        for market_id, answers in prediction.probabilities.items():
            score = hits[_group(market_id)]
            score[0] += pick(answers) == outcome(market_id, result)
            score[1] += 1
        for take in prediction.hot_takes:
            hot_hits += _hot_take_hit(take, outcome(take.market, result))
            hot_scored += 1
    return LogSummary(
        hit_rates={group: (h, n) for group, (h, n) in hits.items()},
        hot_takes=(hot_hits, hot_scored),
        recent=counted[:RECENT],
    )
