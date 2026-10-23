"""Find the Match a user means and turn it into a saved stats-only Prediction."""

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Protocol

from predict.domain import Match, Prediction, Team
from predict.errors import AmbiguousTeam, MatchNotFound, PredictionLocked
from predict.model.markets import market_probabilities
from predict.model.scoreline import scoreline_matrix
from predict.params import SEARCH_WINDOW_DAYS
from predict.ratings import RatedMatch
from predict.store import Store
from predict.teams import TeamResolver, parse_query


class UpcomingSource(Protocol):
    def team_upcoming(self, team_id: int, date_from: date, date_to: date) -> list[Match]: ...


class Rater(Protocol):
    def rate_match(self, match: Match) -> RatedMatch: ...


class Predictor:
    def __init__(
        self,
        fdo: UpcomingSource,
        ratings: Rater,
        resolver: TeamResolver,
        store: Store,
        *,
        now: Callable[[], datetime],
    ) -> None:
        self._fdo = fdo
        self._ratings = ratings
        self._resolver = resolver
        self._store = store
        self._now = now

    def find_match(self, query: str, choose: Callable[[list[Team]], Team]) -> Match:
        """The earliest upcoming Match for "A v B" (either home/away order) or for one team.

        `choose` picks a team when the text could mean several; whatever it raises propagates.
        """
        first_text, second_text = parse_query(query)
        first = self._resolve(first_text, choose)
        second = self._resolve(second_text, choose) if second_text is not None else None

        today = self._now().astimezone(UTC).date()
        upcoming = self._fdo.team_upcoming(first.id, today, today + timedelta(days=SEARCH_WINDOW_DAYS))
        if second is None:
            matches = upcoming
            nothing = f"no upcoming match for {first.name}"
        else:
            matches = [m for m in upcoming if {m.home.id, m.away.id} == {first.id, second.id}]
            nothing = f"no upcoming match between {first.name} and {second.name}"
        if not matches:
            raise MatchNotFound(f"{nothing} in the next {SEARCH_WINDOW_DAYS} days")
        return min(matches, key=lambda m: m.kickoff)

    def predict(self, match: Match) -> Prediction:
        """Rate the Match, compute every Market and save the Prediction. Locked from kickoff on."""
        now = self._now().astimezone(UTC)
        if now >= match.kickoff:
            raise PredictionLocked(match)
        rated = self._ratings.rate_match(match)
        matrix = scoreline_matrix(rated.projected.home, rated.projected.away)
        prediction = Prediction(
            match=match,
            created_at=now,
            projected_base=rated.projected,
            projected=rated.projected,
            probabilities=market_probabilities(matrix),
        )
        return prediction.model_copy(update={"id": self._store.save_prediction(prediction)})

    def _resolve(self, text: str, choose: Callable[[list[Team]], Team]) -> Team:
        try:
            return self._resolver.resolve_query(text)
        except AmbiguousTeam as ambiguous:
            return choose(ambiguous.candidates)
