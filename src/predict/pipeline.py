"""Find the Match a user means and turn it into a saved stats-only Prediction."""

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Protocol

from predict.domain import Match, MatchDossier, Prediction, Probabilities, ProjectedGoals, Team
from predict.errors import (
    AmbiguousTeam,
    JevFailed,
    MatchNotFound,
    OddsUnavailable,
    PredictionLocked,
    ResearchFailed,
    SourceError,
)
from predict.model.markets import market_probabilities
from predict.model.scoreline import scoreline_matrix
from predict.news.adjust import apply_impacts
from predict.news.scoring import JevClient, score_dossier
from predict.odds import hot_takes
from predict.params import SEARCH_WINDOW_DAYS
from predict.ratings import RatedMatch
from predict.sources.odds_api import OddsSnapshot
from predict.store import Store
from predict.teams import TeamResolver, parse_query


class UpcomingSource(Protocol):
    def team_upcoming(self, team_id: int, date_from: date, date_to: date) -> list[Match]: ...


class Rater(Protocol):
    def rate_match(self, match: Match) -> RatedMatch: ...


class OddsSource(Protocol):
    def snapshot(self, match: Match) -> OddsSnapshot: ...


class DossierSource(Protocol):
    def get(self, match: Match, *, fresh: bool, model: str) -> MatchDossier: ...


NO_ODDS_KEY = "no ODDS_API_KEY set"
NO_JEV_KEY = "no TYPESAFE_API_KEY set"
CHEAPER_MODEL = "claude-sonnet-5-5"


class Predictor:
    def __init__(
        self,
        fdo: UpcomingSource,
        ratings: Rater,
        resolver: TeamResolver,
        store: Store,
        odds: OddsSource | None,
        dossiers: DossierSource,
        jev: JevClient | None,
        claude_model: str,
        jev_model: str,
        *,
        now: Callable[[], datetime],
    ) -> None:
        """`odds` is None when no ODDS_API_KEY is set; Predictions then carry that as their `odds_note`.

        `jev` is None when no TYPESAFE_API_KEY is set; news is then not researched and Predictions are stats-only.
        `claude_model` is the default model for research, `--model` overrides it per run.
        """
        self._fdo = fdo
        self._ratings = ratings
        self._resolver = resolver
        self._store = store
        self._odds = odds
        self._dossiers = dossiers
        self._jev = jev
        self._claude_model = claude_model
        self._jev_model = jev_model
        self._now = now

    def find_match(self, query: str, choose: Callable[[list[Team]], Team]) -> Match:
        """The earliest Match still to kick off for "A v B" (either home/away order) or for one team.

        `choose` picks a team when the text could mean several; whatever it raises propagates.
        """
        first_text, second_text = parse_query(query)
        first = self._resolve(first_text, choose)
        second = self._resolve(second_text, choose) if second_text is not None else None

        now = self._now().astimezone(UTC)
        today = now.date()
        # The upcoming list may be cached from before kickoff, so drop any Match that has started since.
        upcoming = [
            m
            for m in self._fdo.team_upcoming(first.id, today, today + timedelta(days=SEARCH_WINDOW_DAYS))
            if m.kickoff > now
        ]
        if second is None:
            matches = upcoming
            nothing = f"no upcoming match for {first.name}"
        else:
            matches = [m for m in upcoming if {m.home.id, m.away.id} == {first.id, second.id}]
            nothing = f"no upcoming match between {first.name} and {second.name}"
        if not matches:
            raise MatchNotFound(f"{nothing} in the next {SEARCH_WINDOW_DAYS} days")
        return min(matches, key=lambda m: m.kickoff)

    def predict(
        self, match: Match, *, fresh: bool = False, model: str | None = None, share: bool = False
    ) -> Prediction:
        """Rate the Match, apply Team News, compute every Market and save the Prediction. Locked from kickoff on.

        `fresh` ignores a saved News Dossier, `model` overrides the Claude model for this run and `share` marks the
        Prediction shared now. When news cannot be had the Prediction goes ahead on the stats model with the reason.
        """
        now = self._now().astimezone(UTC)
        if now >= match.kickoff:
            raise PredictionLocked(match)
        rated = self._ratings.rate_match(match)
        projected, news = self._news(match, rated.projected, now, fresh=fresh, model=model or self._claude_model)
        probabilities = market_probabilities(scoreline_matrix(projected.home, projected.away))
        bookmaker = self._bookmaker_fields(match, probabilities)
        # Research can run for minutes, so check the lock again now and stamp the Prediction with this time.
        saved_at = self._now().astimezone(UTC)
        if saved_at >= match.kickoff:
            raise PredictionLocked(match)
        prediction = Prediction(
            match=match,
            created_at=saved_at,
            projected_base=rated.projected,
            projected=projected,
            probabilities=probabilities,
            shared_at=saved_at if share else None,
            **news,
            **bookmaker,
        )
        return prediction.model_copy(update={"id": self._store.save_prediction(prediction)})

    def _news(
        self, match: Match, base: ProjectedGoals, now: datetime, *, fresh: bool, model: str
    ) -> tuple[ProjectedGoals, dict]:
        """The adjusted Projected Goals and the Prediction fields about news; the base and the reason if it fails."""
        if self._jev is None:
            return base, {"news_note": NO_JEV_KEY}
        try:
            dossier = self._dossiers.get(match, fresh=fresh, model=model)
            impacts = score_dossier(dossier, match, self._jev, today=now.date())
        except (ResearchFailed, JevFailed) as failed:
            return base, {"news_note": _with_advice(failed.reason)}
        except SourceError as failed:
            return base, {"news_note": str(failed)}
        fields = {
            "impacts": impacts,
            "news_applied": True,
            "claude_model": dossier.claude_model,
            "jev_model": self._jev_model,
        }
        return apply_impacts(base, impacts), fields

    def _bookmaker_fields(self, match: Match, probabilities: Probabilities) -> dict:
        """Prediction fields from the bookmakers; if they can't be had the Prediction goes ahead with the reason."""
        if self._odds is None:
            return {"odds_note": NO_ODDS_KEY}
        try:
            snapshot = self._odds.snapshot(match)
        except OddsUnavailable as unavailable:
            return {"odds_note": unavailable.reason}
        return {
            "bookmaker": snapshot.probabilities,
            "hot_takes": hot_takes(probabilities, snapshot.probabilities),
            "credits_remaining": snapshot.credits_remaining,
        }

    def _resolve(self, text: str, choose: Callable[[list[Team]], Team]) -> Team:
        try:
            return self._resolver.resolve_query(text)
        except AmbiguousTeam as ambiguous:
            return choose(ambiguous.candidates)


def _with_advice(reason: str) -> str:
    """The failure reason, with a way out when Claude's usage limit is the cause."""
    if "usage limit" in reason.lower():
        return f"{reason}; rerun with --model {CHEAPER_MODEL}"
    return reason
