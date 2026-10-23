"""Errors raised across the predictor."""

from predict.domain import Match, Team


class SourceError(Exception):
    """A data source failed or returned something unusable."""

    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"{source}: {detail}")
        self.source = source
        self.detail = detail


class TeamNotFound(Exception):
    """No team is close enough to the text the user typed."""

    def __init__(self, text: str, suggestions: list[str]) -> None:
        super().__init__(f"no team matches {text!r}" + (f" (try: {', '.join(suggestions)})" if suggestions else ""))
        self.text = text
        self.suggestions = suggestions


class AmbiguousTeam(Exception):
    """The text fits several teams; `candidates` is ordered best score first, then by name."""

    def __init__(self, text: str, candidates: list[Team]) -> None:
        super().__init__(f"{text!r} could be: {', '.join(t.name for t in candidates)}")
        self.text = text
        self.candidates = candidates


class MatchNotFound(Exception):
    """No upcoming Match fits what the user asked for."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class PredictionLocked(Exception):
    """The Match has kicked off, so it can no longer be predicted."""

    def __init__(self, match: Match) -> None:
        super().__init__("Kickoff has passed — Predictions are locked.")
        self.match = match


class OddsUnavailable(Exception):
    """Bookmaker odds could not be had for this Match; `reason` is fit to show the user."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
