"""Errors raised across the predictor."""

from predict.domain import Team


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
