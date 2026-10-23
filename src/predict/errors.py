"""Errors raised across the predictor."""


class SourceError(Exception):
    """A data source failed or returned something unusable."""

    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"{source}: {detail}")
        self.source = source
        self.detail = detail
