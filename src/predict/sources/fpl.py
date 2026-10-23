"""Fantasy Premier League availability flags: free, official hints about who is injured or doubtful."""

from datetime import timedelta

from predict.domain import Team
from predict.errors import SourceError
from predict.http import CachedHttp
from predict.teams import TeamResolver

SOURCE = "fpl"
URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
TTL = timedelta(hours=1)
_STATUS_WORDS = {"d": "doubtful", "i": "injured", "s": "suspended", "u": "unavailable", "n": "not available"}
_OUT = frozenset("isun")  # statuses that mean the player will not play, so a missing chance is 0%


class Fpl:
    def __init__(self, http: CachedHttp) -> None:
        self._http = http

    def hints(self, team: Team, resolver: TeamResolver) -> list[str]:
        """One line per player FPL does not list as available, e.g. "Saka: doubtful, 50% chance — Thigh injury".

        FPL covers the Premier League only, so a team it does not list has no hints.
        """
        data = self._http.get_json(URL, ttl=TTL)
        try:
            names = {t["id"]: t["name"] for t in data["teams"]}
            elements = list(data["elements"])
            try:
                fpl_name = resolver.find_in_source(SOURCE, team, names.values())
            except SourceError:
                return []
            ids = {team_id for team_id, name in names.items() if name == fpl_name}
            return [_hint(e) for e in elements if e["team"] in ids and e["status"] != "a"]
        except (KeyError, TypeError) as exc:
            raise SourceError(SOURCE, f"unexpected response: {exc!r}") from exc


def _hint(element: dict) -> str:
    status = element["status"]
    chance = element.get("chance_of_playing_next_round")
    if chance is None and status in _OUT:
        chance = 0
    hint = f"{element['web_name']}: {_STATUS_WORDS.get(status, f'status {status}')}"
    if chance is not None:
        hint += f", {chance}% chance"
    if news := element.get("news"):
        hint += f" — {news}"
    return hint
