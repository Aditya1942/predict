"""Team name resolution (aliases + fuzzy matching) and query parsing."""

import re
import tomllib
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from rapidfuzz import fuzz

from predict.domain import Team
from predict.errors import AmbiguousTeam, SourceError, TeamNotFound

QUERY_SPLIT = re.compile(r"\s+(?:v\.?|vs\.?|-|–|—)\s+", re.IGNORECASE)

QUERY_ACCEPT = 85.0  # a clear match
QUERY_LEAD = 5.0  # ... that leads the runner-up by this much
QUERY_AMBIGUOUS = 70.0  # below a clear match but close enough to offer as candidates
QUERY_BAND = 10.0  # candidates within this many points of the top
SOURCE_ACCEPT = 90.0
SOURCE_LEAD = 5.0
FIND_ACCEPT = 85.0
SUGGESTIONS = 3

_CLUB_TOKENS = frozenset(
    {"fc", "afc", "cf", "sc", "ac", "as", "ssc", "rc", "rcd", "cd", "ud", "ca", "club", "de", "calcio", "1"}
)
_FOLD = str.maketrans({"ß": "ss", "ø": "o", "æ": "ae", "ł": "l", "đ": "d"})


def normalize(name: str) -> str:
    """Lowercase, strip accents and punctuation, drop club-form tokens (fc, ac, de, ...)."""
    folded = unicodedata.normalize("NFKD", name.lower().translate(_FOLD))
    plain = "".join(c for c in folded if not unicodedata.combining(c))
    tokens = re.sub(r"[^a-z0-9]+", " ", plain).split()
    kept = [t for t in tokens if t not in _CLUB_TOKENS]
    return " ".join(kept or tokens)


def parse_query(text: str) -> tuple[str, str | None]:
    """Split "A v B" / "A v. B" / "A vs B" / "A - B" (hyphen, en or em dash); one team gives (team, None)."""
    parts = QUERY_SPLIT.split(text.strip(), maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return parts[0].strip(), None


@dataclass(frozen=True)
class Aliases:
    """`user`: normalised alias -> canonical names. `source`: source -> normalised source name -> canonical name."""

    user: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    source: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Aliases":
        if path is None:
            text = (resources.files("predict") / "data" / "aliases.toml").read_text(encoding="utf-8")
        else:
            text = path.read_text(encoding="utf-8")
        data = tomllib.loads(text)
        user = {normalize(alias): tuple(names) for alias, names in data.get("user", {}).items()}
        source = {
            name: {normalize(src): canonical for src, canonical in table.items()}
            for name, table in data.get("source", {}).items()
        }
        return cls(user, source)

    def source_override(self, source: str, name: str) -> str | None:
        return self.source.get(source, {}).get(normalize(name))


def _name_fields(team: Team) -> tuple[str, ...]:
    return tuple(dict.fromkeys(f for f in (normalize(team.name), normalize(team.short_name)) if f))


def _score(query: str, fields: Iterable[str]) -> float:
    return max((fuzz.WRatio(query, f) for f in fields), default=0.0)


def _source_score(query: str, team: Team) -> float:
    return _score(query, _name_fields(team))


def _query_score(query: str, team: Team) -> float:
    """Fuzzy over name and short name; the tla only counts on an exact match (fuzzy tla hits are false friends)."""
    if team.tla and query == normalize(team.tla):
        return 100.0
    return _score(query, _name_fields(team))


class TeamResolver:
    def __init__(self, teams: list[Team], aliases: Aliases) -> None:
        # The same team appears under several Competitions; identity is the team id.
        self._teams = list({t.id: t for t in teams}.values())
        self._aliases = aliases
        self._by_name = {normalize(t.name): t for t in self._teams}

    def resolve_query(self, text: str) -> Team:
        query = normalize(text)
        aliased = self._alias_teams(query)
        if len(aliased) == 1:
            return aliased[0]
        if len(aliased) > 1:
            raise AmbiguousTeam(text, sorted(aliased, key=lambda t: t.name))

        ranked = self._rank(query, _query_score)
        top_score = ranked[0][0] if ranked else 0.0
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        if top_score >= QUERY_ACCEPT and top_score - runner_up >= QUERY_LEAD:
            return ranked[0][1]
        if top_score >= QUERY_AMBIGUOUS:
            raise AmbiguousTeam(text, [t for s, t in ranked if s >= top_score - QUERY_BAND])
        raise TeamNotFound(text, [t.name for _, t in ranked[:SUGGESTIONS]])

    def map_source_names(self, source: str, names: Iterable[str]) -> dict[str, int | None]:
        mapped: dict[str, int | None] = {}
        owner: dict[int, str] = {}
        for name in names:
            team = self._match_source_name(source, name)
            mapped[name] = team.id if team else None
            if team is None:
                continue
            if team.id in owner and owner[team.id] != name:
                raise SourceError(source, f"{owner[team.id]!r} and {name!r} both map to {team.name}")
            owner[team.id] = name
        return mapped

    def find_in_source(self, source: str, team: Team, names: Iterable[str]) -> str:
        candidates = list(names)
        overrides = {n: self._aliases.source_override(source, n) for n in candidates}
        for name, canonical in overrides.items():
            if canonical is not None and normalize(canonical) == normalize(team.name):
                return name
        fields = _name_fields(team)
        best, best_score = None, 0.0
        for name in candidates:
            if overrides[name] is not None:  # claimed by another team
                continue
            score = _score(normalize(name), fields)
            if score > best_score:
                best, best_score = name, score
        if best is None or best_score < FIND_ACCEPT:
            raise SourceError(source, f"no entry for {team.name}")
        return best

    def _alias_teams(self, query: str) -> list[Team]:
        names = self._aliases.user.get(query, ())
        found = {t.id: t for n in names if (t := self._by_name.get(normalize(n)))}
        return list(found.values())

    def _match_source_name(self, source: str, name: str) -> Team | None:
        canonical = self._aliases.source_override(source, name)
        if canonical is not None:
            return self._by_name.get(normalize(canonical))
        ranked = self._rank(normalize(name), _source_score)
        if not ranked:
            return None
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        top_score, top = ranked[0]
        return top if top_score >= SOURCE_ACCEPT and top_score - runner_up >= SOURCE_LEAD else None

    def _rank(self, query: str, score: Callable[[str, Team], float]) -> list[tuple[float, Team]]:
        """Teams by fuzzy score, best first, ties by name so the order never depends on input order."""
        scored = [(score(query, t), t) for t in self._teams]
        return sorted(scored, key=lambda st: (-st[0], st[1].name))
