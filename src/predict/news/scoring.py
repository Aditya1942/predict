"""Scoring News Factors with Jev: how much each one changes a team's chances created and conceded."""

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from typesafe_sdk import Noul, Score, TypeSafeClient, TypeSafeError

from predict.domain import FactorImpact, FactorKind, Match, MatchDossier, NewsFactor, Side
from predict.errors import JevFailed
from predict.params import ATTACK_BOOST, ATTACK_WEAKEN, CONCEDE_BOOST, CONCEDE_WEAKEN

# Jev rates every News Factor on these 4 levels, once for attack and once for defence.
LEVELS = (
    "No real change: a fringe player, or an absence that has already lasted most of the team's last 6 matches",
    "Small change: a rotation player, or a starter with a like-for-like replacement",
    "Clear change: a regular starter whose replacement is noticeably weaker",
    "Major change: the team's key player in this area, such as the top scorer or main creator,"
    " with no comparable replacement",
)
_BOOSTS = frozenset({"return", "hot_streak"})


@dataclass(frozen=True)
class JevAnswers:
    """`nouls` are probabilities of yes; `score_probabilities` hold one probability per level, in level order."""

    nouls: dict[str, float]
    score_probabilities: dict[str, list[float]]


class JevClient(Protocol):
    def ask(self, state: dict, nouls: dict[str, str], scores: dict[str, tuple[str, list[str]]]) -> JevAnswers:
        """Yes/no questions by name, and score questions by name as (instructions, levels in order)."""
        ...


class TypeSafeJev:
    """`JevClient` over the TypeSafe API: one `system_one` request with Noul and Score questions."""

    def __init__(self, api_key: str, model: str, *, client: TypeSafeClient | None = None) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client

    def ask(self, state: dict, nouls: dict[str, str], scores: dict[str, tuple[str, list[str]]]) -> JevAnswers:
        try:
            questions = {
                **{name: Noul(instructions=text) for name, text in nouls.items()},
                **{name: Score(instructions=text, criteria=levels) for name, (text, levels) in scores.items()},
            }
            if self._client is not None:
                response = self._client.system_one(state, questions, model=self._model)
            else:
                with TypeSafeClient(api_key=self._api_key, model=self._model) as client:
                    response = client.system_one(state, questions, model=self._model)
        except (TypeSafeError, ValueError) as exc:  # the SDK's own errors, or a question it refuses to build
            raise JevFailed(f"Jev request failed: {exc}") from exc
        answers = JevAnswers(nouls={}, score_probabilities={})
        for name in nouls:
            if name not in response.nouls:
                raise JevFailed(f"Jev gave no answer for {name}")
            answers.nouls[name] = response.nouls[name].noul
        for name, (_, levels) in scores.items():
            if name not in response.scores:
                raise JevFailed(f"Jev gave no answer for {name}")
            answers.score_probabilities[name] = _by_level(name, response.scores[name].probabilities, len(levels))
        return answers


def _by_level(name: str, probabilities: dict[int, float], count: int) -> list[float]:
    """Probabilities keyed by level number (from 0) as a list in level order, a level left out being 0, summing to 1."""
    if not set(probabilities) <= set(range(count)):
        raise JevFailed(f"Jev scored {name} on levels {sorted(probabilities)}, but {count} were asked")
    ordered = [probabilities.get(level, 0.0) for level in range(count)]
    total = sum(ordered)
    if total <= 0:
        raise JevFailed(f"Jev gave {name} no probability on any level")
    return [p / total for p in ordered]


def _expected(table: tuple[float, ...], probabilities: list[float]) -> float:
    if len(probabilities) != len(table):
        raise ValueError(f"need a probability for each of the {len(table)} levels, got {len(probabilities)}")
    return sum(p * m for p, m in zip(probabilities, table, strict=True))


def attack_multiplier(kind: FactorKind, probabilities: list[float]) -> float:
    """The expected multiplier on the team's own goals: Σ probability of level × that level's multiplier."""
    return _expected(ATTACK_BOOST if kind in _BOOSTS else ATTACK_WEAKEN, probabilities)


def concede_multiplier(kind: FactorKind, probabilities: list[float]) -> float:
    """The expected multiplier on the opponent's goals."""
    return _expected(CONCEDE_BOOST if kind in _BOOSTS else CONCEDE_WEAKEN, probabilities)


def scale_by_start(m: float, p_start: float) -> float:
    """A weakening multiplier softened by the chance the player starts: 1 − (1 − m)(1 − p_start)."""
    return 1 - (1 - m) * (1 - p_start)


def score_dossier(
    dossier: MatchDossier, match: Match, jev: JevClient, *, today: date | None = None
) -> list[FactorImpact]:
    """Ask Jev about every News Factor in one request and turn the answers into multipliers.

    A factor that cannot matter (an absence or doubt for a player who does not miss the Match) is not asked
    about and has no impact. Without any question left no request is made. `today` defaults to the day the
    dossier was made, which is today to within the dossier's reuse window.
    """
    today = today or dossier.created_at.date()
    teams = {"home": (match.home.name, dossier.home), "away": (match.away.name, dossier.away)}
    nouls: dict[str, str] = {}
    scores: dict[str, tuple[str, list[str]]] = {}
    asked: list[tuple[Side, int, NewsFactor]] = []
    for side, (team, news) in teams.items():
        for i, factor in enumerate(news.factors):
            if factor.misses_match == "no" and factor.kind in ("absence", "doubtful"):
                continue
            asked.append((side, i, factor))
            for change, what in (("attack", "creates"), ("defence", "concedes")):
                scores[f"{side}{i}_{change}"] = (
                    f"How much does this news change how many chances {team} {what} in this Match?"
                    f" News: {factor.summary}",
                    list(LEVELS),
                )
            if factor.player is not None and (factor.kind == "doubtful" or factor.misses_match == "likely"):
                nouls[f"{side}{i}_starts"] = f"Will {factor.player} start this Match for {team}?"
    if not asked:
        return []
    state = {
        "match": {
            "home": match.home.name,
            "away": match.away.name,
            "competition": match.competition.display,
            "kickoff": match.kickoff.isoformat(),
            "today": today.isoformat(),
        },
        "team_news": {
            "home": dossier.home.model_dump(mode="json"),
            "away": dossier.away.model_dump(mode="json"),
        },
    }
    answers = jev.ask(state, nouls, scores)
    return [_impact(side, i, factor, answers) for side, i, factor in asked]


def _impact(side: Side, i: int, factor: NewsFactor, answers: JevAnswers) -> FactorImpact:
    def levels(name: str) -> list[float]:
        probabilities = answers.score_probabilities.get(name)
        if probabilities is None:
            raise JevFailed(f"Jev gave no answer for {name}")
        if len(probabilities) != len(LEVELS):
            raise JevFailed(f"Jev answered {name} with {len(probabilities)} levels, expected {len(LEVELS)}")
        return probabilities

    attack = attack_multiplier(factor.kind, levels(f"{side}{i}_attack"))
    concede = concede_multiplier(factor.kind, levels(f"{side}{i}_defence"))
    p_start = answers.nouls.get(f"{side}{i}_starts")
    if p_start is not None:
        p_start = min(max(p_start, 0.0), 1.0)
        # Weakening news bites when the player is out; a return or streak only helps if he plays.
        p_scale = 1 - p_start if factor.kind in _BOOSTS else p_start
        attack, concede = scale_by_start(attack, p_scale), scale_by_start(concede, p_scale)
    return FactorImpact(side=side, factor=factor, attack_mult=attack, concede_mult=concede, p_start=p_start)
