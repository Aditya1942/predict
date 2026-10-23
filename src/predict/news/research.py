"""Team News research: the Claude prompt, the Claude Agent SDK call, and News Dossier reuse."""

import asyncio
import json
from collections.abc import Callable, Mapping
from contextlib import aclosing
from datetime import UTC, datetime
from typing import Any, Protocol

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from pydantic import ValidationError

from predict.domain import Match, MatchDossier, NewsDossier, ResearchOutput, Side, Team
from predict.errors import ResearchFailed, SourceError
from predict.params import DOSSIER_TTL, MAX_FACTORS_PER_TEAM, MAX_SEARCHES_PER_TEAM
from predict.store import Store
from predict.teams import TeamResolver

TOOLS = ["WebSearch", "WebFetch"]
MAX_TURNS = 30


class Researcher(Protocol):
    def research(self, match: Match, hints: dict[Side, list[str]], model: str) -> MatchDossier: ...


class HintSource(Protocol):
    def hints(self, team: Team, resolver: TeamResolver) -> list[str]: ...


def build_prompt(match: Match, hints: Mapping[Side, list[str]], now: datetime) -> str:
    kickoff = match.kickoff.astimezone(UTC)
    lines = [
        "You are researching Team News for one football Match. The result feeds a probability model, so accuracy"
        " matters more than volume.",
        "",
        f"Match: {match.home.name} (home) v {match.away.name} (away)",
        f"Competition: {match.competition.display}",
        f"Kickoff: {kickoff:%A} {kickoff:%Y-%m-%d} at {kickoff:%H:%M} UTC",
        f"Today: {now.astimezone(UTC):%Y-%m-%d}",
        "",
        "For each team find the Team News that changes how it will play in this Match: players who are out"
        " (injury or suspension), doubtful, returning, expected to be rotated, or on a notable hot or cold streak."
        " Also check whether the official starting lineup has been published yet.",
        "",
        "Rules:",
        f"- Use at most {MAX_SEARCHES_PER_TEAM} web searches per team.",
        f"- Report at most {MAX_FACTORS_PER_TEAM} News Factors per team, ranked by importance (most important first)."
        " Leave out news that would not change the team's level.",
        "- For each News Factor, resolve every date into misses_match: compare return dates and suspension lengths"
        " with the kickoff date above, and use yes, likely, no or n/a.",
        "- For an absence, count how many of the team's last 6 matches the player has missed.",
        "- Keep each summary to one or two plain sentences of fact, and include source URLs for it.",
        "- Do not give probabilities or predictions.",
        "- Do not invent news. If a team has no relevant news, return an empty list of News Factors for it.",
    ]
    sections = [(match.home.name, hints.get("home", [])), (match.away.name, hints.get("away", []))]
    if any(team_hints for _, team_hints in sections):
        lines += [
            "",
            "Fantasy Premier League (FPL) availability flags, as hints to check against the news (they can be stale):",
        ]
        for team, team_hints in sections:
            if team_hints:
                lines.append(f"{team}:")
                lines += [f"- {hint}" for hint in team_hints]
    return "\n".join(lines)


def inline_refs(schema: dict) -> dict:
    """A copy of a JSON schema with every `$ref` replaced by the definition it points at, and `$defs` removed."""
    defs = schema.get("$defs", {})

    def resolve(node: Any, active: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, active) for item in node]
        if not isinstance(node, dict):
            return node
        resolved = {key: resolve(value, active) for key, value in node.items() if key not in ("$ref", "$defs")}
        ref = node.get("$ref")
        if ref is None:
            return resolved
        name = ref.removeprefix("#/$defs/")
        if name not in defs or name in active:
            raise ValueError(f"cannot inline {ref!r}: " + ("it is recursive" if name in defs else "no such definition"))
        return {**resolve(defs[name], active | {name}), **resolved}

    return resolve(schema, frozenset())


class ClaudeResearcher:
    """Researches a Match with Claude on the subscription login: web search and fetch only, structured output."""

    def __init__(self, *, now: Callable[[], datetime]) -> None:
        self._now = now

    def research(self, match: Match, hints: dict[Side, list[str]], model: str) -> MatchDossier:
        now = self._now()
        options = ClaudeAgentOptions(
            model=model,
            tools=list(TOOLS),
            allowed_tools=list(TOOLS),
            setting_sources=[],  # nothing from the user's or project's Claude settings
            max_turns=MAX_TURNS,
            output_format={"type": "json_schema", "schema": inline_refs(ResearchOutput.model_json_schema())},
        )
        try:
            result = asyncio.run(_final_result(build_prompt(match, hints, now), options))
        except Exception as exc:  # the SDK raises its own errors, plus whatever the CLI subprocess throws
            raise ResearchFailed(f"Claude research failed: {exc}") from exc
        output = _parse(result)
        return MatchDossier(
            home=_truncated(output.home),
            away=_truncated(output.away),
            created_at=now,
            claude_model=model,
        )


async def _final_result(prompt: str, options: ClaudeAgentOptions) -> ResultMessage | None:
    last: ResultMessage | None = None
    async with aclosing(query(prompt=prompt, options=options)) as messages:
        async for message in messages:
            if isinstance(message, ResultMessage):
                last = message
    return last


def _parse(result: ResultMessage | None) -> ResearchOutput:
    if result is None:
        raise ResearchFailed("Claude research failed: no result message came back")
    if result.is_error:
        detail = "; ".join(result.errors or []) or result.result or ""
        status = f" (HTTP {result.api_error_status})" if result.api_error_status else ""
        raise ResearchFailed(f"Claude research failed: {result.subtype}{status}" + (f": {detail}" if detail else ""))
    raw = result.structured_output
    if raw is None:
        raise ResearchFailed("Claude research failed: no structured output came back")
    try:
        return ResearchOutput.model_validate(json.loads(raw) if isinstance(raw, str) else raw)
    except (ValidationError, ValueError) as exc:
        raise ResearchFailed(f"Claude research failed: the answer did not fit the schema ({exc})") from exc


def _truncated(dossier: NewsDossier) -> NewsDossier:
    return dossier.model_copy(update={"factors": dossier.factors[:MAX_FACTORS_PER_TEAM]})


class DossierService:
    """Hands out the Match's News Dossier, researching only when the saved one is too old or from another model."""

    def __init__(
        self,
        researcher: Researcher,
        fpl: HintSource,
        resolver: TeamResolver,
        store: Store,
        *,
        now: Callable[[], datetime],
    ) -> None:
        self._researcher = researcher
        self._fpl = fpl
        self._resolver = resolver
        self._store = store
        self._now = now

    def get(self, match: Match, *, fresh: bool, model: str) -> MatchDossier:
        if not fresh:
            saved = self._store.latest_dossier(match.id)
            if saved is not None and saved.claude_model == model and self._now() - saved.created_at < DOSSIER_TTL:
                return saved
        dossier = self._researcher.research(match, self._hints(match), model)
        self._store.save_dossier(match.id, dossier)
        return dossier

    def _hints(self, match: Match) -> dict[Side, list[str]]:
        """FPL flags are only hints, so a failing FPL means research goes ahead without them."""
        try:
            return {
                "home": self._fpl.hints(match.home, self._resolver),
                "away": self._fpl.hints(match.away, self._resolver),
            }
        except SourceError:
            return {"home": [], "away": []}


def api_key_warning(env: Mapping[str, str]) -> str | None:
    """A warning when ANTHROPIC_API_KEY is set, as Claude would then bill the API key, not the subscription."""
    if not env.get("ANTHROPIC_API_KEY"):
        return None
    return (
        "ANTHROPIC_API_KEY is set, so Claude research will be billed to that API key instead of your Claude"
        " subscription login. Unset it to use the subscription."
    )
