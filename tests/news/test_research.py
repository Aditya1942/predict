import json
from datetime import UTC, datetime, timedelta

import pytest
from claude_agent_sdk import ResultMessage

from predict.domain import MatchDossier, NewsDossier, NewsFactor, ResearchOutput
from predict.errors import ResearchFailed, SourceError
from predict.news import research
from predict.news.research import ClaudeResearcher, DossierService, api_key_warning, build_prompt, inline_refs
from predict.params import MAX_FACTORS_PER_TEAM
from predict.store import Store
from predict.teams import Aliases, TeamResolver
from tests.helpers import make_match, make_team

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
KICKOFF = datetime(2026, 10, 25, 16, 30, tzinfo=UTC)

ARSENAL = make_team(1, "Arsenal FC")
CHELSEA = make_team(2, "Chelsea FC")
MATCH = make_match(ARSENAL, CHELSEA, kickoff=KICKOFF, id=42)
RESOLVER = TeamResolver([ARSENAL, CHELSEA], Aliases.load())


def factor(summary="Saka out with a hamstring injury", **changes) -> NewsFactor:
    base = {
        "kind": "absence",
        "player": "Saka",
        "summary": summary,
        "misses_match": "yes",
        "matches_missed_of_last_6": 1,
        "sources": ["https://example.com/saka"],
    }
    return NewsFactor(**{**base, **changes})


def dossier(team: str, *factors: NewsFactor, lineup_confirmed=False) -> NewsDossier:
    return NewsDossier(team=team, factors=list(factors), lineup_confirmed=lineup_confirmed)


def match_dossier(model="claude-opus-5-5", created_at=NOW) -> MatchDossier:
    return MatchDossier(
        home=dossier("Arsenal FC", factor()),
        away=dossier("Chelsea FC"),
        created_at=created_at,
        claude_model=model,
    )


# --- prompt ---------------------------------------------------------------------------------------------------------

HINTS = {"home": ["Saka: doubtful, 50% chance — Thigh injury"], "away": ["Palmer: injured, 0% chance — Groin"]}


def test_prompt_has_the_match_and_the_rules():
    prompt = build_prompt(MATCH, HINTS, NOW)
    for needed in (
        "Arsenal FC",
        "Chelsea FC",
        "Premier League",
        "2026-10-25",  # kickoff date
        "2026-10-23",  # today
        "at most 5 web searches per team",
        "at most 6 News Factors per team, ranked by importance",
        "resolve every date into misses_match",
        "include source URLs",
        "Do not give probabilities or predictions",
    ):
        assert needed in prompt, needed


def test_prompt_lists_every_hint_under_its_team():
    prompt = build_prompt(MATCH, HINTS, NOW)
    assert "Saka: doubtful, 50% chance — Thigh injury" in prompt
    assert "Palmer: injured, 0% chance — Groin" in prompt
    assert prompt.index("Saka: doubtful") < prompt.index("Palmer: injured")  # home hints come first


def test_prompt_without_hints_has_no_hint_section():
    assert "FPL" not in build_prompt(MATCH, {"home": [], "away": []}, NOW)


def test_prompt_uses_the_limits_from_params():
    assert f"at most {research.MAX_SEARCHES_PER_TEAM} web searches per team" in build_prompt(MATCH, {}, NOW)


# --- schema ---------------------------------------------------------------------------------------------------------


def test_inline_refs_leaves_no_ref_and_keeps_the_enums():
    schema = inline_refs(ResearchOutput.model_json_schema())
    text = json.dumps(schema)
    assert "$ref" not in text
    assert "$defs" not in schema
    news_factor = schema["properties"]["home"]["properties"]["factors"]["items"]
    assert news_factor["properties"]["misses_match"]["enum"] == ["yes", "likely", "no", "n/a"]
    assert "absence" in news_factor["properties"]["kind"]["enum"]


def test_inline_refs_does_not_change_its_input():
    original = ResearchOutput.model_json_schema()
    snapshot = json.dumps(original, sort_keys=True)
    inline_refs(original)
    assert json.dumps(original, sort_keys=True) == snapshot


def test_inline_refs_rejects_a_dangling_ref():
    with pytest.raises(ValueError, match="Missing"):
        inline_refs({"type": "object", "properties": {"a": {"$ref": "#/$defs/Missing"}}})


# --- dossier service ------------------------------------------------------------------------------------------------


class FakeResearcher:
    def __init__(self, clock) -> None:
        self.clock = clock
        self.calls: list[tuple[dict, str]] = []

    def research(self, match, hints, model) -> MatchDossier:
        self.calls.append((hints, model))
        return match_dossier(model, created_at=self.clock())


class FakeFpl:
    def __init__(self, hints=None, error: Exception | None = None) -> None:
        self.hints_by_team = hints or {}
        self.error = error

    def hints(self, team, resolver) -> list[str]:
        if self.error:
            raise self.error
        return self.hints_by_team.get(team.id, [])


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta) -> None:
        self.now += timedelta(**delta)


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def clock():
    return Clock()


def service(store, clock, *, fpl=None):
    researcher = FakeResearcher(clock)
    return DossierService(researcher, fpl or FakeFpl(), RESOLVER, store, now=clock), researcher


def test_second_get_within_the_ttl_reuses_the_dossier(store, clock):
    svc, researcher = service(store, clock)
    first = svc.get(MATCH, fresh=False, model="m1")
    clock.advance(hours=1)
    second = svc.get(MATCH, fresh=False, model="m1")
    assert len(researcher.calls) == 1
    assert second == first


def test_get_after_the_ttl_researches_again(store, clock):
    svc, researcher = service(store, clock)
    svc.get(MATCH, fresh=False, model="m1")
    clock.advance(hours=7)
    svc.get(MATCH, fresh=False, model="m1")
    assert len(researcher.calls) == 2


def test_dossier_exactly_at_the_ttl_is_stale(store, clock):
    svc, researcher = service(store, clock)
    svc.get(MATCH, fresh=False, model="m1")
    clock.advance(hours=6)
    svc.get(MATCH, fresh=False, model="m1")
    assert len(researcher.calls) == 2


def test_fresh_always_researches(store, clock):
    svc, researcher = service(store, clock)
    svc.get(MATCH, fresh=False, model="m1")
    svc.get(MATCH, fresh=True, model="m1")
    assert len(researcher.calls) == 2


def test_a_different_model_researches_again(store, clock):
    svc, researcher = service(store, clock)
    svc.get(MATCH, fresh=False, model="m1")
    svc.get(MATCH, fresh=False, model="m2")
    assert [model for _, model in researcher.calls] == ["m1", "m2"]


def test_dossiers_are_per_match(store, clock):
    svc, researcher = service(store, clock)
    svc.get(MATCH, fresh=False, model="m1")
    svc.get(make_match(ARSENAL, CHELSEA, kickoff=KICKOFF, id=43), fresh=False, model="m1")
    assert len(researcher.calls) == 2


def test_the_researched_dossier_is_saved_and_returned(store, clock):
    svc, _ = service(store, clock)
    got = svc.get(MATCH, fresh=False, model="m1")
    assert store.latest_dossier(42) == got
    assert got.claude_model == "m1"


def test_fpl_hints_for_both_teams_go_to_the_researcher(store, clock):
    fpl = FakeFpl({1: ["Saka: doubtful"], 2: ["Palmer: injured"]})
    svc, researcher = service(store, clock, fpl=fpl)
    svc.get(MATCH, fresh=False, model="m1")
    assert researcher.calls[0][0] == {"home": ["Saka: doubtful"], "away": ["Palmer: injured"]}


def test_hints_are_skipped_when_fpl_fails(store, clock):
    svc, researcher = service(store, clock, fpl=FakeFpl(error=SourceError("fantasy.premierleague.com", "HTTP 503")))
    svc.get(MATCH, fresh=False, model="m1")
    assert researcher.calls[0][0] == {"home": [], "away": []}


def test_reuse_does_not_call_fpl(store, clock):
    fpl = FakeFpl()
    svc, _ = service(store, clock, fpl=fpl)
    svc.get(MATCH, fresh=False, model="m1")
    fpl.error = AssertionError("fpl should not be asked")
    svc.get(MATCH, fresh=False, model="m1")


def test_a_failed_research_saves_nothing(store, clock):
    class Failing:
        def research(self, match, hints, model):
            raise ResearchFailed("no")

    svc = DossierService(Failing(), FakeFpl(), RESOLVER, store, now=clock)
    with pytest.raises(ResearchFailed):
        svc.get(MATCH, fresh=False, model="m1")
    assert store.latest_dossier(42) is None


# --- api key warning ------------------------------------------------------------------------------------------------


def test_api_key_warning():
    warning = api_key_warning({"ANTHROPIC_API_KEY": "x"})
    assert warning is not None
    assert "ANTHROPIC_API_KEY" in warning
    assert "subscription" in warning


@pytest.mark.parametrize("env", [{}, {"ANTHROPIC_API_KEY": ""}, {"OTHER": "x"}])
def test_no_api_key_no_warning(env):
    assert api_key_warning(env) is None


# --- Claude researcher (the SDK is faked; no query is ever sent) ----------------------------------------------------


def result_message(**changes) -> ResultMessage:
    base = {
        "subtype": "success",
        "duration_ms": 1,
        "duration_api_ms": 1,
        "is_error": False,
        "num_turns": 3,
        "session_id": "s",
    }
    return ResultMessage(**{**base, **changes})


def output(home_factors=1, away_factors=0) -> dict:
    return {
        "home": {
            "team": "Arsenal FC",
            "factors": [factor(f"home {i}").model_dump() for i in range(home_factors)],
            "lineup_confirmed": False,
        },
        "away": {
            "team": "Chelsea FC",
            "factors": [factor(f"away {i}", player="Palmer").model_dump() for i in range(away_factors)],
            "lineup_confirmed": True,
        },
    }


@pytest.fixture
def sdk(monkeypatch):
    """Replaces claude_agent_sdk.query; set `messages` (or `raises`) and read back what the researcher sent."""

    class Fake:
        messages: list = []
        raises: Exception | None = None
        prompt: str | None = None
        options = None

    async def fake_query(*, prompt, options=None, transport=None):
        Fake.prompt, Fake.options = prompt, options
        for message in Fake.messages:
            yield message
        if Fake.raises is not None:
            raise Fake.raises

    monkeypatch.setattr(research, "query", fake_query)
    Fake.messages, Fake.raises, Fake.prompt, Fake.options = [], None, None, None
    return Fake


def run(hints=None) -> MatchDossier:
    return ClaudeResearcher(now=lambda: NOW).research(MATCH, hints or HINTS, "claude-opus-5-5")


def test_researcher_returns_the_structured_output_as_a_dossier(sdk):
    sdk.messages = [result_message(structured_output=output(home_factors=2, away_factors=1))]
    got = run()
    assert got.claude_model == "claude-opus-5-5"
    assert got.created_at == NOW
    assert [f.summary for f in got.home.factors] == ["home 0", "home 1"]
    assert got.away.lineup_confirmed is True
    assert got.away.factors[0].player == "Palmer"


def test_researcher_sends_the_prompt_and_the_search_only_options(sdk):
    sdk.messages = [result_message(structured_output=output())]
    run()
    assert sdk.prompt == build_prompt(MATCH, HINTS, NOW)
    options = sdk.options
    assert options.model == "claude-opus-5-5"
    assert options.tools == ["WebSearch", "WebFetch"]
    assert options.allowed_tools == ["WebSearch", "WebFetch"]
    assert options.setting_sources == []
    assert options.max_turns == 30
    assert options.output_format["type"] == "json_schema"
    assert "$ref" not in json.dumps(options.output_format["schema"])


def test_researcher_keeps_at_most_the_configured_factors_per_team(sdk):
    sdk.messages = [result_message(structured_output=output(home_factors=9, away_factors=7))]
    got = run()
    assert len(got.home.factors) == len(got.away.factors) == MAX_FACTORS_PER_TEAM
    assert got.home.factors[0].summary == "home 0"  # ranked by importance, so the head is kept


def test_researcher_uses_the_last_result_message(sdk):
    sdk.messages = ["chatter", result_message(structured_output=output()), "more chatter"]
    assert run().home.team == "Arsenal FC"


def test_researcher_accepts_a_json_string(sdk):
    sdk.messages = [result_message(structured_output=json.dumps(output()))]
    assert run().home.team == "Arsenal FC"


def test_error_result_fails_with_the_reason(sdk):
    sdk.messages = [result_message(is_error=True, subtype="error_max_turns", errors=["ran out of turns"])]
    with pytest.raises(ResearchFailed, match="error_max_turns.*ran out of turns"):
        run()


def test_error_result_with_an_http_status(sdk):
    sdk.messages = [result_message(is_error=True, api_error_status=529, result="Overloaded")]
    with pytest.raises(ResearchFailed, match="529"):
        run()


def test_missing_structured_output_fails(sdk):
    sdk.messages = [result_message(result="some prose")]
    with pytest.raises(ResearchFailed, match="structured"):
        run()


def test_no_result_message_fails(sdk):
    sdk.messages = ["chatter"]
    with pytest.raises(ResearchFailed, match="no result"):
        run()


@pytest.mark.parametrize("bad", [{"home": {"team": "x"}}, "not json", 7, {"home": {}, "away": {}}])
def test_output_that_does_not_fit_the_schema_fails(sdk, bad):
    sdk.messages = [result_message(structured_output=bad)]
    with pytest.raises(ResearchFailed, match="did not fit"):
        run()


def test_sdk_exception_fails(sdk):
    sdk.raises = RuntimeError("cli exploded")
    with pytest.raises(ResearchFailed, match="cli exploded"):
        run()


def test_researcher_names_a_missing_cli(sdk):
    from claude_agent_sdk import CLINotFoundError

    sdk.raises = CLINotFoundError("Claude Code not found")
    with pytest.raises(ResearchFailed, match="Claude Code"):
        run()
