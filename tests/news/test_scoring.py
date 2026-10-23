import json
from datetime import UTC, date, datetime

import httpx2
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from predict.domain import MatchDossier, NewsDossier, NewsFactor
from predict.errors import JevFailed
from predict.news.scoring import (
    LEVELS,
    JevAnswers,
    TypeSafeJev,
    attack_multiplier,
    concede_multiplier,
    scale_by_start,
    score_dossier,
)
from tests.helpers import make_match, make_team

KICKOFF = datetime(2026, 10, 25, 16, 30, tzinfo=UTC)
CREATED = datetime(2026, 10, 23, 9, 0, tzinfo=UTC)
ARSENAL = make_team(1, "Arsenal FC")
CHELSEA = make_team(2, "Chelsea FC")
MATCH = make_match(ARSENAL, CHELSEA, kickoff=KICKOFF, id=42)


def factor(kind="absence", misses="yes", player="Saka", summary="Saka is out", **changes) -> NewsFactor:
    base = {
        "kind": kind,
        "player": player,
        "summary": summary,
        "misses_match": misses,
        "matches_missed_of_last_6": 0,
        "sources": ["https://example.com"],
    }
    return NewsFactor(**{**base, **changes})


def match_dossier(home=(), away=()) -> MatchDossier:
    return MatchDossier(
        home=NewsDossier(team="Arsenal FC", factors=list(home), lineup_confirmed=False),
        away=NewsDossier(team="Chelsea FC", factors=list(away), lineup_confirmed=False),
        created_at=CREATED,
        claude_model="claude-opus-5-5",
    )


# --- multipliers ----------------------------------------------------------------------------------------------------


def test_expected_multiplier():
    assert attack_multiplier("absence", [0, 0, 1, 0]) == pytest.approx(0.92)
    assert attack_multiplier("absence", [0.5, 0.5, 0, 0]) == pytest.approx(0.985)
    assert attack_multiplier("return", [0, 0, 0, 1]) == pytest.approx(1.10)


@pytest.mark.parametrize("kind", ["absence", "doubtful", "rotation", "cold_streak"])
def test_weakening_kinds_lower_attack_and_raise_concede(kind):
    assert attack_multiplier(kind, [0, 0, 0, 1]) == pytest.approx(0.85)
    assert concede_multiplier(kind, [0, 0, 0, 1]) == pytest.approx(1.15)


@pytest.mark.parametrize("kind", ["return", "hot_streak"])
def test_boosting_kinds_raise_attack_and_lower_concede(kind):
    assert attack_multiplier(kind, [0, 0, 1, 0]) == pytest.approx(1.06)
    assert concede_multiplier(kind, [0, 0, 1, 0]) == pytest.approx(0.94)


def test_level_zero_changes_nothing():
    for kind in ("absence", "return"):
        assert attack_multiplier(kind, [1, 0, 0, 0]) == 1.0
        assert concede_multiplier(kind, [1, 0, 0, 0]) == 1.0


def test_probabilities_must_cover_every_level():
    with pytest.raises(ValueError, match="4"):
        attack_multiplier("absence", [0.5, 0.5])


def test_doubtful_scaled_by_start_chance():
    assert scale_by_start(0.85, p_start=0.6) == pytest.approx(0.94)


@pytest.mark.parametrize(("p_start", "expected"), [(1.0, 1.0), (0.0, 0.85)])
def test_start_chance_extremes(p_start, expected):
    assert scale_by_start(0.85, p_start=p_start) == pytest.approx(expected)


def test_levels_are_the_agreed_rubric():
    assert LEVELS == (
        "No real change: a fringe player, or an absence that has already lasted most of the team's last 6 matches",
        "Small change: a rotation player, or a starter with a like-for-like replacement",
        "Clear change: a regular starter whose replacement is noticeably weaker",
        "Major change: the team's key player in this area, such as the top scorer or main creator,"
        " with no comparable replacement",
    )


# --- score_dossier --------------------------------------------------------------------------------------------------


class FakeJev:
    """Answers every question from `scores` / `nouls` (by name), or with level 0 / 0.5 when not listed."""

    def __init__(self, scores=None, nouls=None) -> None:
        self.scores = scores or {}
        self.nouls = nouls or {}
        self.calls: list[tuple[dict, dict, dict]] = []

    def ask(self, state, nouls, scores) -> JevAnswers:
        self.calls.append((state, nouls, scores))
        return JevAnswers(
            nouls={name: self.nouls.get(name, 0.5) for name in nouls},
            score_probabilities={name: self.scores.get(name, [1.0, 0.0, 0.0, 0.0]) for name in scores},
        )


def test_one_ask_with_the_expected_question_names():
    dossier = match_dossier(
        home=[
            factor(kind="absence", misses="yes"),
            factor(kind="doubtful", misses="no", player="Odegaard"),  # fit, so not asked about
            factor(kind="doubtful", misses="likely", player="Rice"),
        ],
        away=[factor(kind="return", misses="no", player="Palmer")],
    )
    jev = FakeJev()
    score_dossier(dossier, MATCH, jev)
    assert len(jev.calls) == 1
    _, nouls, scores = jev.calls[0]
    assert sorted(scores) == [
        "away0_attack",
        "away0_defence",
        "home0_attack",
        "home0_defence",
        "home2_attack",
        "home2_defence",
    ]
    assert sorted(nouls) == ["home2_starts"]


def test_absence_that_does_not_miss_the_match_gets_no_questions():
    dossier = match_dossier(home=[factor(kind="absence", misses="no"), factor(kind="doubtful", misses="no")])
    jev = FakeJev()
    assert score_dossier(dossier, MATCH, jev) == []
    assert jev.calls == []  # nothing to ask, so no request is sent


def test_doubtful_factor_is_asked_whether_the_player_starts():
    dossier = match_dossier(home=[factor(kind="doubtful", misses="likely", player="Saka")])
    jev = FakeJev()
    score_dossier(dossier, MATCH, jev)
    _, nouls, scores = jev.calls[0]
    assert nouls == {"home0_starts": "Will Saka start this Match for Arsenal FC?"}
    assert sorted(scores) == ["home0_attack", "home0_defence"]


@pytest.mark.parametrize(
    ("kind", "misses", "asked"),
    [
        ("doubtful", "yes", True),
        ("doubtful", "n/a", True),
        ("absence", "likely", True),
        ("rotation", "likely", True),
        ("absence", "yes", False),
        ("return", "yes", False),
        ("rotation", "n/a", False),
        ("hot_streak", "no", False),
    ],
)
def test_start_question_only_for_doubtful_or_likely(kind, misses, asked):
    jev = FakeJev()
    score_dossier(match_dossier(home=[factor(kind=kind, misses=misses)]), MATCH, jev)
    assert ("home0_starts" in jev.calls[0][1]) is asked


def test_no_start_question_without_a_player():
    jev = FakeJev()
    score_dossier(match_dossier(home=[factor(kind="doubtful", misses="likely", player=None)]), MATCH, jev)
    assert jev.calls[0][1] == {}


def test_question_text_and_levels():
    dossier = match_dossier(
        home=[factor(summary="Saka is out for a month")],
        away=[factor(kind="hot_streak", player="Palmer", summary="Palmer has 5 in 3", misses="n/a")],
    )
    jev = FakeJev()
    score_dossier(dossier, MATCH, jev)
    _, _, scores = jev.calls[0]
    question = (
        "How much does this news change how many chances Arsenal FC {} in this Match? News: Saka is out for a month"
    )
    assert scores["home0_attack"] == (question.format("creates"), list(LEVELS))
    assert scores["home0_defence"] == (question.format("concedes"), list(LEVELS))
    assert scores["away0_attack"][0].startswith("How much does this news change how many chances Chelsea FC creates")
    assert scores["away0_attack"][0].endswith("News: Palmer has 5 in 3")


def test_question_names_use_the_factor_position_in_the_dossier():
    dossier = match_dossier(home=[factor(misses="no"), factor(summary="second")])
    jev = FakeJev()
    score_dossier(dossier, MATCH, jev)
    assert sorted(jev.calls[0][2]) == ["home1_attack", "home1_defence"]


def test_state_has_the_news_the_match_and_todays_date():
    dossier = match_dossier(home=[factor()], away=[factor(player="Palmer")])
    jev = FakeJev()
    score_dossier(dossier, MATCH, jev)
    state = jev.calls[0][0]
    assert state["match"] == {
        "home": "Arsenal FC",
        "away": "Chelsea FC",
        "competition": "Premier League",
        "kickoff": "2026-10-25T16:30:00+00:00",
        "today": "2026-10-23",
    }
    assert state["team_news"]["home"]["factors"][0]["summary"] == "Saka is out"
    assert state["team_news"]["away"]["factors"][0]["player"] == "Palmer"
    json.dumps(state)  # plain JSON data


def test_today_can_be_given():
    jev = FakeJev()
    score_dossier(match_dossier(home=[factor()]), MATCH, jev, today=date(2026, 10, 24))
    assert jev.calls[0][0]["match"]["today"] == "2026-10-24"


def test_impacts_follow_the_levels_asked():
    dossier = match_dossier(home=[factor()], away=[factor(kind="return", misses="no", player="Palmer")])
    jev = FakeJev(
        scores={
            "home0_attack": [0, 0, 1, 0],
            "home0_defence": [0.5, 0.5, 0, 0],
            "away0_attack": [0, 0, 0, 1],
            "away0_defence": [0, 0, 0, 1],
        }
    )
    home, away = score_dossier(dossier, MATCH, jev)
    assert home.side == "home" and home.factor == dossier.home.factors[0]
    assert home.attack_mult == pytest.approx(0.92)
    assert home.concede_mult == pytest.approx(1.015)
    assert home.p_start is None
    assert away.side == "away"
    assert away.attack_mult == pytest.approx(1.10)
    assert away.concede_mult == pytest.approx(0.90)


def test_start_chance_softens_a_weakening_factor():
    dossier = match_dossier(home=[factor(kind="doubtful", misses="likely")])
    jev = FakeJev(scores={"home0_attack": [0, 0, 0, 1], "home0_defence": [0, 0, 0, 1]}, nouls={"home0_starts": 0.6})
    [impact] = score_dossier(dossier, MATCH, jev)
    assert impact.p_start == pytest.approx(0.6)
    assert impact.attack_mult == pytest.approx(1 - 0.15 * 0.4)
    assert impact.concede_mult == pytest.approx(1 + 0.15 * 0.4)


def test_start_chance_scales_a_boosting_factor_by_the_chance_he_plays():
    """A return or streak only helps if the player plays, so the effect shrinks as the start chance falls."""
    dossier = match_dossier(home=[factor(kind="return", misses="likely", player="Haaland")])
    jev = FakeJev(scores={"home0_attack": [0, 0, 0, 1], "home0_defence": [0, 0, 0, 1]}, nouls={"home0_starts": 0.25})
    [impact] = score_dossier(dossier, MATCH, jev)
    assert impact.attack_mult == pytest.approx(1 + 0.10 * 0.25)
    assert impact.concede_mult == pytest.approx(1 - 0.10 * 0.25)


def test_a_missing_answer_is_a_jev_failure():
    class Silent:
        def ask(self, state, nouls, scores):
            return JevAnswers(nouls={}, score_probabilities={})

    with pytest.raises(JevFailed, match="home0_attack"):
        score_dossier(match_dossier(home=[factor()]), MATCH, Silent())


def test_wrong_number_of_levels_is_a_jev_failure():
    with pytest.raises(JevFailed, match="levels"):
        score_dossier(match_dossier(home=[factor()]), MATCH, FakeJev(scores={"home0_attack": [0.5, 0.5]}))


def test_jev_failure_propagates():
    class Down:
        def ask(self, state, nouls, scores):
            raise JevFailed("down")

    with pytest.raises(JevFailed, match="down"):
        score_dossier(match_dossier(home=[factor()]), MATCH, Down())


# --- the TypeSafe adapter (SDK over a fake transport; nothing leaves the machine) -------------------------------------


def answers(**by_name) -> dict:
    return {"model": "jev-1.13.0", "usage": {"input_tokens": 10, "output_tokens": 2}, "answers": by_name}


def noul(p: float) -> dict:
    return {"type": "noul", "noul": p}


def score(probabilities: dict[str, float]) -> dict:
    return {
        "type": "score",
        "score": 1.0,
        "confidence": 0.9,
        "legend": {str(i): level for i, level in enumerate(LEVELS)},
        "probabilities": probabilities,
    }


class Api:
    def __init__(self, body=None, status=200) -> None:
        self.body, self.status = body, status
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx2.Response(self.status, json={"message": "no"})
        return httpx2.Response(200, json=self.body)


def jev_over(api: Api) -> TypeSafeJev:
    client = TypeSafeClient(
        api_key="secret", transport=httpx2.MockTransport(api), compression=False, retry=RetryPolicy(max_retries=0)
    )
    return TypeSafeJev("secret", "jev-1.13.0", client=client)


QUESTIONS = {"s1": ("How much?", list(LEVELS))}


def test_adapter_sends_noul_and_score_questions_and_maps_the_answers():
    api = Api(answers(n1=noul(0.7), s1=score({"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.1})))
    got = jev_over(api).ask({"match": "x"}, {"n1": "Will he start?"}, QUESTIONS)
    assert got == JevAnswers(nouls={"n1": 0.7}, score_probabilities={"s1": [0.1, 0.2, 0.6, 0.1]})
    [request] = api.requests
    body = json.loads(request.content)
    assert body["state"] == {"match": "x"}
    assert body["model"] == "jev-1.13.0"
    assert body["questions"]["n1"] == {"type": "noul", "instructions": "Will he start?"}
    assert body["questions"]["s1"] == {"type": "score", "instructions": "How much?", "criteria": list(LEVELS)}
    assert request.headers["authorization"] == "Bearer secret"


def test_adapter_orders_probabilities_by_level_number_not_by_arrival():
    api = Api(answers(s1=score({"3": 0.1, "2": 0.6, "1": 0.2, "0": 0.1})))
    assert jev_over(api).ask({}, {}, QUESTIONS).score_probabilities == {"s1": [0.1, 0.2, 0.6, 0.1]}


def test_adapter_treats_a_missing_level_as_zero_and_renormalises():
    api = Api(answers(s1=score({"0": 0.2, "2": 0.6})))
    got = jev_over(api).ask({}, {}, QUESTIONS).score_probabilities["s1"]
    assert got == pytest.approx([0.25, 0.0, 0.75, 0.0])


def test_adapter_rejects_levels_it_did_not_ask_for():
    api = Api(answers(s1=score({"0": 0.5, "4": 0.5})))
    with pytest.raises(JevFailed, match="s1"):
        jev_over(api).ask({}, {}, QUESTIONS)


def test_adapter_fails_when_a_question_is_unanswered():
    with pytest.raises(JevFailed, match="n1"):
        jev_over(Api(answers())).ask({}, {"n1": "Will he start?"}, {})


@pytest.mark.parametrize("status", [401, 422, 400])
def test_adapter_turns_sdk_errors_into_jev_failures(status):
    with pytest.raises(JevFailed) as err:
        jev_over(Api(status=status)).ask({}, {"n1": "q"}, {})
    assert "secret" not in str(err.value)


def test_adapter_fails_on_an_unreadable_response():
    with pytest.raises(JevFailed):
        jev_over(Api({"answers": "nope"})).ask({}, {"n1": "q"}, {})


def test_adapter_with_a_blank_key_is_a_jev_failure_not_a_crash(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevFailed):
        TypeSafeJev("", "jev-1.13.0").ask({}, {"n1": "q"}, {})
