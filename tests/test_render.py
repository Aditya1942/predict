from datetime import UTC, datetime, timedelta, timezone

import pytest

from predict.domain import MARKET_IDS, Prediction, ProjectedGoals
from predict.model.markets import market_probabilities
from predict.model.scoreline import scoreline_matrix
from predict.render import answer_label, format_kickoff, full_view, market_label
from tests.helpers import make_match, make_team

ARSENAL = make_team(1, "Arsenal")
CHELSEA = make_team(2, "Chelsea")
KICKOFF = datetime(2026, 10, 25, 16, 30, tzinfo=UTC)
MATCH = make_match(ARSENAL, CHELSEA, kickoff=KICKOFF)
IST = timezone(timedelta(hours=5, minutes=30), "IST")


def prediction(**changes) -> Prediction:
    projected = ProjectedGoals(home=1.62, away=1.1)
    base = {
        "match": MATCH,
        "created_at": KICKOFF - timedelta(days=1),
        "projected_base": projected,
        "projected": projected,
        "probabilities": market_probabilities(scoreline_matrix(projected.home, projected.away)),
    }
    return Prediction(**{**base, **changes})


@pytest.mark.parametrize(
    ("market_id", "label"),
    [
        ("result", "Match Result"),
        ("btts", "Both teams to score"),
        ("total:2.5", "Total goals 2.5"),
        ("team:home:1.5", "Arsenal goals 1.5"),
        ("team:away:0.5", "Chelsea goals 0.5"),
    ],
)
def test_market_label(market_id, label):
    assert market_label(market_id, MATCH) == label


@pytest.mark.parametrize(
    ("market_id", "answer", "label"),
    [
        ("result", "home", "Arsenal to win"),
        ("result", "draw", "Draw"),
        ("result", "away", "Chelsea to win"),
        ("btts", "yes", "BTTS: Yes"),
        ("btts", "no", "BTTS: No"),
        ("total:2.5", "over", "Over 2.5 goals"),
        ("total:1.5", "under", "Under 1.5 goals"),
        ("team:home:1.5", "over", "Arsenal over 1.5 goals"),
        ("team:away:1.5", "under", "Chelsea under 1.5 goals"),
    ],
)
def test_answer_label(market_id, answer, label):
    assert answer_label(market_id, answer, MATCH) == label


def test_every_market_and_answer_has_a_label():
    for market_id, answers in prediction().probabilities.items():
        assert market_label(market_id, MATCH)
        for answer in answers:
            assert answer_label(market_id, answer, MATCH)


@pytest.mark.parametrize("market_id", ["nonsense", "team:side:1.5", "total"])
def test_unknown_market_is_an_error(market_id):
    with pytest.raises(ValueError, match="market"):
        market_label(market_id, MATCH)
    with pytest.raises(ValueError, match="market"):
        answer_label(market_id, "over", MATCH)


def test_unknown_answer_is_an_error():
    with pytest.raises(ValueError, match="answer"):
        answer_label("result", "nobody", MATCH)


def test_format_kickoff_is_local():
    assert format_kickoff(KICKOFF, IST) == "Sun 25 Oct, 22:00 IST"
    assert format_kickoff(KICKOFF, UTC) == "Sun 25 Oct, 16:30 UTC"


def test_full_view_header_and_projected_goals():
    text = full_view(prediction(), IST)
    assert "Arsenal v Chelsea" in text
    assert "Premier League" in text
    assert "Sun 25 Oct, 22:00 IST" in text
    assert "Projected Goals" in text
    assert "1.62 → 1.62" in text
    assert "1.10 → 1.10" in text


def test_full_view_shows_base_then_adjusted_goals():
    text = full_view(prediction(projected=ProjectedGoals(home=1.40, away=1.25)), IST)
    assert "1.62 → 1.40" in text
    assert "1.10 → 1.25" in text


def test_full_view_has_a_row_per_market_with_whole_percentages():
    p = prediction()
    text = full_view(p, IST)
    lines = text.splitlines()
    for market_id in MARKET_IDS:
        label = market_label(market_id, MATCH)
        assert sum(1 for line in lines if line.lstrip().startswith(label)) == 1, label
    assert "Match Result" in text
    home = round(p.probabilities["result"]["home"] * 100)
    assert f"Arsenal to win {home}%" in text
    assert "." not in next(line for line in lines if "Both teams to score" in line)


def test_full_view_marks_a_neutral_venue():
    text = full_view(prediction(match=make_match(ARSENAL, CHELSEA, kickoff=KICKOFF, neutral=True)), IST)
    assert "neutral venue" in text


def test_full_view_is_plain_text():
    assert "\x1b" not in full_view(prediction(), IST)
