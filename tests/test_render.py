from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from predict.domain import MARKET_IDS, HotTake, Prediction, ProjectedGoals
from predict.model.markets import market_probabilities
from predict.model.scoreline import scoreline_matrix
from predict.render import answer_label, format_kickoff, full_view, market_label
from tests.helpers import make_match, make_team, system_zone

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


LONDON = ZoneInfo("Europe/London")
BST_KICKOFF = datetime(2026, 10, 24, 15, 0, tzinfo=UTC)  # the clocks go back on 25 Oct 2026
GMT_KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)


def test_format_kickoff_follows_the_daylight_saving_rules_of_a_named_zone():
    assert format_kickoff(BST_KICKOFF, LONDON) == "Sat 24 Oct, 16:00 BST"
    assert format_kickoff(GMT_KICKOFF, LONDON) == "Sat 07 Nov, 15:00 GMT"


def test_format_kickoff_without_a_zone_uses_the_system_zone_rules_for_each_instant():
    with system_zone("Europe/London"):
        assert format_kickoff(BST_KICKOFF, None) == "Sat 24 Oct, 16:00 BST"
        assert format_kickoff(GMT_KICKOFF, None) == "Sat 07 Nov, 15:00 GMT"
    with system_zone("Asia/Kolkata"):
        assert format_kickoff(GMT_KICKOFF, None) == "Sat 07 Nov, 20:30 IST"


def test_full_view_without_a_zone_shows_local_time():
    with system_zone("Europe/London"):
        text = full_view(prediction(match=make_match(ARSENAL, CHELSEA, kickoff=GMT_KICKOFF)), None)
    assert "Arsenal v Chelsea · Premier League · Sat 07 Nov, 15:00 GMT" in text


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


BOOK = {
    "result": {"home": 0.40, "draw": 0.26, "away": 0.34},
    "btts": {"yes": 0.51, "no": 0.49},
}


def with_bookmaker(**changes) -> Prediction:
    return prediction(bookmaker=BOOK, hot_takes=[], credits_remaining=480, **changes)


def test_full_view_without_bookmaker_has_no_bookies_column_and_says_why():
    text = full_view(prediction(odds_note="no ODDS_API_KEY set"), IST)
    assert "Bookies" not in text
    assert "Hot Takes" not in text
    assert "Bookmaker odds unavailable: no ODDS_API_KEY set" in text


def test_full_view_shows_bookmaker_probabilities_beside_ours():
    text = full_view(with_bookmaker(), IST)
    lines = text.splitlines()
    assert "Bookies" in lines[next(i for i, line in enumerate(lines) if line.lstrip().startswith("Market"))]
    result = next(line for line in lines if line.lstrip().startswith("Match Result"))
    assert "40%" in result and "26%" in result and "34%" in result
    btts = next(line for line in lines if line.lstrip().startswith("Both teams to score"))
    assert "51%" in btts and "49%" in btts


def test_markets_the_bookmakers_do_not_list_show_a_dash():
    text = full_view(with_bookmaker(), IST)
    total = next(line for line in text.splitlines() if line.lstrip().startswith("Total goals 2.5"))
    assert total.rstrip().endswith("—")


def test_full_view_lists_hot_takes_and_credits():
    p = with_bookmaker()
    take = HotTake(market="result", answer="home", model_p=0.54, book_p=0.40)
    away = HotTake(market="btts", answer="no", model_p=0.36, book_p=0.49)
    text = full_view(p.model_copy(update={"hot_takes": [take, away]}), IST)
    lines = text.splitlines()
    start = lines.index("Hot Takes")
    assert lines[start + 1].strip() == "Arsenal to win: us 54% vs bookies 40% (+14 pts)"
    assert lines[start + 2].strip() == "BTTS: No: us 36% vs bookies 49% (-13 pts)"
    assert "Odds credits remaining: 480" in text


def test_full_view_says_when_there_are_no_hot_takes():
    text = full_view(with_bookmaker(), IST)
    assert "Hot Takes: none" in text


def test_full_view_hides_credits_when_unknown():
    text = full_view(with_bookmaker().model_copy(update={"credits_remaining": None}), IST)
    assert "credits" not in text
