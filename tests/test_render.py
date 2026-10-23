from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from predict.domain import MARKET_IDS, FactorImpact, HotTake, Prediction, ProjectedGoals
from predict.model.markets import market_probabilities
from predict.model.scoreline import scoreline_matrix
from predict.render import (
    answer_label,
    copy_to_clipboard,
    format_kickoff,
    full_view,
    market_label,
    share_text,
)
from tests.helpers import make_factor, make_match, make_team, system_zone

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


# --- Team News in the full view -------------------------------------------------------------------------------------


def impact(side="home", attack=0.92, concede=1.08, **factor) -> FactorImpact:
    return FactorImpact(side=side, factor=make_factor(**factor), attack_mult=attack, concede_mult=concede, p_start=None)


def test_full_view_lists_the_reasons_per_team():
    saka = impact(sources=("https://example.com/saka", "https://example.com/other"))
    palmer = impact("away", 1.04, 0.995, kind="return", player=None, summary="Back from suspension.")
    text = full_view(prediction(impacts=[saka, palmer], news_applied=True), IST)
    lines = [line.strip() for line in text.splitlines()]
    assert "Team News" in lines
    assert "Saka: Out for three weeks with a hamstring injury. → Arsenal goals -8% → Chelsea goals +8%" in lines
    assert "https://example.com/saka" in lines
    assert "https://example.com/other" not in text
    # no player: the kind names it; a concede change under 1 point is left out; no source: no URL line
    assert "return: Back from suspension. → Chelsea goals +4%" in lines


def test_full_view_leaves_out_the_opponent_when_it_barely_moves():
    text = full_view(prediction(impacts=[impact(concede=1.009)], news_applied=True), IST)
    assert "→ Arsenal goals -8%" in text
    assert "→ Chelsea goals" not in text
    text = full_view(prediction(impacts=[impact(concede=0.99)], news_applied=True), IST)
    assert "→ Chelsea goals -1%" in text


def test_full_view_shows_only_the_top_three_reasons_per_team():
    attacks = (("A", 0.99), ("B", 0.9), ("C", 0.95), ("D", 0.8))
    impacts = [impact(player=name, attack=a, concede=1.0) for name, a in attacks]
    text = full_view(prediction(impacts=impacts, news_applied=True), IST)
    assert [name for name in "ABCD" if f"{name}: " in text] == ["B", "C", "D"]


def test_full_view_without_news_factors_says_so():
    text = full_view(prediction(news_applied=True), IST)
    assert "Team News" in text
    assert "nothing that changes the numbers" in text


def test_footer_says_news_was_applied_with_the_models():
    text = full_view(prediction(news_applied=True, claude_model="claude-opus-5-5", jev_model="jev-1.13.0"), IST)
    assert "news applied ✓" in text
    assert "claude-opus-5-5" in text
    assert "jev-1.13.0" in text
    assert "not applied" not in text


def test_footer_says_why_news_was_not_applied():
    text = full_view(prediction(news_note="no TYPESAFE_API_KEY set"), IST)
    assert "⚠️ news not applied (no TYPESAFE_API_KEY set)" in text
    assert "Team News" not in text
    assert "news applied ✓" not in text


# --- the share view -------------------------------------------------------------------------------------------------

SHARE_KICKOFF = datetime(2025, 10, 25, 16, 30, tzinfo=UTC)  # a Saturday
KOLKATA = ZoneInfo("Asia/Kolkata")


def share_prediction(**changes) -> Prediction:
    probabilities = {
        "result": {"home": 0.54, "draw": 0.24, "away": 0.22},
        "btts": {"yes": 0.52, "no": 0.48},
        "total:2.5": {"over": 0.51, "under": 0.49},
        "team:home:1.5": {"over": 0.58, "under": 0.42},
        "team:away:1.5": {"over": 0.34, "under": 0.66},
    }
    changes = {
        "match": make_match(ARSENAL, CHELSEA, kickoff=SHARE_KICKOFF),
        "probabilities": probabilities,
        "hot_takes": [HotTake(market="result", answer="home", model_p=0.54, book_p=0.44)],
        **changes,
    }
    return prediction(**changes)


def test_share_text_exact():
    assert share_text(share_prediction(), (21, 32), KOLKATA) == "\n".join(
        [
            "⚽ Arsenal v Chelsea · Premier League · Sat 25 Oct, 22:00",
            "🏆 Arsenal 54% · Draw 24% · Chelsea 22%",
            "🎯 Both teams to score: Yes 52%",
            "🥅 Total goals: Over 2.5 · 51%",
            "👟 Arsenal over 1.5 goals 58% · Chelsea under 1.5 goals 66%",
            "🔥 Hot Take: Arsenal to win (54% vs bookies 44%)",
            "📈 Record: 21/32 results called",
        ]
    )


def test_share_text_without_hot_takes_or_a_record_is_five_lines():
    text = share_text(share_prediction(hot_takes=[]), None, KOLKATA)
    assert len(text.splitlines()) == 5
    assert "🔥" not in text
    assert "📈" not in text


def test_share_text_leaves_out_a_record_with_nothing_scored():
    assert "📈" not in share_text(share_prediction(), (0, 0), KOLKATA)


def test_share_text_shows_the_pick_of_each_market():
    probabilities = {
        **share_prediction().probabilities,
        "btts": {"yes": 0.45, "no": 0.55},
        "total:2.5": {"over": 0.40, "under": 0.60},
    }
    lines = share_text(share_prediction(probabilities=probabilities), None, KOLKATA).splitlines()
    assert lines[2] == "🎯 Both teams to score: No 55%"
    assert lines[3] == "🥅 Total goals: Under 2.5 · 60%"


def test_share_text_has_at_most_three_hot_takes():
    takes = [HotTake(market="result", answer="home", model_p=0.5 + i / 100, book_p=0.3) for i in range(4)]
    lines = share_text(share_prediction(hot_takes=takes), None, KOLKATA).splitlines()
    assert sum(line.startswith("🔥") for line in lines) == 3


def test_share_text_without_a_zone_uses_the_system_zone():
    with system_zone("Asia/Kolkata"):
        assert share_text(share_prediction(), None, None).splitlines()[0].endswith("Sat 25 Oct, 22:00")


def test_copy_to_clipboard_pipes_the_text_to_pbcopy(monkeypatch, tmp_path):
    out = tmp_path / "clip"
    script = tmp_path / "bin" / "pbcopy"
    script.parent.mkdir()
    script.write_text(f"#!/bin/sh\ncat > {out}\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{script.parent}:/usr/bin:/bin")  # ahead of any real pbcopy
    assert copy_to_clipboard("⚽ Arsenal v Chelsea") is True
    assert out.read_text() == "⚽ Arsenal v Chelsea"


def test_copy_to_clipboard_is_false_without_pbcopy(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert copy_to_clipboard("text") is False
