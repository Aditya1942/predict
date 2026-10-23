from datetime import UTC, datetime, timedelta

import pytest

from predict.domain import MARKET_IDS, HotTake, Prediction, Probabilities, ProjectedGoals, Result
from predict.log import VOID_STATUSES, LogSummary, counted_prediction, outcome, settle_pending, summarize
from predict.params import SETTLE_AFTER
from predict.sources.football_data_org import MatchStatus
from predict.store import Store
from tests.helpers import FakeFdo, make_match, make_team

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
KICKOFF = NOW - timedelta(hours=3)  # long enough ago to settle

ARSENAL = make_team(1, "Arsenal")
CHELSEA = make_team(2, "Chelsea")


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def match(id: int = 1, kickoff: datetime = KICKOFF):
    return make_match(ARSENAL, CHELSEA, kickoff=kickoff, id=id)


def probabilities(result="home", btts="yes", total="over", team="over") -> Probabilities:
    """Every Market, with the named answer the Pick (60%) and the rest sharing 40%."""
    chosen = {"result": result, "btts": btts}
    out: Probabilities = {}
    for market_id in MARKET_IDS:
        answers = {
            "result": ["home", "draw", "away"],
            "btts": ["yes", "no"],
        }.get(market_id, ["over", "under"])
        pick = chosen.get(market_id) or (total if market_id.startswith("total") else team)
        rest = [a for a in answers if a != pick]
        out[market_id] = {pick: 0.6, **{a: 0.4 / len(rest) for a in rest}}
    return out


def prediction(m=None, *, created: datetime | None = None, shared: datetime | None = None, **changes) -> Prediction:
    m = m or match()
    goals = ProjectedGoals(home=1.5, away=1.2)
    fields = {
        "match": m,
        "created_at": created or m.kickoff - timedelta(days=1),
        "projected_base": goals,
        "projected": goals,
        "probabilities": probabilities(),
        "shared_at": shared,
        **changes,
    }
    return Prediction(**fields)


def finished(m, home: int, away: int) -> MatchStatus:
    return MatchStatus(match=m, status="FINISHED", result=Result(home_goals=home, away_goals=away))


def status(m, name: str) -> MatchStatus:
    return MatchStatus(match=m, status=name)


# --- outcome ---------------------------------------------------------------------------------------------------------


def test_outcomes():
    r = Result(home_goals=2, away_goals=1)
    assert [outcome(m, r) for m in ("result", "btts", "total:2.5", "total:3.5", "team:away:0.5", "team:away:1.5")] == [
        "home",
        "yes",
        "over",
        "under",
        "over",
        "under",
    ]


@pytest.mark.parametrize(
    ("home", "away", "expected"),
    [
        (0, 0, ("draw", "no", "under", "under", "under")),
        (0, 3, ("away", "no", "over", "under", "over")),
        (1, 1, ("draw", "yes", "over", "under", "over")),
        (3, 0, ("home", "no", "over", "over", "under")),
    ],
)
def test_outcomes_for_other_scores(home, away, expected):
    r = Result(home_goals=home, away_goals=away)
    markets = ("result", "btts", "total:1.5", "team:home:1.5", "team:away:0.5")
    assert tuple(outcome(m, r) for m in markets) == expected


def test_a_line_is_beaten_only_by_more_goals():
    r = Result(home_goals=2, away_goals=0)
    assert outcome("team:home:1.5", r) == "over"
    assert outcome("team:home:2.5", r) == "under"
    assert outcome("total:1.5", r) == "over"


def test_unknown_market_is_an_error():
    with pytest.raises(ValueError, match="market"):
        outcome("nonsense", Result(home_goals=0, away_goals=0))


# --- counted_prediction ----------------------------------------------------------------------------------------------

TUESDAY = datetime(2026, 10, 20, 18, 0, tzinfo=UTC)
SATURDAY = datetime(2026, 10, 24, 9, 0, tzinfo=UTC)
M = match(kickoff=datetime(2026, 10, 24, 15, 0, tzinfo=UTC))


def test_counted_is_the_shared_one_over_a_later_unshared_one():
    shared = prediction(M, created=TUESDAY, shared=TUESDAY, id=1)
    later = prediction(M, created=SATURDAY, id=2)
    assert counted_prediction([shared, later]) == shared
    assert counted_prediction([later, shared]) == shared


def test_counted_is_the_latest_shared_one():
    first = prediction(M, created=TUESDAY, shared=TUESDAY, id=1)
    second = prediction(M, created=SATURDAY, shared=SATURDAY, id=2)
    assert counted_prediction([second, first]) == second


def test_counted_without_a_shared_one_is_the_latest_created():
    first = prediction(M, created=TUESDAY, id=1)
    second = prediction(M, created=SATURDAY, id=2)
    assert counted_prediction([second, first]) == second


def test_counted_ignores_a_prediction_created_after_kickoff():
    before = prediction(M, created=TUESDAY, id=1)
    after = prediction(M, created=M.kickoff + timedelta(minutes=1), shared=M.kickoff + timedelta(minutes=1), id=2)
    at = prediction(M, created=M.kickoff, id=3)
    assert counted_prediction([before, after, at]) == before
    assert counted_prediction([after, at]) is None


def test_counted_is_none_without_predictions():
    assert counted_prediction([]) is None


# --- settle_pending --------------------------------------------------------------------------------------------------


def settle(store, statuses, now=NOW):
    fdo = FakeFdo([], statuses)
    return settle_pending(store, fdo, now), fdo


def test_a_finished_match_is_settled_with_its_result(store):
    m = match()
    store.save_prediction(prediction(m))
    count, fdo = settle(store, [finished(m, 2, 1)])
    assert count == 1
    assert fdo.status_calls == [[1]]
    assert store.results() == {1: ("finished", Result(home_goals=2, away_goals=1))}
    assert store.unsettled_match_ids(NOW) == []


def test_a_postponed_match_is_void(store):
    m = match()
    store.save_prediction(prediction(m))
    count, _ = settle(store, [status(m, "POSTPONED")])
    assert count == 1
    assert store.results() == {1: ("void", None)}


@pytest.mark.parametrize("name", sorted(VOID_STATUSES))
def test_every_void_status_voids(store, name):
    m = match()
    store.save_prediction(prediction(m))
    settle(store, [status(m, name)])
    assert store.results() == {1: ("void", None)}


def test_void_statuses():
    assert VOID_STATUSES == {"POSTPONED", "CANCELLED", "SUSPENDED", "AWARDED"}


def test_a_rescheduled_match_is_void_not_scored_against_the_new_date(store):
    m = match()
    store.save_prediction(prediction(m))
    moved = match(kickoff=KICKOFF + timedelta(days=5))
    count, _ = settle(store, [status(moved, "TIMED")])
    assert count == 1
    assert store.results() == {1: ("void", None)}


def test_a_rescheduled_match_that_has_since_finished_is_still_void(store):
    m = match()
    store.save_prediction(prediction(m))
    moved = match(kickoff=KICKOFF + timedelta(days=5))
    settle(store, [finished(moved, 3, 0)])
    assert store.results() == {1: ("void", None)}


def test_a_small_kickoff_change_is_left_waiting(store):
    m = match()
    store.save_prediction(prediction(m))
    count, _ = settle(store, [status(match(kickoff=KICKOFF + timedelta(hours=2)), "TIMED")])
    assert count == 0
    assert store.results() == {}


def test_a_kickoff_move_of_exactly_48_hours_is_not_void(store):
    m = match()
    store.save_prediction(prediction(m))
    settle(store, [status(match(kickoff=KICKOFF + timedelta(hours=48)), "TIMED")])
    assert store.results() == {}


def test_a_match_still_in_play_is_skipped(store):
    m = match()
    store.save_prediction(prediction(m))
    count, _ = settle(store, [status(m, "IN_PLAY")])
    assert count == 0
    assert store.results() == {}
    assert store.unsettled_match_ids(NOW) == [1]  # tried again next run


def test_a_match_the_source_does_not_return_is_skipped(store):
    store.save_prediction(prediction(match()))
    count, _ = settle(store, [])
    assert count == 0
    assert store.results() == {}


def test_nothing_is_asked_before_the_settle_delay(store):
    store.save_prediction(prediction(match()))
    count, fdo = settle(store, [finished(match(), 1, 0)], now=KICKOFF + SETTLE_AFTER - timedelta(minutes=1))
    assert (count, fdo.status_calls) == (0, [])
    count, fdo = settle(store, [finished(match(), 1, 0)], now=KICKOFF + SETTLE_AFTER + timedelta(minutes=1))
    assert (count, fdo.status_calls) == (1, [[1]])


def test_a_settled_match_is_not_asked_again(store):
    m = match()
    store.save_prediction(prediction(m))
    settle(store, [finished(m, 1, 0)])
    count, fdo = settle(store, [finished(m, 5, 5)])
    assert (count, fdo.status_calls) == (0, [])
    assert store.results() == {1: ("finished", Result(home_goals=1, away_goals=0))}


def test_each_match_is_asked_once_however_many_predictions(store):
    m1, m2 = match(1), match(2, KICKOFF - timedelta(hours=1))
    for m in (m1, m1, m2):
        store.save_prediction(prediction(m))
    count, fdo = settle(store, [finished(m1, 1, 0), finished(m2, 0, 0)])
    assert count == 2
    assert fdo.status_calls == [[1, 2]]


def test_a_failing_source_leaves_everything_unsettled(store):
    from predict.errors import SourceError

    store.save_prediction(prediction(match()))
    fdo = FakeFdo([])
    fdo.status_error = SourceError("api.football-data.org", "boom")
    with pytest.raises(SourceError):
        settle_pending(store, fdo, NOW)
    assert store.results() == {}


# --- summarize -------------------------------------------------------------------------------------------------------


def settled(store, m, *, result: tuple[int, int] | None, **changes) -> Prediction:
    saved = prediction(m, **changes)
    saved = saved.model_copy(update={"id": store.save_prediction(saved)})
    if result is None:
        store.save_result(m.id, "void", None, NOW)
    else:
        store.save_result(m.id, "finished", Result(home_goals=result[0], away_goals=result[1]), NOW)
    return saved


def test_hit_rates_count_picks_that_matched_the_result(store):
    # Picks: home, BTTS yes, over every Total Goals line, over every Team Goals line (both teams).
    settled(store, match(1), result=(2, 1))
    settled(store, match(2), result=(0, 1))
    rates = summarize(store).hit_rates
    assert rates["Match Result"] == (1, 2)
    assert rates["BTTS"] == (1, 2)
    assert rates["Total Goals 1.5"] == (1, 2)  # 3 goals beat it, 1 does not
    assert rates["Total Goals 2.5"] == (1, 2)
    assert rates["Total Goals 3.5"] == (0, 2)  # 3 goals do not beat 3.5
    assert rates["Team Goals 0.5"] == (3, 4)  # 2, 1 and 1 goals beat it, 0 does not
    assert rates["Team Goals 1.5"] == (1, 4)  # only the 2
    assert rates["Team Goals 2.5"] == (0, 4)


def test_every_group_is_present_even_with_nothing_scored(store):
    assert summarize(store) == LogSummary(
        hit_rates={
            key: (0, 0)
            for key in (
                "Match Result",
                "BTTS",
                "Total Goals 1.5",
                "Total Goals 2.5",
                "Total Goals 3.5",
                "Team Goals 0.5",
                "Team Goals 1.5",
                "Team Goals 2.5",
            )
        },
        hot_takes=(0, 0),
        recent=[],
    )


def test_only_the_counted_prediction_of_a_match_is_scored(store):
    m = match()
    store.save_prediction(prediction(m, created=m.kickoff - timedelta(days=2), probabilities=probabilities("away")))
    settled(store, m, result=(1, 0), created=m.kickoff - timedelta(days=1))  # the later one, a home Pick
    summary = summarize(store)
    assert summary.hit_rates["Match Result"] == (1, 1)
    assert len(summary.recent) == 1


def test_a_void_match_and_an_unsettled_match_are_excluded(store):
    settled(store, match(1), result=(2, 1))
    settled(store, match(2), result=None)
    store.save_prediction(prediction(match(3)))  # no result yet
    summary = summarize(store)
    assert summary.hit_rates["Match Result"] == (1, 1)
    assert [p.match.id for p, _ in summary.recent] == [1]


def hot_take_record(store, take: HotTake, result: tuple[int, int]) -> tuple[int, int]:
    settled(store, match(), result=result, hot_takes=[take])
    return summarize(store).hot_takes


@pytest.mark.parametrize(
    ("model_p", "book_p", "result", "hit"),
    [
        (0.60, 0.45, (2, 0), True),  # rated higher than the bookies and it happened
        (0.60, 0.45, (0, 2), False),  # rated higher and it did not happen
        (0.30, 0.45, (0, 2), True),  # rated lower and it did not happen
        (0.30, 0.45, (2, 0), False),  # rated lower and it happened
    ],
)
def test_hot_take_hit_rule_both_directions(store, model_p, book_p, result, hit):
    take = HotTake(market="result", answer="home", model_p=model_p, book_p=book_p)
    assert hot_take_record(store, take, result) == (int(hit), 1)


def test_hot_takes_count_each_take_of_each_counted_prediction(store):
    takes = [
        HotTake(market="result", answer="home", model_p=0.60, book_p=0.45),
        HotTake(market="btts", answer="yes", model_p=0.65, book_p=0.50),
    ]
    settled(store, match(1), result=(2, 0), hot_takes=takes)  # home yes, btts no
    settled(store, match(2), result=None, hot_takes=takes)  # void: not counted
    assert summarize(store).hot_takes == (1, 2)


def test_recent_holds_the_last_ten_counted_newest_first(store):
    for i in range(1, 13):
        settled(store, match(i, KICKOFF - timedelta(days=13 - i)), result=(1, 0))
    recent = summarize(store).recent
    assert [p.match.id for p, _ in recent] == list(range(12, 2, -1))
    assert recent[0][1] == Result(home_goals=1, away_goals=0)
