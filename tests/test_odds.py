import pytest

from predict.domain import HotTake
from predict.odds import devig, hot_takes


def test_devig():
    p = devig({"home": 2.0, "draw": 3.5, "away": 4.0})
    assert (p["home"], p["draw"], p["away"]) == (
        pytest.approx(0.48276, abs=1e-5),
        pytest.approx(0.27586, abs=1e-5),
        pytest.approx(0.24138, abs=1e-5),
    )


def test_devig_sums_to_one_and_keeps_the_keys():
    p = devig({"over": 1.85, "under": 2.05})
    assert list(p) == ["over", "under"]
    assert sum(p.values()) == pytest.approx(1.0)


def test_devig_of_a_fair_book_changes_nothing():
    assert devig({"yes": 2.0, "no": 2.0}) == {"yes": 0.5, "no": 0.5}


@pytest.mark.parametrize("prices", [{}, {"home": 2.0, "draw": 0.0}, {"home": -1.5, "away": 2.0}])
def test_devig_rejects_unusable_prices(prices):
    with pytest.raises(ValueError, match="price"):
        devig(prices)


def test_hot_take_diff_is_model_minus_book():
    take = HotTake(market="total:2.5", answer="over", model_p=0.62, book_p=0.50)
    assert take.diff == pytest.approx(0.12)


def test_two_way_market_reports_only_the_answer_the_model_favours_more():
    model = {"total:2.5": {"over": 0.62, "under": 0.38}}
    book = {"total:2.5": {"over": 0.50, "under": 0.50}}
    [take] = hot_takes(model, book)
    assert (take.market, take.answer) == ("total:2.5", "over")
    assert take.diff == pytest.approx(0.12)


def test_difference_below_the_threshold_is_not_a_hot_take():
    model = {"total:2.5": {"over": 0.55, "under": 0.45}}
    book = {"total:2.5": {"over": 0.50, "under": 0.50}}
    assert hot_takes(model, book) == []


def test_exactly_ten_points_is_a_hot_take_despite_float_noise():
    model = {"btts": {"yes": 0.60, "no": 0.40}}
    book = {"btts": {"yes": 0.50, "no": 0.50}}  # 0.60 - 0.50 is 0.0999... in floating point
    assert [(t.answer) for t in hot_takes(model, book)] == ["yes"]


def test_result_reports_every_answer_that_differs_either_way_largest_first():
    model = {"result": {"home": 0.54, "draw": 0.24, "away": 0.22}}
    book = {"result": {"home": 0.40, "draw": 0.25, "away": 0.35}}
    takes = hot_takes(model, book)
    assert [(t.answer, round(t.diff, 2)) for t in takes] == [("home", 0.14), ("away", -0.13)]


def test_markets_the_book_does_not_list_are_skipped():
    model = {"result": {"home": 0.7, "draw": 0.2, "away": 0.1}, "btts": {"yes": 0.8, "no": 0.2}}
    book = {"btts": {"yes": 0.5, "no": 0.5}}
    assert [(t.market, t.answer) for t in hot_takes(model, book)] == [("btts", "yes")]


def test_sorted_by_absolute_difference_across_markets():
    model = {
        "btts": {"yes": 0.62, "no": 0.38},
        "total:2.5": {"over": 0.35, "under": 0.65},
        "result": {"home": 0.3, "draw": 0.3, "away": 0.4},
    }
    book = {
        "btts": {"yes": 0.50, "no": 0.50},
        "total:2.5": {"over": 0.52, "under": 0.48},
        "result": {"home": 0.3, "draw": 0.3, "away": 0.4},
    }
    assert [(t.market, t.answer) for t in hot_takes(model, book)] == [("total:2.5", "under"), ("btts", "yes")]


def test_custom_threshold():
    model = {"btts": {"yes": 0.55, "no": 0.45}}
    book = {"btts": {"yes": 0.50, "no": 0.50}}
    assert [t.answer for t in hot_takes(model, book, threshold=0.05)] == ["yes"]
