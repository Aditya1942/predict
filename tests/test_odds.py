import pytest

from predict.odds import devig


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
