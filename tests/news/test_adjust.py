import pytest

from predict.domain import FactorImpact, NewsFactor, ProjectedGoals
from predict.news.adjust import apply_impacts, top_reasons

F = NewsFactor(kind="absence", player="Saka", summary="Out", misses_match="yes", matches_missed_of_last_6=0, sources=[])


def impact(side="home", attack=1.0, concede=1.0, summary="Out", p_start=None) -> FactorImpact:
    factor = F.model_copy(update={"summary": summary})
    return FactorImpact(side=side, factor=factor, attack_mult=attack, concede_mult=concede, p_start=p_start)


def test_cap():
    imp = [impact(attack=0.85)] * 3
    assert apply_impacts(ProjectedGoals(home=2.0, away=1.0), imp).home == pytest.approx(1.5)


def test_cap_applies_upwards_too():
    imp = [impact(side="away", attack=1.10)] * 4  # 1.4641
    assert apply_impacts(ProjectedGoals(home=2.0, away=1.0), imp).away == pytest.approx(1.25)


def test_a_custom_cap():
    imp = [impact(attack=0.85)] * 3
    assert apply_impacts(ProjectedGoals(home=2.0, away=1.0), imp, cap=(0.9, 1.1)).home == pytest.approx(1.8)


def test_attack_and_opponent_defence_combine():
    imp = [impact(side="home", attack=0.92), impact(side="away", concede=1.08)]
    assert apply_impacts(ProjectedGoals(home=2.0, away=1.0), imp).home == pytest.approx(2.0 * 0.9936)


def test_each_side_uses_its_own_attack_and_the_others_concede():
    imp = [
        impact(side="home", attack=0.90, concede=1.10),
        impact(side="away", attack=0.95, concede=1.05),
    ]
    adjusted = apply_impacts(ProjectedGoals(home=2.0, away=1.0), imp)
    assert adjusted.home == pytest.approx(2.0 * 0.90 * 1.05)
    assert adjusted.away == pytest.approx(1.0 * 0.95 * 1.10)


def test_no_impacts_changes_nothing():
    base = ProjectedGoals(home=1.7, away=1.1)
    assert apply_impacts(base, []) == base


def test_base_is_not_modified():
    base = ProjectedGoals(home=2.0, away=1.0)
    apply_impacts(base, [impact(attack=0.9)])
    assert base == ProjectedGoals(home=2.0, away=1.0)


def test_top_reasons_are_the_largest_effects_of_that_side_first():
    imps = [
        impact("home", attack=0.97, summary="small"),
        impact("home", attack=0.85, summary="big"),
        impact("away", attack=0.80, summary="other side"),
        impact("home", concede=1.08, attack=1.0, summary="defence"),
        impact("home", attack=1.0, summary="none"),
    ]
    got = top_reasons(imps, "home")
    assert [i.factor.summary for i in got] == ["big", "defence", "small"]


def test_top_reasons_measure_the_larger_of_attack_and_concede():
    a, b = impact(attack=0.95, concede=1.12, summary="a"), impact(attack=0.90, concede=1.0, summary="b")
    assert [i.factor.summary for i in top_reasons([b, a], "home")] == ["a", "b"]


def test_top_reasons_n():
    imps = [impact(attack=1 - 0.01 * k, summary=str(k)) for k in range(1, 6)]
    assert len(top_reasons(imps, "home")) == 3
    assert [i.factor.summary for i in top_reasons(imps, "home", n=1)] == ["5"]
    assert top_reasons(imps, "away") == []
