import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from predict import backtest
from predict.backtest import BacktestReport, Forecast, brier, format_reports, run_backtest, walk_forward
from predict.domain import LEAGUES, Competition, MatchRecord
from predict.errors import SourceError
from predict.http import CachedHttp
from predict.model.markets import pick
from predict.odds import devig
from predict.sources.football_data_couk import ClosingOdds, FootballDataCoUk
from predict.store import Store
from predict.teams import Aliases

PL = Competition.PL
FIXTURES = Path(__file__).parent / "fixtures"
START = datetime(2025, 8, 1, tzinfo=UTC)  # day 0 of the 2025/26 season
LAST_START = datetime(2024, 8, 1, tzinfo=UTC)
NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
NO_ALIASES = Aliases()

# --- a tiny league: four teams, a double round robin in six rounds of two matches ---

ROUNDS = [
    [("Alpha", "Bravo"), ("Charlie", "Delta")],
    [("Alpha", "Charlie"), ("Bravo", "Delta")],
    [("Alpha", "Delta"), ("Bravo", "Charlie")],
    [("Bravo", "Alpha"), ("Delta", "Charlie")],
    [("Charlie", "Alpha"), ("Delta", "Bravo")],
    [("Delta", "Alpha"), ("Charlie", "Bravo")],
]


def goals(season_shift: int, round_no: int, slot: int, side: int) -> int:
    return (season_shift + 2 * round_no + 3 * slot + 5 * side + round_no * slot) % 4


def understat(start: datetime, season_shift: int, *, change: dict[tuple[int, int], tuple[int, int]] | None = None):
    """Understat records of one season: round r is on day 7r, played at 15:00."""
    records = []
    for round_no, matches in enumerate(ROUNDS):
        for slot, (home, away) in enumerate(matches):
            hg = goals(season_shift, round_no, slot, 0)
            ag = goals(season_shift, round_no, slot, 1)
            hg, ag = (change or {}).get((round_no, slot), (hg, ag))
            records.append(
                MatchRecord(
                    kickoff=start + timedelta(days=7 * round_no, hours=15),
                    competition=PL,
                    home_name=home,
                    away_name=away,
                    home_goals=hg,
                    away_goals=ag,
                    home_xg=hg * 0.8 + 0.4,
                    away_xg=ag * 0.8 + 0.4,
                )
            )
    return records


def closing_odds(start: datetime, season_shift: int, *, rounds=None, **prices) -> list[ClosingOdds]:
    out = []
    for round_no in range(len(ROUNDS)) if rounds is None else rounds:
        for slot, (home, away) in enumerate(ROUNDS[round_no]):
            out.append(
                ClosingOdds(
                    kickoff=start + timedelta(days=7 * round_no),  # the date only: midnight UTC
                    home_name=home,
                    away_name=away,
                    home_goals=goals(season_shift, round_no, slot, 0),
                    away_goals=goals(season_shift, round_no, slot, 1),
                    **{"odds_home": 2.2, "odds_draw": 3.4, "odds_away": 3.3, "odds_over25": 1.9, "odds_under25": 1.95}
                    | prices,
                )
            )
    return out


class FakeUnderstat:
    def __init__(self, seasons: dict[int, list[MatchRecord]]) -> None:
        self.seasons = seasons

    def league_matches(self, comp: Competition, season: int) -> list[MatchRecord]:
        return [r.model_copy(update={"competition": comp}) for r in self.seasons.get(season, [])]


class FakeCouk:
    def __init__(self, odds: list[ClosingOdds]) -> None:
        self.odds = odds

    def season(self, comp: Competition, season: int) -> list[ClosingOdds]:
        return self.odds


def league(*, change=None, odds=None):
    understat_seasons = {2024: understat(LAST_START, 1), 2025: understat(START, 0, change=change)}
    return FakeUnderstat(understat_seasons), FakeCouk(odds if odds is not None else closing_odds(START, 0))


def forecasts(**kwargs) -> list[Forecast]:
    source, couk = league(**{k: kwargs.pop(k) for k in ("change", "odds") if k in kwargs})
    return walk_forward(PL, 2025, source, couk, NO_ALIASES, **kwargs)


# --- Brier ---


def test_brier():
    assert brier({"home": 0.5, "draw": 0.3, "away": 0.2}, "home") == pytest.approx(0.38)


def test_brier_perfect_and_worst():
    assert brier({"over": 1.0, "under": 0.0}, "over") == 0.0
    assert brier({"over": 0.0, "under": 1.0}, "over") == pytest.approx(2.0)


def test_brier_rejects_an_outcome_that_is_not_an_answer():
    with pytest.raises(ValueError, match="outcome"):
        brier({"home": 0.5, "away": 0.5}, "draw")


# --- football-data.co.uk ---

HEADER = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,B365H,B365D,B365A,"
    "AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5,AvgCH,AvgCD,AvgCA,AvgC>2.5,AvgC<2.5"
)
CSV = "\n".join(
    [
        HEADER,
        "E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H,22,1.3,6,9.5,1.3,5.9,9.4,1.5,2.6,1.28,6.4,10.2,1.45,2.75",
        "E0,16/08/2025,12:30,Aston Villa,Newcastle,0,0,D,10,2.5,3.4,2.9,2.4,3.4,3.0,2.0,1.8,2.45,3.5,2.95,2.1,1.78",
        "E0,17/08/2025,15:00,Brentford,Fulham,,,,,2,3.5,3.6,2,3.5,3.6,1.9,1.9,,,,,",  # not played, no closing odds
        "",
    ]
)


class CsvRig:
    def __init__(self, body: str = CSV, status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.now = NOW
        self.store = Store(":memory:")

        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(status, text=body)

        http = CachedHttp(self.store, httpx.Client(transport=httpx.MockTransport(handle)), now=lambda: self.now)
        self.couk = FootballDataCoUk(http)


@pytest.fixture
def csv_rig():
    rigs: list[CsvRig] = []

    def make(*args, **kwargs) -> CsvRig:
        rigs.append(CsvRig(*args, **kwargs))
        return rigs[-1]

    yield make
    for rig in rigs:
        rig.store.close()


def test_csv_with_the_listed_header_parses(csv_rig):
    rig = csv_rig()
    rows = rig.couk.season(PL, 2025)
    assert rows == [
        ClosingOdds(
            kickoff=datetime(2025, 8, 15, tzinfo=UTC),
            home_name="Liverpool",
            away_name="Bournemouth",
            home_goals=4,
            away_goals=2,
            odds_home=1.28,
            odds_draw=6.4,
            odds_away=10.2,
            odds_over25=1.45,
            odds_under25=2.75,
        ),
        ClosingOdds(
            kickoff=datetime(2025, 8, 16, tzinfo=UTC),
            home_name="Aston Villa",
            away_name="Newcastle",
            home_goals=0,
            away_goals=0,
            odds_home=2.45,
            odds_draw=3.5,
            odds_away=2.95,
            odds_over25=2.1,
            odds_under25=1.78,
        ),
    ]


def test_csv_url_is_the_season_and_division(csv_rig):
    rig = csv_rig()
    rig.couk.season(PL, 2025)
    rig.couk.season(Competition.PD, 2024)
    assert [str(r.url) for r in rig.requests] == [
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    ]


def test_csv_is_cached_for_seven_days(csv_rig):
    rig = csv_rig()
    rig.couk.season(PL, 2025)
    rig.now = NOW + timedelta(days=7, seconds=-1)
    rig.couk.season(PL, 2025)
    assert len(rig.requests) == 1
    rig.now = NOW + timedelta(days=7)
    rig.couk.season(PL, 2025)
    assert len(rig.requests) == 2


def test_csv_with_a_byte_order_mark_and_two_digit_years(csv_rig):
    body = "﻿" + CSV.replace("15/08/2025", "15/08/25")
    assert csv_rig(body).couk.season(PL, 2025)[0].kickoff == datetime(2025, 8, 15, tzinfo=UTC)


def test_csv_missing_a_column_is_a_source_error(csv_rig):
    body = CSV.replace("AvgC>2.5", "Other")
    with pytest.raises(SourceError, match=r"AvgC>2\.5"):
        csv_rig(body).couk.season(PL, 2025)


def test_csv_that_is_not_a_csv_is_a_source_error(csv_rig):
    with pytest.raises(SourceError, match="unexpected response"):
        csv_rig("<html>Not found</html>").couk.season(PL, 2025)


def test_csv_with_a_bad_number_is_a_source_error(csv_rig):
    with pytest.raises(SourceError, match="unexpected response"):
        csv_rig(CSV.replace("1.28,6.4", "oops,6.4")).couk.season(PL, 2025)


def test_csv_http_failure_is_a_source_error(csv_rig):
    with pytest.raises(SourceError, match="404"):
        csv_rig(status=404).couk.season(PL, 2025)


def test_the_champions_league_has_no_csv(csv_rig):
    rig = csv_rig()
    with pytest.raises(SourceError, match="Champions League"):
        rig.couk.season(Competition.CL, 2025)
    assert rig.requests == []


# --- the walk-forward ---


def test_every_match_is_forecast_with_probabilities_that_sum_to_one():
    found = forecasts()
    assert len(found) == 12
    for f in found:
        assert set(f.result) == {"home", "draw", "away"}
        assert set(f.total25) == {"over", "under"}
        assert sum(f.result.values()) == pytest.approx(1.0)
        assert sum(f.total25.values()) == pytest.approx(1.0)


def test_a_result_on_a_date_never_reaches_another_match_that_date():
    """Round 3 (index 3) has two matches on the same day: neither may see the other's result."""
    before = forecasts()
    after = forecasts(change={(3, 0): (0, 6)})  # the first match of that day ends differently
    changed = {i for i, (b, a) in enumerate(zip(before, after, strict=True)) if b.result != a.result}
    same_day = {6, 7}  # forecasts are in couk order: two per round
    assert changed.isdisjoint(same_day)
    assert changed.isdisjoint(set(range(0, 6)))  # earlier rounds cannot change either
    assert changed == set(range(8, 12))  # ... but the later rounds learn from it


def test_a_result_from_before_the_date_does_change_the_forecast():
    before = forecasts()
    after = forecasts(change={(0, 0): (5, 0)})
    assert before[0].result == after[0].result  # round 0 sees only last season
    assert before[2].result != after[2].result  # round 1 sees round 0


def test_the_season_before_is_used_and_a_match_with_no_history_is_skipped():
    source, couk = league()
    source.seasons.pop(2024)
    found = walk_forward(PL, 2025, source, couk, NO_ALIASES)
    assert len(found) == 10  # round 0 has neither season before it
    assert found[0].odds.kickoff == START + timedelta(days=7)


def test_matches_with_no_closing_odds_rows_are_not_forecast():
    odds = closing_odds(START, 0, rounds=[1, 2])
    assert len(forecasts(odds=odds)) == 4


def test_couk_names_map_to_understat_names_through_the_overrides():
    def rename(name: str) -> str:
        return "The Reds" if name == "Alpha" else name

    odds = [
        o.model_copy(update={"home_name": rename(o.home_name), "away_name": rename(o.away_name)})
        for o in closing_odds(START, 0)
    ]
    source, couk = league(odds=odds)
    with pytest.raises(SourceError, match="The Reds"):
        walk_forward(PL, 2025, source, couk, NO_ALIASES)
    aliases = Aliases(source={"football_data_couk": {"the reds": "Alpha"}})
    assert len(walk_forward(PL, 2025, source, couk, aliases)) == 12
    assert [f.result for f in walk_forward(PL, 2025, source, couk, aliases)] == [f.result for f in forecasts()]


def test_the_unmapped_error_names_the_alias_table():
    odds = [o.model_copy(update={"away_name": "Zed"}) for o in closing_odds(START, 0)]
    source, couk = league(odds=odds)
    with pytest.raises(SourceError, match=r"Zed.*football_data_couk"):
        walk_forward(PL, 2025, source, couk, NO_ALIASES)


def test_the_keyword_overrides_reach_the_model():
    base = forecasts()
    assert forecasts(rho=0.0) != base
    assert forecasts(form_weight=0.0) != base
    assert forecasts(full=4) != base
    assert forecasts(rho=-0.05, form_weight=0.3, full=18) == base  # the defaults


# --- the report ---


def test_report_from_hand_worked_odds():
    # Two matches on separate days; the 1st: home favourite wins, the 2nd: away favourite but a draw.
    first = ClosingOdds(
        kickoff=START + timedelta(days=7),
        home_name="Alpha",
        away_name="Charlie",
        home_goals=3,
        away_goals=0,
        odds_home=1.5,
        odds_draw=4.0,
        odds_away=6.0,
        odds_over25=1.8,
        odds_under25=2.0,
    )
    second = first.model_copy(
        update={"kickoff": START + timedelta(days=14), "home_name": "Bravo", "away_name": "Charlie",
                "home_goals": 1, "away_goals": 1, "odds_home": 5.0, "odds_draw": 4.0, "odds_away": 1.6,
                "odds_over25": 2.2, "odds_under25": 1.65}
    )
    source, _ = league()
    report = run_backtest(PL, 2025, source, FakeCouk([first, second]), NO_ALIASES)

    book1 = devig({"home": 1.5, "draw": 4.0, "away": 6.0})
    book2 = devig({"home": 5.0, "draw": 4.0, "away": 1.6})
    ou1 = devig({"over": 1.8, "under": 2.0})
    ou2 = devig({"over": 2.2, "under": 1.65})
    assert report.n == 2
    assert report.book_brier_result == pytest.approx((brier(book1, "home") + brier(book2, "draw")) / 2)
    assert report.book_brier_total25 == pytest.approx((brier(ou1, "over") + brier(ou2, "under")) / 2)
    assert report.book_hits_result == 1
    found = walk_forward(PL, 2025, source, FakeCouk([first, second]), NO_ALIASES)
    assert report.model_brier_result == pytest.approx(
        (brier(found[0].result, "home") + brier(found[1].result, "draw")) / 2
    )
    assert report.model_brier_total25 == pytest.approx(
        (brier(found[0].total25, "over") + brier(found[1].total25, "under")) / 2
    )
    assert report.model_hits_result == int(pick(found[0].result) == "home") + int(pick(found[1].result) == "draw")


def test_a_league_with_nothing_to_evaluate_is_a_source_error():
    source, _ = league()
    with pytest.raises(SourceError, match="no matches"):
        run_backtest(PL, 2025, source, FakeCouk([]), NO_ALIASES)


# --- the printout ---


def report(model: float, book: float, n: int = 380) -> BacktestReport:
    return BacktestReport(
        n=n,
        model_brier_result=model,
        book_brier_result=book,
        model_brier_total25=0.5,
        book_brier_total25=0.49,
        model_hits_result=190,
        book_hits_result=200,
    )


def test_format_reports_passes_when_the_mean_gap_is_within_0_02():
    reports = {comp: report(0.60 + 0.02 * i, 0.60) for i, comp in enumerate(LEAGUES)}  # gaps 0, .02, .04, .06, .08
    text, passed = format_reports(2025, {c: r for c, r in reports.items() if c in (PL, Competition.PD)})
    assert passed  # mean gap 0.01
    assert text.rstrip().endswith("PASS")
    assert "Premier League" in text and "La Liga" in text and "mean" in text


def test_format_reports_fails_above_0_02():
    text, passed = format_reports(2025, {PL: report(0.64, 0.60), Competition.PD: report(0.62, 0.60)})  # mean 0.03
    assert not passed
    assert text.rstrip().endswith("FAIL")


def test_format_reports_the_boundary_is_a_pass():
    _, passed = format_reports(2025, {PL: report(0.62, 0.60)})  # 0.62 - 0.60 is 0.02 up to rounding error
    assert passed


# --- the command ---


def outcome_of(o: ClosingOdds) -> str:
    return "home" if o.home_goals > o.away_goals else "away" if o.home_goals < o.away_goals else "draw"


def priced(odds: list[ClosingOdds], *, right: float, wrong: float) -> list[ClosingOdds]:
    """Odds that price what actually happened at `right` and everything else at `wrong`."""
    out = []
    for o in odds:
        prices = {a: right if a == outcome_of(o) else wrong for a in ("home", "draw", "away")}
        out.append(
            o.model_copy(
                update={f"odds_{a}": price for a, price in prices.items()}
                | {"odds_over25": wrong, "odds_under25": wrong}
            )
        )
    return out


@pytest.fixture
def run_main(monkeypatch, tmp_path):
    def run(odds=None, *, failure: SourceError | None = None) -> int:
        source, couk = league(odds=odds)
        if failure is not None:
            couk.season = lambda comp, season: (_ for _ in ()).throw(failure)
        monkeypatch.setattr(backtest, "Understat", lambda http: source)
        monkeypatch.setattr(backtest, "FootballDataCoUk", lambda http: couk)
        return backtest.main(["--season", "2025", "--home", str(tmp_path)])

    return run


def test_main_prints_a_row_per_league_and_exits_0_on_pass(run_main, capsys):
    assert run_main(priced(closing_odds(START, 0), right=50.0, wrong=1.01)) == 0  # hopeless bookmakers
    out = capsys.readouterr().out
    assert [c.display in out for c in LEAGUES] == [True] * 5
    assert out.rstrip().endswith("PASS")


def test_main_exits_1_on_fail(run_main, capsys):
    assert run_main(priced(closing_odds(START, 0), right=1.01, wrong=50.0)) == 1  # clairvoyant bookmakers
    assert capsys.readouterr().out.rstrip().endswith("FAIL")


def test_main_exits_2_when_a_source_fails(run_main, capsys):
    assert run_main(failure=SourceError("www.football-data.co.uk", "timed out")) == 2
    captured = capsys.readouterr()
    assert "timed out" in captured.err
    assert captured.out == ""


def test_main_needs_a_season(capsys):
    with pytest.raises(SystemExit) as caught:
        backtest.main([])
    assert caught.value.code == 2


# --- alias data ---


def test_couk_aliases_point_at_real_understat_names():
    by_league = json.loads((FIXTURES / "understat_team_names.json").read_text())
    understat_names = {name for names in by_league.values() for name in names}
    targets = Aliases.load().source["football_data_couk"].values()
    assert set(targets) <= understat_names, sorted(set(targets) - understat_names)
