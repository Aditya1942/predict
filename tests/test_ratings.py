from datetime import UTC, date, datetime, timedelta

import pytest

from predict.domain import Competition, LeagueBase, MatchRecord, Team
from predict.errors import SourceError
from predict.model.goals import CL_BASE, projected_goals
from predict.model.strength import combine, form_rating, league_base, season_rating
from predict.ratings import RatedMatch, RatingsService
from predict.sources.clubelo import ClubElo, ClubEloRow
from predict.teams import Aliases, TeamResolver
from tests.helpers import make_match, make_team

PL, PD, CL = Competition.PL, Competition.PD, Competition.CL

ARSENAL = make_team(1, "Arsenal FC", "England")
EVERTON = make_team(2, "Everton FC", "England")
CHELSEA = make_team(3, "Chelsea FC", "England")
BURNLEY = make_team(4, "Burnley FC", "England")
REAL = make_team(10, "Real Madrid CF", "Spain")
BARCA = make_team(11, "FC Barcelona", "Spain")
BENFICA = make_team(50, "Benfica", "Portugal")
LEAGUE_TEAMS = {PL: [ARSENAL, EVERTON, CHELSEA, BURNLEY], PD: [REAL, BARCA]}
IDS = {"Arsenal": 1, "Everton": 2, "Chelsea": 3, "Burnley": 4, "Real Madrid": 10, "Barcelona": 11}

KICKOFF = datetime(2026, 10, 24, 15, 0, tzinfo=UTC)
CL_MEAN = 1.45


def day(n: int) -> datetime:
    """A time in the current (2026) season: day 0 is 1 August."""
    return datetime(2026, 8, 1, 15, 0, tzinfo=UTC) + timedelta(days=n)


def last(n: int) -> datetime:
    """A time in the previous (2025) season."""
    return datetime(2025, 9, 1, 15, 0, tzinfo=UTC) + timedelta(days=n)


def urec(home, away, hg, ag, hxg, axg, when, comp=PL) -> MatchRecord:
    """An Understat record: names and xG, no ids."""
    return MatchRecord(
        kickoff=when,
        competition=comp,
        home_name=home,
        away_name=away,
        home_goals=hg,
        away_goals=ag,
        home_xg=hxg,
        away_xg=axg,
    )


def frec(home_id, away_id, hg, ag, when, comp=PL) -> MatchRecord:
    """A football-data.org result: ids and goals, no xG."""
    return MatchRecord(
        kickoff=when,
        competition=comp,
        home_name=f"T{home_id}",
        away_name=f"T{away_id}",
        home_goals=hg,
        away_goals=ag,
        home_id=home_id,
        away_id=away_id,
    )


CURRENT = [
    urec("Arsenal", "Everton", 3, 0, 2.4, 0.5, day(14)),
    urec("Chelsea", "Burnley", 2, 1, 1.8, 1.0, day(15)),
    urec("Everton", "Chelsea", 0, 2, 0.6, 1.9, day(28)),
    urec("Burnley", "Arsenal", 0, 2, 0.5, 2.2, day(29)),
    urec("Arsenal", "Chelsea", 2, 1, 2.0, 1.1, day(42)),
    urec("Everton", "Burnley", 1, 1, 1.0, 0.9, day(43)),
]
LAST = [
    urec("Arsenal", "Everton", 4, 0, 2.8, 0.4, last(0)),
    urec("Chelsea", "Arsenal", 2, 2, 1.7, 1.9, last(7)),
    urec("Everton", "Chelsea", 1, 3, 0.9, 2.1, last(14)),
    urec("Arsenal", "Ghost Town", 3, 1, 2.5, 0.8, last(21)),
    urec("Ghost Town", "Everton", 1, 1, 1.1, 1.0, last(28)),
    urec("Chelsea", "Ghost Town", 2, 0, 1.9, 0.6, last(35)),
    urec("Everton", "Arsenal", 0, 2, 0.7, 2.3, last(42)),
    urec("Chelsea", "Everton", 1, 0, 1.4, 0.8, last(49)),
]
PD_CURRENT = [
    urec("Real Madrid", "Barcelona", 2, 1, 1.9, 1.2, day(10), PD),
    urec("Barcelona", "Real Madrid", 3, 3, 2.2, 2.0, day(40), PD),
]
PD_LAST = [urec("Barcelona", "Real Madrid", 1, 0, 1.3, 0.7, last(10), PD)]


class FakeFdo:
    def __init__(self, results: dict[int, list[MatchRecord]] | None = None) -> None:
        self._results = results or {}
        self.result_calls: list[tuple[int, date, date]] = []

    def teams(self, comp: Competition) -> list[Team]:
        return LEAGUE_TEAMS.get(comp, [])

    def team_results(self, team_id: int, date_from: date, date_to: date) -> list[MatchRecord]:
        self.result_calls.append((team_id, date_from, date_to))
        return [r for r in self._results.get(team_id, []) if date_from <= r.kickoff.date() <= date_to]


class FakeUnderstat:
    def __init__(self, current=CURRENT, previous=LAST, pd_current=PD_CURRENT, pd_previous=PD_LAST) -> None:
        self._data = {(PL, 2026): current, (PL, 2025): previous, (PD, 2026): pd_current, (PD, 2025): pd_previous}
        self.calls: list[tuple[Competition, int]] = []

    def league_matches(self, comp: Competition, season: int) -> list[MatchRecord]:
        self.calls.append((comp, season))
        return list(self._data.get((comp, season), []))


class FakeClubElo:
    def __init__(self, rows: list[ClubEloRow] | None = None) -> None:
        self._rows = {r.club: r for r in rows or []}
        self.calls: list[date] = []

    def ratings(self, on: date) -> dict[str, ClubEloRow]:
        self.calls.append(on)
        return dict(self._rows)

    def league_elo(self, rows: dict[str, ClubEloRow], country: str) -> float:
        return ClubElo.league_elo(rows, country)


ELO_ROWS = [
    ClubEloRow(club="Arsenal", country="ENG", level=1, elo=2000.0),
    ClubEloRow(club="Man City", country="ENG", level=1, elo=2100.0),
    ClubEloRow(club="Real Madrid", country="ESP", level=1, elo=1990.0),
    ClubEloRow(club="Barcelona", country="ESP", level=1, elo=1950.0),
    ClubEloRow(club="Benfica", country="POR", level=1, elo=1800.0),
]


def service(fdo=None, understat=None, clubelo=None) -> RatingsService:
    resolver = TeamResolver([*LEAGUE_TEAMS[PL], *LEAGUE_TEAMS[PD], BENFICA], Aliases())
    return RatingsService(fdo or FakeFdo(), understat or FakeUnderstat(), clubelo or FakeClubElo(ELO_ROWS), resolver)


def with_ids(records: list[MatchRecord]) -> list[MatchRecord]:
    return [r.model_copy(update={"home_id": IDS.get(r.home_name), "away_id": IDS.get(r.away_name)}) for r in records]


def league_match(home=ARSENAL, away=EVERTON, kickoff=KICKOFF):
    return make_match(home, away, kickoff=kickoff)


def cl_match(home=ARSENAL, away=BENFICA):
    return make_match(home, away, kickoff=KICKOFF, competition=CL)


# --- league Matches ---


def test_league_match_projects_more_goals_for_the_stronger_home_team():
    rated = service().rate_match(league_match())
    assert isinstance(rated, RatedMatch)
    assert rated.projected.home > rated.projected.away > 0
    assert rated.home.attack > rated.away.attack
    assert rated.home.concede < rated.away.concede


def test_league_match_has_no_elo_and_never_asks_clubelo():
    clubelo = FakeClubElo(ELO_ROWS)
    rated = service(clubelo=clubelo).rate_match(league_match())
    assert rated.home.elo is None and rated.away.elo is None
    assert clubelo.calls == []


def test_league_match_reads_this_and_last_season_of_its_own_league():
    understat = FakeUnderstat()
    service(understat=understat).rate_match(league_match())
    assert sorted(understat.calls) == [(PL, 2025), (PL, 2026)]


def test_neutral_flag_reaches_projected_goals():
    neutral = make_match(ARSENAL, EVERTON, kickoff=KICKOFF, neutral=True)
    home_ground = service().rate_match(league_match())
    on_neutral = service().rate_match(neutral)
    assert on_neutral.projected != home_ground.projected
    assert on_neutral.home == home_ground.home


def test_form_window_is_120_days_up_to_the_day_before_kickoff():
    fdo = FakeFdo()
    service(fdo=fdo).rate_match(league_match())
    window = (date(2026, 6, 26), date(2026, 10, 23))
    assert sorted(fdo.result_calls) == [(1, *window), (2, *window)]


def test_matches_after_kickoff_do_not_change_the_output():
    baseline = service().rate_match(league_match())
    future = [
        urec("Everton", "Arsenal", 50, 0, 9.0, 0.0, KICKOFF + timedelta(days=3)),
        urec("Arsenal", "Everton", 0, 50, 0.0, 9.0, KICKOFF),  # kickoff itself is not before kickoff
    ]
    with_future = service(understat=FakeUnderstat(current=[*CURRENT, *future])).rate_match(league_match())
    assert with_future == baseline
    # the same records would matter if they had been played before the Match
    later = service(understat=FakeUnderstat(current=[*CURRENT, *future])).rate_match(
        league_match(kickoff=KICKOFF + timedelta(days=10))
    )
    assert later != baseline


def test_form_results_after_kickoff_are_ignored():
    stray = frec(1, 2, 20, 0, KICKOFF + timedelta(hours=2))

    class Sloppy(FakeFdo):  # a source that ignores the date range
        def team_results(self, team_id, date_from, date_to):
            return self._results.get(team_id, [])

    baseline = service().rate_match(league_match())
    assert service(fdo=Sloppy({1: [stray], 2: [stray]})).rate_match(league_match()) == baseline


def test_promoted_team_blends_towards_the_promoted_defaults():
    rated = service().rate_match(league_match(home=BURNLEY, away=ARSENAL))
    assert rated.home.attack < 1 < rated.home.concede
    assert rated.projected.home > 0


def test_without_current_season_matches_only_last_season_counts():
    rated = service(understat=FakeUnderstat(current=[])).rate_match(league_match())
    # unmapped names ("Ghost Town") still count toward the league average
    assert rated.base == LeagueBase(
        home_goals=sum(r.home_goals for r in LAST) / len(LAST),
        away_goals=sum(r.away_goals for r in LAST) / len(LAST),
    )
    assert rated.projected.home > 0 and rated.projected.away > 0


def test_no_understat_data_at_all_raises():
    with pytest.raises(SourceError, match="understat"):
        service(understat=FakeUnderstat(current=[], previous=[])).rate_match(league_match())


def test_ratings_follow_the_strength_model_with_xg_filled_into_form():
    # Football-data.org results carry goals only; Understat holds the xG of the same Match on the same date.
    arsenal_form = frec(1, 3, 2, 1, day(42).replace(hour=14, minute=30))  # = Arsenal 2-1 Chelsea in CURRENT
    everton_form = frec(4, 2, 0, 1, day(60))  # no Understat counterpart: stays goals only
    fdo = FakeFdo({1: [arsenal_form], 2: [everton_form]})
    rated = service(fdo=fdo).rate_match(league_match())

    current, previous = with_ids(CURRENT), with_ids(LAST)
    base = league_base(current, previous)
    comp_avg = {PL: (base.home_goals + base.away_goals) / 2, CL: CL_MEAN}
    filled = arsenal_form.model_copy(update={"home_xg": 2.0, "away_xg": 1.1})
    home = combine(season_rating(1, current, previous), form_rating(1, [filled], comp_avg))
    away = combine(season_rating(2, current, previous), form_rating(2, [everton_form], comp_avg))
    assert rated == RatedMatch(home=home, away=away, base=base, projected=projected_goals(home, away, base))


# --- Champions League Matches ---


def test_cl_top_five_team_gets_its_league_mean_elo_and_the_other_team_its_own():
    understat, clubelo = FakeUnderstat(), FakeClubElo(ELO_ROWS)
    rated = service(understat=understat, clubelo=clubelo).rate_match(cl_match())
    assert rated.home.elo == pytest.approx(2050.0)  # mean of the two level 1 English clubs
    assert rated.away.elo == pytest.approx(1800.0)  # Benfica's own rating
    assert rated.base == CL_BASE
    assert sorted(understat.calls) == [(PL, 2025), (PL, 2026)]  # Arsenal's league only
    assert clubelo.calls == [KICKOFF.date()]
    assert rated.projected.home > rated.projected.away > 0  # Arsenal: stronger Elo and stronger league ratings


def test_cl_team_outside_the_top_five_starts_from_neutral_ratings():
    rated = service().rate_match(cl_match())
    assert (rated.away.attack, rated.away.concede) == (1.0, 1.0)


def test_cl_results_feed_the_form_of_both_teams():
    cl_result = frec(1, 50, 0, 3, day(55), CL)
    league_result = frec(1, 3, 2, 1, day(60))
    plain = service().rate_match(cl_match())
    with_form = service(fdo=FakeFdo({1: [cl_result, league_result], 50: [cl_result]})).rate_match(cl_match())
    assert with_form.home != plain.home and with_form.away != plain.away
    # Benfica's only record is a CL one: attack = 3 goals vs the CL mean, geometric weight 0.3
    assert with_form.away.attack == pytest.approx((3 / CL_MEAN) ** 0.3)
    assert with_form.home.elo == plain.home.elo and with_form.away.elo == plain.away.elo


def test_cl_between_two_top_five_teams_uses_each_ones_own_league():
    understat, clubelo = FakeUnderstat(), FakeClubElo(ELO_ROWS)
    rated = service(understat=understat, clubelo=clubelo).rate_match(cl_match(home=ARSENAL, away=REAL))
    assert rated.home.elo == pytest.approx(2050.0)
    assert rated.away.elo == pytest.approx(1970.0)  # mean of Real Madrid and Barcelona
    assert sorted(understat.calls) == sorted([(PL, 2025), (PL, 2026), (PD, 2025), (PD, 2026)])
    assert rated.base == CL_BASE


def test_cl_club_missing_from_clubelo_raises_naming_the_club():
    others = [r for r in ELO_ROWS if r.club != "Benfica"]
    with pytest.raises(SourceError, match="Benfica"):
        service(clubelo=FakeClubElo(others)).rate_match(cl_match())


def test_form_ignores_results_from_competitions_without_a_league_average():
    stray = frec(50, 99, 5, 0, day(60), Competition.SA)  # a league the team's rating knows nothing about
    plain = service().rate_match(cl_match())
    assert service(fdo=FakeFdo({50: [stray]})).rate_match(cl_match()) == plain
