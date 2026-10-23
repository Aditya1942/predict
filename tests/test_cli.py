import io
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

from predict import cli
from predict.cli import choose_team, main
from predict.errors import AmbiguousTeam, SourceError
from predict.pipeline import Predictor
from predict.store import Store
from predict.teams import Aliases, TeamResolver
from predict.wiring import Services
from tests.helpers import FakeFdo, StubRater, make_match, make_team, system_zone

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)

ARSENAL = make_team(1, "Arsenal")
CHELSEA = make_team(2, "Chelsea")
REAL_MADRID = make_team(10, "Real Madrid CF")
REAL_BETIS = make_team(11, "Real Betis Balompié")
REAL_SOCIEDAD = make_team(12, "Real Sociedad de Fútbol")
TEAMS = [ARSENAL, CHELSEA, REAL_MADRID, REAL_BETIS, REAL_SOCIEDAD]


@pytest.fixture
def services_for():
    """Builds Services over fakes; every Store is closed after the test."""
    stores: list[Store] = []

    def make(matches, *, now=NOW, clock=None, rater=None) -> Services:
        store = Store(":memory:")
        stores.append(store)
        fdo = FakeFdo(matches)
        resolver = TeamResolver(TEAMS, Aliases.load())
        ratings = rater or StubRater()
        return Services(
            store=store,
            fdo=fdo,
            resolver=resolver,
            ratings=ratings,
            predictor=Predictor(fdo, ratings, resolver, store, None, now=clock or (lambda: now)),
        )

    yield make
    for store in stores:
        store.close()


def stdin(monkeypatch, text: str) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(text))


def test_predict_prints_the_full_view_and_saves(capsys, services_for):
    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3), id=5)])
    assert main(["Arsenal", "v", "Chelsea"], services) == 0
    out = capsys.readouterr().out
    assert "Chelsea v Arsenal" in out
    assert "Projected Goals" in out
    assert "Match Result" in out
    assert [p.match.id for p in services.store.all_predictions()] == [5]


def test_query_may_be_one_quoted_argument_and_flags_may_come_anywhere(capsys, services_for):
    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3))])
    assert main(["--fresh", "Arsenal v Chelsea", "--model", "claude-sonnet-5-5", "--share"], services) == 0
    assert "Chelsea v Arsenal" in capsys.readouterr().out


def test_ambiguous_team_prompts_a_numbered_choice(monkeypatch, capsys, services_for):
    services = services_for([make_match(REAL_MADRID, ARSENAL, kickoff=NOW + timedelta(days=3))])
    stdin(monkeypatch, "2\n")
    assert main(["real"], services) == 0
    captured = capsys.readouterr()
    assert "1. Real Betis Balompié" in captured.err
    assert "2. Real Madrid CF" in captured.err
    assert "Real Madrid CF v Arsenal" in captured.out


def test_ambiguous_team_declined_exits_1(monkeypatch, capsys, services_for):
    services = services_for([make_match(REAL_MADRID, ARSENAL, kickoff=NOW + timedelta(days=3))])
    stdin(monkeypatch, "\n")
    assert main(["real"], services) == 1
    assert "No team chosen" in capsys.readouterr().err
    assert services.store.all_predictions() == []


def test_choose_team_reprompts_on_a_bad_number(monkeypatch, capsys):
    stdin(monkeypatch, "9\nx\n1\n")
    assert choose_team([CHELSEA, ARSENAL]) == CHELSEA
    assert capsys.readouterr().err.count("Pick a number") == 3


def test_choose_team_with_one_candidate_asks_to_confirm(monkeypatch, capsys):
    stdin(monkeypatch, "y\n")
    assert choose_team([CHELSEA]) == CHELSEA
    assert "Did you mean Chelsea? [y/N]" in capsys.readouterr().err


@pytest.mark.parametrize("answer", ["n\n", "\n", ""])
def test_choose_team_with_one_candidate_declined(monkeypatch, answer):
    stdin(monkeypatch, answer)
    with pytest.raises(cli.Declined):
        choose_team([CHELSEA])


def test_single_candidate_ambiguity_goes_through_the_confirmation(monkeypatch, capsys, services_for):
    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3))])

    def low_confidence(text):
        raise AmbiguousTeam(text, [CHELSEA])

    services.resolver.resolve_query = low_confidence
    stdin(monkeypatch, "y\n")
    assert main(["chelsa"], services) == 0
    assert "Did you mean Chelsea?" in capsys.readouterr().err


def test_kickoff_passing_during_the_run_exits_1_with_locked(capsys, services_for):
    """The Match is upcoming when found and has started by the time it is predicted."""
    match = make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(minutes=1))
    times = iter([NOW, NOW + timedelta(minutes=2)])
    services = services_for([match], clock=lambda: next(times))
    assert main(["Arsenal v Chelsea"], services) == 1
    err = capsys.readouterr().err
    assert "Kickoff has passed — Predictions are locked." in err
    assert services.store.all_predictions() == []


def test_a_match_that_has_started_is_not_offered(capsys, services_for):
    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=NOW - timedelta(minutes=5))])
    assert main(["Arsenal v Chelsea"], services) == 1
    assert "no upcoming match between Arsenal and Chelsea" in capsys.readouterr().err
    assert services.store.all_predictions() == []


@pytest.mark.parametrize(
    ("kickoff", "shown"),
    [
        (datetime(2026, 10, 24, 15, 0, tzinfo=UTC), "Sat 24 Oct, 16:00 BST"),
        (datetime(2026, 11, 7, 15, 0, tzinfo=UTC), "Sat 07 Nov, 15:00 GMT"),
    ],
)
def test_kickoff_is_shown_in_the_system_zone_with_its_own_daylight_saving(capsys, services_for, kickoff, shown):
    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=kickoff)])
    with system_zone("Europe/London"):
        assert main(["Arsenal v Chelsea"], services) == 0
    assert shown in capsys.readouterr().out


def test_unknown_team_exits_1_with_suggestions(capsys, services_for):
    services = services_for([])
    assert main(["Zzzzzz"], services) == 1
    err = capsys.readouterr().err
    assert "no team matches 'Zzzzzz'" in err
    assert "try:" in err


def test_no_match_in_window_exits_1(capsys, services_for):
    services = services_for([])
    assert main(["Arsenal"], services) == 1
    assert "no upcoming match for Arsenal" in capsys.readouterr().err


def test_source_error_exits_1(capsys, services_for):
    class Broken(StubRater):
        def rate_match(self, match):
            raise SourceError("understat", "boom")

    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3))], rater=Broken())
    assert main(["Arsenal"], services) == 1
    assert "understat: boom" in capsys.readouterr().err


def test_log_is_reserved(capsys, services_for):
    services = services_for([])
    assert main(["log"], services) == 1
    assert services.store.all_predictions() == []
    assert capsys.readouterr().err


def test_no_query_is_a_usage_error(capsys, services_for):
    assert main([], services_for([])) == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_help_works_without_a_token(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOOTBALL_DATA_ORG_TOKEN", raising=False)
    assert main(["--help"]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_missing_token_is_a_clear_error(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOOTBALL_DATA_ORG_TOKEN", raising=False)
    assert main(["Arsenal"]) == 1
    assert "FOOTBALL_DATA_ORG_TOKEN" in capsys.readouterr().err


def test_importing_the_cli_needs_no_token(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "FOOTBALL_DATA_ORG_TOKEN"}
    done = subprocess.run(
        [sys.executable, "-c", "import predict.cli"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
