import io
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

from predict import cli
from predict.cli import choose_team, main
from predict.domain import Prediction, ProjectedGoals, Result
from predict.errors import AmbiguousTeam, SourceError
from predict.model.markets import market_probabilities
from predict.model.scoreline import scoreline_matrix
from predict.pipeline import Predictor
from predict.sources.football_data_org import MatchStatus
from predict.store import Store
from predict.teams import Aliases, TeamResolver
from predict.wiring import Services
from tests.helpers import (
    FakeDossiers,
    FakeFdo,
    StubJev,
    StubRater,
    make_dossier,
    make_factor,
    make_match,
    make_team,
    system_zone,
)

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)

ARSENAL = make_team(1, "Arsenal")
CHELSEA = make_team(2, "Chelsea")
REAL_MADRID = make_team(10, "Real Madrid CF")
REAL_BETIS = make_team(11, "Real Betis Balompié")
REAL_SOCIEDAD = make_team(12, "Real Sociedad de Fútbol")
TEAMS = [ARSENAL, CHELSEA, REAL_MADRID, REAL_BETIS, REAL_SOCIEDAD]


@pytest.fixture(autouse=True)
def clipboard(monkeypatch):
    """Stands in for pbcopy, so a test never touches the real clipboard. Holds the texts copied."""
    copied: list[str] = []
    monkeypatch.setattr(cli, "copy_to_clipboard", lambda text: copied.append(text) or True)
    return copied


@pytest.fixture
def services_for():
    """Builds Services over fakes; every Store is closed after the test."""
    stores: list[Store] = []

    def make(matches, *, now=NOW, clock=None, rater=None, dossiers=None, jev=None, statuses=None) -> Services:
        store = Store(":memory:")
        stores.append(store)
        fdo = FakeFdo(matches, statuses)
        resolver = TeamResolver(TEAMS, Aliases.load())
        ratings = rater or StubRater()
        return Services(
            store=store,
            fdo=fdo,
            resolver=resolver,
            ratings=ratings,
            now=lambda: now,  # settling reads this, so a `clock` that counts its calls is left to the Predictor
            predictor=Predictor(
                fdo,
                ratings,
                resolver,
                store,
                None,
                dossiers or FakeDossiers(make_dossier(created_at=now, model="claude-opus-5-5")),
                jev,
                "claude-opus-5-5",
                "jev-1.13.0",
                now=clock or (lambda: now),
            ),
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


def test_share_prints_the_share_text_copies_it_and_marks_the_prediction_shared(capsys, services_for, clipboard):
    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3), id=5)])
    assert main(["Arsenal", "v", "Chelsea", "--share"], services) == 0
    captured = capsys.readouterr()
    shared = services.store.all_predictions()[0].shared_at
    assert shared == NOW
    assert captured.out.index("Projected Goals") < captured.out.index("⚽ Chelsea v Arsenal")
    assert len(clipboard) == 1
    assert clipboard[0] in captured.out
    assert clipboard[0].startswith("⚽ Chelsea v Arsenal · Premier League · ")
    assert "copied" in captured.err.lower()


def test_without_share_nothing_is_printed_copied_or_marked(capsys, services_for, clipboard):
    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3))])
    assert main(["Arsenal v Chelsea"], services) == 0
    assert "⚽" not in capsys.readouterr().out
    assert clipboard == []
    assert services.store.all_predictions()[0].shared_at is None


def test_share_still_prints_when_there_is_no_clipboard(monkeypatch, capsys, services_for):
    monkeypatch.setattr(cli, "copy_to_clipboard", lambda text: False)
    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3))])
    assert main(["Arsenal v Chelsea", "--share"], services) == 0
    captured = capsys.readouterr()
    assert "⚽ Chelsea v Arsenal" in captured.out
    assert "pbcopy" in captured.err
    assert services.store.all_predictions()[0].shared_at == NOW


def test_fresh_and_model_reach_the_dossier_service(services_for):
    match = make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3))
    dossiers = FakeDossiers(make_dossier(created_at=NOW, model="claude-sonnet-5-5"))
    services = services_for([match], dossiers=dossiers, jev=StubJev())
    assert main(["Arsenal v Chelsea", "--fresh", "--model", "claude-sonnet-5-5"], services) == 0
    assert main(["Arsenal v Chelsea"], services) == 0
    assert [(fresh, model) for _, fresh, model in dossiers.calls] == [
        (True, "claude-sonnet-5-5"),
        (False, "claude-opus-5-5"),
    ]


def test_news_is_shown_in_the_full_view(capsys, services_for):
    saka = make_factor("absence", "Saka", sources=("https://example.com/saka",))
    dossier = make_dossier(away=(saka,), created_at=NOW, model="claude-opus-5-5")
    major = [0.0, 0.0, 0.0, 1.0]
    jev = StubJev({"away0_attack": major, "away0_defence": major})
    services = services_for(
        [make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3))], dossiers=FakeDossiers(dossier), jev=jev
    )
    assert main(["Arsenal v Chelsea"], services) == 0
    out = capsys.readouterr().out
    assert "Team News" in out
    assert "Saka: Out for three weeks with a hamstring injury. → Arsenal goals -15%" in out
    assert "news applied ✓" in out


def test_news_not_applied_without_a_jev_key_is_said_in_the_footer(capsys, services_for):
    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3))])
    assert main(["Arsenal v Chelsea"], services) == 0
    assert "⚠️ news not applied (no TYPESAFE_API_KEY set)" in capsys.readouterr().out


def test_an_anthropic_api_key_in_the_environment_is_warned_about(monkeypatch, capsys, services_for):
    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3))])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert main(["Arsenal v Chelsea"], services) == 0
    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY is set" in captured.err
    assert "ANTHROPIC_API_KEY" not in captured.out


def test_no_warning_without_an_anthropic_api_key(monkeypatch, capsys, services_for):
    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3))])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["Arsenal v Chelsea"], services) == 0
    assert "ANTHROPIC_API_KEY" not in capsys.readouterr().err


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


def played(services, match_id: int, kickoff=NOW - timedelta(days=1)) -> None:
    """A saved Prediction of a home win for a Match that has kicked off."""
    projected = ProjectedGoals(home=1.8, away=0.9)
    services.store.save_prediction(
        Prediction(
            match=make_match(CHELSEA, ARSENAL, kickoff=kickoff, id=match_id),
            created_at=kickoff - timedelta(days=1),
            projected_base=projected,
            projected=projected,
            probabilities=market_probabilities(scoreline_matrix(projected.home, projected.away)),
        )
    )


def final(match_id: int, home: int, away: int) -> MatchStatus:
    match = make_match(CHELSEA, ARSENAL, kickoff=NOW - timedelta(days=1), id=match_id)
    return MatchStatus(match=match, status="FINISHED", result=Result(home_goals=home, away_goals=away))


def test_log_prints_the_prediction_log(capsys, services_for):
    services = services_for([])
    played(services, 1)
    played(services, 2)
    services.store.save_result(1, "finished", Result(home_goals=2, away_goals=0), NOW)
    services.store.save_result(2, "finished", Result(home_goals=0, away_goals=1), NOW)
    assert main(["log"], services) == 0
    out = capsys.readouterr().out
    assert "Match Result" in out
    assert "1/2" in out
    assert services.store.all_predictions()[0].shared_at is None
    assert len(services.store.all_predictions()) == 2  # log makes no Prediction


def test_log_settles_finished_matches_first(capsys, services_for):
    services = services_for([], statuses=[final(1, 2, 0), final(2, 0, 1)])
    played(services, 1)
    played(services, 2)
    assert main(["log"], services) == 0
    assert "1/2" in capsys.readouterr().out
    assert services.store.results() == {
        1: ("finished", Result(home_goals=2, away_goals=0)),
        2: ("finished", Result(home_goals=0, away_goals=1)),
    }


def test_predicting_settles_past_matches_first(services_for):
    services = services_for(
        [make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3), id=9)], statuses=[final(1, 2, 0)]
    )
    played(services, 1)
    assert main(["Arsenal v Chelsea"], services) == 0
    assert services.store.results() == {1: ("finished", Result(home_goals=2, away_goals=0))}


def test_a_failing_settle_is_a_one_line_warning_and_the_command_goes_on(capsys, services_for):
    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3), id=9)])
    played(services, 1)
    services.fdo.status_error = SourceError("api.football-data.org", "boom")
    assert main(["Arsenal v Chelsea"], services) == 0
    captured = capsys.readouterr()
    assert "Chelsea v Arsenal" in captured.out
    warning = [line for line in captured.err.splitlines() if "boom" in line]
    assert len(warning) == 1
    assert "settle" in warning[0].lower()

    assert main(["log"], services) == 0
    assert "Match Result" in capsys.readouterr().out


def test_share_text_carries_the_match_result_record(capsys, services_for):
    services = services_for(
        [make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3), id=9)],
        statuses=[final(1, 2, 0), final(2, 0, 1)],
    )
    played(services, 1)
    played(services, 2)
    assert main(["Arsenal v Chelsea", "--share"], services) == 0
    assert "📈 Record: 1/2 results called" in capsys.readouterr().out


def test_share_text_has_no_record_before_anything_is_scored(capsys, services_for):
    services = services_for([make_match(CHELSEA, ARSENAL, kickoff=NOW + timedelta(days=3))])
    assert main(["Arsenal v Chelsea", "--share"], services) == 0
    assert "📈" not in capsys.readouterr().out


def test_log_takes_no_arguments(capsys, services_for):
    assert main(["log", "arsenal"], services_for([])) == 2
    assert "usage" in capsys.readouterr().err.lower()


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
