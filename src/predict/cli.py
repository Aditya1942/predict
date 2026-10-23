"""Command line: `predict <query...> [--share] [--fresh] [--model ID]`."""

import argparse
import os
import sys
from datetime import UTC, datetime

from pydantic import ValidationError

from predict.config import Settings
from predict.domain import Team
from predict.errors import MatchNotFound, PredictionLocked, SourceError, TeamNotFound
from predict.news.research import api_key_warning
from predict.render import copy_to_clipboard, full_view, share_text
from predict.wiring import Services, build


class Declined(Exception):
    """The user turned down every team offered."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _ask(prompt: str) -> str | None:
    """One line from stdin (prompts go to stderr so stdout stays the Prediction); None at end of input."""
    print(prompt, end="", file=sys.stderr, flush=True)
    line = sys.stdin.readline()
    return line.strip() if line else None


def choose_team(candidates: list[Team]) -> Team:
    """Ask which team the user meant. One candidate is a low-confidence guess, so it is confirmed."""
    if len(candidates) == 1:
        answer = _ask(f"Did you mean {candidates[0].name}? [y/N] ")
        if answer is not None and answer.lower() in ("y", "yes"):
            return candidates[0]
        raise Declined
    for number, team in enumerate(candidates, start=1):
        print(f"  {number}. {team.name}", file=sys.stderr)
    while True:
        answer = _ask(f"Pick a number 1-{len(candidates)} (blank to cancel): ")
        if not answer:
            raise Declined
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            return candidates[int(answer) - 1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict",
        description='Probabilities for the next Match: predict "Arsenal v Chelsea", or one team: predict arsenal.',
    )
    parser.add_argument("query", nargs="+", help='a team, or two teams: "Arsenal v Chelsea" (also vs, -)')
    parser.add_argument("--share", action="store_true", help="also print the group-chat version and mark it shared")
    parser.add_argument("--fresh", action="store_true", help="ignore a cached News Dossier")
    parser.add_argument("--model", metavar="ID", help="Claude model for this run")
    return parser


def _missing_token(exc: ValidationError) -> str:
    if any(error["loc"] == ("football_data_org_token",) for error in exc.errors()):
        return "FOOTBALL_DATA_ORG_TOKEN is not set. Put it in .env (see .env.example) or the environment."
    return f"Invalid settings: {exc}"


def main(argv: list[str] | None = None, services: Services | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["log"]:  # reserved for the Prediction Log
        print("predict log is not available yet.", file=sys.stderr)
        return 1
    try:
        options = _parser().parse_intermixed_args(args)
    except SystemExit as done:  # argparse: 0 after --help, 2 for bad usage
        return done.code if isinstance(done.code, int) else 1
    warning = api_key_warning(os.environ)
    if warning:
        print(warning, file=sys.stderr)

    owned = services is None
    try:
        if services is None:
            try:
                settings = Settings()
            except ValidationError as exc:
                print(_missing_token(exc), file=sys.stderr)
                return 1
            services = build(settings, _utc_now)
        match = services.predictor.find_match(" ".join(options.query), choose_team)
        prediction = services.predictor.predict(match, fresh=options.fresh, model=options.model, share=options.share)
        print(full_view(prediction, None))  # None: the system zone, correct for each kickoff date
        if options.share:
            text = share_text(prediction, None, None)
            print()
            print(text)
            if copy_to_clipboard(text):
                print("Copied to clipboard.", file=sys.stderr)
            else:
                print("Could not copy to the clipboard (pbcopy not available).", file=sys.stderr)
        return 0
    except Declined:
        print("No team chosen.", file=sys.stderr)
        return 1
    except (TeamNotFound, MatchNotFound, PredictionLocked, SourceError) as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        if owned and services is not None:
            services.close()
