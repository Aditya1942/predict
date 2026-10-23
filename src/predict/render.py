"""Text output: labels for Markets and answers, and the full Prediction view."""

import io
from datetime import datetime, tzinfo

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from predict.domain import Match, Prediction

_WIDTH = 100
_SIDES = ("home", "away")


def format_kickoff(kickoff: datetime, tz: tzinfo) -> str:
    """"Sun 25 Oct, 22:00 IST": the kickoff in the viewer's own time."""
    return kickoff.astimezone(tz).strftime("%a %d %b, %H:%M %Z").strip()


def _team_name(side: str, match: Match, market_id: str) -> str:
    if side not in _SIDES:
        raise ValueError(f"unknown market {market_id!r}")
    return (match.home if side == "home" else match.away).short_name


def market_label(market_id: str, match: Match) -> str:
    kind, *rest = market_id.split(":")
    if kind == "result" and not rest:
        return "Match Result"
    if kind == "btts" and not rest:
        return "Both teams to score"
    if kind == "total" and len(rest) == 1:
        return f"Total goals {rest[0]}"
    if kind == "team" and len(rest) == 2:
        return f"{_team_name(rest[0], match, market_id)} goals {rest[1]}"
    raise ValueError(f"unknown market {market_id!r}")


def answer_label(market_id: str, answer: str, match: Match) -> str:
    kind, *rest = market_id.split(":")
    if kind == "result" and not rest:
        labels = {"home": f"{match.home.short_name} to win", "draw": "Draw", "away": f"{match.away.short_name} to win"}
    elif kind == "btts" and not rest:
        labels = {"yes": "BTTS: Yes", "no": "BTTS: No"}
    elif kind == "total" and len(rest) == 1:
        labels = {"over": f"Over {rest[0]} goals", "under": f"Under {rest[0]} goals"}
    elif kind == "team" and len(rest) == 2:
        team = _team_name(rest[0], match, market_id)
        labels = {"over": f"{team} over {rest[1]} goals", "under": f"{team} under {rest[1]} goals"}
    else:
        raise ValueError(f"unknown market {market_id!r}")
    if answer not in labels:
        raise ValueError(f"unknown answer {answer!r} for market {market_id!r}")
    return labels[answer]


def full_view(p: Prediction, tz: tzinfo) -> str:
    match = p.match
    console = Console(file=io.StringIO(), record=True, width=_WIDTH, force_terminal=False, color_system=None)

    venue = " · neutral venue" if match.neutral else ""
    console.print(
        Text(
            f"{match.home.short_name} v {match.away.short_name} · {match.competition.display}"
            f" · {format_kickoff(match.kickoff, tz)}{venue}",
            style="bold",
        )
    )
    base, adjusted = p.projected_base, p.projected
    console.print(
        Text(
            "Projected Goals (base → adjusted): "
            f"{match.home.short_name} {base.home:.2f} → {adjusted.home:.2f}"
            f" · {match.away.short_name} {base.away:.2f} → {adjusted.away:.2f}"
        )
    )
    console.print()

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("Market", no_wrap=True)
    table.add_column("Probabilities")
    for market_id, answers in p.probabilities.items():
        cells = (f"{answer_label(market_id, answer, match)} {prob * 100:.0f}%" for answer, prob in answers.items())
        table.add_row(Text(market_label(market_id, match)), Text("  ·  ".join(cells)))
    console.print(table)
    console.print(Text("Stats model only; no news applied.", style="dim"))
    return "\n".join(line.rstrip() for line in console.export_text().splitlines())
