"""Text output: labels for Markets and answers, and the full Prediction view."""

import io
from datetime import datetime, tzinfo

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from predict.domain import HotTake, Match, Prediction
from predict.params import HOT_TAKE_THRESHOLD

_WIDTH = 100
_SIDES = ("home", "away")


def format_kickoff(kickoff: datetime, tz: tzinfo | None) -> str:
    """"Sun 25 Oct, 22:00 IST": the kickoff in local time.

    `tz=None` means the system zone, applying its daylight-saving rules to this kickoff's own date.
    """
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


def _bookies_cell(listed: dict[str, float] | None, answers: list[str]) -> str:
    """The bookmakers' percentages in the same answer order as ours; a dash where they list nothing."""
    if listed is None:
        return "—"
    return " · ".join(f"{listed[answer] * 100:.0f}%" if answer in listed else "—" for answer in answers)


def _print_hot_takes(console: Console, takes: list[HotTake], match: Match) -> None:
    if not takes:
        points = round(HOT_TAKE_THRESHOLD * 100)
        console.print(Text(f"Hot Takes: none (nothing is {points}+ points from the bookmakers)"))
        return
    console.print(Text("Hot Takes", style="bold"))
    for take in takes:
        console.print(
            Text(
                f"  {answer_label(take.market, take.answer, match)}: us {take.model_p * 100:.0f}%"
                f" vs bookies {take.book_p * 100:.0f}% ({round(take.diff * 100):+d} pts)"
            )
        )


def full_view(p: Prediction, tz: tzinfo | None) -> str:
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
    if p.bookmaker is not None:
        table.add_column("Bookies")
    for market_id, answers in p.probabilities.items():
        cells = (f"{answer_label(market_id, answer, match)} {prob * 100:.0f}%" for answer, prob in answers.items())
        row = [Text(market_label(market_id, match)), Text("  ·  ".join(cells))]
        if p.bookmaker is not None:
            row.append(Text(_bookies_cell(p.bookmaker.get(market_id), list(answers))))
        table.add_row(*row)
    console.print(table)
    if p.bookmaker is not None:
        _print_hot_takes(console, p.hot_takes, match)
    console.print(Text("Stats model only; no news applied.", style="dim"))
    if p.bookmaker is None and p.odds_note:
        console.print(Text(f"Bookmaker odds unavailable: {p.odds_note}", style="dim"))
    if p.bookmaker is not None and p.credits_remaining is not None:
        console.print(Text(f"Odds credits remaining: {p.credits_remaining}", style="dim"))
    return "\n".join(line.rstrip() for line in console.export_text().splitlines())
