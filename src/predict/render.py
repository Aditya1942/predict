"""Text output: labels for Markets and answers, and the full Prediction view."""

import io
import os
import subprocess
from datetime import datetime, tzinfo

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from predict.domain import HotTake, Match, Prediction
from predict.log import LogSummary, outcome
from predict.model.markets import pick
from predict.news.adjust import top_reasons
from predict.params import HOT_TAKE_THRESHOLD

_WIDTH = 100
_SIDES = ("home", "away")
_SHARE_TOTAL_LINE = 2.5
_SHARE_TEAM_LINE = 1.5
_SHARE_HOT_TAKES = 3
_OPPONENT_SHOWN_FROM = 0.01  # a concede multiplier closer to 1 than this is not worth a mention
_LOG_LABEL_WIDTH = 17


def format_kickoff(kickoff: datetime, tz: tzinfo | None) -> str:
    """ "Sun 25 Oct, 22:00 IST": the kickoff in local time.

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


def _change(multiplier: float) -> str:
    """A goals multiplier as a signed whole percentage: 0.92 is -8%."""
    return f"{round((multiplier - 1) * 100):+d}%"


def _print_team_news(console: Console, p: Prediction) -> None:
    match = p.match
    console.print(Text("Team News", style="bold"))
    sides = [
        (team, opponent, top_reasons(p.impacts, side))
        for side, team, opponent in (("home", match.home, match.away), ("away", match.away, match.home))
    ]
    if not any(reasons for _, _, reasons in sides):
        console.print(Text("  nothing that changes the numbers"))
        return
    for team, opponent, reasons in sides:
        if not reasons:
            continue
        console.print(Text(team.short_name, style="bold"))
        for impact in reasons:
            factor = impact.factor
            subject = factor.player or factor.kind
            line = f"  {subject}: {factor.summary} → {team.short_name} goals {_change(impact.attack_mult)}"
            if round(abs(impact.concede_mult - 1), 6) >= _OPPONENT_SHOWN_FROM:
                line += f" → {opponent.short_name} goals {_change(impact.concede_mult)}"
            console.print(Text(line), soft_wrap=True)
            if factor.sources:
                console.print(Text(f"    {factor.sources[0]}", style="dim"), soft_wrap=True)


def _print_footer(console: Console, p: Prediction) -> None:
    if p.news_applied:
        status = "news applied ✓"
    else:
        status = f"⚠️ news not applied ({p.news_note})" if p.news_note else "⚠️ news not applied"
    models = [f"{name} {model}" for name, model in (("Claude", p.claude_model), ("Jev", p.jev_model)) if model]
    console.print(Text(" · ".join([status, *models]), style="dim"))
    if p.bookmaker is None and p.odds_note:
        console.print(Text(f"Bookmaker odds unavailable: {p.odds_note}", style="dim"))
    if p.bookmaker is not None and p.credits_remaining is not None:
        console.print(Text(f"Odds credits remaining: {p.credits_remaining}", style="dim"))


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
    if p.news_applied:
        _print_team_news(console, p)
    _print_footer(console, p)
    return "\n".join(line.rstrip() for line in console.export_text().splitlines())


def _percent(probability: float) -> str:
    return f"{probability * 100:.0f}%"


def share_text(p: Prediction, record: tuple[int, int] | None, tz: tzinfo | None) -> str:
    """The short version for a group chat. `record` is (results called right, results scored) from the Log."""
    match = p.match
    home, away = match.home.short_name, match.away.short_name
    result = p.probabilities["result"]
    btts = p.probabilities["btts"]
    total = p.probabilities[f"total:{_SHARE_TOTAL_LINE}"]
    lines = [
        f"⚽ {home} v {away} · {match.competition.display} · {match.kickoff.astimezone(tz):%a %d %b, %H:%M}",
        f"🏆 {home} {_percent(result['home'])} · Draw {_percent(result['draw'])} · {away} {_percent(result['away'])}",
        f"🎯 Both teams to score: {pick(btts).capitalize()} {_percent(btts[pick(btts)])}",
        f"🥅 Total goals: {pick(total).capitalize()} {_SHARE_TOTAL_LINE} · {_percent(total[pick(total)])}",
    ]
    team_lines = []
    for side in _SIDES:
        market_id = f"team:{side}:{_SHARE_TEAM_LINE}"
        answers = p.probabilities[market_id]
        team_lines.append(f"{answer_label(market_id, pick(answers), match)} {_percent(answers[pick(answers)])}")
    lines.append("👟 " + " · ".join(team_lines))
    for take in p.hot_takes[:_SHARE_HOT_TAKES]:
        lines.append(
            f"🔥 Hot Take: {answer_label(take.market, take.answer, match)}"
            f" ({_percent(take.model_p)} vs bookies {_percent(take.book_p)})"
        )
    if record is not None and record[1] > 0:
        lines.append(f"📈 Record: {record[0]}/{record[1]} results called")
    return "\n".join(lines)


def copy_to_clipboard(text: str) -> bool:
    """Put the text on the macOS clipboard. False when pbcopy is missing or fails."""
    # pbcopy reads its input in the locale's encoding, which is not UTF-8 when LANG is unset.
    env = {**os.environ, "LANG": "en_US.UTF-8"}
    try:
        subprocess.run(["pbcopy"], input=text.encode(), env=env, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _rate(hits: int, scored: int) -> str:
    return "—" if scored == 0 else f"{hits}/{scored} ({round(hits / scored * 100)}%)"


def log_view(summary: LogSummary) -> str:
    """The Prediction Log: Hit Rate per Market group, the Hot Take record and the latest Counted Predictions."""
    lines = ["Prediction Log", ""]
    rows = [*summary.hit_rates.items(), ("Hot Takes", summary.hot_takes)]
    lines += [f"{label:<{_LOG_LABEL_WIDTH}}{_rate(hits, scored)}" for label, (hits, scored) in rows]
    lines.append("")
    if not summary.recent:
        lines.append("No scored Predictions yet.")
        return "\n".join(lines)
    lines.append("Recent")
    for prediction, result in summary.recent:
        match = prediction.match
        called = pick(prediction.probabilities["result"])
        mark = "✓" if called == outcome("result", result) else "✗"
        lines.append(
            f"  {match.kickoff.astimezone():%d %b}  {match.home.short_name} {result.home_goals}-{result.away_goals}"
            f" {match.away.short_name}  ·  {answer_label('result', called, match)} {mark}"
        )
    return "\n".join(lines)
