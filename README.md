# predict

Football match probabilities from the terminal. Name a match (or one team) and it finds the next
fixture in the top five leagues and the Champions League, rates both teams from current form and
season stats, researches team news (injuries, suspensions, lineups) with Claude, and prints
probabilities for Match Result, BTTS, total goals and each team's goals. It sets them next to the bookmakers'
odds and flags the biggest disagreements as Hot Takes. Every Prediction is saved to a log and scored
once the match is played. Predictions lock at kickoff.

## Setup

1. `uv sync`
2. Log in to Claude Code once by running `claude`. Research runs on that subscription login.
3. Put your keys in `~/.predict/.env` (a `.env` in the current directory overrides it; see `.env.example`):

```
FOOTBALL_DATA_ORG_TOKEN=...   # required, free signup at football-data.org
ODDS_API_KEY=...              # optional, free at the-odds-api.com; no key, no bookmaker odds
TYPESAFE_API_KEY=...          # optional; no key, no team news adjustment
```

Do not set `ANTHROPIC_API_KEY`. Claude would then bill that API key instead of your subscription
(the CLI warns if it is set).

## Usage

```
uv run predict "Arsenal v Chelsea"    # the next meeting of the two teams
uv run predict arsenal --share        # one team: its next match; also prints a group-chat version and copies it
uv run predict arsenal --fresh        # ignore the cached team news and research again
uv run predict arsenal --model ID     # Claude model for this run (e.g. claude-sonnet-5-5)
uv run predict log                    # the Prediction Log: hit rates, Hot Takes, recent results
```

The first run downloads team lists (about 1 minute); later runs reuse the cache in `~/.predict`.

## Before trusting the numbers

Run the backtest, which compares the stats model with bookmaker closing odds:

```
uv run python -m predict.backtest --season 2025
```

See [docs/backtest-2025.md](docs/backtest-2025.md) for what it checks and its status.

## More

- [How to run and use it](docs/HOW-TO-RUN.md): full step-by-step guide
- [GLOSSARY.md](GLOSSARY.md): the terms used here (Match, Market, Hot Take, News Factor, ...)
- [Design spec](docs/superpowers/specs/2026-10-22-predict-design.md)
