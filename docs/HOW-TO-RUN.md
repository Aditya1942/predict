# How to run and use `predict`

A step-by-step guide, from a fresh checkout to sharing a prediction in the group chat.

## 1. One-time setup

### Install

```
cd ~/Documents/personal/predict
uv sync
```

Needs Python 3.12+ and [uv](https://docs.astral.sh/uv/). Check it works:

```
uv run predict --help
```

### Get the keys

| Key | Where | Cost | What breaks without it |
|---|---|---|---|
| `FOOTBALL_DATA_ORG_TOKEN` | Register at football-data.org; the token arrives by email | Free | Everything. It's required. |
| `ODDS_API_KEY` | Sign up at the-odds-api.com | Free, 500 credits/month | No bookmaker odds and no Hot Takes |
| `TYPESAFE_API_KEY` | console.typesafe.ai (Jev early access) | Pay per use, very cheap | No Team News adjustment; you get stats-only numbers |

### Save the keys

Create `~/.predict/.env`:

```
mkdir -p ~/.predict
cat > ~/.predict/.env <<'EOF'
FOOTBALL_DATA_ORG_TOKEN=your-token
ODDS_API_KEY=your-key
TYPESAFE_API_KEY=your-key
EOF
```

- A `.env` in the folder you run from overrides this file. Don't leave blank keys in one (e.g. a copied `.env.example`), because a blank value wins over the real one.
- Optional extras: `CLAUDE_MODEL` (default `claude-opus-5-5`), `JEV_MODEL` (default `jev-1.13.0`), `PREDICT_HOME` (default `~/.predict`).

### Log in to Claude

Run `claude` once and log in with your Pro account. Team News research uses that login, so it counts against your Claude usage.

**Do not set `ANTHROPIC_API_KEY`.** If it's set, research is billed to that API key instead of your subscription, and the CLI warns you.

## 2. Predict a match

```
uv run predict "Arsenal v Chelsea"   # next meeting of the two, either home/away order
uv run predict arsenal               # Arsenal's next Match in any covered Competition
uv run predict "barca vs real"       # nicknames and typos work; "v", "vs", "-" all split
```

- **Covered Competitions:** Champions League, Premier League, La Liga, Bundesliga, Ligue 1, Serie A. Nothing else.
- **Ambiguous names:** for things like `real` or `milan`, it lists the options and asks you to pick a number. It never guesses.
- **First run:** takes about 1 minute while it downloads team lists. After that they're cached for 24h.
- **Each run:** costs one Claude research run (when news isn't cached) and about 4 odds credits.

### Flags

| Flag | Use it when |
|---|---|
| `--share` | You want the WhatsApp version. It prints it, copies it to the clipboard, and makes this the Prediction that counts in your record. |
| `--fresh` | News has changed, e.g. lineups were just announced. It ignores the cached News Dossier (otherwise reused for 6 hours). |
| `--model claude-sonnet-5-5` | You hit your Claude usage limit, or want a cheaper run. |

### Best routine

1. Run it the day before for a first look.
2. About an hour before kickoff, once lineups are out, run `uv run predict "<match>" --fresh --share`.
3. Paste the result into the group.

Predictions lock at kickoff.

## 3. Reading the output

**Full view, top to bottom:**

- **Header:** teams, Competition, kickoff in your local time.
- **Projected Goals (base → adjusted):** the stats model's goals per team, then after Team News. News can move each team by at most ±25%.
- **Markets table:** your % next to the bookies' %, for:
  - Match Result
  - BTTS
  - Total goals 1.5 / 2.5 / 3.5
  - each team's goals 0.5 / 1.5 / 2.5
- **Hot Takes:** where you differ from the bookmakers by 10+ points. These are your "hero" calls.
- **Team News:** the top 3 News Factors per team, their effect (e.g. `Saka: hamstring → Arsenal goals −8%`) and a source link.
- **Footer:**
  - `news applied ✓` or `⚠️ news not applied (reason)`
  - the models used
  - odds credits left

**Share view (`--share`):**

```
⚽ Arsenal v Chelsea · Premier League · Sat 25 Oct, 22:00
🏆 Arsenal 54% · Draw 24% · Chelsea 22%
🎯 Both teams to score: Yes 52%
🥅 Total goals: Over 2.5 · 51%
👟 Arsenal over 1.5 goals 58% · Chelsea under 1.5 goals 66%
🔥 Hot Take: Arsenal to win (54% vs bookies 44%)
📈 Record: 21/32 results called
```

## 4. Your record: `predict log`

```
uv run predict log
```

It shows:
- Hit Rate per Market: Match Result, BTTS, Total Goals per Line, Team Goals per Line
- your Hot Take record
- the last 10 scored Matches

How scoring works:
- Results fill in automatically every time you run any `predict` command.
- **One Prediction counts per Match:** the latest `--share`d one before kickoff, otherwise your last one before kickoff.
- **Result** = the score after 90 minutes. Extra time and penalties don't count.
- A postponed, abandoned or rescheduled (moved 48h+) Match is **Void** and never scored.

## 5. Before trusting the numbers (do once)

Run the backtest. It needs no keys:

```
uv run python -m predict.backtest --season 2025
```

It replays last season and compares the model with bookmaker closing odds. Read the result:
- `PASS`: the model is within 0.02 Brier of the bookmakers. Good enough to trust.
- `FAIL`: tune `RHO`, `FORM_WEIGHT` and `SEASON_FULL_WEIGHT_MATCHES` in `src/predict/params.py`. A model-only run suggested trying `RHO=-0.08`, `FORM_WEIGHT=0.1`, `SEASON_FULL_WEIGHT_MATCHES=26`. Record the result in `docs/backtest-2025.md`.

## 6. First-run checklist

These parts were built from API documentation, not tested against the real services. Check them on your first runs:

- [ ] `uv run predict arsenal` finds a Match and prints sensible %.
- [ ] `uv run predict real` and `uv run predict milan` ask you to choose.
- [ ] A Champions League Match works. This tests ClubElo, which was flaky during the build.
- [ ] The footer says `news applied ✓`, and the Team News links look real. This tests Claude and Jev.
- [ ] Bookies' % appear for Match Result, Total Goals, BTTS and Team Goals.
- [ ] After a Match ends, `uv run predict log` shows it scored.

If one fails, the error message names the source. For example, `understat: no Understat matches found for X; add an alias` means you add a line to `src/predict/data/aliases.toml` under the matching `[source.<name>]` table.

## 7. Troubleshooting

| Message | Fix |
|---|---|
| `FOOTBALL_DATA_ORG_TOKEN is not set.` | Add it to `~/.predict/.env`, and check there's no blank `.env` in the current folder |
| Team not found, with suggestions | Use a suggested name, or add a nickname under `[user]` in `src/predict/data/aliases.toml` |
| `no upcoming match ... in the next 60 days` | It searches 60 days ahead, covered Competitions only |
| `Kickoff has passed — Predictions are locked.` | Too late for this Match |
| `⚠️ news not applied (... usage limit ...; rerun with --model claude-sonnet-5-5)` | Claude usage limit hit; rerun with `--model claude-sonnet-5-5` or wait for the reset |
| `⚠️ news not applied (no TYPESAFE_API_KEY set)` | Add the Jev key |
| `odds quota used up or key invalid (HTTP 401)` | Monthly 500 credits used up, or a wrong key. The Prediction is still made without Hot Takes. |
| `bookmakers don't list ...` | No odds for that Match yet. Try again closer to kickoff. |
| `clubelo: unavailable and no saved ratings` | ClubElo is down and nothing is cached yet. Retry later (affects Champions League only). |
| `Could not settle past Predictions: ...` | A temporary football-data.org problem. The log catches up next run. |

## 8. Where things live

- `~/.predict/predict.db`: SQLite holding the cache, News Dossiers, odds snapshots, your Predictions and Results. Delete it to start fresh; **this erases your record**.
- `src/predict/params.py`: every model constant.
- `src/predict/data/aliases.toml`: nicknames and per-source team-name fixes.
- `GLOSSARY.md`: what each term means.
- `docs/superpowers/specs/2026-10-22-predict-design.md`: the full design.

## 9. Development

```
uv run pytest -q        # 561 tests, no network
uv run ruff check && uv run ruff format --check
```
