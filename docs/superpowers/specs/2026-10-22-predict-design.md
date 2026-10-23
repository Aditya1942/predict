# Predict — Design Spec

Vocabulary: see `GLOSSARY.md`. Core decision: `docs/adr/0001-stats-model-sets-probabilities-ai-only-adjusts.md`.

## Purpose

A personal Python CLI that gives a football fan the probabilities for an upcoming Match so they can impress their friends' group chat. No money is involved, but each Prediction is compared with bookmakers (Hot Takes) and tracked (Prediction Log) so the bragging is backed by a record.

## Scope (v1)

- **Competitions:** Champions League (CL), Premier League (PL), La Liga (PD), Bundesliga (BL1), Ligue 1 (FL1), Serie A (SA). Any Match in these can be predicted. Nothing else (no Europa League, domestic cups, internationals).
- **Single Match only**, before kickoff only. No weekly batch, no scheduling, no live/in-play.
- **Markets and Lines:**
  - Match Result: home / draw / away
  - BTTS: yes / no
  - Total Goals: over/under 1.5, 2.5, 3.5
  - Team Goals: over/under 0.5, 1.5, 2.5 for each team
- **Out of v1:** corners, cards, referees, player markets, halftime, correct score, double chance, clean sheet.

## Commands

- `predict "Arsenal v Chelsea"`: the next Match between the two teams, in either home/away order. Also accepts `vs`, `-`.
- `predict arsenal`: that team's next Match in any covered Competition.
- `--share`: also print the group-chat version, copy it to the clipboard (`pbcopy`), and mark this Prediction as shared.
- `--fresh`: ignore a cached News Dossier.
- `--model <id>`: Claude model for this run (default from config, `claude-opus-5-5`).
- `predict log`: Hit Rate per Market, Hot Take record, recent Counted Predictions with Results.

Team names: fuzzy matching plus nicknames (`barca`, `inter`, `psg`, typos like `chelsia`, `leverkusin`). If a name is ambiguous (`real` → Madrid/Betis/Sociedad; `milan` → AC Milan/Inter), ask the user to choose; never guess.

## Output

**Full view:** Match header (local time), Projected Goals before and after news, every Market's probabilities next to Bookmaker Probability, Hot Takes, top 3 News Factors per team with their effect ("Saka out → Arsenal goals −8%") and source links, and a footer showing whether news was applied, odds credits remaining, and the models used.

**Share view:** short WhatsApp-ready text with emojis: Match Result, BTTS, Total Goals 2.5, each team's 1.5 Team Goals, up to 3 Hot Takes, and the Match Result Hit Rate.

## Probability engine (stats model)

- **Performance per Match** = 0.7 × xG + 0.3 × goals (for and against). Goals only where no xG exists (Champions League).
- **Season Strength:** attack and concede ratios relative to the league average, from league Matches. Blended with last season by weight w = min(1, matches played / 18): about 50/50 in late October, current season only from around January. A promoted team (no last-season data) uses last-season attack 0.85 and concede 1.15.
- **Form:** the last 6 Matches in covered Competitions, weights 6,5,4,3,2,1 (most recent first), relative to that Competition's per-team goal average. Combined with Season Strength geometrically: rating = season^0.7 × form^0.3.
- **Projected Goals, same league:** home = league home-goals average × home attack × away concede; away = league away-goals average × away attack × home concede.
- **Projected Goals, Champions League:** base 1.60 home / 1.30 away. Teams from different leagues are adjusted by league strength from ClubElo: factor = exp(0.0025 × (EloA − EloB) / 2) multiplies team A and divides team B. Elo is the league's average Elo for a top-5 team, and the team's own Elo for any other club (whose attack/concede are then 1.0).
- **Neutral venue** (Champions League final): both teams use the mean of the home and away bases.
- **Scoreline probabilities:** Dixon-Coles, ρ = −0.05, 0–10 goals each side, normalised. Every Market is computed from this one matrix.
- All constants are tunable; a backtest against bookmaker closing odds (2025/26 top-5 leagues) checks the model before it's trusted.

## Team News

- **Research (Claude Opus 5.5 via Claude Agent SDK, Pro subscription login):** one run per Match covering both teams; at most 5 web searches per team; returns a News Dossier per team with at most 6 News Factors ranked by importance. Each factor has a kind (absence, doubtful, return, rotation, hot_streak, cold_streak), player, summary, sources, and dates already resolved to "misses this Match: yes / likely / no". Premier League injury data from the FPL API is given to Claude as hints.
- **Reuse:** a News Dossier younger than 6 hours is reused unless `--fresh`.
- **Scoring (Jev):** for each factor, Jev scores the impact magnitude on the team's attack and on its defending (4 situation-described levels each). For doubtful players Jev also answers "Will <player> start?" and the effect is scaled by the chance he doesn't.
- **Adjustment:** factor multipliers multiply together per team. Each team's Projected Goals change is capped at ±25%.
- Jev never produces Market probabilities. Exact Claude and Jev model versions are saved with each Prediction.

## Bookmakers and Hot Takes

- Odds from The Odds API (free tier, 500 credits/month): h2h, totals, btts, team_totals for the one Match (about 4 credits), cached 60 minutes.
- Bookmaker Probability: per bookmaker, 1/odds normalised to remove margin; median across bookmakers; renormalised.
- Hot Take: an answer where the Prediction's probability and the Bookmaker Probability differ by 10 points or more. For two-way Markets only the answer the model favours more is reported.
- Markets bookmakers don't list simply have no Hot Takes. Remaining credits are shown.

## Prediction Log

- Every run saves a Prediction. Predictions can't be made after kickoff.
- **Counted Prediction:** the latest shared Prediction before kickoff, otherwise the last Prediction before kickoff.
- **Result** = score after 90 minutes plus stoppage time (extra time and penalties ignored).
- **Void:** the Match was postponed, cancelled, abandoned or awarded, or its kickoff moved more than 48 hours from the Prediction's kickoff. Never scored.
- Pick = answer with the highest probability; Hit = Pick matched the Result.
- Hit Rate is shown for Match Result, BTTS, Total Goals at each Line, and Team Goals at each Line (both teams combined).
- Hot Take record: a Hot Take is a hit if the answer happened when the model rated it higher than bookmakers, or didn't happen when the model rated it lower.
- Results fill in automatically on every command run.

## Data sources

| Data | Source |
|---|---|
| Matches, teams, Results (all 6 Competitions) | football-data.org (free, 10 requests/min, token) |
| xG and goals per Match, current and last season (top 5) | Understat `getLeagueData` JSON |
| Cross-league strength | ClubElo daily ratings CSV (cache last good copy) |
| Premier League injuries | FPL API `bootstrap-static` |
| Other Team News | Claude web search |
| Odds | The Odds API |
| Backtest history and closing odds | football-data.co.uk CSVs |

## Failure behaviour

- If Claude or Jev fails, the Prediction is still produced from the stats model and marked "⚠️ news not applied (reason)".
- If odds are unavailable (quota, network, Match not listed), the Prediction is produced without Hot Takes, with the reason shown.
- If ClubElo is down, use the last saved ratings. If none were ever saved, a cross-league Champions League Match fails with a clear message.
- A team name from a source that can't be matched causes an error naming the source and the name, so an alias can be added.
- If `ANTHROPIC_API_KEY` is set, warn that it will bill the API instead of the subscription.

## Storage and config

- SQLite at `~/.predict/predict.db`: HTTP cache, ClubElo snapshots, News Dossiers, odds snapshots, Predictions, Results.
- `.env`: `FOOTBALL_DATA_ORG_TOKEN`, `ODDS_API_KEY`, `TYPESAFE_API_KEY`, optional `CLAUDE_MODEL`, `JEV_MODEL`, `PREDICT_HOME`.
- Times stored in UTC, shown in local time.
