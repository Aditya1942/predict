# Backtest 2025/26 against bookmaker closing odds

Status: **the comparison against the bookmakers has not been run yet.** `www.football-data.co.uk` timed out
from the machine that built this (22 Oct 2026; Understat and other hosts were reachable). The backtest, its tests
and the football-data.co.uk client are done and tested against the documented CSV shape. The model parameters in
`src/predict/params.py` are unchanged (`RHO = -0.05`, `FORM_WEIGHT = 0.3`, `SEASON_FULL_WEIGHT_MATCHES = 18`).

## Run it

```
uv run python -m predict.backtest --season 2025
```

It needs no keys. It prints a row per league, the mean gap and `PASS` (exit 0) when the mean of
`model - book` Match Result Brier is at most 0.02, otherwise `FAIL` (exit 1); a failing source exits 2.

Before trusting the first live run:

- Check that the CSV has the columns the client reads: `Date` (dd/mm/yyyy), `HomeTeam`, `AwayTeam`, `FTHG`,
  `FTAG`, `AvgCH`, `AvgCD`, `AvgCA`, `AvgC>2.5`, `AvgC<2.5`. A missing column is a `SourceError` naming it.
- football-data.co.uk team names are mapped to Understat names by fuzzy matching plus
  `[source.football_data_couk]` in `src/predict/data/aliases.toml`. Those aliases were written from memory of the
  2025/26 names (their Understat targets are checked by a test against the live Understat names). A name that
  maps to nothing stops the run with the name and the table to add it to.

## Result

Not run. Fill in the table printed by the command above:

| League | Matches | Result: model | book | gap | Total 2.5: model | book | gap | Result hits: model | book |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |

Mean Match Result Brier gap: _pending_ (limit 0.02). Verdict: _pending_.

## What is known without the bookmakers

Understat alone gives the model's side of the comparison. At the default parameters, on every played 2025/26
Match Understat has (so up to a few Matches more than the odds files will have), the mean Match Result Brier is:

| League | Matches | Model Brier |
| --- | --- | --- |
| Premier League | 380 | 0.6223 |
| La Liga | 380 | 0.5896 |
| Bundesliga | 306 | 0.5806 |
| Serie A | 380 | 0.6038 |
| Ligue 1 | 306 | 0.5937 |
| mean | | 0.5980 |

For scale, always predicting each league's own home/draw/away frequencies (an in-sample baseline that cannot be
used for real) scores 0.6534, 0.6299, 0.6477, 0.6579 and 0.6411 on the same Matches.

## Parameter grid (model side only, for use if the run says FAIL)

The bookmakers' Brier score does not depend on the model parameters, so ranking parameter sets by the model's own
mean Match Result Brier gives the same order as ranking them by the gap. The brief's grid (`rho` in
{-0.13, -0.08, -0.05, 0}, `form_weight` in {0.1, 0.2, 0.3, 0.4}, `full` in {10, 18, 26}; 48 sets, all five
leagues) gives:

| Rank | rho | form_weight | full | Mean model Brier |
| --- | --- | --- | --- | --- |
| 1 | -0.08 | 0.1 | 26 | 0.5927 |
| 2 | -0.05 | 0.1 | 26 | 0.5927 |
| 3 | -0.13 | 0.1 | 26 | 0.5928 |
| 31 of 48 | -0.05 | 0.3 | 18 (the current defaults) | 0.5980 |

The worst set scores 0.6041. `rho` barely matters for the Match Result Brier (it moves the draw mostly). The best
sets sit on the edge of the grid (`form_weight` 0.1 is its smallest value and `full` 26 its largest), so if the
live run fails, extend the grid past those edges before setting new values. These values were tuned and scored on
the same season, so treat any gain as optimistic.

If the live run prints `FAIL`: set the best values in `src/predict/params.py` (`RHO`, `FORM_WEIGHT`,
`SEASON_FULL_WEIGHT_MATCHES`), rerun, and record the table and the chosen values here.
