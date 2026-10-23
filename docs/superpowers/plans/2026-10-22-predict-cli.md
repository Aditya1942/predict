# Predict CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A personal Python CLI that predicts probabilities for every v1 Market of one upcoming football Match, adjusts them for Team News, compares them with bookmakers, and keeps a Prediction Log.

**Architecture:** A pure stats core (Season Strength + Form → Projected Goals → Dixon-Coles scoreline matrix → Markets) gets its data from thin source clients behind a SQLite-backed HTTP cache. Team News is researched by Claude (Agent SDK), scored by Jev, and turned into capped multipliers on Projected Goals; AI never outputs Market probabilities (ADR 0001). The CLI is a thin layer over `Predictor` and the log functions.

**Tech Stack:** Python 3.12, uv, httpx, pydantic v2, pydantic-settings, typer, rich, rapidfuzz, numpy, claude-agent-sdk, typesafe-sdk, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-22-predict-design.md`. Vocabulary: `GLOSSARY.md`. Decision: `docs/adr/0001-stats-model-sets-probabilities-ai-only-adjusts.md`.

## Global Constraints

- Python ≥ 3.12, uv-managed, package `predict` under `src/predict`, console script `predict = "predict.cli:main"`.
- Runtime dependencies are only the ones in Tech Stack. No pandas.
- Unit tests never use the network. HTTP goes through `httpx.MockTransport`; Claude and Jev go through fakes behind Protocols. Steps marked **(live)** are manual checks against real services and need `.env` filled in.
- Code names follow `GLOSSARY.md`: Match (not fixture/event), Market, Prediction, Projected Goals (not expected goals), News Factor, News Dossier, Hot Take, Counted Prediction, Result, Void, Pick, Hit, Hit Rate.
- Competition codes are football-data.org codes: `PL, PD, BL1, SA, FL1, CL`. Team identity is the football-data.org team id everywhere.
- All model constants live in `src/predict/params.py` (values in Task 1). Functions take them as keyword defaults so the backtest can override them.
- Datetimes are timezone-aware UTC internally and are converted to local time only in `render.py`.
- Probabilities are addressed only via the Market ids and answers from Task 1.
- Secrets come from `.env`: `FOOTBALL_DATA_ORG_TOKEN`, `ODDS_API_KEY`, `TYPESAFE_API_KEY`. Claude uses the subscription login, never an API key.

## Review Focus

1. **A source's team name doesn't match** (e.g. "Inter" vs "FC Internazionale Milano"). Expected: a clear error naming the source and the name, or odds marked unavailable. It must never be silently mapped to the wrong club. Tests: Task 6 (threshold, duplicate mapping), Task 9 (club missing from ClubElo), Task 12 (no matching bookmaker event).
2. **A Match is postponed and rescheduled under the same id.** Expected: the Prediction is Void, not scored against the new date. Test: Task 16.
3. **`predict` is run after kickoff or during the Match.** Expected: refused with "Predictions are locked". Test: Task 10.
4. **A team with little or no data:** 0 matches played this season, newly promoted, or a Champions League club from outside the top 5 leagues. Expected: ratings are still produced. Tests: Task 9.
5. **An external failure:** odds quota used up, Claude usage limit hit, Jev down, ClubElo down. Expected: a stats-only or no-Hot-Takes Prediction with a note, not a crash. Tests: Tasks 8, 12, 15.

---

### Task 1: Scaffold, domain types, scoreline matrix, Markets

**Files:**
- Create: `pyproject.toml`, `.gitignore` (`.venv/`, `.env`, `__pycache__/`), `.env.example`, `src/predict/__init__.py`, `src/predict/params.py`, `src/predict/domain.py`, `src/predict/model/__init__.py`, `src/predict/model/scoreline.py`, `src/predict/model/markets.py`, `tests/helpers.py`
- Test: `tests/model/test_scoreline.py`, `tests/model/test_markets.py`, `tests/test_domain.py`

**Interfaces:**
- Produces:
  - `params.py` constants (exact values):
    `XG_WEIGHT=0.7, SEASON_FULL_WEIGHT_MATCHES=18, PROMOTED_ATTACK=0.85, PROMOTED_CONCEDE=1.15, FORM_MATCHES=6, FORM_WEIGHTS=(6,5,4,3,2,1), FORM_WEIGHT=0.3, CL_HOME_GOALS=1.60, CL_AWAY_GOALS=1.30, ELO_KAPPA=0.0025, RHO=-0.05, MAX_GOALS=10, TOTAL_LINES=(1.5,2.5,3.5), TEAM_LINES=(0.5,1.5,2.5), NEWS_CAP=(0.75,1.25), MAX_FACTORS_PER_TEAM=6, MAX_SEARCHES_PER_TEAM=5, DOSSIER_TTL=timedelta(hours=6), ODDS_TTL=timedelta(minutes=60), HOT_TAKE_THRESHOLD=0.10, VOID_KICKOFF_SHIFT=timedelta(hours=48), SETTLE_AFTER=timedelta(minutes=150), SEARCH_WINDOW_DAYS=60`
  - `domain.py`:
    - `Competition(StrEnum)` with members `PL, PD, BL1, SA, FL1, CL` and properties `display` ("Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Champions League"), `is_league` (False only for CL), `understat_league` ("EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1", None), `odds_sport` ("soccer_epl", "soccer_spain_la_liga", "soccer_germany_bundesliga", "soccer_italy_serie_a", "soccer_france_ligue_one", "soccer_uefa_champs_league"), `clubelo_country` ("ENG","ESP","GER","ITA","FRA", None), `couk_division` ("E0","SP1","D1","I1","F1", None)
    - `LEAGUES: tuple[Competition, ...]` = the five leagues
    - frozen pydantic models: `Team(id: int, name: str, short_name: str, tla: str | None = None, country: str | None = None)`, `Match(id: int, competition: Competition, home: Team, away: Team, kickoff: datetime, stage: str | None = None, neutral: bool = False)`, `Result(home_goals: int, away_goals: int)`, `MatchRecord(kickoff: datetime, competition: Competition, home_name: str, away_name: str, home_goals: int, away_goals: int, home_xg: float | None = None, away_xg: float | None = None, home_id: int | None = None, away_id: int | None = None)`, `ProjectedGoals(home: float, away: float)`, `TeamRating(attack: float, concede: float, elo: float | None = None)`, `LeagueBase(home_goals: float, away_goals: float)`
    - `Side = Literal["home", "away"]`, `Probabilities = dict[str, dict[str, float]]`
    - `MARKET_IDS: tuple[str, ...]` in display order: `"result", "btts", "total:1.5", "total:2.5", "total:3.5", "team:home:0.5", "team:home:1.5", "team:home:2.5", "team:away:0.5", "team:away:1.5", "team:away:2.5"`. Answers: result → `home/draw/away`, btts → `yes/no`, total/team → `over/under`.
    - `season_of(dt: datetime) -> int`: the season's start year (`dt.year` if month ≥ 7, otherwise `dt.year - 1`)
  - `model/scoreline.py`: `scoreline_matrix(home: float, away: float, *, rho: float = RHO, max_goals: int = MAX_GOALS) -> np.ndarray` with shape `(max_goals+1, max_goals+1)`, `[i, j] = P(home i, away j)`, normalised to sum 1
  - `model/markets.py`: `market_probabilities(matrix: np.ndarray) -> Probabilities` (all `MARKET_IDS`) and `pick(answers: dict[str, float]) -> str` (highest probability; ties go to the first in dict order)
  - `tests/helpers.py`: `make_team(id: int, name: str, country: str | None = None) -> Team` (short_name = name), `make_match(home: Team, away: Team, *, kickoff: datetime, competition: Competition = Competition.PL, id: int = 1, neutral: bool = False) -> Match`, `rec(home_id: int, away_id: int, hg: int, ag: int, hxg: float | None = None, axg: float | None = None, *, day: int, competition: Competition = Competition.PL) -> MatchRecord` (kickoff = 2026-08-01 UTC + `day` days, names `f"T{id}"`, ids set)

- [ ] **Step 1: Scaffold.** Run `git init`. Write `pyproject.toml` (hatchling, src layout, the deps above, dev group pytest + ruff, script entry). Write `.env.example` with the three keys. Run `uv sync`.
- [ ] **Step 2: Write failing tests**

```python
# tests/model/test_scoreline.py
def test_independent_poisson_when_rho_zero():
    m = scoreline_matrix(1.0, 1.0, rho=0.0)
    assert m.shape == (11, 11)
    assert m.sum() == pytest.approx(1.0)
    assert m[0, 0] == pytest.approx(math.exp(-2), abs=1e-6)


def test_negative_rho_moves_probability_into_low_draws():
    dc, ind = scoreline_matrix(1.4, 1.1, rho=-0.05), scoreline_matrix(1.4, 1.1, rho=0.0)
    assert dc[0, 0] > ind[0, 0] and dc[1, 1] > ind[1, 1]
    assert dc[1, 0] < ind[1, 0] and dc[0, 1] < ind[0, 1]


# tests/model/test_markets.py
def test_markets_from_known_projected_goals():
    p = market_probabilities(scoreline_matrix(1.5, 1.0, rho=0.0))
    assert set(p) == set(MARKET_IDS)
    assert all(sum(a.values()) == pytest.approx(1.0) for a in p.values())
    assert p["btts"]["yes"] == pytest.approx(0.4911, abs=1e-4)
    assert p["total:2.5"]["over"] == pytest.approx(0.4562, abs=1e-4)
    assert p["team:home:0.5"]["over"] == pytest.approx(0.7769, abs=1e-4)
    assert p["team:home:1.5"]["over"] == pytest.approx(0.4422, abs=1e-4)


def test_equal_teams_give_equal_win_chances():
    p = market_probabilities(scoreline_matrix(1.3, 1.3))
    assert p["result"]["home"] == pytest.approx(p["result"]["away"])


def test_pick():
    assert pick({"home": 0.4, "draw": 0.3, "away": 0.3}) == "home"
    assert pick({"over": 0.45, "under": 0.55}) == "under"


# tests/test_domain.py
def test_season_of():
    assert season_of(datetime(2026, 10, 22, tzinfo=UTC)) == 2026
    assert season_of(datetime(2027, 3, 1, tzinfo=UTC)) == 2026
```

- [ ] **Step 3: Run** `uv run pytest -q`. Expected: FAIL (import errors).
- [ ] **Step 4: Implement** `params.py`, `domain.py`, `tests/helpers.py`, `scoreline_matrix` and `market_probabilities`. Dixon-Coles τ for low scores: τ(0,0)=1−λμρ, τ(0,1)=1+λρ, τ(1,0)=1+μρ, τ(1,1)=1−ρ, otherwise 1. Multiply the independent Poisson matrix by τ, then normalise.
- [ ] **Step 5: Run** `uv run pytest -q && uv run ruff check`. Expected: all pass.
- [ ] **Step 6: Commit** `git add -A && git commit -m "feat: domain types, Dixon-Coles scoreline matrix and Markets"`

---

### Task 2: Projected Goals

**Files:**
- Create: `src/predict/model/goals.py`
- Test: `tests/model/test_goals.py`

**Interfaces:**
- Consumes: `TeamRating`, `LeagueBase`, `ProjectedGoals`, params `ELO_KAPPA`, `CL_HOME_GOALS`, `CL_AWAY_GOALS`
- Produces: `CL_BASE: LeagueBase`; `projected_goals(home: TeamRating, away: TeamRating, base: LeagueBase, *, neutral: bool = False, kappa: float = ELO_KAPPA) -> ProjectedGoals`
  - home = base.home_goals × home.attack × away.concede; away = base.away_goals × away.attack × home.concede
  - `neutral`: both sides use (home_goals + away_goals) / 2
  - Elo factor `exp(kappa × (home.elo − away.elo) / 2)` multiplies home and divides away, applied only when both `elo` are set

- [ ] **Step 1: Write failing tests**

```python
def test_league_match():
    g = projected_goals(
        TeamRating(attack=1.2, concede=0.8),
        TeamRating(attack=0.9, concede=1.1),
        LeagueBase(home_goals=1.5, away_goals=1.2),
    )
    assert (g.home, g.away) == (pytest.approx(1.98), pytest.approx(0.864))


def test_neutral_venue_uses_mean_base():
    g = projected_goals(
        TeamRating(attack=1, concede=1),
        TeamRating(attack=1, concede=1),
        LeagueBase(home_goals=1.5, away_goals=1.2),
        neutral=True,
    )
    assert g.home == pytest.approx(1.35) and g.away == pytest.approx(1.35)


def test_cross_league_elo_factor():
    g = projected_goals(TeamRating(attack=1, concede=1, elo=1750), TeamRating(attack=1, concede=1, elo=1600), CL_BASE)
    assert g.home == pytest.approx(1.9300, abs=1e-4) and g.away == pytest.approx(1.0777, abs=1e-4)


def test_no_elo_factor_when_either_elo_missing():
    g = projected_goals(TeamRating(attack=1, concede=1, elo=1750), TeamRating(attack=1, concede=1), CL_BASE)
    assert g.home == pytest.approx(1.60)
```

- [ ] **Step 2: Run** `uv run pytest tests/model/test_goals.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `projected_goals` and `CL_BASE`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: Projected Goals with home base, neutral venue and Elo league factor`

---

### Task 3: Season Strength and Form

**Files:**
- Create: `src/predict/model/strength.py`
- Test: `tests/model/test_strength.py`

**Interfaces:**
- Consumes: `MatchRecord`, `TeamRating`, `LeagueBase`, `Competition`, params
- Produces:
  - `performance(goals: int, xg: float | None, *, xg_weight: float = XG_WEIGHT) -> float`: 0.7·xG + 0.3·goals, or goals when xG is None
  - `season_weight(matches_played: int, *, full: int = SEASON_FULL_WEIGHT_MATCHES) -> float`: min(1, n / full)
  - `league_base(current: list[MatchRecord], last: list[MatchRecord], *, full: int = ...) -> LeagueBase`: mean actual home and away goals per season, blended by `season_weight` of the current season's average matches per team. Raises `ValueError` if both lists are empty.
  - `season_rating(team_id: int, current: list[MatchRecord], last: list[MatchRecord], *, full: int = ...) -> TeamRating`. Per season: attack = the team's mean performance-for ÷ the league's mean performance per team per match, and concede the same for performance-against. Blended with w = `season_weight(team's current matches)`. If the team is absent from `last`, it uses `PROMOTED_ATTACK` / `PROMOTED_CONCEDE`.
  - `form_rating(team_id: int, recent: list[MatchRecord], comp_avg: Mapping[Competition, float]) -> TeamRating | None`: the team's latest ≤ 6 records by kickoff, weights `FORM_WEIGHTS` truncated to the count and renormalised, each performance divided by `comp_avg[record.competition]`. Returns None if there are no records.
  - `combine(season: TeamRating, form: TeamRating | None, *, weight: float = FORM_WEIGHT) -> TeamRating`: season^(1−w) × form^w for attack and concede; keeps `season.elo`; returns `season` if form is None

- [ ] **Step 1: Write failing tests** (use `rec` from `tests/helpers.py`)

```python
CUR = [rec(1, 2, 2, 1, 2.0, 1.0, day=1), rec(2, 1, 1, 2, 1.0, 2.0, day=8)]


def test_performance():
    assert performance(2, 1.0) == pytest.approx(1.3) and performance(2, None) == 2.0


def test_season_weight():
    assert (season_weight(0), season_weight(9), season_weight(25)) == (0.0, 0.5, 1.0)


def test_promoted_team_blends_toward_promoted_defaults():
    r = season_rating(1, CUR, last=[])
    assert r.attack == pytest.approx(0.9037, abs=1e-4) and r.concede == pytest.approx(1.0963, abs=1e-4)


def test_same_numbers_both_seasons_ignore_weight():
    assert season_rating(1, CUR, last=CUR).attack == pytest.approx(4 / 3)


def test_form_weights_recent_matches_more():
    recent = [rec(1, 2, 2, 1, 2.0, 1.0, day=1), rec(2, 1, 1, 1, 1.0, 1.0, day=8)]  # newest is the weaker game
    f = form_rating(1, recent, {Competition.PL: 1.5})
    assert f.attack == pytest.approx(0.9697, abs=1e-4)


def test_form_uses_goals_when_no_xg():
    f = form_rating(1, [rec(1, 9, 3, 0, day=1, competition=Competition.CL)], {Competition.CL: 1.5})
    assert f.attack == pytest.approx(2.0)


def test_combine_geometric():
    c = combine(TeamRating(attack=1.1, concede=1.0), TeamRating(attack=1.4, concede=1.0))
    assert c.attack == pytest.approx(1.18253, abs=1e-5)
    assert combine(TeamRating(attack=1.1, concede=1.0), None).attack == 1.1


def test_league_base_blends_home_away_goals():
    b = league_base(CUR, last=CUR)
    assert (b.home_goals, b.away_goals) == (pytest.approx(1.5), pytest.approx(1.5))
```

- [ ] **Step 2: Run** `uv run pytest tests/model/test_strength.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the six functions.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: Season Strength and Form ratings`

---

### Task 4: Store and cached HTTP

**Files:**
- Create: `src/predict/store.py`, `src/predict/http.py`, `src/predict/errors.py`
- Test: `tests/test_store.py`, `tests/test_http.py`

**Interfaces:**
- Produces:
  - `errors.py`: `SourceError(source: str, detail: str)` (str = `f"{source}: {detail}"`)
  - `Store(path: Path)`: creates tables on init and accepts `":memory:"`. Methods: `cache_get(key: str, max_age: timedelta, now: datetime) -> str | None`, `cache_put(key: str, body: str, now: datetime) -> None`, `cache_latest(key: str) -> tuple[str, datetime] | None` (ignores age). Later tasks add tables and methods.
  - `CachedHttp(store: Store, client: httpx.Client, *, now: Callable[[], datetime], sleep: Callable[[float], None] = time.sleep)`:
    - `get_text(url: str, *, ttl: timedelta, params: dict | None = None, headers: dict | None = None, min_interval: float = 0.0) -> str`
    - `get_json(...)`, same arguments
    - The cache key is the URL plus sorted params, excluding any param named `apiKey`. Header values are never part of the key.
    - `min_interval` is enforced per host between real requests only.
    - A non-2xx response raises `SourceError(host, f"HTTP {status}")`.

- [ ] **Step 1: Write failing tests**: cache hit within ttl makes 0 transport calls; an expired entry refetches; HTTP 502 raises `SourceError` with "502" in the message; `apiKey` isn't in any stored key; two uncached calls to one host with `min_interval=6.5` call `sleep` with ≈6.5 when the fake clock hasn't advanced; `cache_latest` returns the most recent body regardless of age.
- [ ] **Step 2: Run** `uv run pytest tests/test_store.py tests/test_http.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** with stdlib `sqlite3`. Table `http_cache(key TEXT PRIMARY KEY, body TEXT, fetched_at TEXT)`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: SQLite store and cached HTTP client`

---

### Task 5: football-data.org client

**Files:**
- Create: `src/predict/sources/__init__.py`, `src/predict/sources/football_data_org.py`, `tests/fixtures/fdo_*.json`
- Test: `tests/sources/test_football_data_org.py`

**Interfaces:**
- Consumes: `CachedHttp`, domain types
- Produces: `FootballDataOrg(http: CachedHttp, token: str)`, base `https://api.football-data.org/v4`, header `X-Auth-Token`, `min_interval=6.5`
  - `teams(comp: Competition) -> list[Team]`: `/competitions/{code}/teams`, ttl 24h, `country` = `area.name`
  - `team_upcoming(team_id: int, date_from: date, date_to: date) -> list[Match]`: `/teams/{id}/matches?status=SCHEDULED,TIMED&competitions=PL,PD,BL1,SA,FL1,CL&dateFrom&dateTo`, ttl 1h, sorted by kickoff
  - `team_results(team_id: int, date_from: date, date_to: date) -> list[MatchRecord]`: same endpoint with `status=FINISHED`, goals from `regular_time_score`, ids and names set, xG None
  - `matches_by_ids(ids: list[int]) -> list[MatchStatus]`: `/matches?ids=...`, ttl 0
  - `MatchStatus` (frozen pydantic): `match: Match, status: str, result: Result | None` (result only when FINISHED)
  - `regular_time_score(score: dict) -> tuple[int, int]`: `duration == "REGULAR"` → `fullTime`, otherwise `regularTime`
  - `Match.neutral` is True when competition is CL and `stage == "FINAL"`

- [ ] **Step 1 (live): Record real responses.** With the token in `.env`, save trimmed responses as fixtures:
  - `fdo_teams_pl.json`: 3 teams
  - `fdo_team_matches.json`: 1 scheduled + 1 finished Match
  - `fdo_extra_time.json`: a 2025/26 Champions League knockout Match that went to extra time, found via `/competitions/CL/matches?season=2025`

  Confirm the `score.duration` / `score.regularTime` field names. If they differ, update `regular_time_score`'s rule in this plan and the code.
- [ ] **Step 2: Write failing tests**:
  - `teams` parses id/name/short_name/tla/country
  - `team_upcoming` parses kickoff as aware UTC
  - extra-time fixture → `team_results` gives the 90-minute score
  - REGULAR → fullTime; PENALTY_SHOOTOUT with regularTime 0–0 → (0, 0)
  - CL `stage="FINAL"` → `neutral` True
  - request carries `X-Auth-Token`
  - `matches_by_ids` maps `POSTPONED` with `result` None
- [ ] **Step 3: Run** `uv run pytest tests/sources/test_football_data_org.py -q`. Expected: FAIL.
- [ ] **Step 4: Implement** the client.
- [ ] **Step 5: Run** the same command. Expected: PASS.
- [ ] **Step 6: Commit** `feat: football-data.org client with 90-minute Results`

---

### Task 6: Team name resolution and query parsing

**Files:**
- Create: `src/predict/teams.py`, `src/predict/data/aliases.toml`
- Modify: `src/predict/errors.py`
- Test: `tests/test_teams.py`

**Interfaces:**
- Produces:
  - errors: `TeamNotFound(text: str, suggestions: list[str])`, `AmbiguousTeam(text: str, candidates: list[Team])`
  - `parse_query(text: str) -> tuple[str, str | None]`: splits on `\s+(v|vs\.?|-)\s+` (case-insensitive); a single team gives `(team, None)`
  - `Aliases.load(path: Path | None = None) -> Aliases`, reading `aliases.toml`. Sections:
    - `[user]`: alias → list of canonical football-data.org names (one entry = direct, several = ambiguous)
    - `[source.<name>]`: source name → canonical name. `[source.football_data_couk]` maps to Understat names, for the backtest only.
  - `TeamResolver(teams: list[Team], aliases: Aliases)`:
    - `resolve_query(text: str) -> Team`. Order:
      1. user alias (normalised exact)
      2. fuzzy `rapidfuzz` WRatio over normalised name, short_name, tla: top ≥ 85 and lead ≥ 5 → that team
      3. top ≥ 70 → `AmbiguousTeam` with the candidates within 10 points of the top
      4. otherwise `TeamNotFound` with the top 3 suggestions
    - `map_source_names(source: str, names: Iterable[str]) -> dict[str, int | None]`: source override first, then fuzzy ≥ 90 with lead ≥ 5, otherwise None. Raises `SourceError(source, ...)` naming both names if two names map to one team id.
    - `find_in_source(source: str, team: Team, names: Iterable[str]) -> str`: the name in `names` for this team (override first, then best fuzzy ≥ 85). Raises `SourceError(source, f"no entry for {team.name}")`.
  - `normalize(name: str) -> str`: lowercase, strip accents, drop club-form tokens (fc, afc, cf, sc, ac, as, ssc, club, de, calcio and similar)

- [ ] **Step 1 (live): Dump real names.** Print football-data.org team names for all 6 Competitions. Fill `[user]` with at least: barca, inter, psg, man city, man utd, atleti, bayern, leverkusen, juve, real (Real Madrid, Real Betis, Real Sociedad), milan (AC Milan, Inter).
- [ ] **Step 2: Write failing tests** with a fixed team list (Arsenal FC, Chelsea FC, Manchester City FC, Manchester United FC, Brighton & Hove Albion FC, Liverpool FC, FC Barcelona, Real Madrid CF, Real Betis Balompié, Real Sociedad de Fútbol, Club Atlético de Madrid, FC Bayern München, Bayer 04 Leverkusen, Paris Saint-Germain FC, AS Monaco FC, AS Roma, FC Internazionale Milano, AC Milan, Como 1907):

```python
@pytest.mark.parametrize(
    "q,expected",
    [
        ("chelsia", "Chelsea FC"),
        ("bayer leverkusin", "Bayer 04 Leverkusen"),
        ("barca", "FC Barcelona"),
        ("psg", "Paris Saint-Germain FC"),
        ("inter", "FC Internazionale Milano"),
        ("man city", "Manchester City FC"),
    ],
)
def test_resolves(q, expected):
    assert resolver.resolve_query(q).name == expected


def test_real_is_ambiguous():
    with pytest.raises(AmbiguousTeam) as e:
        resolver.resolve_query("real")
    assert {t.name for t in e.value.candidates} == {"Real Madrid CF", "Real Betis Balompié", "Real Sociedad de Fútbol"}


def test_milan_is_ambiguous(): ...  # {"AC Milan", "FC Internazionale Milano"}
def test_unknown(): ...  # "xyzzy" raises TeamNotFound


@pytest.mark.parametrize(
    "q,expected",
    [
        ("Arsenal v Chelsea", ("Arsenal", "Chelsea")),
        ("arsenal vs chelsea", ("arsenal", "chelsea")),
        ("Arsenal - Chelsea", ("Arsenal", "Chelsea")),
        ("Brighton & Hove Albion v Real Betis", ("Brighton & Hove Albion", "Real Betis")),
        ("arsenal", ("arsenal", None)),
    ],
)
def test_parse_query(q, expected):
    assert parse_query(q) == expected


def test_source_names_unmatched_is_none():
    m = resolver.map_source_names("understat", ["Manchester City", "Arsenal", "Ipswich"])
    assert m["Ipswich"] is None and m["Arsenal"] == ARSENAL_ID


def test_two_source_names_for_one_team_raises(): ...  # ["Manchester City", "Man City FC"] → SourceError naming both
```

- [ ] **Step 3: Run** `uv run pytest tests/test_teams.py -q`. Expected: FAIL.
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run** the same command. Expected: PASS.
- [ ] **Step 6: Commit** `feat: fuzzy team resolution with nicknames and ambiguity`

---

### Task 7: Understat client

**Files:**
- Create: `src/predict/sources/understat.py`, `tests/fixtures/understat_epl.json`
- Test: `tests/sources/test_understat.py`

**Interfaces:**
- Produces: `Understat(http: CachedHttp)`
  - `league_matches(comp: Competition, season: int) -> list[MatchRecord]`
    - GET `https://understat.com/getLeagueData/{comp.understat_league}/{season}` with header `X-Requested-With: XMLHttpRequest`, ttl 6h
    - only played Matches; names from the home/away team titles; xG as floats; ids None
    - raises `SourceError("understat", ...)` for CL

- [ ] **Step 1 (live): Record a response.** Fetch EPL 2026 and save a trimmed fixture (2 played + 1 unplayed Match). Note the real keys for the played flag, team titles, goals, xG and datetime; the parser follows them.
- [ ] **Step 2: Write failing tests**:
  - fixture → 2 records
  - xG floats and goals ints correct
  - the unplayed Match is excluded
  - kickoff is aware UTC
  - CL raises `SourceError`
- [ ] **Step 3: Run** `uv run pytest tests/sources/test_understat.py -q`. Expected: FAIL.
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run** the same command. Expected: PASS.
- [ ] **Step 6: Commit** `feat: Understat league xG client`

---

### Task 8: ClubElo client with saved-ratings fallback

**Files:**
- Create: `src/predict/sources/clubelo.py`
- Test: `tests/sources/test_clubelo.py`

**Interfaces:**
- Produces: `ClubElo(http: CachedHttp, store: Store)`
  - `ClubEloRow(club: str, country: str, level: int, elo: float)`
  - `ratings(on: date) -> dict[str, ClubEloRow]`
    - GET `http://api.clubelo.com/{YYYY-MM-DD}` CSV (`Rank,Club,Country,Level,Elo,From,To`), ttl 24h
    - on success, also saves the body under `"clubelo:latest"`
    - on `SourceError`, falls back to `store.cache_latest("clubelo:latest")`
    - with nothing saved, raises `SourceError("clubelo", "unavailable and no saved ratings")`
  - `league_elo(rows: dict[str, ClubEloRow], country: str) -> float`: mean Elo of the country's level-1 rows

- [ ] **Step 1: Write failing tests**:
  - CSV parse
  - `league_elo` mean
  - HTTP 502 with a saved copy → saved ratings returned
  - HTTP 502 with nothing saved → `SourceError` with "no saved ratings"
- [ ] **Step 2: Run** `uv run pytest tests/sources/test_clubelo.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: ClubElo ratings with saved fallback`

---

### Task 9: Ratings for a Match

**Files:**
- Create: `src/predict/ratings.py`
- Test: `tests/test_ratings.py` (in-memory fakes for the three sources)

**Interfaces:**
- Consumes: `FootballDataOrg.teams/team_results`, `Understat.league_matches`, `ClubElo.ratings/league_elo`, `TeamResolver.map_source_names/find_in_source`, strength functions, `projected_goals`, `CL_BASE`
- Produces:
  - `RatedMatch(home: TeamRating, away: TeamRating, base: LeagueBase, projected: ProjectedGoals)`
  - `RatingsService(fdo, understat, clubelo, resolver: TeamResolver)` with `rate_match(match: Match) -> RatedMatch`
  - **League Match:**
    - Understat records for `season_of(kickoff)` (only kickoff < Match kickoff) and the season before
    - names mapped to ids via `map_source_names("understat", ...)`; unmapped names are kept with id None, so they still count toward league averages
    - `league_base`, then `season_rating` per team
    - form from `team_results(team, kickoff−120d, kickoff−1d)`, with xG filled from the current Understat records with the same date and ids
    - `comp_avg` = {own league: mean of base, CL: (CL_HOME_GOALS + CL_AWAY_GOALS) / 2}
    - `combine`, then `projected_goals(..., neutral=match.neutral)`
  - **CL Match:** each team's league = the first of `LEAGUES` whose `teams()` contains it.
    - Top-5 team: rated as in its league (season from that league; form includes CL results), with `elo = league_elo(country)`.
    - Any other team: `combine(TeamRating(attack=1, concede=1), form)` with `elo` = its own ClubElo rating via `find_in_source("clubelo", ...)`.
    - Uses `CL_BASE`.

- [ ] **Step 1: Write failing tests**:
  - league Match → positive Projected Goals, home > away for a stronger home team
  - a record dated after kickoff with absurd goals doesn't change the output (no lookahead)
  - team absent last season (promoted) → rating produced, attack < 1 with a small average current sample
  - 0 current-season matches → last season only, no crash
  - CL top-5 vs non-top-5 → home `elo` = league mean, away `elo` = own Elo
  - CL club missing from ClubElo → `SourceError` containing the club name
- [ ] **Step 2: Run** `uv run pytest tests/test_ratings.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat: assemble team ratings and Projected Goals for a Match`

---

### Task 10: Stats-only Prediction, persistence, full view, CLI

**Files:**
- Create: `src/predict/config.py`, `src/predict/pipeline.py`, `src/predict/render.py`, `src/predict/wiring.py`, `src/predict/cli.py`
- Modify: `src/predict/domain.py`, `src/predict/store.py`, `src/predict/errors.py`
- Test: `tests/test_pipeline.py`, `tests/test_render.py`, `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `config.py`: `Settings(BaseSettings)` reading `.env`, with `football_data_org_token: str`, `odds_api_key: str = ""`, `typesafe_api_key: str = ""`, `claude_model: str = "claude-opus-5-5"`, `jev_model: str = "jev-1.13.0"`, `predict_home: Path = ~/.predict`
  - `domain.Prediction` (pydantic): `id: int | None = None, match: Match, created_at: datetime, projected_base: ProjectedGoals, projected: ProjectedGoals, probabilities: Probabilities, shared_at: datetime | None = None`. Later tasks add fields with defaults.
  - errors: `MatchNotFound(detail: str)`, `PredictionLocked(match: Match)`
  - Store: table `predictions(id, match_id, kickoff, created_at, shared_at, body JSON)`, with `save_prediction(p) -> int`, `update_prediction(p) -> None`, `predictions_for_match(match_id) -> list[Prediction]`, `all_predictions() -> list[Prediction]`
  - `Predictor(fdo, ratings: RatingsService, resolver: TeamResolver, store: Store, *, now: Callable[[], datetime])`:
    - `find_match(query: str, choose: Callable[[list[Team]], Team]) -> Match`. Calls `choose` on `AmbiguousTeam`. With two teams: the earliest upcoming Match between them, in either order, within `SEARCH_WINDOW_DAYS`. With one team: that team's earliest upcoming Match. Otherwise raises `MatchNotFound`.
    - `predict(match: Match) -> Prediction`: raises `PredictionLocked` if now ≥ kickoff; otherwise rates the Match, computes Markets, saves and returns the Prediction
  - `render.py`:
    - `market_label(market_id, match) -> str`: "Match Result", "Both teams to score", "Total goals 2.5", "Arsenal goals 1.5"
    - `answer_label(market_id, answer, match) -> str`: "Arsenal to win", "Draw", "BTTS: Yes", "Over 2.5 goals", "Chelsea under 1.5 goals"
    - `full_view(p: Prediction, tz: tzinfo) -> str`: rich Console with `record=True`, exported as text. Includes the header with local kickoff, Projected Goals (base → adjusted), and one row per Market with whole-number percentages.
  - `wiring.py`: `build(settings: Settings, now) -> Services` (dataclass holding store at `settings.predict_home / "predict.db"`, sources, resolver over the union of `teams()` for all 6 Competitions, ratings, predictor)
  - `cli.py`: `main(argv: list[str] | None = None, services: Services | None = None) -> int`
    - `predict <query...> [--share] [--fresh] [--model ID]` (flags parsed here, wired in Tasks 12/15)
    - `predict log` is reserved for Task 16
    - an ambiguous team prompts a numbered choice
    - `TeamNotFound` (with suggestions), `MatchNotFound`, `PredictionLocked` ("Kickoff has passed — Predictions are locked.") and `SourceError` print a message and return 1

- [ ] **Step 1: Write failing tests**:
  - `find_match("Arsenal v Chelsea")` finds the Match when Chelsea is home
  - one-team query returns the earliest Match
  - no Match in the window → `MatchNotFound`
  - `predict` after kickoff → `PredictionLocked`
  - `predict` saves a row whose Markets each sum to 1
  - `answer_label` examples above
  - `full_view` contains both team names, "Projected Goals" and "Match Result"
  - CLI ambiguous "real" with input "2\n" picks the second candidate
  - CLI after kickoff exits 1 with "locked"
- [ ] **Step 2: Run** `uv run pytest -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5 (live):** `uv run predict "<an upcoming PL Match>"` prints a plausible full view, and `uv run predict <a CL team>` works.
- [ ] **Step 6: Commit** `feat: stats-only predict command with Prediction persistence`

---

### Task 11: Backtest against bookmaker closing odds

**Files:**
- Create: `src/predict/sources/football_data_couk.py`, `src/predict/odds.py`, `src/predict/backtest.py`, `docs/backtest-2025.md`
- Test: `tests/test_backtest.py`, `tests/test_odds.py`

**Interfaces:**
- Produces:
  - `ClosingOdds(kickoff, home_name, away_name, home_goals, away_goals, odds_home, odds_draw, odds_away, odds_over25, odds_under25)` from columns `Date (dd/mm/yyyy), HomeTeam, AwayTeam, FTHG, FTAG, AvgCH, AvgCD, AvgCA, AvgC>2.5, AvgC<2.5`
  - `FootballDataCoUk(http)` with `season(comp: Competition, season: int) -> list[ClosingOdds]`: URL `https://www.football-data.co.uk/mmz4281/{yy}{yy+1}/{comp.couk_division}.csv`, ttl 7 days
  - `odds.devig(prices: dict[str, float]) -> dict[str, float]`: 1/price, normalised
  - `backtest.brier(probs: dict[str, float], outcome: str) -> float`
  - `run_backtest(comp, season, understat, couk, aliases, *, rho=RHO, form_weight=FORM_WEIGHT, full=SEASON_FULL_WEIGHT_MATCHES) -> BacktestReport(n, model_brier_result, book_brier_result, model_brier_total25, book_brier_total25, model_hits_result, book_hits_result)`
    - walk forward: each Match is rated only from same-league Understat records strictly before its date, plus the season before
    - names mapped through a `TeamResolver` built from Understat names (pseudo-ids), using `[source.football_data_couk]` overrides
  - `python -m predict.backtest --season 2025` prints a table per league and a mean, then `PASS` if mean(model − book Match Result Brier) ≤ 0.02, otherwise `FAIL`

- [ ] **Step 1: Write failing tests**:

```python
def test_brier():
    assert brier({"home": 0.5, "draw": 0.3, "away": 0.2}, "home") == pytest.approx(0.38)


def test_devig():
    p = devig({"home": 2.0, "draw": 3.5, "away": 4.0})
    assert (p["home"], p["draw"], p["away"]) == (
        pytest.approx(0.48276, abs=1e-5),
        pytest.approx(0.27586, abs=1e-5),
        pytest.approx(0.24138, abs=1e-5),
    )
```

  Also test that changing the result of a Match on date d doesn't change the backtest prediction for another Match on date d, and that a CSV with the listed header parses.
- [ ] **Step 2: Run** `uv run pytest tests/test_backtest.py tests/test_odds.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5 (live): Run the backtest.** Run `uv run python -m predict.backtest --season 2025` and confirm the CSV header columns exist. If the result is FAIL, grid-search `rho ∈ {−0.13, −0.08, −0.05, 0}`, `form_weight ∈ {0.1, 0.2, 0.3, 0.4}` and `full ∈ {10, 18, 26}`, then set the best values in `params.py`. Record the table and the chosen values in `docs/backtest-2025.md`.
- [ ] **Step 6: Commit** `feat: walk-forward backtest vs closing odds`

---

### Task 12: Bookmaker Probabilities and Hot Takes

**Files:**
- Create: `src/predict/sources/odds_api.py`, `tests/fixtures/odds_event.json`
- Modify: `src/predict/odds.py`, `src/predict/domain.py`, `src/predict/errors.py`, `src/predict/pipeline.py`, `src/predict/render.py`, `src/predict/wiring.py`
- Test: `tests/sources/test_odds_api.py`, `tests/test_odds.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - errors: `OddsUnavailable(reason: str)`
  - `domain.HotTake(market: str, answer: str, model_p: float, book_p: float)` with property `diff`
  - `Prediction` gains: `bookmaker: Probabilities | None = None, hot_takes: list[HotTake] = [], odds_note: str | None = None, credits_remaining: int | None = None`
  - `OddsSnapshot(probabilities: Probabilities, credits_remaining: int | None, fetched_at: datetime)`
  - `OddsApi(client: httpx.Client, store: Store, api_key: str, resolver: TeamResolver, *, now)` with `snapshot(match: Match) -> OddsSnapshot`:
    1. `GET /v4/sports/{odds_sport}/events`. The event matches when both names map, via `map_source_names("odds_api", ...)`, to the Match's team ids and commence time is within 3h of kickoff. Otherwise `OddsUnavailable(f"bookmakers don't list {home} v {away}")`.
    2. `GET /v4/sports/{sport}/events/{id}/odds?regions=eu&markets=h2h,totals,btts,team_totals&oddsFormat=decimal`.
    - Credits come from the `x-requests-remaining` header.
    - HTTP 401/429 → `OddsUnavailable("odds quota used up or key invalid (HTTP n)")`.
    - Snapshots are cached in the store per Match id for `ODDS_TTL`.
    - Parsing:
      - h2h outcome names compared with the event's home_team/away_team/"Draw"
      - totals/team_totals only at the Lines in `TOTAL_LINES`/`TEAM_LINES`; team_totals side taken from `description`
      - btts Yes/No
      - devig per bookmaker per Market, median per answer across bookmakers, renormalise
  - `odds.hot_takes(model: Probabilities, book: Probabilities, *, threshold=HOT_TAKE_THRESHOLD) -> list[HotTake]`
    - two-answer Markets: only the answer with diff ≥ threshold
    - "result": every answer with |diff| ≥ threshold
    - Markets missing in `book` are skipped
    - sorted by |diff| descending
  - `Predictor` gains constructor arg `odds: OddsApi`. `predict` calls `odds.snapshot(match)`, then `hot_takes`. On `OddsUnavailable` it sets `odds_note` and continues.
  - `full_view` adds a bookies column, a Hot Takes section and the remaining credits.

- [ ] **Step 1 (live): Record an odds fixture.** Fetch events and event odds for one upcoming PL Match. Save a trimmed fixture with 2 bookmakers, and confirm the outcome shapes for `btts` and `team_totals`.
- [ ] **Step 2: Write failing tests**:
  - fixture → result/btts/total:2.5/team:home:1.5 present, and a 4.5 Line ignored
  - median aggregation
  - Hot Takes:
    - model over 0.62 vs book 0.50 → one Hot Take ("over", +0.12)
    - 0.55 vs 0.50 → none
    - result home 0.54/0.40 and away 0.22/0.35 → two Hot Takes, home first
  - 401 → `OddsUnavailable`
  - no matching event → `OddsUnavailable` naming both teams
  - a second call within 60 min makes 0 HTTP calls
  - `predict` with odds failing still saves a Prediction with `odds_note` set
- [ ] **Step 3: Run** `uv run pytest -q`. Expected: the new tests FAIL.
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 6: Commit** `feat: bookmaker probabilities and Hot Takes`

---

### Task 13: Team News research (Claude) with dossier reuse

**Files:**
- Create: `src/predict/news/__init__.py`, `src/predict/news/research.py`, `src/predict/sources/fpl.py`, `tests/fixtures/fpl_bootstrap.json`
- Modify: `src/predict/domain.py`, `src/predict/store.py`, `src/predict/errors.py`
- Test: `tests/news/test_research.py`, `tests/sources/test_fpl.py`

**Interfaces:**
- Produces:
  - domain:
    - `FactorKind = Literal["absence", "doubtful", "return", "rotation", "hot_streak", "cold_streak"]`
    - `NewsFactor(kind, player: str | None, summary: str, misses_match: Literal["yes", "likely", "no", "n/a"], matches_missed_of_last_6: int | None, sources: list[str])`
    - `NewsDossier(team: str, factors: list[NewsFactor], lineup_confirmed: bool)`
    - `ResearchOutput(home: NewsDossier, away: NewsDossier)`
    - `MatchDossier(home: NewsDossier, away: NewsDossier, created_at: datetime, claude_model: str)`
  - errors: `ResearchFailed(reason: str)`
  - `Fpl(http)` with `hints(team: Team, resolver: TeamResolver) -> list[str]`
    - `bootstrap-static`, ttl 1h
    - players with status ≠ "a" become `"{web_name}: {status word}, {chance}% chance — {news}"`
    - a team not in FPL → `[]`
  - `Researcher` Protocol: `research(match: Match, hints: dict[Side, list[str]], model: str) -> MatchDossier`
  - `build_prompt(match: Match, hints: dict[Side, list[str]], now: datetime) -> str`. It must contain:
    - both team names, the Competition display name, the kickoff date and today's date
    - "at most 5 web searches per team"
    - at most 6 News Factors per team, ranked by importance
    - "resolve every date into misses_match"
    - "include source URLs"
    - "Do not give probabilities or predictions"
    - every hint
  - `inline_refs(schema: dict) -> dict`: inlines `$defs`, leaving no `$ref`
  - `ClaudeResearcher(*, now)`: `claude_agent_sdk.query` with `ClaudeAgentOptions(model=model, tools=["WebSearch","WebFetch"], allowed_tools=["WebSearch","WebFetch"], setting_sources=[], max_turns=30, output_format={"type": "json_schema", "schema": inline_refs(ResearchOutput.model_json_schema())})`
    - reads `ResultMessage.structured_output`
    - an error result, missing output or SDK exception → `ResearchFailed(reason)`
    - truncates factors to `MAX_FACTORS_PER_TEAM`
  - Store: `save_dossier(match_id, dossier)`, `latest_dossier(match_id) -> MatchDossier | None`
  - `DossierService(researcher, fpl, resolver, store, *, now)` with `get(match, *, fresh: bool, model: str) -> MatchDossier`: reuses the latest dossier if younger than `DOSSIER_TTL`, made with the same model, and `fresh` is False
  - `api_key_warning(env: Mapping[str, str]) -> str | None`: a warning when `ANTHROPIC_API_KEY` is set

- [ ] **Step 1 (live): Check the SDKs and record FPL data.**
  - Run `uv run python -c "import inspect, claude_agent_sdk as s; print(inspect.signature(s.ClaudeAgentOptions))"`. Confirm the option names above; if they differ, use the real names and note them here.
  - Save a trimmed `bootstrap-static` as the FPL fixture.
- [ ] **Step 2: Write failing tests**:
  - prompt contains each required phrase and every hint
  - `inline_refs` output has no `"$ref"` and keeps NewsFactor's `misses_match` enum
  - with a fake researcher:
    - two `get` calls 1h apart → 1 research call
    - 7h apart → 2 calls
    - `fresh=True` → 2 calls
    - a different model → 2 calls
  - FPL: injured player at 25% → hint string; status "a" excluded
  - `api_key_warning({"ANTHROPIC_API_KEY": "x"})` is not None
- [ ] **Step 3: Run** `uv run pytest tests/news tests/sources/test_fpl.py -q`. Expected: FAIL.
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run** the same command. Expected: PASS.
- [ ] **Step 6 (live):** Research one upcoming Match through `DossierService` on the subscription login and check that the dossier is sensible and has sources.
- [ ] **Step 7: Commit** `feat: Claude Team News research with dossier reuse`

---

### Task 14: Jev scoring and capped adjustment

**Files:**
- Create: `src/predict/news/scoring.py`, `src/predict/news/adjust.py`
- Modify: `src/predict/domain.py`, `src/predict/errors.py`
- Test: `tests/news/test_scoring.py`, `tests/news/test_adjust.py`

**Interfaces:**
- Produces:
  - domain: `FactorImpact(side: Side, factor: NewsFactor, attack_mult: float, concede_mult: float, p_start: float | None)`. `attack_mult` applies to this team's goals, `concede_mult` to the opponent's.
  - errors: `JevFailed(reason: str)`
  - `JevAnswers(nouls: dict[str, float], score_probabilities: dict[str, list[float]])`
  - `JevClient` Protocol: `ask(state: dict, nouls: dict[str, str], scores: dict[str, tuple[str, list[str]]]) -> JevAnswers`. `scores` maps name → (instructions, levels in order); `score_probabilities` are aligned to the levels.
  - `TypeSafeJev(api_key: str, model: str)`: adapter over `typesafe_sdk.TypeSafeClient(...).system_one(state=..., questions={...})` using `Noul`/`Score`. Any SDK error → `JevFailed`.
  - `LEVELS` (the same 4 for attack and defence, exact copy):
    1. "No real change: a fringe player, or an absence that has already lasted most of the team's last 6 matches"
    2. "Small change: a rotation player, or a starter with a like-for-like replacement"
    3. "Clear change: a regular starter whose replacement is noticeably weaker"
    4. "Major change: the team's key player in this area, such as the top scorer or main creator, with no comparable replacement"
  - Multipliers by level:
    - attack: weaken `(1.00, 0.97, 0.92, 0.85)`, boost `(1.00, 1.03, 1.06, 1.10)`
    - concede: weaken `(1.00, 1.03, 1.08, 1.15)`, boost `(1.00, 0.97, 0.94, 0.90)`
    - weaken = absence, doubtful, rotation, cold_streak; boost = return, hot_streak
  - `score_dossier(dossier: MatchDossier, match: Match, jev: JevClient) -> list[FactorImpact]`
    - makes one `ask` call; the state is the dossier JSON plus Match info and today's date
    - questions per factor i of side s:
      - `f"{s}{i}_attack"`: "How much does this news change how many chances {team} creates in this Match? News: {summary}"
      - `f"{s}{i}_defence"`: the same, about chances {team} concedes
      - `f"{s}{i}_starts"` (noul "Will {player} start this Match for {team}?") only when kind is doubtful or misses_match is "likely"
    - factors with misses_match "no" and kind absence/doubtful get no questions
    - multiplier = Σ p_level × table_level; with a start probability, m_eff = 1 − (1 − m)(1 − p_start)
  - `apply_impacts(base: ProjectedGoals, impacts: list[FactorImpact], *, cap=NEWS_CAP) -> ProjectedGoals`: home × clamp(Π home attack × Π away concede), and the mirror for away
  - `top_reasons(impacts: list[FactorImpact], side: Side, n: int = 3) -> list[FactorImpact]`: sorted by max(|attack−1|, |concede−1|), descending

- [ ] **Step 1 (live): Check Jev's response mapping.** Make one `TypeSafeJev` call with a tiny state, one noul and one 4-level score. Confirm how score probabilities map to levels (list order or keyed by level text) and adjust the adapter.
- [ ] **Step 2: Write failing tests**:

```python
def test_expected_multiplier():
    assert attack_multiplier("absence", [0, 0, 1, 0]) == pytest.approx(0.92)
    assert attack_multiplier("absence", [0.5, 0.5, 0, 0]) == pytest.approx(0.985)
    assert attack_multiplier("return", [0, 0, 0, 1]) == pytest.approx(1.10)


def test_doubtful_scaled_by_start_chance():
    assert scale_by_start(0.85, p_start=0.6) == pytest.approx(0.94)


def test_cap():
    imp = [FactorImpact(side="home", factor=F, attack_mult=0.85, concede_mult=1.0, p_start=None)] * 3
    assert apply_impacts(ProjectedGoals(home=2.0, away=1.0), imp).home == pytest.approx(1.5)


def test_attack_and_opponent_defence_combine():
    imp = [
        FactorImpact(side="home", factor=F, attack_mult=0.92, concede_mult=1.0, p_start=None),
        FactorImpact(side="away", factor=F, attack_mult=1.0, concede_mult=1.08, p_start=None),
    ]
    assert apply_impacts(ProjectedGoals(home=2.0, away=1.0), imp).home == pytest.approx(2.0 * 0.9936)
```

  Also: `score_dossier` with a FakeJev makes exactly 1 `ask` call with the expected question names, sends no questions for a misses_match "no" absence, and asks `_starts` for a doubtful factor. Expose `attack_multiplier(kind, probs)`, `concede_multiplier(kind, probs)` and `scale_by_start(m, p_start)` in `scoring.py`.
- [ ] **Step 3: Run** `uv run pytest tests/news -q`. Expected: FAIL.
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run** the same command. Expected: PASS.
- [ ] **Step 6: Commit** `feat: Jev scoring of News Factors and capped adjustment`

---

### Task 15: News in the pipeline, share view, flags, fallbacks

**Files:**
- Modify: `src/predict/domain.py`, `src/predict/pipeline.py`, `src/predict/render.py`, `src/predict/wiring.py`, `src/predict/cli.py`
- Test: `tests/test_pipeline.py`, `tests/test_render.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `DossierService`, `score_dossier`, `apply_impacts`, `top_reasons`, `JevClient`, `api_key_warning`
- Produces:
  - `Prediction` gains: `impacts: list[FactorImpact] = [], news_applied: bool = False, news_note: str | None = None, claude_model: str | None = None, jev_model: str | None = None`
  - `Predictor` gains constructor args `dossiers: DossierService, jev: JevClient, claude_model: str, jev_model: str`
  - `predict(match, *, fresh: bool = False, model: str | None = None, share: bool = False) -> Prediction`
    - order: lock check → rate → dossier + `score_dossier` → `apply_impacts` → Markets from the adjusted Projected Goals → odds → save
    - `ResearchFailed`, `JevFailed` or `SourceError` during news → `news_applied=False`, `news_note=reason`, stats-only Markets
    - a reason that mentions a usage limit adds the suggestion "rerun with --model claude-sonnet-5-5"
    - `share=True` sets `shared_at=now`
  - `render.share_text(p: Prediction, record: tuple[int, int] | None, tz: tzinfo) -> str`, exact format:

```
⚽ Arsenal v Chelsea · Premier League · Sat 25 Oct, 22:00
🏆 Arsenal 54% · Draw 24% · Chelsea 22%
🎯 Both teams to score: Yes 52%
🥅 Total goals: Over 2.5 · 51%
👟 Arsenal over 1.5 goals 58% · Chelsea under 1.5 goals 66%
🔥 Hot Take: Arsenal to win (54% vs bookies 44%)
📈 Record: 21/32 results called
```

    - team short names; local time format `%a %d %b, %H:%M`
    - BTTS, Total Goals 2.5 and each team's 1.5 Line show the Pick and its %
    - up to 3 🔥 lines (none if no Hot Takes); the 📈 line is omitted when the record is None or has 0 scored
  - `render.copy_to_clipboard(text: str) -> bool` uses `pbcopy`; returns False if it's missing
  - `full_view` adds:
    - a Team News section: per team, `top_reasons` as `"{player or kind}: {summary} → {team} goals {±x%}"`, plus `"→ {opponent} goals {±x%}"` when |concede−1| ≥ 0.01, and the first source URL
    - footer: "news applied ✓" or "⚠️ news not applied ({note})", the Claude/Jev model ids, and odds credits
  - CLI wires `--share` (prints the share text after the full view, copies it, marks shared), `--fresh` and `--model`, and prints `api_key_warning` if any

- [ ] **Step 1: Write failing tests**:
  - researcher raises `ResearchFailed("usage limit reached")` → Prediction saved, `news_applied` False, note contains "--model claude-sonnet-5-5", Markets sum to 1
  - Jev raises → same fallback
  - successful news → `projected` equals `apply_impacts(projected_base, impacts)` and the impacts are saved
  - `share=True` → `shared_at` set
  - `share_text` for a fixed Prediction equals the block above exactly
  - `share_text` with no Hot Takes and no record has 5 lines
  - CLI with `ANTHROPIC_API_KEY` set prints the warning
- [ ] **Step 2: Run** `uv run pytest -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5 (live):** `uv run predict "<upcoming Match>" --share`. News is applied, the share text is in the clipboard, and a rerun within 6h reuses the dossier (no new Claude run).
- [ ] **Step 6: Commit** `feat: Team News adjustment, share view and CLI flags`

---

### Task 16: Settlement and Prediction Log

**Files:**
- Create: `src/predict/log.py`
- Modify: `src/predict/store.py`, `src/predict/render.py`, `src/predict/cli.py`
- Test: `tests/test_log.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `FootballDataOrg.matches_by_ids`, `MatchStatus`, `pick`, `HotTake`
- Produces:
  - Store: table `results(match_id PK, status TEXT, home_goals, away_goals, settled_at)`, with `save_result(match_id, status: Literal["finished", "void"], result: Result | None, at: datetime)`, `results() -> dict[int, tuple[str, Result | None]]`, `unsettled_match_ids(kicked_off_before: datetime) -> list[int]`
  - `log.VOID_STATUSES = {"POSTPONED", "CANCELLED", "SUSPENDED", "AWARDED"}`
  - `settle_pending(store, fdo, now) -> int`
    - checks Match ids with kickoff < now − `SETTLE_AFTER` and no result
    - FINISHED → finished with the 90-minute Result
    - a status in `VOID_STATUSES`, or a kickoff that moved more than `VOID_KICKOFF_SHIFT` from the predicted kickoff → void
    - anything else is skipped
    - returns the number settled
  - `counted_prediction(preds: list[Prediction]) -> Prediction | None`: only Predictions created before kickoff; the latest shared one, otherwise the latest created
  - `outcome(market_id: str, result: Result) -> str`: the answer that happened
  - `LogSummary(hit_rates: dict[str, tuple[int, int]], hot_takes: tuple[int, int], recent: list[tuple[Prediction, Result]])`
    - hit-rate group keys: "Match Result", "BTTS", "Total Goals 1.5/2.5/3.5", "Team Goals 0.5/1.5/2.5" (both sides combined)
    - `recent` holds the last 10 counted
  - `summarize(store) -> LogSummary`: Counted Predictions with finished Results only; Void never counts
    - a Hot Take is a hit when diff > 0 and outcome == answer, or diff < 0 and outcome ≠ answer
  - `render.log_view(summary) -> str`
  - CLI:
    - `predict log` prints `log_view`
    - every command runs `settle_pending` first (on `SourceError`, print a one-line warning and continue)
    - `share_text` gets `summary.hit_rates["Match Result"]` as its record

- [ ] **Step 1: Write failing tests**:

```python
def test_outcomes():
    r = Result(home_goals=2, away_goals=1)
    assert [outcome(m, r) for m in ("result", "btts", "total:2.5", "total:3.5", "team:away:0.5", "team:away:1.5")] == [
        "home",
        "yes",
        "over",
        "under",
        "over",
        "under",
    ]
```

  Also test:
  - counted: shared Tuesday + unshared Saturday → Tuesday; none shared → latest; a Prediction created after kickoff is excluded
  - settle: FINISHED → finished; POSTPONED → void; TIMED with kickoff +5 days → void; TIMED +2h → skipped; IN_PLAY → skipped
  - summarize with 2 counted Predictions and known Results gives exact "Match Result" (1, 2); a Void Match is excluded; the Hot Take hit rule in both directions
  - `predict log` output contains "Match Result" and "1/2"
- [ ] **Step 2: Run** `uv run pytest tests/test_log.py tests/test_cli.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q && uv run ruff check`. Expected: all PASS.
- [ ] **Step 5 (live):** Predict a Match that kicks off soon. After it ends, `uv run predict log` shows it settled.
- [ ] **Step 6: Commit** `feat: settlement and Prediction Log`
