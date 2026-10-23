"""All model constants. Functions take these as keyword defaults."""

from datetime import timedelta

# Season strength
XG_WEIGHT = 0.7
SEASON_FULL_WEIGHT_MATCHES = 18
PROMOTED_ATTACK = 0.85
PROMOTED_CONCEDE = 1.15

# Form
FORM_MATCHES = 6
FORM_WEIGHTS = (6, 5, 4, 3, 2, 1)
FORM_WEIGHT = 0.3
FORM_FLOOR = 0.25
FORM_WINDOW_DAYS = 120

# Projected Goals
CL_HOME_GOALS = 1.60
CL_AWAY_GOALS = 1.30
ELO_KAPPA = 0.0025

# Scoreline and Markets
RHO = -0.05
MAX_GOALS = 10
TOTAL_LINES = (1.5, 2.5, 3.5)
TEAM_LINES = (0.5, 1.5, 2.5)

# News
NEWS_CAP = (0.75, 1.25)
MAX_FACTORS_PER_TEAM = 6
MAX_SEARCHES_PER_TEAM = 5
DOSSIER_TTL = timedelta(hours=6)
ODDS_TTL = timedelta(minutes=60)

# Hot Takes and settlement
HOT_TAKE_THRESHOLD = 0.10
VOID_KICKOFF_SHIFT = timedelta(hours=48)
SETTLE_AFTER = timedelta(minutes=150)
SEARCH_WINDOW_DAYS = 60
