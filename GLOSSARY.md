# Predict

A tool that estimates outcome probabilities for an upcoming football match from current form, season performance and the latest team news.

## Language

**Match**:
A single scheduled football game between a home team and an away team.
_Avoid_: Event, fixture, game

**Market**:
A question about a Match's outcome that has a fixed set of answers, each given a probability.
_Avoid_: Stat, bet type

**Match Result**:
The Market with answers Home win / Draw / Away win, whose probabilities add up to 100%.
_Avoid_: Winning ratio, winner, 1X2

**BTTS**:
The yes/no Market asking whether both teams score at least one goal.
_Avoid_: Both teams goal, GG/NG

**Total Goals**:
The Market asking whether the Match's combined goals go over or under a Line.
_Avoid_: Goal target, goals ratio

**Team Goals**:
The Market asking whether one team's goals in a Match go over or under a Line.
_Avoid_: Goal target, team total

**Line**:
The half-goal threshold an over/under Market is measured against, e.g. 2.5.
_Avoid_: Target, threshold

**Projected Goals**:
The number of goals one team is predicted to score, on average, in an upcoming Match.
_Avoid_: Expected goals, xG

**xG**:
A measure of chance quality, counted from shots in Matches already played.
_Avoid_: Expected goals (ambiguous with Projected Goals)

**Competition**:
A league or cup the tool covers: Champions League, Premier League, La Liga, Bundesliga, Ligue 1, Serie A.
_Avoid_: League (when the cup is included), tournament

## Team Strength

**Season Strength**:
A team's level across the current season, blended with last season while the current sample is still small.
_Avoid_: Season performance, table position

**Form**:
A team's recent level over its last 6 Matches in the covered Competitions, weighted toward the most recent.
_Avoid_: Momentum, streak, current form

**Team News**:
Absences, suspensions, expected rotation and player hot or cold streaks that change a team's level for one Match.
_Avoid_: Injuries (too narrow), player form

**News Factor**:
One item of Team News, such as a player's absence, a return from injury or a scoring streak.
_Avoid_: Reason, news item

**News Dossier**:
The written summary of one team's Team News for one Match, with sources, that the AI scores.
_Avoid_: Report, research, context

## Predictions

**Prediction**:
The probabilities for every Market of one Match, made at a specific time before kickoff.
_Avoid_: Forecast, tip

**Counted Prediction**:
The one Prediction per Match that the Prediction Log scores: the latest shared one, otherwise the last one before kickoff.
_Avoid_: Final prediction, official pick

**Bookmaker Probability**:
The chance of an answer implied by a bookmaker's odds, with the bookmaker's margin removed.
_Avoid_: Odds (when the percentage is meant), market price

**Hot Take**:
An answer where the Prediction's probability differs from the Bookmaker Probability by 10 points or more.
_Avoid_: Value bet, edge, upset pick

**Prediction Log**:
The permanent record of Predictions, locked at kickoff and scored against the Result.
_Avoid_: History, track record

**Result**:
A Match's score after 90 minutes plus stoppage time; extra time and penalties are ignored.
_Avoid_: Final score, outcome

**Void**:
The status of a Prediction whose Match was postponed, cancelled, abandoned, awarded, or moved more than 48 hours; it is never scored.
_Avoid_: Cancelled, push

**Pick**:
The answer with the highest probability in a Market.
_Avoid_: Tip, call, favourite

**Hit**:
A Market whose Pick matched the Result.
_Avoid_: Win, correct, success

**Hit Rate**:
Hits divided by scored Markets in the Prediction Log, reported per Market.
_Avoid_: Accuracy, win rate, winning ratio
