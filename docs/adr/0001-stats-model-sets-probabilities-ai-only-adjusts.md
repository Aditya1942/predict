# Stats model sets the probabilities; AI only adjusts them

Every Market's probability comes from one goals model: Projected Goals for each team, built from season xG, form and cross-league strength, with every Market (Match Result, BTTS, Team Goals, Total Goals) calculated from the resulting scoreline probabilities. The AI models never output a final probability: Claude researches Team News into a News Dossier, Jev scores its impact, and code turns those scores into capped adjustments to Projected Goals. We chose this over asking an AI "who wins?" because AI-guessed percentages aren't calibrated, and asking about each Market separately produces contradictions (e.g. 75% Over 2.5 but 30% BTTS). One scoreline distribution keeps every Market consistent and lets the Prediction Log test the model.

## Considered Options

- **AI only** (web search + "give me the %"): fastest to build, numbers unreliable and inconsistent across Markets.
- **Jev answers each Market directly**: calibrated per question, but still inconsistent across Markets.
- **Stats model only**: consistent, but blind to injuries, suspensions and rotation.
