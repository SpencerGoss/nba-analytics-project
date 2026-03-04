<!-- TL;DR: NBA-specific rules. Auto-loaded every session. -->

# NBA Domain Rules

## Data Leakage (CRITICAL — never violate)
- All rolling/lag features must use `shift(1)` before `.rolling()` or `.expanding()`
- Row N must only see rows 0..N-1 — no same-game information
- Lineup, injury, and opponent features must also be shifted

## Model Training
- Expanding-window validation only — never use future seasons in training window
- After retraining any model, always run `calibration.py` immediately after
- `fetch_odds.py` must load `game_outcome_model_calibrated.pkl` — not the base model
- Feature list comes from `models/artifacts/game_outcome_features.pkl` — use it for inference

## Data Pipeline
- Never modify files in `data/raw/` — they are source of truth
- Season code format is always an integer: `202425` (not "2024-25" or 2024)
- nba_api calls need `time.sleep(0.6)` between requests to avoid rate limits
- Lineup features are 0.0 for seasons before 2015-16 — this is expected

## Database
- `database/nba.db` — 18 tables, season as integer
- `database/predictions_history.db` — WAL mode, prediction store
- Always use parameterized queries — no string formatting in SQL

## Secrets
- Never commit `.env`
- `ODDS_API_KEY` and `BALLDONTLIE_API_KEY` live in `.env` only
- `.env.example` must be kept in sync with `.env` (no real values)
