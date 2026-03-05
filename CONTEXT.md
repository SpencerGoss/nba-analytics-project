<!-- TL;DR: Full project context. Read when AI_INDEX.md says to. -->
# Context — NBA Analytics Project

## What Problem This Solves
Identify NBA games where the model's win probability meaningfully disagrees with Vegas lines,
producing profitable against-the-spread picks. Current v1 baseline: 66.8% game accuracy,
51.2% ATS (below 52.4% vig breakeven). v2 targets both above those thresholds.

## Key Constraints
- No data leakage: rolling features must use `shift(1)` so row N sees only rows 0..N-1
- Expanding-window validation only — never split on future data
- Calibrated model must be used in fetch_odds.py (was a v1 bug — fixed in phase 6)
- Season code format is integer: `202425` not "2024-25"
- nba_api rate limits: add `time.sleep(0.6)` between calls

## Known Gotchas
- Lineup features are 0.0 for seasons before 2015-16 (expected — no data)
- Basketball Reference scraper blocked by Cloudflare in Windows dev (workaround: use nba_api)
- `fetch_odds.py` must load calibrated model first — see `models/artifacts/game_outcome_model_calibrated.pkl`
- After retraining any model, always regenerate calibration via `calibration.py`
- STATE.md can drift from reality — git log is the source of truth for phase status
- `game_matchup_features.csv` is currently 291 columns (272 base + 19 lineup net rating features)

## External Dependencies
- `nba_api` — primary data source, rate-limited (~0.6s between calls)
- BallDontLie API — stub in place, needs `BALLDONTLIE_API_KEY` in `.env`
- Odds API — `fetch_odds.py`, needs `ODDS_API_KEY` in `.env`
- SQLite `database/nba.db` — 18 tables, season format integer `202425`
- SQLite `database/predictions_history.db` — prediction store
- Kaggle injury dataset — used in training as proxy (not real-time)

## Model Artifacts
- `models/artifacts/game_outcome_model.pkl` — base classifier
- `models/artifacts/game_outcome_model_calibrated.pkl` — load this first for inference
- `models/artifacts/game_outcome_features.pkl` — feature list for inference
- `models/artifacts/ats_model.pkl` — ATS logistic regression

## v2.0 Phase Status (actual — git is truth)
| Phase | Status |
|-------|--------|
| 6 — Production Fixes & Injury Data | COMPLETE |
| 7 — New Data Sources | COMPLETE |
| 8 — Feature Engineering | COMPLETE |
| 9 — Model Retraining & ATS Optimization | COMPLETE |
| 10 — Real Absence Features & Pipeline Fix | IN PROGRESS |
