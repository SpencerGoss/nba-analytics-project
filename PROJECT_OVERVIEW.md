# NBA Analytics Project — Overview

End-to-end NBA analytics system: data ingestion → feature engineering → game outcome prediction (67.1%, AUC 0.7406) → ATS betting model (54.9%, Brier-optimized) → prediction store → web dashboard (fully live data) + CLV tracking.

## v2.2 Results (March 2026)
- Game outcome model: **67.1% accuracy, AUC 0.7406** — LightGBM added as candidate (guarded import); gradient_boosting still selected; 11 injury features active
- ATS model: **54.9% accuracy** — Brier-optimized; 2021-22 held-out calibration split; logistic_l1 selected
- Feature matrix: **296-column** matchup CSV; includes `diff_pythagorean_win_pct_roll10` (Morey exponent 14.3, 10-game rolling)
- Value bet output: now includes `kelly_fraction` (fractional Kelly, 0.5x scale) per bet
- CLV tracking: `clv_tracking` table in `predictions_history.db`; opening lines logged automatically from `fetch_odds.py`; `CLVTracker.get_clv_summary()` reports mean CLV and edge flag
- `player_absences.csv`: 1,098,538 rows, 75 seasons, 12.6% absence rate — fully wired into injury pipeline
- Tests: 573 passing (29 new in build_line_movement; NULL guard fix)
- Pipeline: fully operational; daily `update.py` refreshes data + features + predictions in one command
- **Odds source:** Pinnacle guest API (free, keyless, no quota) — game_lines.csv populates live with NBA moneylines + spreads
- **Dashboard v3:** full Linear/Coinbase redesign; 1,710 players from `player_comparison.json` with search + pagination; Players tab filter (Current Season / Legends / All); Standings default tab = Conference; 7 audit bugs fixed (null crashes, missing era buttons, dead stubs); Betting Tools drawer, Power Rankings, Injuries, H2H, Teams, Value Bets all live; live at https://spencergoss.github.io/nba-analytics-project/
- **HPO result:** Optuna 100 trials — LightGBM 0.7116, XGBoost 0.7115 — both lose to gradient_boosting (0.7406); production model unchanged

## Key Links
- **Architecture & full description:** [`docs/project_overview.md`](docs/project_overview.md)
- **Pipeline stage reference:** [`docs/PIPELINE.md`](docs/PIPELINE.md)
- **Current progress / session state:** [`.planning/STATE.md`](.planning/STATE.md)
- **Known bugs & tech debt:** [`.planning/codebase/CONCERNS.md`](.planning/codebase/CONCERNS.md)
- **Session log:** [`PROJECT_JOURNAL.md`](PROJECT_JOURNAL.md)

## Quick Commands
```bash
pytest -v                                           # run tests
python update.py                                    # daily data refresh (Stages 1+2)
python src/features/team_game_features.py           # build features (Stage 3)
python src/models/train_all_models.py               # train models (Stage 4)
python src/models/calibration.py                    # calibrate (Stage 5)
python src/models/predict_cli.py game --home BOS --away LAL   # predict
python -m http.server 8080 --directory dashboard    # serve dashboard
```

## Stack
Python 3.12+, pandas, scikit-learn, SQLite, Chart.js. No npm/Node.
