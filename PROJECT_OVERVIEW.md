# NBA Analytics Project — Overview

End-to-end NBA analytics system: data ingestion → feature engineering → game outcome prediction (67.4%, AUC 0.742) → ATS betting model (54.9%, Brier-optimized) → prediction store → web dashboard.

## v2.1 Results (March 2026)
- Game outcome model: **67.4% accuracy, AUC 0.7419** — injury features active for the first time (home_rotation_availability rank #5 in importances); 11 injury features now contributing
- ATS model: **54.9% accuracy** (up from 53.5%) — now selects on Brier score (calibration), not accuracy; 2021-22 held-out calibration split
- Feature matrix: 291-column matchup CSV; 60.9% of games have non-zero missing_minutes (was 0% before Phase 1)
- `player_absences.csv`: 1,098,538 rows, 75 seasons, 12.6% absence rate — fully wired into injury pipeline
- Tests: 145 passing
- Pipeline: fully operational; daily `update.py` refreshes data + features + predictions in one command
- **Odds source:** Pinnacle guest API (free, keyless, no quota) — game_lines.csv populates live with NBA moneylines + spreads

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
