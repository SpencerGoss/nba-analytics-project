# NBA Analytics Project — Overview

End-to-end NBA analytics system: data ingestion → feature engineering → game outcome prediction (68%) → ATS betting model (53.5%, +2.2% ROI) → prediction store → web dashboard.

## v2.0 Results (March 2026)
- Game outcome model: **68% accuracy** (up from 66.8% in v1)
- ATS model: **53.5%** (+2.2% holdout ROI — above 52.4% breakeven)
- Feature matrix: 291-column matchup CSV (incl. injury proxy features), rebuilt daily via `update.py`
- Prediction store: `predictions_history.db` — 9 predictions written for Mar 5 games; healthy
- Tests: 145 passing
- Pipeline: fully operational; daily `update.py` refreshes data + features + predictions in one command
- **Odds source:** Pinnacle guest API (free, keyless, no quota) — game_lines.csv now populates live with NBA moneylines + spreads; model_vs_odds.csv compares model probabilities to market lines

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
