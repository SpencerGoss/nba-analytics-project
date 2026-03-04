# NBA Analytics Project — Overview

End-to-end NBA analytics system: data ingestion → feature engineering → game outcome prediction (66.8%) → ATS betting model → prediction store → web dashboard.

## v1.0 Results (March 2026)
- Game outcome model: **66.8% accuracy** on 2023-24 / 2024-25 holdout
- ATS backtest: 18,233 games, **+1.28% ROI** on value-bet filtered subset
- Prediction store: WAL-mode SQLite + daily JSON snapshots
- Pipeline: 6 documented stages, all independently runnable

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
