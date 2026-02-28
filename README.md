# NBA Analytics Project

End-to-end NBA analytics pipeline: ingest data from NBA API, build cleaned datasets,
load SQLite tables, and train/evaluate prediction models.

## What This Repo Does
- Fetches NBA data across player, team, game log, standings, and playoffs endpoints.
- Rebuilds cleaned datasets in `data/processed/`.
- Loads all processed tables into `database/nba.db`.
- Trains and evaluates models for game outcomes and player performance.

## Quick Runbook
1. Run daily refresh for the current season:
   `python update.py`
2. Run one-time historical backfill (long run):
   `python backfill.py`
3. Train models:
   `python src/models/game_outcome_model.py`
   `python src/models/player_performance_model.py`
4. Generate explainability outputs:
   `python src/models/model_explainability.py`

## Project Layout
- `src/data/`: API ingestion scripts
- `src/processing/`: preprocessing and SQLite loading
- `src/features/`: feature engineering
- `src/models/`: model training, calibration, backtesting, explainability
- `data/raw/`: raw API pulls by endpoint and season
- `data/processed/`: cleaned CSVs used by models and SQL loader
- `database/nba.db`: local SQLite database
- `reports/explainability/`: SHAP plots and feature direction exports
- `docs/project_overview.md`: longer-form project notes

## Notes
- `update.py` is for ongoing season maintenance.
- `backfill.py` is for historical gaps and should be run rarely.
- Processed data and database files can be regenerated from raw pulls.
