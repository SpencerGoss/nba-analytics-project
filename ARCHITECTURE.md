<!-- TL;DR: How the system is structured. Read before touching code. -->
# Architecture — NBA Analytics Project

## Overview
NBA data flows from the nba_api through fetchers into `data/raw/`, then preprocessing
builds `data/processed/`, feature engineering produces `data/features/`, and models
train on those features. Inference runs daily via `update.py`, which writes predictions
to SQLite and JSON for the dashboard.

## Data Pipeline
```
nba_api / external APIs
    ↓
src/data/*.py  →  data/raw/           (fetchers — never modify output)
    ↓
src/processing/preprocessing.py  →  data/processed/
    ↓
src/features/team_game_features.py  →  data/features/game_matchup_features.csv (291 cols)
src/features/lineup_features.py    →  data/features/lineup_team_features.csv
    ↓
src/models/game_outcome_model.py   →  models/artifacts/*.pkl
src/models/calibration.py          →  models/artifacts/*_calibrated.pkl
    ↓
update.py (daily)
    ↓
database/predictions_history.db + data/dashboard/*.json
    ↓
dashboard/index.html (Chart.js)
```

## Key Directories
| Path | Purpose |
|------|---------|
| `src/data/` | 20+ fetcher scripts (nba_api, odds, injuries, lineups) |
| `src/features/` | Feature engineering (rolling windows, lineup nets, injury proxy) |
| `src/models/` | Training, calibration, ATS model, predict CLI |
| `src/processing/` | Preprocessing pipeline |
| `src/validation/` | Data integrity validators (v2) |
| `data/raw/` | Source of truth — never modify |
| `data/processed/` | Cleaned game logs, player stats |
| `data/features/` | Feature matrices for training/inference |
| `models/artifacts/` | Serialized models (gitignored — large files) |
| `database/` | nba.db (18 tables) + predictions_history.db |
| `dashboard/` | Static HTML dashboard |
| `.planning/` | GSD roadmap, STATE.md, phase plans |
| `tests/` | pytest suite (115 tests passing) |

## Key Files
| File | Purpose |
|------|---------|
| `update.py` | Daily pipeline entrypoint |
| `backfill.py` | Full historical rebuild |
| `fetch_odds.py` | Loads calibrated model → queries odds API → writes predictions |
| `src/features/team_game_features.py` | Main feature matrix builder |
| `src/features/lineup_features.py` | Lineup net rating aggregates |
| `src/models/game_outcome_model.py` | Game outcome model training |
| `src/models/calibration.py` | Probability calibration (run after every retrain) |
| `src/models/ats_model.py` | ATS logistic regression |
| `src/models/retrain_all.py` | Convenience script — retrains all models in order |
| `src/models/value_bet_detector.py` | Detects value bets using game-outcome edge; `get_strong_value_bets()` filters by confidence threshold |
| `src/models/model_explainability.py` | SHAP-based feature importance and single-prediction explanation |
| `src/data/get_historical_absences.py` | Generates player absence features from game logs (Phase 10) |
| `src/data/get_balldontlie.py` | BallDontLie API client (injuries, stats, teams) |
| `src/data/get_injury_data.py` | Injury report normalization from PDF and nba_api |
| `src/data/get_lineup_data.py` | On-court lineup data from nba_py |

## External Services
| Service | Used For | Config |
|---------|---------|--------|
| nba_api | Game logs, team stats, lineups, play-by-play | No key needed |
| Odds API | Game spreads and moneylines | `ODDS_API_KEY` in `.env` |
| BallDontLie API | Supplemental player/game data | `BALLDONTLIE_API_KEY` in `.env` (optional) |
| Kaggle injury dataset | Historical injury proxy for training | Manual download |

## Season Code Convention
Integer format: `202425` = 2024-25 season. Not "2024-25". Enforced across all scripts.
