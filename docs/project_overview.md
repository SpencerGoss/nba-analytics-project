# NBA Analytics Project — Project Overview

This document describes the current state of the NBA analytics project as of v1.0 (March 2026).

---

## Table of Contents
- [1. Project Goals](#1-project-goals)
- [2. Architecture](#2-architecture)
- [3. Data Pipeline](#3-data-pipeline)
- [4. Feature Engineering](#4-feature-engineering)
- [5. Predictive Models](#5-predictive-models)
- [6. Evaluation Suite](#6-evaluation-suite)
- [7. ATS Betting Model](#7-ats-betting-model)
- [8. External Data Sources](#8-external-data-sources)
- [9. Known Limitations and Tech Debt](#9-known-limitations-and-tech-debt)

---

## 1. Project Goals

This project is a full end-to-end NBA analytics system. It combines data engineering, feature engineering, and machine learning to predict game outcomes, player performance, and against-the-spread (ATS) betting value.

**Core Goals:**
- Build a reproducible data pipeline from the NBA API through to trained model predictions
- Predict game outcomes with >66% accuracy on modern-era holdout seasons
- Identify ATS value bets that show positive ROI over a large backtest
- Store prediction history in a queryable, web-ready format
- Document all pipeline stages so each is self-contained and independently runnable

**v1.0 Milestone Results (31/31 requirements satisfied):**
- Game outcome model: 66.8% test accuracy on 2023-24 / 2024-25 holdout
- ATS backtest: 18,233 games, +1.28% ROI on value-bet filtered subset
- Prediction store: WAL-mode SQLite + daily JSON snapshots
- Pipeline: 6 documented stages, all runnable independently

---

## 2. Architecture

```
nba-analytics-project/
├── data/
│   ├── raw/              # One CSV per season per table (from NBA API)
│   ├── processed/        # Cleaned, combined CSVs (all seasons per table)
│   ├── features/         # Model-ready feature tables
│   ├── odds/             # Sportsbook data (The Odds API + Kaggle historical)
│   └── outputs/          # Daily JSON prediction snapshots
├── database/
│   └── predictions_history.db   # WAL-mode SQLite prediction store
├── models/
│   └── artifacts/        # Trained model files + metadata JSON
├── reports/
│   ├── calibration/      # Calibration curves, Brier scores, by-era/season
│   └── explainability/   # SHAP values, feature importance charts
├── src/
│   ├── data/             # API ingestion scripts (get_*.py)
│   │   └── external/     # Basketball Reference scraper, injury report fetcher
│   ├── features/         # Feature engineering modules
│   ├── models/           # ML models, evaluation, prediction CLI
│   ├── outputs/          # Prediction store and JSON export
│   └── processing/       # Preprocessing pipeline
├── scripts/
│   ├── fetch_odds.py     # The Odds API caller
│   └── run_update.bat    # Windows Task Scheduler batch file
├── docs/
│   ├── PIPELINE.md       # Authoritative pipeline reference (6 stages)
│   └── project_overview.md   # This file
├── update.py             # Daily data refresh orchestrator (Stages 1+2)
├── backfill.py           # One-time historical backfill
└── requirements.txt
```

**Data Flow:**
```
NBA API → data/raw/ → preprocessing.py → data/processed/ → feature modules → data/features/ → models → predictions
```

**Database:**
- The project uses CSV-based data flow for the main pipeline
- `predictions_history.db` stores all prediction outputs (WAL mode for concurrent access)

**18 data tables** spanning:
- Player stats (regular + advanced + clutch + scoring + playoffs): 1996-97+
- Player game logs: 1946-47+
- Team stats (regular + advanced + playoffs): 1996-97+
- Team game logs: 1946-47+
- Hustle stats: 2015-16+
- Standings: 1979-80+
- Player/team dimension tables: all-time

Season code format: 6-digit integer (e.g., `202425` for 2024-25).

---

## 3. Data Pipeline

The pipeline has two operating modes:
- **Daily refresh** (`python update.py`): fetches current-season data, rebuilds processed CSVs. Runs Stages 1+2 only. Scheduled daily at 7:00 AM via Windows Task Scheduler.
- **Full rebuild** (manual): all 6 stages in order, required for model retraining.

| Stage | Command | Runtime |
|-------|---------|---------|
| 1. Fetch | `python update.py` | 10-15 min |
| 2. Preprocess | (called by update.py) | 1-2 min |
| 3. Feature Build | `python src/features/team_game_features.py` | 5-10 min |
| 4. Train | `python src/models/train_all_models.py` | 5-10 min |
| 5. Calibrate | `python src/models/calibration.py` | 1-2 min |
| 6. Predict | `python src/models/predict_cli.py game --home BOS --away LAL` | <1 min |

See [PIPELINE.md](PIPELINE.md) for full stage details, dependency graph, and common operations.

---

## 4. Feature Engineering

All features enforce **no data leakage** via `shift(1)` before any rolling computation.

### Team Game Features (`src/features/team_game_features.py`)
- Rolling means (5/10/20 games) for pts, fg_pct, fg3_pct, ft_pct, reb, ast, stl, blk, tov, pf, plus_minus
- Rolling win percentage (5/10/20 games)
- Opponent points allowed (defensive signal)
- Strength of schedule (rolling opponent win%)
- Context: is_home, days_rest, is_back_to_back, season_game_num, cum_win_pct
- Injury proxy features: missing_minutes, missing_usg_pct, rotation_availability, star_player_out

### Game Matchup Features (`data/features/game_matchup_features.csv`)
- Home/away prefixed team features merged into single-row-per-game format
- Differential columns (home minus away) for key stats

### Player Features (`src/features/player_features.py`)
- Rolling means and standard deviations (5/10/20 games) for all box score stats
- Season-to-date expanding averages
- Advanced stats (usage rate, true shooting %, net rating, PIE)
- Opponent defensive context
- Player bio context (age, height, weight from 2020-21+)

### Era Labels (`src/features/era_labels.py`)
Maps seasons to 6 historical NBA eras anchored to rule changes (1946-present). Applied automatically to all feature tables.

### Injury Proxy (`src/features/injury_proxy.py`)
Detects absent rotation players from game log gaps. Computes missing minutes, missing usage, rotation availability, and star-player-out flags. Separate training path (historical proxy) and inference path (live NBA injury report).

### ATS Features (`src/features/ats_features.py`)
Builds ATS-specific feature table including Vegas spread, for the spread-covering model.

### Referee Features (`src/features/referee_features.py`)
Wired but awaiting data — Basketball Reference scraper is complete but Cloudflare-blocked in Windows environment. Features degrade gracefully to NaN.

---

## 5. Predictive Models

### Game Outcome (`src/models/game_outcome_model.py`)
- Binary classification: home win vs loss
- GradientBoosting / RandomForest / Logistic (best selected automatically)
- Modern-era filter option (2014-15+ only)
- **v1.0 result:** 66.8% accuracy on 2023-24/2024-25 holdout

### Player Performance (`src/models/player_performance_model.py`)
- Multi-target regression: pts, reb, ast
- Separate GBM per stat target, Ridge baseline comparison
- Filtered to players with 20+ training games

### Playoff Odds (`src/models/playoff_odds_model.py`)
- Monte Carlo simulation (10,000 iterations)
- Bradley-Terry strength model with home-court advantage
- Simulates regular season remainder + playoff bracket

### Custom GBM (`src/models/numpy_gbm.py`)
- Pure-numpy gradient boosting implementation (699 lines)
- Used as internal dependency to avoid serialization compatibility issues

---

## 6. Evaluation Suite

### Walk-Forward Backtesting (`src/models/backtesting.py`)
Rolling one-season-at-a-time backtest. Trains on all prior seasons, tests on next. Tracks accuracy, ROC-AUC, Brier score per season. Breaks results down by era.

### Calibration Analysis (`src/models/calibration.py`)
Reliability diagram, Brier score, Expected Calibration Error (ECE). Fits isotonic regression calibration wrapper. Saves calibrated model to `models/artifacts/`.

### SHAP Explainability (`src/models/model_explainability.py`)
Global SHAP summary plots, directional bar charts, per-prediction waterfall charts. Falls back to permutation importance if SHAP not installed.

### Value Bet Detection (`src/models/value_bet_detector.py`)
Identifies games where model probability diverges from implied market probability. Configurable edge threshold.

---

## 7. ATS Betting Model

### ATS Classifier (`src/models/ats_model.py`)
- Logistic regression predicting whether a team covers the spread
- Walk-forward validation with 11 expanding splits
- **v1.0 result:** 51.2% OOS accuracy (below 52.4% vig breakeven — baseline model, improvement opportunities exist)

### ATS Backtest (`src/models/ats_backtest.py`)
- 18,233-game historical backtest (2009-2025)
- Metrics: hit rate, ROI, CLV, Kelly fraction, max drawdown
- Baseline ROI: +0.67%, value-bet filtered: +1.28%
- Historical odds from Kaggle dataset (`src/data/get_historical_odds.py`)

### Odds Integration
- Live odds: The Odds API (`scripts/fetch_odds.py`), refreshed by `update.py`
- Historical odds: Kaggle NBA historical spreads dataset
- API key stored in `.env` (never committed to git)

---

## 8. External Data Sources

### Basketball Reference Referee Scraper (`src/data/external/bref_scraper.py`)
- Scrapes referee crew assignments from box scores
- Rate limited: 3 sec/request (respects robots.txt)
- Not in daily pipeline — run manually for historical backfill
- Currently blocked by Cloudflare in Windows dev environment

### NBA Injury Report Fetcher (`src/data/external/injury_report.py`)
- Fetches current day's official NBA injury report
- Primary: `nba_api` LeagueInjuryReport endpoint; fallback: PDF from nba.com
- Used at inference time only (training uses historical proxy)

---

## 9. Known Limitations and Tech Debt

### Integration Gaps
1. ~~`train_all_models.py` does not call `train_ats_model()` or `run_calibration_analysis()`~~ — `train_all_models.py` now includes ATS training and calibration (resolved)
2. ~~`predict_cli.py` has no `ats` or `value-bet` subcommand~~ — `predict_cli.py` now supports `ats` and `value-bet` subcommands (resolved)
3. No referee data scraped yet — features are all NaN in current model

### Model Limitations
- ATS model at 51.2% is below the 52.4% vig breakeven threshold
- CLV computation returns 0.0 in backtest (Kaggle data may use opening rather than closing lines)
- SHAP reports in `reports/explainability/` are stale (built from older model version) — regenerate with `python src/models/run_evaluation.py`

### Pipeline Gaps
- ~~Full retrain flow requires 4 manual commands~~ — Full retrain: `python src/models/train_all_models.py --rebuild-features` (resolved — single command covers features, training, calibration, and ATS)
- Modern-era filter shows 66.84% vs full-history 67.54% — accepted as noise
