# NBA Analytics Project

End-to-end NBA analytics system: data ingestion, feature engineering, game outcome prediction, player performance forecasting, and ATS betting analysis.

## What It Does

- **Game Outcome Prediction** — 66.8% accuracy on modern-era holdout seasons (GradientBoosting classifier)
- **Player Performance Forecasting** — Next-game pts/reb/ast predictions per player (GBM regressors)
- **Playoff Odds Simulation** — Monte Carlo simulation of remaining season + playoff bracket
- **ATS Value Bet Detection** — Identifies spread-covering opportunities (+1.28% ROI on 18,233-game backtest)
- **ATS Betting Model** — Against-the-spread classifier with 18,233-game backtest (+1.28% ROI on value-bet subset)
- **Web Dashboard** — Static Chart.js dashboard: today's picks, accuracy history, value bets, player lookup
- **Data Integrity Validation** — Stage-by-stage validators catch silent failures before they propagate
- **Automated Data Pipeline** — Daily refresh from NBA API via Windows Task Scheduler

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# First-time setup: fetch all historical data (~30-45 min)
python backfill.py

# Build features, train models, calibrate
python src/features/team_game_features.py
python src/models/train_all_models.py
python src/models/calibration.py

# Predict a game
python src/models/predict_cli.py game --home BOS --away LAL

# Predict player performance
python src/models/predict_cli.py player --name "LeBron James"
```

## Daily Updates

```bash
python update.py    # Fetches current-season data + rebuilds processed CSVs
```

Runs automatically at 7:00 AM via `scripts/run_update.bat` (Windows Task Scheduler).

## Project Structure

```text
├── data/
│   ├── raw/              # Raw API pulls (one CSV per season per table)
│   ├── processed/        # Cleaned combined CSVs
│   ├── features/         # Model-ready feature tables
│   └── odds/             # Sportsbook data
├── src/
│   ├── data/             # API ingestion scripts
│   │   └── external/     # Referee scraper, injury report fetcher
│   ├── features/         # Feature engineering (team, player, injury, ATS, era)
│   ├── models/           # ML models, evaluation, prediction CLI
│   ├── outputs/          # Prediction store + JSON export
│   └── processing/       # Preprocessing pipeline
├── src/
│   └── validation/       # Data integrity validators (stage-by-stage)
├── dashboard/
│   └── index.html        # Static web dashboard (Chart.js, no build step)
├── models/artifacts/     # Trained models + metadata (gitignored)
├── reports/              # Calibration, explainability, backtest reports
├── docs/
│   ├── PIPELINE.md       # Pipeline reference (6 stages with commands)
│   └── project_overview.md  # Full project description
├── update.py             # Daily refresh orchestrator
├── backfill.py           # One-time historical data fetch
└── requirements.txt
```

## Pipeline Stages

| Stage | Command | What It Does |
| ----- | ------- | ------------ |
| 1. Fetch | `python update.py` | Pull current-season data from NBA API |
| 2. Preprocess | (called by update.py) | Clean and combine raw CSVs |
| 3. Features | `python src/features/team_game_features.py` | Build rolling/matchup features |
| 4. Train | `python src/models/train_all_models.py` | Train game, player, and playoff models |
| 5. Calibrate | `python src/models/calibration.py` | Calibrate probability outputs |
| 6. Predict | `python src/models/predict_cli.py` | Generate predictions |

See [docs/PIPELINE.md](docs/PIPELINE.md) for full details.

## Key Features

- **No data leakage**: All rolling features use `shift(1)` — only prior-game data used
- **18 data tables** covering 1946-present (varies by endpoint)
- **6 NBA eras** mapped to rule changes for era-aware modeling
- **Injury proxy**: Detects absent rotation players from game log gaps
- **Walk-forward backtesting**: Season-by-season expanding window evaluation
- **Calibration**: Isotonic regression wrapper for trustworthy probability outputs
- **Prediction store**: WAL-mode SQLite + daily JSON snapshots

## Tech Stack

- Python, pandas, scikit-learn, numpy
- nba_api (NBA Stats endpoints)
- SQLite (prediction history)
- The Odds API (live sportsbook lines)

## Dashboard

```bash
python scripts/export_dashboard_data.py     # Export DB → dashboard/data/*.json
python -m http.server 8080 --directory dashboard   # Serve at localhost:8080
```

## Documentation

- [Pipeline Reference](docs/PIPELINE.md) — Stage-by-stage commands, inputs, outputs, runtimes
- [Project Overview](docs/project_overview.md) — Architecture, models, evaluation, known limitations
- [Session Journal](PROJECT_JOURNAL.md) — Dated session log, what was done and what's next
