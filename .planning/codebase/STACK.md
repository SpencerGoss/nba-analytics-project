# Technology Stack

**Analysis Date:** 2025-03-01

## Languages

**Primary:**
- Python 3.14.3 - All data processing, model training, API integrations, and pipeline orchestration

## Runtime

**Environment:**
- Python 3.14.3 via virtual environment (`.venv`)

**Package Manager:**
- pip - Dependency management
- Lockfile: `requirements.txt` (present)

## Frameworks & Core Libraries

**Data Processing:**
- pandas 2.0.0+ - Data transformation, CSV manipulation, and processing pipelines
- numpy 1.24.0+ - Numerical computations and array operations

**Machine Learning:**
- scikit-learn 1.3.0+ - Classification and regression models
  - GradientBoostingClassifier, RandomForestClassifier for game outcome prediction
  - GradientBoostingRegressor, ExtraTreesRegressor for player performance predictions
  - LogisticRegression for baseline classifiers
  - Pipeline, StandardScaler, SimpleImputer for preprocessing
  - Permutation importance for model explainability

**Model Explainability:**
- shap 0.44.0+ - SHAP values for feature importance analysis (optional, with fallback to sklearn)

**Visualization:**
- matplotlib 3.7.0+ - Plotting and chart generation for reports

**API Integration:**
- nba_api 1.4.0+ - Official NBA Stats API client for data fetching
- requests 2.28.0+ - HTTP library for external API calls (used by The Odds API integration)
- python-dotenv 1.0.0+ - Environment variable management for API keys

## Key Data Pipeline Dependencies

**nba_api Endpoints Used:**
- `leaguedashplayerstats` - Player regular season statistics
- `leaguedashplayerhottest` - Advanced player statistics
- `leaguegamelog` - Game-level logs (player and team)
- `leaguestandingsv3` - Season standings
- `playerhuSTLESDAT` - Hustle/tracking statistics (2015-16+)
- `leaguedashplayerbiostats` - Player biographical stats
- `leaguehuSTLESTAT` - Team hustle statistics
- Shot chart endpoint (via shot chart data module)

## Configuration & Environment

**Environment Variables:**
- `ODDS_API_KEY` - Required for daily sportsbook odds refresh (The Odds API)
  - Location: `.env` file at project root
  - Not required to run core pipeline; odds refresh is non-fatal if missing

**Logging:**
- Standard Python logging module
- Error logs written to `logs/pipeline_errors.log`
- Pipeline outputs to stdout/console

## Data Storage

**File Format:**
- CSV files for raw and processed data
  - Raw: `data/raw/[dataset_type]/` subdirectories
  - Processed: `data/processed/` consolidated files

**Local Storage Paths:**
- Raw data: `data/raw/` (player_stats, team_stats, player_game_logs, team_game_logs, standings, etc.)
- Processed data: `data/processed/` (cleaned, consolidated CSVs)
- Sportsbook odds: `data/odds/` (game_lines.csv, player_props.csv, model_vs_odds.csv)
- Features: `data/features/` (game_matchup_features.csv, player_features.csv, etc.)
- Model artifacts: `models/artifacts/` (trained model pickles, feature importances)

## Platform Requirements

**Development:**
- Python 3.14.3
- pip and virtual environment
- Windows (11 Pro) or compatible OS with bash/Unix shell
- Storage: ~400+ MB for nba_data.zip and processed datasets

**Production/Scheduling:**
- Windows Task Scheduler or equivalent (configured in `scripts/run_update.bat`)
- Scheduled daily execution capability
- Network access to NBA Stats API and optional The Odds API

## Build & Automation

**Orchestrators:**
- `update.py` - Main daily refresh script (runs current season only, ~2-5 min)
- `backfill.py` - Historical data backfill (one-time or occasional use)
- `scripts/run_update.bat` - Windows Task Scheduler batch wrapper

**Model Training:**
- `src/models/train_all_models.py` - Trains all three prediction models
- `src/models/predict_cli.py` - Inference CLI for predictions

**Data Preprocessing:**
- `src/processing/preprocessing.py` - Converts raw CSVs to processed consolidated files
- Called by update.py as part of daily pipeline

## Version Control & Documentation

**Git:**
- Repository: GitHub
- Ignores: `.env`, `logs/`, `__pycache__/`, `.venv/`

**Documentation:**
- README.md - Project overview
- docs/ directory - Architecture and integration notes
- Inline docstrings in Python modules

---

*Stack analysis: 2025-03-01*
