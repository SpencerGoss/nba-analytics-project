# Architecture

**Analysis Date:** 2026-03-01

## Pattern Overview

**Overall:** ETL Pipeline with Feature Engineering and Multi-Task ML Modeling

**Key Characteristics:**
- **Three-stage pipeline**: NBA API data fetch → CSV preprocessing → ML feature engineering
- **Decoupled task layers**: Each predictive task (game outcome, player performance, playoff odds) has isolated feature and model modules
- **Time-series respecting design**: Data leakage prevention via shifted rolling windows across all feature engineering
- **Daily/on-demand orchestration**: `update.py` coordinates data refresh; `backfill.py` handles historical backfills; models trained separately
- **Modular data fetching**: One file per API endpoint, enabling targeted incremental updates

## Layers

**Data Fetching:**
- Purpose: Retrieve raw NBA statistics from nba_api endpoints
- Location: `src/data/get_*.py` (20+ modules, one per endpoint)
- Contains: API call wrappers with retry logic, season-based fetching loops
- Depends on: nba_api, pandas, time/os utilities
- Used by: `update.py`, `backfill.py`
- Output: CSV files to `data/raw/{table_name}/{table_name}_{season}.csv`

**Raw Data Storage:**
- Purpose: Organize season-specific raw API responses before processing
- Location: `data/raw/` (24 subdirectories matching table names)
- Contains: One CSV per season per dataset type (e.g., `player_stats_202425.csv`, `player_stats_202324.csv`)
- Column naming: UPPERCASE (as returned by nba_api)
- Indexing: Season embedded in filename

**Preprocessing:**
- Purpose: Consolidate all seasonal CSVs into clean, unified processed tables
- Location: `src/processing/preprocessing.py`
- Contains: `run_preprocessing()` function that orchestrates 20+ table processing tasks
- Pattern: Load all seasons from one raw folder → apply column cleaning (lowercase_underscore) → rename specific columns per table → type conversion → deduplication → save to `data/processed/`
- Used by: `update.py`, `backfill.py` as final step before modeling
- Output: `data/processed/{table_name}.csv` (all seasons combined, one row type per file)

**Feature Engineering:**
- Purpose: Build ML-ready features from processed tables, respecting temporal ordering
- Location: `src/features/` (4 modules: team_game_features, player_features, era_labels, injury_proxy)
- Contains:
  - `team_game_features.py`: Computes rolling windows [5, 10, 20], strength of schedule, rest/fatigue, trend volatility, matchup differentials
  - `player_features.py`: Aggregates player rolling stats, season priors, role opportunity, opponent context
  - `era_labels.py`: Assigns era tags (e.g., "3-point revolution" post-2014)
  - `injury_proxy.py`: Estimates missing player minutes and usage availability
- Depends on: Processed CSV tables, era labeling system
- Used by: `game_outcome_model.py`, `player_performance_model.py` during training
- Output: `data/features/team_game_features.csv`, `data/features/game_matchup_features.csv`, `data/features/player_game_features.csv`

**Model Training:**
- Purpose: Train classification/regression models for each prediction task
- Location: `src/models/` (8 core modules: game_outcome_model, player_performance_model, playoff_odds_model, plus backtesting/calibration/explainability)
- Contains:
  - `game_outcome_model.py`: GradientBoosting/RandomForest classifier for `home_win` binary target, includes decision-threshold tuning, season-based expanding validation splits
  - `player_performance_model.py`: Per-target (pts, reb, ast) Ridge/GBM/RandomForest regressors, validation-season model selection
  - `playoff_odds_model.py`: Monte Carlo simulation for playoff/play-in/title probabilities
- Depends on: Feature tables from `src/features/`, sklearn pipelines
- Used by: `train_all_models.py` orchestrator
- Output: Pickled models to `models/artifacts/`, evaluation metrics logged

**Prediction/Inference:**
- Purpose: Apply trained models to new matchups and player games
- Location: `src/models/predict_cli.py` (CLI interface)
- Contains: `predict_game()`, `predict_player_next_game()` functions
- Used by: End-user CLI for interactive predictions
- Output: JSON-formatted predictions (probabilities for games, point/rebound/assist projections for players)

**Orchestration (Root Level):**
- Purpose: Coordinate multi-step pipelines
- Location: `update.py`, `backfill.py` (root-level scripts)
- `update.py`: Fetch current season only → preprocess → refresh odds; scheduled daily
- `backfill.py`: Fetch historical ranges (1996-1999 for stats, 1946-1999 for game logs) → preprocess; run once or rarely
- Output: Populated SQLite database + processed CSVs ready for modeling

## Data Flow

**Daily Update Flow:**

1. **Detect season**: `update.py` → `get_current_season_year()` returns integer season start year based on month
2. **Fetch current season**: Call all `get_*.py` functions for current season only (e.g., `get_player_stats_all_seasons(2025, 2025)`)
3. **Conditional fetches**:
   - If month >= 4: fetch playoff data (game logs, stats, advanced)
   - If year >= 2015: fetch hustle stats (tracking data)
4. **Rebuild all processed CSVs**: `preprocessing.py` → `run_preprocessing()` loads ALL raw seasons, consolidates, and regenerates `data/processed/` tables
5. **Refresh odds**: `refresh_odds_data()` calls external odds service (non-fatal if API key missing)
6. **Time**: ~2-5 minutes for current season fetch + 1-2 minutes for preprocessing = ~10-15 minutes total

**Historical Backfill Flow (Rare):**

1. Fetch stats endpoints for 1996-1999 (earliest reliable leaguedash data)
2. Fetch game logs for 1946-1999 (earliest available)
3. Fetch standings for 1979-1999 (most reliable from 1979-80 onward)
4. Preprocess all accumulated raw data
5. Time: ~30-45 minutes depending on network

**Model Training Flow:**

1. `train_all_models.py` orchestrator (or `--rebuild-features` to recompute feature tables first)
2. Call `build_team_game_features()` → outputs `data/features/team_game_features.csv` + `game_matchup_features.csv`
3. Call `build_player_game_features()` → outputs `data/features/player_game_features.csv`
4. Train game outcome model: Feature selection → candidate models → expanding validation splits → decision threshold tuning → final test evaluation
5. Train player models: Per-target (pts, reb, ast) → validation-season selection → final test evaluation
6. Simulate playoff odds: Monte Carlo on current standings
7. Time: ~5-10 minutes depending on feature data size

**Prediction/Inference Flow:**

1. User calls `predict_cli.py game --home BOS --away LAL` or `player --name "LeBron James"`
2. Load last trained game/player model from `models/artifacts/`
3. Construct feature row for matchup/player-game from current processed data
4. Apply trained model → return probabilities/projections as JSON

**State Management:**

- **Raw data state**: Season-keyed CSV files in `data/raw/{endpoint}/` with automatic overwrite on re-fetch (no versioning)
- **Processed data state**: Consolidated CSVs in `data/processed/` fully regenerated by `run_preprocessing()` each pipeline run
- **Feature data state**: `data/features/` regenerated before each model training (not persisted long-term, deterministic from processed data)
- **Model artifacts state**: Pickled sklearn pipelines + metadata in `models/artifacts/` (versioned by filename, e.g., `game_outcome_gb_20260301.pkl`)
- **Error logging**: `logs/pipeline_errors.log` appends all exceptions from `update.py` and `backfill.py` for monitoring

## Key Abstractions

**Fetcher Pattern:**
- Purpose: Standardized API call wrapper with retry logic and season-loop orchestration
- Examples: `src/data/get_player_stats.py`, `src/data/get_team_game_logs.py`
- Pattern:
  - Import nba_api endpoint class
  - Define `fetch_with_retry()` helper (3 retries, 10s delay between attempts)
  - Define `get_*_all_seasons(start_year, end_year)` public function
  - Loop years → format season as "YYYY-YY" → call API → save CSV
- Why: Centralizes error handling, rate-limiting respect, and season iteration

**CSV Consolidation Pattern:**
- Purpose: Merge all seasonal CSV files from one raw folder into a single processed table
- Examples: `preprocessing.py` uses `load_season_folder()` helper 20+ times
- Pattern:
  - Glob all files in raw folder matching prefix pattern
  - Extract season code from filename (e.g., "202425" from "player_stats_202425.csv")
  - Read each file → add season column → concatenate
  - Apply column normalization (lowercase, remove dashes/slashes)
  - Type conversion (player_id→int, dates→datetime, etc.)
  - Deduplication by full row
  - Save to `data/processed/`
- Why: Ensures consistent naming convention across data sources, season-trackable, handles optional tables gracefully (bio_stats, shot_chart)

**Rolling Feature Pattern:**
- Purpose: Compute lagged rolling statistics without time-series leakage
- Examples: `src/features/team_game_features.py` uses `_rolling_mean_shift()`, `src/features/player_features.py` uses `_compute_player_rolling()`
- Pattern:
  - Group by entity (player or team)
  - **Shift(1)** the source column first
  - Apply rolling window (e.g., `.rolling(window=5, min_periods=1)`)
  - Compute aggregation (mean, std, etc.)
  - Store as new column (e.g., `pts_roll5`, `pts_std10`)
- Why: Shift(1) ensures only prior games contribute to feature value for game N (prevents knowing outcome before prediction)
- Windows: [5, 10, 20] games as standard rolling horizons

**Feature Set Selection Pattern:**
- Purpose: Select numeric predictors while excluding leakage columns
- Examples: `game_outcome_model.py` → `get_feature_cols()`, `player_performance_model.py` → `get_feature_cols()`
- Pattern:
  - Define exclusion set (target, IDs, dates, outcome columns like `win`)
  - Filter DataFrame columns to numeric type only
  - Prioritize specific signal groups (e.g., rolling stats, opponent context, schedule features)
  - Return sorted unique list
- Why: Flexible to handle missing optional columns (bio_stats, clutch tables), prevents data leakage

**Model Selection Pattern:**
- Purpose: Choose best candidate model via validation-split evaluation
- Examples: `game_outcome_model.py` → train with expanding validation splits; `player_performance_model.py` → per-target validation-season selection
- Pattern for game outcome:
  - Split data by season (train on seasons S1..Sn-1, validate on Sn)
  - Expand window iteratively (first split: S1..S2 train, S3 validate; next: S1..S3 train, S4 validate)
  - For each split, try multiple candidate models (GradientBoosting, RandomForest, LogisticRegression)
  - Track metrics (accuracy, AUC, best threshold)
  - Select model with best validation performance
  - Retrain on full train set, evaluate on held-out test seasons
- Why: Season-based split respects temporal order; expanding validation simulates online learning

## Entry Points

**update.py:**
- Location: `/c/Users/spenc/OneDrive/Desktop/GIT/nba-analytics-project/update.py`
- Triggers: Manual execution or daily scheduler (Windows Task Scheduler via `scripts/run_update.bat`)
- Responsibilities:
  - Auto-detect current NBA season
  - Conditionally fetch regular/playoff/hustle data based on month and year
  - Rebuild all processed CSVs
  - Refresh sportsbook odds (non-fatal)
  - Log errors to `logs/pipeline_errors.log`

**backfill.py:**
- Location: `/c/Users/spenc/OneDrive/Desktop/GIT/nba-analytics-project/backfill.py`
- Triggers: Manual execution (one-time setup or rare historical backfill)
- Responsibilities:
  - Fetch historical stats (1996-1999)
  - Fetch historical game logs (1946-1999)
  - Fetch standings (1979-1999)
  - Rebuild all processed CSVs
  - Log errors

**train_all_models.py:**
- Location: `/c/Users/spenc/OneDrive/Desktop/GIT/nba-analytics-project/src/models/train_all_models.py`
- Triggers: Manual execution after data refresh or on schedule for periodic retraining
- Responsibilities:
  - Optionally rebuild feature tables (`--rebuild-features` flag)
  - Train game outcome classifier
  - Train player performance regressors (pts, reb, ast)
  - Simulate playoff odds
  - Log summary metrics

**predict_cli.py:**
- Location: `/c/Users/spenc/OneDrive/Desktop/GIT/nba-analytics-project/src/models/predict_cli.py`
- Triggers: User CLI invocation
- Responsibilities:
  - Parse `game` or `player` subcommand + arguments
  - Load trained model artifact from `models/artifacts/`
  - Construct feature row from current processed data
  - Apply model → output JSON predictions

## Error Handling

**Strategy:** Non-fatal exception handling with logging; pipeline continues on optional data failures.

**Patterns:**

- **API fetch errors**: `fetch_with_retry()` in each `get_*.py` retries 3 times with 10-second delays; if all fail, skips season and logs to `logs/pipeline_errors.log`
- **Missing optional tables**: `preprocessing.py` gracefully skips tables with no raw files (e.g., `shot_chart` requires explicit run of 3-4 hour script)
- **Odds API failure**: `refresh_odds_data()` in `update.py` returns False on error; main flow continues (non-blocking)
- **Model training failure**: Unhandled exceptions in `train_all_models.py` halt training; orchestrator should catch and retry

**Logging:**
- Root errors logged to `logs/pipeline_errors.log` by `log_pipeline_error()` helper in orchestrators
- Format: `[YYYY-MM-DD HH:MM:SS] {script_name} failed: {error}`

## Cross-Cutting Concerns

**Logging:**
- `update.py` and `backfill.py` print timestamped progress to stdout; errors appended to `logs/pipeline_errors.log`
- Feature engineering modules print row counts to stdout
- Model training prints candidate metrics, validation splits, and final test results to stdout

**Validation:**
- Type conversion in preprocessing enforces schema (player_id→int, game_date→datetime)
- Feature selection filters to numeric columns only, preventing string/object columns from entering sklearn pipelines
- Data leakage prevention via shift(1) before rolling window computation in all feature modules

**Authentication:**
- NBA API: Uses standard Mozilla User-Agent headers defined in `HEADERS` constant in each data script
- Odds API: Respects `ODDS_API_KEY` environment variable; skips if not set

**Configuration:**
- Hard-coded constants in module docstrings and at top of files (HEADERS, ROLL_WINDOWS, TEST_SEASONS, ARTIFACTS_DIR)
- Season format: integer codes (202425 for 2024-25 season) throughout pipeline
- Paths: Relative to project root (e.g., `data/raw/`, `models/artifacts/`) enable easy migration

---

*Architecture analysis: 2026-03-01*
