# Codebase Structure

**Analysis Date:** 2026-03-01

## Directory Layout

```
nba-analytics-project/
├── update.py                          # Daily orchestrator: fetch current season → preprocess → refresh odds
├── backfill.py                        # Historical backfill: fetch 1946-1999 data → preprocess
├── README.md                          # Project overview
├── requirements.txt                   # Python dependencies
│
├── src/                               # All application code
│   ├── data/                          # Data fetching layer (20+ API wrapper modules)
│   │   ├── get_player_stats.py        # Player basic stats all seasons
│   │   ├── get_player_stats_advanced.py    # Advanced player stats (PER, USG%, etc.)
│   │   ├── get_player_stats_clutch.py      # Clutch-time player stats
│   │   ├── get_player_stats_scoring.py     # Scoring-focused player stats
│   │   ├── get_player_stats_playoffs.py    # Player playoff stats (+ advanced)
│   │   ├── get_player_game_logs.py   # Individual player game-level logs
│   │   ├── get_player_game_logs_playoffs.py   # Player playoff game logs
│   │   ├── get_player_master.py      # Player master reference table
│   │   ├── get_player_bio_stats.py   # Player biography (height, weight, age)
│   │   ├── get_player_positions.py   # Player position lookup (supplemental)
│   │   ├── get_team_stats.py         # Team basic stats all seasons
│   │   ├── get_team_stats_advanced.py      # Advanced team stats
│   │   ├── get_team_stats_playoffs.py      # Team playoff stats (+ advanced)
│   │   ├── get_team_game_logs.py    # Team game-level logs (regular season)
│   │   ├── get_team_game_logs_playoffs.py  # Team playoff game logs
│   │   ├── get_standings.py         # League standings by season
│   │   ├── get_teams.py             # Static team reference (id, name, abbreviation)
│   │   ├── get_hustle_stats.py      # Tracking stats (2015-16+): speed, touches, contested
│   │   ├── get_odds.py              # Sportsbook odds refresh (external API integration)
│   │   ├── get_shot_chart.py        # Shot-level detail (3-4 hours to fetch all seasons)
│   │   └── __init__.py              # (empty)
│   │
│   ├── processing/                    # Data preprocessing layer
│   │   ├── preprocessing.py          # Consolidate all raw CSVs → clean & unified processed tables
│   │   └── __pycache__/
│   │
│   ├── features/                      # Feature engineering layer
│   │   ├── team_game_features.py     # Team rolling stats, strength of schedule, rest/fatigue, matchup differentials
│   │   ├── player_features.py        # Player rolling stats, season priors, role opportunity, opponent context
│   │   ├── era_labels.py             # Assign era tags (e.g., "3-point revolution" post-2014)
│   │   ├── injury_proxy.py           # Estimate missing player minutes & usage availability
│   │   ├── __init__.py               # Feature module marker
│   │   └── __pycache__/
│   │
│   ├── models/                        # Model training & prediction layer
│   │   ├── README.md                 # Model workflows documentation
│   │   ├── game_outcome_model.py     # Binary classifier: home_win (GBM, RF, LR candidates)
│   │   ├── player_performance_model.py   # Regressors per target (pts, reb, ast)
│   │   ├── playoff_odds_model.py     # Monte Carlo playoff/play-in/title simulation
│   │   ├── train_all_models.py       # Consolidated training orchestrator
│   │   ├── predict_cli.py            # CLI interface for game & player predictions
│   │   ├── backtesting.py            # Walk-forward validation robustness checks
│   │   ├── calibration.py            # Probability calibration & ECE/Brier analysis
│   │   ├── model_explainability.py   # SHAP & permutation importance reports
│   │   ├── run_evaluation.py         # Runs analysis suite post-training
│   │   ├── numpy_gbm.py              # Custom gradient boosting implementation
│   │   ├── __init__.py               # Model module marker
│   │   └── __pycache__/
│   │
│   └── __pycache__/
│
├── data/                              # All pipeline data outputs
│   ├── raw/                           # Raw API responses (season-keyed CSV files)
│   │   ├── player_stats/             # e.g., player_stats_202425.csv, player_stats_202324.csv
│   │   ├── player_stats_advanced/    # Advanced player stats by season
│   │   ├── player_stats_clutch/      # Clutch-time player stats by season
│   │   ├── player_stats_scoring/     # Scoring-focused player stats by season
│   │   ├── player_stats_playoffs/    # Player playoff stats + advanced by season
│   │   ├── player_stats_advanced_playoffs/   # Advanced player playoff stats
│   │   ├── player_game_logs/         # Player game-level logs by season
│   │   ├── player_game_logs_playoffs/       # Player playoff game logs by season
│   │   ├── player_hustle_stats/      # Tracking stats by season (2015-16+)
│   │   ├── player_bio_stats/         # Player bio stats (optional)
│   │   ├── players/                  # Player master reference (single file)
│   │   ├── team_stats/               # Team basic stats by season
│   │   ├── team_stats_advanced/      # Advanced team stats by season
│   │   ├── team_stats_playoffs/      # Team playoff stats by season
│   │   ├── team_stats_advanced_playoffs/    # Advanced team playoff stats
│   │   ├── team_game_logs/           # Team game-level logs by season
│   │   ├── team_game_logs_playoffs/  # Team playoff game logs by season
│   │   ├── team_hustle_stats/        # Team tracking stats by season (2015-16+)
│   │   ├── standings/                # League standings by season
│   │   ├── teams/                    # Static team reference (single file)
│   │   ├── shot_chart/               # Shot-level detail (optional, long fetch)
│   │   └── odds/                     # Sportsbook odds snapshots
│   │
│   ├── processed/                     # Consolidated, cleaned tables (all seasons combined)
│   │   ├── players.csv               # All players ever (from player_master, deduplicated)
│   │   ├── player_stats.csv          # All player season stats (1996-97+)
│   │   ├── player_stats_advanced.csv
│   │   ├── player_stats_clutch.csv
│   │   ├── player_stats_scoring.csv
│   │   ├── player_stats_playoffs.csv
│   │   ├── player_stats_advanced_playoffs.csv
│   │   ├── player_game_logs.csv      # All player game logs (1946-47+)
│   │   ├── player_game_logs_playoffs.csv
│   │   ├── player_hustle_stats.csv   # (2015-16+)
│   │   ├── player_bio_stats.csv      # (optional, if fetched)
│   │   ├── teams.csv                 # All teams (static reference, optional)
│   │   ├── team_stats.csv            # All team season stats (1996-97+)
│   │   ├── team_stats_advanced.csv
│   │   ├── team_stats_playoffs.csv
│   │   ├── team_stats_advanced_playoffs.csv
│   │   ├── team_game_logs.csv        # All team game logs (1946-47+)
│   │   ├── team_game_logs_playoffs.csv
│   │   ├── team_hustle_stats.csv     # (2015-16+)
│   │   ├── standings.csv             # All standings by season (1979-80+)
│   │   └── shot_chart.csv            # (optional, if fetched)
│   │
│   ├── features/                      # ML feature tables (regenerated before each training)
│   │   ├── team_game_features.csv    # Team rolling stats + context (one row = team-game before matchup)
│   │   ├── game_matchup_features.csv # Matchup-level differential features (home vs away)
│   │   └── player_game_features.csv  # Player rolling stats + opponent/team context (one row = player-game before game)
│   │
│   └── odds/                          # Sportsbook odds snapshots
│       └── (odds data storage, structure varies by API)
│
├── models/                            # Model artifacts & evaluation reports
│   ├── artifacts/                     # Trained pickle files + metadata
│   │   ├── game_outcome_gb_20260301.pkl     # Example: serialized GradientBoosting model
│   │   ├── game_outcome_metadata.json       # Model metadata (features used, params, test metrics)
│   │   ├── player_pts_gbr_20260301.pkl      # Player PTS regressor
│   │   ├── player_pts_metadata.json
│   │   ├── player_reb_gbr_20260301.pkl      # Player REB regressor
│   │   ├── player_reb_metadata.json
│   │   ├── player_ast_gbr_20260301.pkl      # Player AST regressor
│   │   └── player_ast_metadata.json
│   │
│   └── (reports/ subdirectories created by run_evaluation.py)
│
├── reports/                           # Evaluation & analysis outputs
│   ├── backtest/                      # Walk-forward validation results
│   ├── calibration/                   # Probability calibration plots & metrics
│   └── explainability/                # SHAP plots, feature importances
│
├── scripts/                           # Utility & deployment scripts
│   ├── fetch_odds.py                 # External script for sportsbook API integration
│   ├── run_update.bat                # Windows Task Scheduler batch wrapper for daily update.py execution
│   └── package_for_colab.py          # Utility to package project for Google Colab
│
├── notebooks/                         # (Jupyter notebooks, if any — not primary code)
│   └── (optional analysis notebooks)
│
├── docs/                              # Documentation
│   ├── agent_starter_prompts.md
│   └── agent_task_plan.md
│
├── logs/                              # Pipeline logs
│   ├── pipeline_errors.log            # Errors from update.py & backfill.py (append-only)
│   └── update.log                     # Full output log from scheduled run_update.bat
│
├── .planning/                         # GSD planning documents
│   └── codebase/                      # Generated architecture/structure analyses
│       ├── ARCHITECTURE.md
│       ├── STRUCTURE.md
│       ├── CONVENTIONS.md
│       └── TESTING.md
│
├── .env                               # Environment variables (ODDS_API_KEY, etc.)
├── .gitignore                         # Version control exclusions
├── .git/                              # Version control
├── .venv/                             # Python virtual environment
│
└── __pycache__/                       # Python bytecode cache
```

## Directory Purposes

**src/data/:**
- Purpose: API data fetching layer — one module per NBA Stats endpoint
- Contains: Functions named `get_*_all_seasons(start_year, end_year)` that orchestrate year loops, retry logic, and CSV output
- Key files: 20+ `get_*.py` modules following standardized pattern with nba_api imports

**src/processing/:**
- Purpose: Raw-to-processed data transformation
- Contains: Single `preprocessing.py` orchestrator with 20+ table-loading tasks
- Pattern: Load seasonal CSVs from `data/raw/` → clean columns → rename/convert types → deduplicate → save to `data/processed/`

**src/features/:**
- Purpose: Feature engineering from processed tables → ML-ready feature sets
- Contains: Four specialized modules for different feature types
- Key outputs: `team_game_features.csv`, `game_matchup_features.csv`, `player_game_features.csv`

**src/models/:**
- Purpose: Model training, evaluation, and prediction
- Contains: Core task models (game_outcome, player_performance, playoff_odds) + supporting analysis (backtesting, calibration, explainability)
- Key files: `train_all_models.py` (orchestrator), `predict_cli.py` (inference)

**data/raw/:**
- Purpose: Store season-keyed raw API responses
- Structure: Subdirectory per table name (e.g., `player_stats/`, `team_game_logs/`)
- File naming: `{table_name}_{season_code}.csv` (e.g., `player_stats_202425.csv`)
- Lifecycle: Overwritten on re-fetch; no versioning

**data/processed/:**
- Purpose: Clean, consolidated tables ready for feature engineering
- Structure: Flat directory with one CSV per data type (all seasons combined)
- File naming: `{table_name}.csv` (e.g., `player_stats.csv` contains all seasons from 1996-97+)
- Columns: lowercase_underscore format with standardized types (int, float, datetime)
- Lifecycle: Fully regenerated by `run_preprocessing()` each pipeline run

**data/features/:**
- Purpose: Staging area for ML feature tables
- Structure: Three core feature CSVs + supporting files
- Lifecycle: Regenerated before each model training; not long-term versioned

**models/artifacts/:**
- Purpose: Store serialized trained models + metadata
- Structure: One `.pkl` file per model + one `_metadata.json` per model
- Naming: `{model_name}_{algorithm}_{timestamp}.pkl` (e.g., `game_outcome_gb_20260301.pkl`)
- Lifecycle: Manually managed (new file created per training run)

**logs/:**
- Purpose: Error and execution logs
- Structure: `pipeline_errors.log` (append-only) and `update.log` (full run output)
- Lifecycle: Append indefinitely; monitor for cleanup

## Key File Locations

**Entry Points:**
- `update.py`: Root-level daily orchestrator
- `backfill.py`: Root-level historical backfill orchestrator
- `src/models/train_all_models.py`: Model training entrypoint
- `src/models/predict_cli.py`: Prediction CLI entrypoint

**Configuration:**
- `requirements.txt`: Python package dependencies
- `.env`: Environment variables (ODDS_API_KEY, etc.) — not tracked in git
- `src/data/get_*.py`: Hard-coded headers, retry logic, season ranges per endpoint

**Core Logic:**
- `src/processing/preprocessing.py`: CSV consolidation orchestrator (20+ table loads)
- `src/features/team_game_features.py`: Team-level feature engineering
- `src/features/player_features.py`: Player-level feature engineering
- `src/models/game_outcome_model.py`: Game outcome binary classifier
- `src/models/player_performance_model.py`: Player stat regressors (pts, reb, ast)
- `src/models/playoff_odds_model.py`: Playoff odds simulation

**Testing:**
- No dedicated test files or test framework detected; validation happens via pipeline execution logs

**Data Reference:**
- `data/raw/players/player_master.csv`: Player ID → name mapping (fetched via `get_player_master.py`)
- `data/raw/teams/teams.csv`: Team ID → abbreviation mapping (fetched via `get_teams.py`)
- `data/processed/players.csv`: Cleaned player reference (all players, deduplicated)
- `data/processed/teams.csv`: Cleaned team reference (all teams)

## Naming Conventions

**Files:**

- **Data fetchers**: `get_*.py` (e.g., `get_player_stats.py`, `get_team_game_logs.py`)
- **Raw data**: `{table_name}_{season_code}.csv` (e.g., `player_stats_202425.csv`)
  - Season code format: integer YYYYYY (202425 for 2024-25 season), not "YYYY-YY"
  - One file per table per season in subdirectory structure
- **Processed data**: `{table_name}.csv` (e.g., `player_stats.csv`)
- **Features**: `{domain}_{context}.csv` (e.g., `team_game_features.csv`, `game_matchup_features.csv`, `player_game_features.csv`)
- **Models**: `{task}_{algorithm}_{timestamp}.pkl` (e.g., `game_outcome_gb_20260301.pkl`)

**Directories:**

- **Data layer**: `src/data/` (all data fetching)
- **Processing**: `src/processing/` (CSV consolidation)
- **Features**: `src/features/` (feature engineering)
- **Models**: `src/models/` (training & inference)
- **Raw storage**: `data/raw/{endpoint_name}/` (e.g., `data/raw/player_stats/`, `data/raw/team_game_logs/`)
- **Processed storage**: `data/processed/` (consolidated tables)
- **Feature storage**: `data/features/` (ML feature sets)
- **Model artifacts**: `models/artifacts/` (serialized models)

**Functions:**

- **Data fetchers**: `get_*_all_seasons(start_year, end_year)` (e.g., `get_player_stats_all_seasons()`)
- **Feature builders**: `build_*_features()` or `build_*_dataset()` (e.g., `build_team_game_features()`)
- **Preprocessing**: `run_preprocessing()` (single orchestrator function)
- **Model training**: `train_*_model()` (e.g., `train_game_outcome_model()`)
- **Prediction**: `predict_*()` (e.g., `predict_game()`, `predict_player_next_game()`)

**Columns (Processed Data):**

- **Format**: lowercase_underscore (not UPPERCASE from raw API)
- **Examples**: `player_id`, `team_id`, `game_id`, `game_date`, `pts`, `reb`, `ast`, `plus_minus`, `fg_pct`
- **Standardized renames** (see `preprocessing.py`):
  - Raw `person_id` → Processed `player_id`
  - Raw `DISPLAY_FIRST_LAST` → Processed `player_name`
  - Raw `FROM_YEAR` → Processed `from_season`
  - Raw `teamid` → Processed `team_id`
  - Raw `wins` → Processed `w`
  - Raw `winpct` → Processed `w_pct`

**Feature Columns:**

- **Rolling windows**: `{stat}_roll{window}` (e.g., `pts_roll5`, `reb_roll10`, `fg_pct_roll20`)
- **Rolling volatility**: `{stat}_std{window}` (e.g., `pts_std10`)
- **Opponent context**: `opp_{stat}` (e.g., `opp_pts`, `opp_fg_pct`)
- **Differential features**: `diff_{stat}` (e.g., `diff_pts_roll5` = home rolling pts − away rolling pts)
- **Era tags**: `era` (e.g., "pre-3pt", "3pt-revolution", "modern")
- **Injury/availability**: `*_missing_minutes`, `*_star_player_out`, `*_rotation_availability`

## Where to Add New Code

**New Data Source (API Endpoint):**
- Create: `src/data/get_{endpoint_name}.py`
- Template: Copy structure from existing `get_*.py` (HEADERS, fetch_with_retry, get_*_all_seasons function)
- Output: Save season CSVs to `data/raw/{endpoint_name}/{endpoint_name}_{season}.csv`
- Integration: Add to `update.py` or `backfill.py` orchestrator list
- Processing: Add table-loading task to `src/processing/preprocessing.py` (load_season_folder call + cleanup)

**New Feature Engineering:**
- Create: New function in `src/features/` (or new module if large)
- Pattern: Load from processed CSVs → compute rolling/aggregate/contextual features → output to `data/features/{name}.csv`
- Integration: Call from `build_*_features()` or from `train_all_models.py` feature engineering step
- Leakage prevention: Use shift(1) before rolling windows; exclude future outcomes

**New Model Task:**
- Create: `src/models/{task}_model.py`
- Pattern: Define feature selection function → candidate models → training logic → evaluation → artifact saving
- Integration: Add to `train_all_models.py` orchestrator
- Artifacts: Save pickled model to `models/artifacts/{task}_{algorithm}.pkl`
- Prediction: Add inference function, optionally expose via `predict_cli.py`

**New Utility Script:**
- Create: `scripts/{purpose}.py` or add to `scripts/` directory
- Integration: Call from `update.py` / `backfill.py` if part of pipeline, or document in README for manual execution

**Shared Helpers:**
- Location: Add to existing module if small, or create `src/utils/` for larger shared library
- Example: `fetch_with_retry()` is duplicated across data fetchers — could be centralized to `src/utils/api.py`

## Special Directories

**data/raw/:**
- Purpose: Store season-keyed API responses before processing
- Generated: By `update.py`, `backfill.py`, and individual `get_*.py` scripts
- Committed: No — .gitignore excludes `data/`
- Lifecycle: Overwritten on re-fetch; not versioned
- Size consideration: Can grow to GBs if all historical seasons present

**data/processed/:**
- Purpose: Consolidated, cleaned tables
- Generated: By `run_preprocessing()` from raw files
- Committed: No — generated from raw
- Lifecycle: Fully regenerated each pipeline run
- Dependency: Must exist before feature engineering; must exist before modeling

**data/features/:**
- Purpose: Feature tables for ML training
- Generated: By feature engineering modules in `src/features/`
- Committed: No — generated from processed data
- Lifecycle: Regenerated before each model training
- Dependency: Must exist before model training

**models/artifacts/:**
- Purpose: Serialized trained models
- Generated: By `train_all_models.py` and component model training functions
- Committed: No (pickle files not portable across Python versions)
- Lifecycle: Manually managed; new file per training run
- Dependency: Must exist before prediction/inference

**logs/:**
- Purpose: Error and execution logs
- Generated: By `update.py`, `backfill.py`
- Committed: No — .gitignore excludes logs/
- Lifecycle: Append indefinitely (monitor for growth)

---

*Structure analysis: 2026-03-01*
