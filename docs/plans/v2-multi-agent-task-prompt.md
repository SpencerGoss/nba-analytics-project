# NBA Analytics Project — v2.0 Multi-Agent Task Prompt

Paste everything below the line into Claude Code as a single prompt.

---

## PROMPT START — COPY FROM HERE

You are working on an NBA analytics project that has completed its v1.0 milestone. The project has a working end-to-end pipeline: data ingestion from NBA API, feature engineering, game outcome prediction (66.8% accuracy), player performance forecasting, ATS betting model (51.2% — below 52.4% vig breakeven), and a prediction store. All code is Python with pandas/scikit-learn.

Read these files first to understand the full project state:
- `.planning/STATE.md` (current progress)
- `.planning/codebase/CONCERNS.md` (all known issues — tech debt, bugs, fragile areas, test gaps)
- `docs/project_overview.md` (architecture overview)
- `docs/PIPELINE.md` (pipeline stages)
- `README.md`

Then execute the following 7 tasks using parallel subagents where possible. Each task is independent unless noted. Use git worktrees for isolation so agents don't conflict with each other.

---

### TASK 1: Shared API Client Refactor
**Priority: High | Files: `src/data/`**
**Branch: `v2/shared-api-client`**

All 20+ data fetcher scripts in `src/data/` duplicate the same retry logic, headers, and error handling. Every file independently defines `HEADERS`, `MAX_RETRIES`, `RETRY_DELAY`, and `fetch_with_retry()`. This is documented in `.planning/codebase/CONCERNS.md` under "Duplicated Error Handling and Logging" and "API Headers Hardcoded Across Scripts".

**Before coding — review the existing pattern:**
Read `src/data/get_player_stats.py` as a reference. Note the duplicated `HEADERS` dict (lines 8-17), `MAX_RETRIES`/`RETRY_DELAY` constants, and the `fetch_with_retry()` function. Then scan 3-4 other `get_*.py` files to confirm they all copy this same pattern. Understand what varies between files (just the endpoint call) vs what's identical (everything else).

**Implementation:**
1. Create `src/data/api_client.py` with:
   - A single `fetch_with_retry(fetch_fn, label, max_retries=3, retry_delay=10)` function
   - Centralized `HEADERS` dict and rate limit config
   - Return a structured result: `{"success": bool, "data": pd.DataFrame or None, "error": str or None}` instead of raw None on failure
   - Use Python `logging` module instead of print statements
   - Include a `configure_logging(level=logging.INFO)` helper so scripts can set verbosity
2. Refactor every `src/data/get_*.py` file to import from `api_client.py` instead of defining their own retry logic. The list of files to update:
   - `get_game_log.py`, `get_hustle_stats.py`, `get_odds.py`, `get_player_bio_stats.py`
   - `get_player_game_logs.py`, `get_player_game_logs_playoffs.py`, `get_player_master.py`
   - `get_player_positions.py`, `get_player_stats.py`, `get_player_stats_advanced.py`
   - `get_player_stats_clutch.py`, `get_player_stats_playoffs.py`, `get_player_stats_scoring.py`
   - `get_shot_chart.py`, `get_standings.py`, `get_team_game_logs_playoffs.py`
   - `get_team_stats.py`, `get_team_stats_advanced.py`, `get_team_stats_playoffs.py`, `get_teams.py`
3. Keep the external interface the same — each script should still be runnable standalone with `python src/data/get_player_stats.py`
4. Don't touch `src/data/external/` (those are scrapers, not NBA API callers)
5. **Verify** nothing breaks by importing each module: run `python -c "from src.data import get_player_stats"` for every refactored file

---

### TASK 2: Unit Test Suite
**Priority: High | New dir: `tests/`**
**Branch: `v2/unit-tests`**

There are zero tests in this project. This is the biggest risk — one nba_api change or pandas update and everything breaks silently.

**Before coding — design the test strategy:**
Read the source files you'll be testing to understand their interfaces:
- `src/features/team_game_features.py` (rolling windows, joins, shift logic)
- `src/features/injury_proxy.py` (merge_asof, absent detection, star_player_out)
- `src/processing/preprocessing.py` (column cleaning, type coercion, concatenation)

Decide on these design choices before writing anything:
- **Fixtures over factories**: Use pytest fixtures with small synthetic DataFrames (5-10 rows). Don't depend on actual data files.
- **Mock boundaries**: Mock only external I/O (file reads, API calls). Don't mock pandas or the actual computation logic — that's what you're testing.
- **Isolation**: Each test should create its own data. No shared mutable state between tests.
- **Naming**: `test_{function_name}_{scenario}_{expected_result}` pattern (e.g., `test_rolling_mean_first_game_returns_nan`)

**Implementation:**
1. Create `tests/` directory with `conftest.py` for shared fixtures (synthetic team game logs, player game logs, raw stat DataFrames)
2. `tests/test_team_game_features.py`:
   - Test rolling window calculations produce correct values on synthetic data
   - **Critical**: Test that `shift(1)` is applied before rolling — for row N, features must only use data from rows 0 through N-1. This is the no-leakage guarantee.
   - Test that inner joins don't silently drop rows — assert `len(result) == len(input)` or document expected drops
   - Test edge cases: first game of season (should have NaN rolling features), back-to-back games (days_rest=1), team with only 1 game
3. `tests/test_injury_proxy.py`:
   - Test absent rotation player detection: create game logs where player A plays games 1-5, misses 6-8, plays 9-10. Verify games 6-8 flag player A as absent.
   - Test `missing_minutes` and `star_player_out` flags compute correctly
   - Test `merge_asof` with `MAX_STALE_DAYS=25`: player absent for 30 days should NOT show as missing (trade/retirement), player absent for 10 days SHOULD show as missing (injury)
4. `tests/test_preprocessing.py`:
   - Test column cleaning (CamelCase → snake_case, spaces → underscores)
   - Test type coercion: DataFrame with NaN in an integer column should not silently drop rows
   - Test duplicate row removal preserves unique rows
   - Test multi-file concatenation: 3 small CSVs with 5 rows each → 15 row combined DataFrame
5. `tests/test_api_client.py` (depends on Task 1 completing first — skip if api_client.py doesn't exist yet):
   - Test retry logic with a mock callable that raises Exception N times then returns a DataFrame
   - Test structured result has `success=True` and `data=DataFrame` on success
   - Test structured result has `success=False` and `error=str` when max retries exceeded
6. Add `pyproject.toml` section:
   ```toml
   [tool.pytest.ini_options]
   testpaths = ["tests"]
   python_files = "test_*.py"
   python_functions = "test_*"
   ```
7. Run `pytest -v` and ensure all tests pass. Fix any failures before committing.

---

### TASK 3: Incremental Preprocessing
**Priority: Medium | Files: `src/processing/preprocessing.py`, `update.py`**
**Branch: `v2/incremental-preprocessing`**
**Depends on: Task 1 should be done first (stable API client)**

`preprocessing.py` rebuilds ALL processed CSVs from scratch every run (~25 seasons of data) even when only the current season changed. Documented in CONCERNS.md under "Preprocessing Always Rebuilds All CSVs".

**Before coding — design the change detection approach:**
Read `src/processing/preprocessing.py` fully. Map out:
- What raw file paths look like: `data/raw/{endpoint_name}/{season_code}.csv`
- What processed file paths look like: `data/processed/{table_name}.csv`
- How the current rebuild works: reads all raw CSVs → concatenates → cleans → writes processed

Choose this approach for change detection: **file mtime comparison**. For each processed table, check if any of its source raw CSVs have a newer modification time than the processed output. This is simple, reliable, and doesn't require maintaining a separate manifest.

**Implementation:**
1. Add a `get_stale_seasons(raw_dir, processed_path)` function that:
   - Lists all raw CSVs in the endpoint's raw directory
   - Gets mtime of the processed output file (if it exists)
   - Returns list of season files newer than the processed file
   - Returns ALL seasons if processed file doesn't exist (first run / full rebuild)
2. Modify each table's processing function to accept an optional `seasons_to_rebuild` parameter:
   - If provided, only read and process those season files
   - Merge new processed rows with existing processed CSV (drop duplicates on primary key)
   - If not provided, rebuild everything (existing behavior)
3. Add a `--full-rebuild` CLI flag: `python src/processing/preprocessing.py --full-rebuild`
4. Add logging: `"Skipping player_stats (no changes)"` or `"Rebuilding player_stats: 2024-25, 2025-26 (2 seasons updated)"`
5. Wire into `update.py`: daily runs use incremental mode by default
6. Wire into `backfill.py`: always use full rebuild mode
7. **Verify**: Run `update.py` twice in a row — second run should skip everything and finish in seconds

---

### TASK 4: ATS Model Improvement
**Priority: High | Files: `src/models/ats_model.py`, `src/features/ats_features.py`**
**Branch: `v2/ats-model-v2`**

The ATS model is at 51.2% accuracy, below the 52.4% vig breakeven. Current approach is logistic regression on a basic feature set. Documented in `docs/project_overview.md` section 7 and CONCERNS.md under "Model Limitations".

**Before coding — explore the data and understand what you're working with:**

Step 1: Profile the ATS feature table.
```python
import pandas as pd
df = pd.read_csv('data/features/game_ats_features.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.describe())
print(f"Target distribution: {df['covers_spread'].value_counts(normalize=True)}")
# Check for class imbalance, null rates, feature correlations
print(df.isnull().sum().sort_values(ascending=False).head(20))
print(df.corr()['covers_spread'].sort_values(ascending=False).head(20))
```

Step 2: Read current model code and baseline results.
- Read `src/models/ats_model.py` fully — understand the expanding-window validation
- Read `reports/ats_backtest_summary.txt` for current metrics
- Read `src/features/ats_features.py` to see what features exist now

Step 3: Identify candidate improvements based on the data profile. Specifically look for:
- Are there features with near-zero correlation to covers_spread? (candidates for removal)
- Is the spread distribution balanced or skewed? (might need spread-bucket features)
- Are there obvious signals in the correlation matrix that aren't being used?

**Implementation — run experiments systematically:**

1. **Feature engineering** — add these to `src/features/ats_features.py`:
   - `home_ats_record_5g` / `away_ats_record_5g`: Rolling ATS cover rate over last 5 and 10 games per team
   - `spread_bucket`: Categorical buckets (-10+, -7 to -3, -3 to +3, +3 to +7, +7+)
   - `home_dog`: Binary flag when home team is underdog (spread > 0)
   - `rest_advantage`: home days_rest minus away days_rest
   - `record_vs_spread_expectation`: season win% minus implied win% from average spread
   - **Important**: All new features must use `shift(1)` or equivalent to prevent leakage. No future data.

2. **Model alternatives** — try in `src/models/ats_model.py`:
   - Keep logistic regression as baseline
   - Add GradientBoostingClassifier (sklearn)
   - Add MLPClassifier (sklearn, 2 hidden layers, early stopping)
   - Try XGBoost if `xgboost` is installed, skip if not
   - **Use the exact same expanding-window splits** for all models — do not change the validation methodology

3. **Feature selection**:
   - Run permutation importance on the best model
   - Try dropping bottom 30% of features by importance and retrain
   - Compare: does fewer features help or hurt?

4. **Report results**:
   - Create `reports/ats_model_v2_experiments.txt` with a table:
     ```
     Model              | Features    | Mean Val Acc | Holdout Acc | Holdout ROI
     ---------------------|-------------|-------------|-------------|------------
     LogReg (baseline)    | v1 original | 0.5287      | 0.5120      | +0.67%
     LogReg              | v2 expanded | ...         | ...         | ...
     GradientBoosting    | v2 expanded | ...         | ...         | ...
     MLP                 | v2 expanded | ...         | ...         | ...
     Best + feat select  | v2 pruned   | ...         | ...         | ...
     ```
   - If any model beats 52.4% holdout accuracy, save it as new production model
   - If nothing beats breakeven, document findings honestly — what worked, what didn't, what to try next

5. **Validate results**:
   - Check for data leakage: no features should use information from the game being predicted
   - Check for survivorship bias: are you only testing on games where odds data exists?
   - Verify expanding window never trains on future data: for each split, assert max(train_dates) < min(test_dates)

---

### TASK 5: Data Integrity Validation
**Priority: Medium | New file: `src/validation/data_integrity.py`**
**Branch: `v2/data-validation`**

There's no validation anywhere in the pipeline. Silent data corruption propagates through everything. Documented across multiple sections of CONCERNS.md.

**Before coding — understand what can go wrong at each stage:**
Read CONCERNS.md sections: "Inconsistent Data Type Coercion", "Feature Engineering Pipeline Has Silent Data Joins", "Missing Injury Features in Game Outcome Model". These tell you exactly what kinds of corruption happen silently.

The validation framework should check these things at each stage:

**Implementation:**
1. Create `src/validation/__init__.py` and `src/validation/data_integrity.py`
2. Implement stage-specific validators:

   **`validate_fetch(season)`** — runs after data fetch:
   - Every expected raw CSV exists in `data/raw/{endpoint}/` for the target season
   - Each file has > 0 rows
   - No file is suspiciously small (< 10 rows for a full season — probably a failed fetch)
   - Log: "PASS: fetch validation — 18 tables, all present for season 202526"

   **`validate_preprocess()`** — runs after preprocessing:
   - `game_id` is unique in each processed game-level table
   - No dates in the future (compared to today)
   - Season codes fall within expected range (194647 to current)
   - No column has > 95% null values (flag which column and what % null)
   - Row counts are within expected range (e.g., team_game_logs should have ~1300 rows per season)
   - Log each check: "PASS: game_id uniqueness in team_game_logs" or "FAIL: player_stats.fg3_pct is 97.2% null"

   **`validate_features()`** — runs after feature engineering:
   - `data/features/game_matchup_features.csv` row count is within 5% of `data/processed/team_game_logs.csv` row count (some drop is OK from joins, but >5% means something broke)
   - All expected feature columns present (check against a list defined in the validator)
   - Value range checks: win percentages between 0.0 and 1.0, no negative shooting percentages, days_rest >= 0, rolling means within plausible NBA ranges
   - Injury features (`home_missing_minutes`, `away_missing_minutes`, etc.) are NOT all null — this was a known bug documented in CONCERNS.md
   - Log: "PASS: 142 features present, all ranges valid" or "FAIL: home_missing_minutes is 100% null"

   **`validate_train()`** — runs after model training:
   - `models/artifacts/game_outcome_model.pkl` exists and is loadable via joblib
   - Feature importance CSV has same column count as model's expected features
   - No feature importance is exactly 0.0 for all features (would mean dead features)

   **`validate_predict(game_predictions)`** — runs after prediction:
   - All probabilities between 0.0 and 1.0
   - Home + away probabilities sum to ~1.0 (within 0.01 tolerance)
   - Every requested game has a prediction result

3. Add `validate_stage(stage_name, **kwargs)` dispatcher function
4. Add `--strict` flag: when True, raise `ValidationError` on any FAIL. When False (default), log warnings only.
5. Wire into `update.py`:
   ```python
   # After fetch
   validate_stage('fetch', season=current_season)
   # After preprocess
   validate_stage('preprocess')
   ```
6. **Verify**: Run the validator against current data and document any failures it finds (there will probably be some — that's the point)

---

### TASK 6: Regenerate Stale Reports + Fix Player Backtest
**Priority: Medium | Files: `src/models/run_evaluation.py`, `src/models/backtesting.py`, `reports/`**
**Branch: `v2/fix-reports`**

Two separate issues documented in CONCERNS.md:

**Issue 1: Stale SHAP reports.** Files in `reports/explainability/` reference features like `fantasy_pts`, raw `dreb`, raw `oreb` that no longer exist in the current trained model. Fix is just re-running `run_evaluation.py`.

**Issue 2: Player backtest stops at 2015-16.** `reports/backtest_player_pts.csv` only covers through 2015-16, missing 10 years of modern NBA data. The game outcome backtest runs through 2024-25, so the player backtest loop has a different (broken) boundary.

**Debugging the player backtest — structured approach:**
1. Read `src/models/backtesting.py` and find the player backtest function
2. Look for the loop that iterates over seasons — check:
   - Is there a hardcoded end season? (e.g., `range(start, 201516)`)
   - Is it reading available seasons from a data file that only goes to 2015-16?
   - Is there an exception being silently caught that stops the loop?
   - Does it depend on a data file (`data/features/player_game_features.csv`) that only has data through 2015-16?
3. Check the player feature data: `python -c "import pandas as pd; df = pd.read_csv('data/features/player_game_features.csv'); print(sorted(df['season'].unique()))"`
   - If seasons stop at 2015-16, the issue is in feature generation, not backtesting
   - If seasons go through 2024-25, the issue is in the backtest loop boundary
4. Fix the root cause, not just the symptom

**Implementation:**
1. Fix the player backtest loop boundary so it runs through all available seasons
2. Run `python src/models/run_evaluation.py` — regenerate SHAP/explainability reports
3. Run the updated player backtest
4. **Verify**:
   - Check `reports/explainability/` files reference current features, not stale ones
   - Check `reports/backtest_player_pts.csv` has rows for seasons through 2024-25
   - Sanity check: player model accuracy should be reasonable (not 0% or 100%) for recent seasons
5. Commit updated reports

---

### TASK 7: Web Dashboard
**Priority: Low (but fun) | New dir: `dashboard/`**
**Branch: `v2/dashboard`**

Build a static web dashboard for viewing predictions and model performance. No build step, no npm, no frameworks that need compilation — just HTML + JS + CSS that can be opened in a browser.

**Before coding — understand available data:**
1. Check the predictions database schema:
   ```bash
   sqlite3 database/predictions_history.db ".schema"
   sqlite3 database/predictions_history.db "SELECT * FROM predictions LIMIT 5"
   ```
2. Check JSON snapshot format: `ls data/outputs/` and read one file
3. Understand what data is available: predictions with probabilities, ATS picks, value bet flags, timestamps

**Dashboard architecture:**
- Single `dashboard/index.html` file with inline CSS and JS (keep it simple)
- Use Chart.js from CDN (`https://cdn.jsdelivr.net/npm/chart.js`) for charts
- Data source: JSON files in `dashboard/data/` (NOT direct SQLite access)
- A Python export script generates the JSON files from the DB

**Implementation:**
1. Create `scripts/export_dashboard_data.py` that:
   - Queries `database/predictions_history.db`
   - Writes `dashboard/data/todays_picks.json` — today's predictions with win prob, ATS pick, value bet flag
   - Writes `dashboard/data/accuracy_history.json` — daily/weekly accuracy over time
   - Writes `dashboard/data/value_bets.json` — recent value bet alerts with edge size
   - Writes `dashboard/data/player_predictions.json` — latest player performance predictions
   - Handles empty data gracefully (no games today = empty array, not crash)

2. Build `dashboard/index.html` with these sections:
   - **Header**: "NBA Analytics Dashboard" with last-updated timestamp
   - **Today's Picks panel**: Card for each game showing teams, win probability bar, ATS pick (cover/don't cover), value bet badge if applicable. Use team colors if you want to be fancy.
   - **Model Accuracy chart**: Line chart (Chart.js) showing rolling accuracy over time. Include a horizontal reference line at 66.8% (current baseline).
   - **Value Bets panel**: Table of games where model edge > threshold, showing: game, model prob, market prob, edge %, recommended side
   - **Player Lookup**: Text input, search through player predictions, show pts/reb/ast projections as cards
   - Style it clean — dark theme works well for sports dashboards. Use CSS grid for layout.

3. Add to `update.py`:
   ```python
   # After predictions
   subprocess.run([sys.executable, 'scripts/export_dashboard_data.py'])
   ```

4. Add a simple serve option: `python -m http.server 8080 --directory dashboard` note in the README

5. **Verify**: Open `dashboard/index.html` in a browser and confirm all panels render. If no prediction data exists yet, it should show "No predictions available" rather than crashing.

---

## Execution Strategy

Run these in parallel where possible:
- **Wave 1 (parallel)**: Task 1 (API client), Task 2 (tests), Task 5 (validation), Task 6 (reports)
- **Wave 2 (after Task 1 done)**: Task 3 (incremental preprocessing) — depends on API client being stable
- **Wave 3 (parallel)**: Task 4 (ATS improvement), Task 7 (dashboard) — both independent

For each task:
1. Use a git worktree for isolation
2. Create a feature branch named `v2/{task-name}` (e.g., `v2/shared-api-client`)
3. Make atomic commits with descriptive messages as you go
4. When complete, merge to main

If any task takes more than the context allows, document exactly where you stopped and what's left in a `{TASK}_STATUS.md` file at the project root so the next session can pick up cleanly.

After all tasks complete, update `README.md` and `docs/project_overview.md` to reflect the v2 changes.

## PROMPT END
