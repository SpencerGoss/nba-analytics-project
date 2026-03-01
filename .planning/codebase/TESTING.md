# Testing Patterns

**Analysis Date:** 2026-03-01

## Test Framework

**Status:** No formal unit testing framework detected

**Finding:**
- No `pytest.ini`, `setup.cfg`, or `tox.ini` configuration
- No test files (`*_test.py`, `test_*.py`) in project source
- No testing dependencies in `requirements.txt` (no pytest, unittest, mock, etc.)
- Single exception: `src/models/backtesting.py` which is walk-forward validation, not unit tests

**Recommendation:**
- Testing is not currently automated
- Code changes rely on manual verification and downstream model evaluation
- Future: Consider adding pytest + fixtures for data pipeline validation

## Validation Through Backtesting

**Primary Validation Method:** Walk-forward model backtesting (`src/models/backtesting.py`)

**Framework:** sklearn metrics + manual pandas grouping (no dedicated backtesting library)

**Approach:**
```python
# Roll forward one season at a time
for i in range(max(1, MIN_TRAIN_SEASONS - 1), len(seasons)):
    train_seasons = seasons[:i]
    valid_season = seasons[i]
    tr = train_df[train_df["season"].astype(str).isin(train_seasons)].copy()
    va = train_df[train_df["season"].astype(str) == valid_season].copy()

    # Train on tr, evaluate on va
    pipe.fit(X_sub, y_sub)
    val_proba = pipe.predict_proba(X_val)[:, 1]
```

**Metrics Collected:**
- Classification: `accuracy_score()`, `roc_auc_score()`, `brier_score_loss()`
- Regression: `mean_absolute_error()`, `root_mean_squared_error()`

**Output:**
- Per-season performance captured in CSV reports (e.g., `reports/backtest_game_outcome.csv`)
- Human-readable summary in `reports/backtest_summary.txt`

## Data Pipeline Validation

**Integration Testing Pattern:** Scripts execute and produce expected CSV outputs

**Verification Methods:**

1. **Row counts and schema** (implicit):
   - Each `get_*.py` script prints row counts: `print(f"  Saved {season} ({len(data)} rows)")`
   - Column presence checked in preprocessing via `clean_columns()` and field renames

2. **Error logging for pipeline failures** (`update.py`, `backfill.py`):
   ```python
   def log_pipeline_error(script_name: str, error: Exception) -> None:
       ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
       timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
       with ERROR_LOG_PATH.open("a", encoding="utf-8") as log_file:
           log_file.write(f"[{timestamp}] {script_name} failed: {error}\n")
   ```

3. **Non-fatal subprocess handling** (`src/data/get_odds.py`):
   - Refresh failures logged but don't abort pipeline
   - Return boolean indicates success/skip rather than raising exceptions
   ```python
   result = subprocess.run([sys.executable, str(_FETCH_SCRIPT)], capture_output=True)
   if result.returncode == 0:
       return True
   else:
       log.warning(f"Odds refresh failed (exit code {result.returncode})")
       return False
   ```

## Model Validation Patterns

**Decision Threshold Tuning** (`src/models/game_outcome_model.py`):
```python
def _best_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple:
    """Find probability threshold that maximizes accuracy."""
    best_t, best_acc = 0.50, -1.0
    for t in np.arange(0.35, 0.66, 0.01):
        pred = (proba >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_t = float(round(t, 2))
    return best_t, best_acc
```

**Season-Based Splits** (`src/models/game_outcome_model.py`):
```python
def _season_splits(train_df: pd.DataFrame) -> list:
    """Create expanding season splits: train up to season i-1, validate on season i."""
    seasons = sorted(train_df["season"].astype(str).unique())
    splits = []
    for i in range(max(1, MIN_TRAIN_SEASONS_FOR_TUNING - 1), len(seasons)):
        train_seasons = seasons[:i]
        valid_season = seasons[i]
        tr = train_df[train_df["season"].astype(str).isin(train_seasons)].copy()
        va = train_df[train_df["season"].astype(str) == valid_season].copy()
        if not tr.empty and not va.empty:
            splits.append((tr, va, valid_season))

    # fallback when dataset has very few seasons
    if not splits:
        cutoff = int(len(train_df) * 0.85)
        tr = train_df.iloc[:cutoff].copy()
        va = train_df.iloc[cutoff:].copy()
        if not tr.empty and not va.empty:
            splits.append((tr, va, "date_fallback"))
    return splits
```

**Candidate Model Selection** (same file):
```python
model_scores = {}
for name, pipe in candidates.items():
    split_accs, split_aucs, split_thresholds = [], [], []

    for tr, va, split_name in splits:
        X_sub, y_sub = tr[feat_cols], tr[TARGET]
        X_val, y_val = va[feat_cols], va[TARGET]

        pipe.fit(X_sub, y_sub)
        val_proba = pipe.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        best_t, val_acc = _best_threshold(y_val, val_proba)

        split_accs.append(val_acc)
        split_aucs.append(val_auc)
        split_thresholds.append(best_t)

    model_scores[name] = {
        "mean_val_acc": float(np.mean(split_accs)),
        "mean_val_auc": float(np.mean(split_aucs)),
        "threshold": float(round(np.mean(split_thresholds), 2)),
    }

best_name = max(model_scores, key=lambda k: model_scores[k]["mean_val_acc"])
```

## Feature Validation

**Presence Checks** (`src/features/player_features.py`):
```python
for col in ROLL_STATS:
    if col not in g.columns:
        continue
    shifted = g[col].shift(1)
    for w in windows:
        g[f"{col}_roll{w}"] = shifted.rolling(window=w, min_periods=1).mean()
```

**Column Dependency Checks** (same file):
```python
ADV_CONTEXT_COLS = [
    "player_id", "season",
    "usg_pct", "ts_pct", "net_rating", "pie",
    "ast_pct", "oreb_pct", "dreb_pct",
]
# Later merged via pd.merge on player_id/season with error handling implicit
```

## Data Quality Checks

**Dropna Patterns:**
```python
n_before = len(df)
df = df.dropna(subset=[TARGET])
if len(df) < n_before:
    print(f"  Dropped {n_before - len(df):,} rows with missing target")
```

**Deduplication:**
```python
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["player_id"])
```

**Type Assertions (Implicit):**
```python
df["player_id"] = df["player_id"].astype(int)
df["season"]    = df["season"].astype(str)
```

## Manual Testing / Verification

**Console Output Inspection:**
- Model training prints per-split metrics to stdout
- Feature engineering scripts show row counts per season
- Pipeline orchestrators (`update.py`, `backfill.py`) timestamp each step

**Example Output Pattern:**
```
============================================================
GAME OUTCOME PREDICTION MODEL
============================================================

Loading matchup features...
  Total games: 12,345 | Seasons: 20

  Training data from 200001: 11,000 games
  Train: 8,500 games | Test: 2,500 games
  Test seasons: ['202324', '202425']

--- Model selection across 15 validation split(s) ---
  logistic         | split=200001 | acc=0.6234 | auc=0.6891 | th=0.50
  gradient_boosting | split=200001 | acc=0.6456 | auc=0.7012 | th=0.52
  ...

Selected model: gradient_boosting (mean val acc=0.6523, mean val auc=0.7145, threshold=0.52)
  Test Accuracy : 0.6478
  Test ROC-AUC  : 0.7089
```

## What Is Tested

**Implicitly validated:**
- Raw API data fetch + save to CSV (via row count output)
- CSV preprocessing pipeline (via preprocessing.py successful completion)
- Model training and threshold selection (via backtest walk-forward)
- Feature engineering presence and shape (via model training success)

**NOT explicitly tested:**
- Individual function logic (no unit tests)
- Data correctness beyond row counts (no schema/value validation)
- Edge cases in feature calculations
- API rate limit handling (only logged, not tested)
- Missing data imputation correctness

## Test Coverage Gaps

**High Priority:**
- `src/processing/preprocessing.py`: No validation that column renames are correct or that type conversions succeed uniformly
  - Files: `src/processing/preprocessing.py`
  - Risk: Silent type conversion failures on future API schema changes

- `src/features/player_features.py`: No tests for rolling window calculations or lag features
  - Files: `src/features/player_features.py`
  - Risk: Data leakage or incorrect lags could go undetected

- `src/data/get_*.py` scripts: No validation that API responses match expected schema
  - Files: `src/data/get_*.py` (all API fetching modules)
  - Risk: API changes or rate limits could silently produce malformed data

**Medium Priority:**
- Model feature selection (`get_feature_cols()` functions)
  - Files: `src/models/game_outcome_model.py`, `src/models/player_performance_model.py`
  - Risk: Accidental inclusion of leakage columns

- Era labeling logic
  - Files: `src/features/era_labels.py`
  - Risk: Misaligned season boundaries could produce incorrect era assignments

**Low Priority:**
- Subprocess/odds refresh error handling
  - Files: `src/data/get_odds.py`
  - Risk: Logged but does not abort pipeline, acceptable non-fatal failure mode

## How to Run Validation

**Manual Validation Currently Used:**

1. **Run pipeline orchestrator:**
   ```bash
   python update.py              # Fetch current season + preprocess
   python backfill.py            # Historical backfill (one-time)
   ```
   - Inspect console output for errors and row counts
   - Check `logs/pipeline_errors.log` for failures

2. **Run model backtests:**
   ```bash
   python src/models/backtesting.py
   ```
   - View per-season accuracy metrics in `reports/backtest_*.csv`
   - Compare accuracy trends across eras

3. **Retrain models:**
   ```bash
   python src/models/train_all_models.py --rebuild-features
   ```
   - Inspects console output for train/validation/test splits
   - Saves model artifacts and feature importance to `models/artifacts/`

**Recommended Future Addition:**

Create `tests/test_preprocessing.py`:
```python
import pytest
import pandas as pd
from src.processing.preprocessing import clean_columns, load_season_folder

def test_clean_columns():
    df = pd.DataFrame({
        "Player Name": [1, 2],
        "TEAM-ID": [3, 4],
        "PPG / 30": [5, 6]
    })
    result = clean_columns(df)
    assert "player_name" in result.columns
    assert "team_id" in result.columns
    assert "ppg_30" in result.columns
```

---

*Testing analysis: 2026-03-01*
