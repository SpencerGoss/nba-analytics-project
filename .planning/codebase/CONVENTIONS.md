# Coding Conventions

**Analysis Date:** 2026-03-01

## Naming Patterns

**Files:**
- Module files: `lowercase_underscore.py` (e.g., `get_player_stats.py`, `game_outcome_model.py`)
- Entry points: lowercase underscored with clear purpose (e.g., `update.py`, `backfill.py`, `train_all_models.py`)
- Data script pattern: `get_*.py` for API fetching modules
- Model pattern: `*_model.py` for prediction/analysis models (e.g., `game_outcome_model.py`)
- Feature pattern: `*_features.py` for feature engineering (e.g., `player_game_features.py`, `team_game_features.py`)

**Functions:**
- Lowercase with underscores: `get_player_stats_all_seasons()`, `clean_columns()`, `_best_threshold()`
- Private helpers prefixed with single underscore: `_to_season_int()`, `_season_splits()`, `_parse_home_away()`
- Descriptive names reflecting action: `fetch_with_retry()`, `run_preprocessing()`, `label_eras()`

**Variables:**
- Lowercase underscored: `season`, `player_id`, `team_id`, `n_games`, `split_accs`
- Constants: UPPERCASE (e.g., `HEADERS`, `MAX_RETRIES`, `RETRY_DELAY`, `MIN_TRAIN_SEASONS`)
- Loop counters: `for attempt in range()`, `for year in range()`, `for w in windows`
- DataFrame column shortcuts: `df`, `tr`, `va`, `te` (train/validation/test subsets)

**Types:**
- Type hints used in function signatures: `def get_feature_cols(df: pd.DataFrame) -> list:`
- Return type annotations: functions returning tuples `-> tuple:`, lists `-> list:`, dicts `-> dict:`
- No widespread class-based OOP; functions dominate

## Code Style

**Formatting:**
- No explicit linting configuration detected (no `.eslintrc`, `.flake8`, `.pylintrc`)
- Follows PEP 8 implicitly: 4-space indentation, blank lines between sections
- Column width: most lines stay under 100 characters; longer lines accepted for long strings/constants
- Operator spacing: consistent single spaces around operators (`+`, `-`, `=`, `==`)

**Line Length:**
- Typical range: 80–100 characters
- Longer lines used for comprehensions, long docstrings, or string constants
- No hard enforced limit detected, but readability prioritized

**Comments and Docstrings:**
- Module-level docstrings present in most files with purpose and usage
- Sections delimited by `# ── Section Name ───────────────────────` (emoji-style dividers)
- Inline comments sparse; logic generally self-documenting through clear function names
- JSDoc-style docstrings in some functions (triple-quoted descriptions)

## Import Organization

**Order:**
1. Standard library imports (`import os`, `import sys`, `from datetime import`, `from pathlib import`)
2. Third-party imports (`import pandas as pd`, `import numpy as np`, `from sklearn.ensemble import`)
3. Local/project imports (`from src.data.get_player_stats import`, `from src.processing.preprocessing import`)

**Pattern in Data Scripts:**
```python
# Import required libraries
from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import time
import os

# Constants
HEADERS = {...}
MAX_RETRIES = 3
```

**Pattern in Model/Feature Scripts:**
```python
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
```

**Path Aliases:**
- No `sys.path` aliases in most modules
- Exception: `src/models/train_all_models.py` manually adds project root to sys.path for imports
- Imports use relative `src.` prefix (e.g., `from src.data.get_player_stats import get_player_stats_all_seasons`)

## Error Handling

**Strategy:**
- Try/except with graceful fallback preferred over raising exceptions immediately
- Retry pattern used for API calls with exponential-like delays

**API Call Pattern (`src/data/get_*.py`):**
```python
def fetch_with_retry(fetch_fn, season):
    for attempt in range(MAX_RETRIES):
        try:
            return fetch_fn()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Attempt {attempt + 1} failed for {season}: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  All {MAX_RETRIES} retries failed for {season}. Skipping.")
                return None
```

**Pipeline Error Logging (`update.py`, `backfill.py`):**
```python
def log_pipeline_error(script_name: str, error: Exception) -> None:
    """Append pipeline errors to logs/pipeline_errors.log."""
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {script_name} failed: {error}\n")
```

**Subprocess Non-Fatal Error Handling (`src/data/get_odds.py`):**
```python
if result.returncode == 0:
    log.info("Odds refresh completed successfully.")
    return True
else:
    # Check for specific error patterns, log appropriately, return False
    return False
```

**Validation with Exceptions (`src/features/era_labels.py`):**
```python
if season_col not in df.columns:
    raise KeyError(
        f"Column '{season_col}' not found in DataFrame. "
        f"Available columns: {list(df.columns)}"
    )
```

## Logging

**Framework:** Python's `logging` module used sparingly; mostly console `print()` statements

**Patterns:**
- Data fetching scripts: `print(f"  Saved {season} ({len(data)} rows)")`
- Pipeline orchestrators: `print(f"[{timestamp}] Starting...")`
- Model training: Large print sections with `print("=" * 60)` dividers
- Non-fatal issues logged via `logging` in `src/data/get_odds.py`: `log.warning()`, `log.info()`

**Timestamp Format:**
```python
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

## DataFrame Operations

**Column Naming Transitions:**
- Raw data: UPPERCASE with spaces/dashes (from NBA API)
- Processed data: lowercase_underscore

**Cleaning Function (`src/processing/preprocessing.py`):**
```python
def clean_columns(df):
    """Standardize all column names to lowercase_underscore format."""
    df.columns = (
        df.columns
            .str.lower()
            .str.strip()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
            .str.replace("/", "_", regex=False)
    )
    return df
```

**Type Conversion Pattern:**
```python
df["player_id"]   = df["player_id"].astype(int)
df["team_id"]     = df["team_id"].astype(int)
df["age"]         = df["age"].astype(float)
df["game_date"]   = pd.to_datetime(df["game_date"])
```

## Feature Engineering

**Seasonal Data Organization:**
- One CSV file per season in raw folders (e.g., `player_stats_202425.csv`)
- Season extracted from filename as 6-digit integer (e.g., `202425`)
- All seasons combined in processed single file (e.g., `player_stats.csv`)

**Rolling Window Pattern (`src/features/player_features.py`):**
```python
for col in ROLL_STATS:
    if col not in g.columns:
        continue
    shifted = g[col].shift(1)
    for w in windows:
        g[f"{col}_roll{w}"] = shifted.rolling(window=w, min_periods=1).mean()
```

**Differential Features:**
- Matchup features use `diff_` prefix for home-away differentials
- Context features (injury, rest, schedule) kept in home/away pairs

## Model Training

**Pipeline Pattern (`src/models/game_outcome_model.py`):**
```python
candidates = {
    "logistic": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(...)),
    ]),
    "gradient_boosting": Pipeline([...]),
}
```

**Configuration Constants:**
- Model hyperparameters defined as module-level constants
- Train/test season codes as module lists (e.g., `TEST_SEASONS = ["202324", "202425"]`)
- Feature selection logic in `get_feature_cols()` functions per model

## Code Organization

**Top-level sections marked with:**
```python
# ── Config ───────────────────────────────────
# ── Helpers ───────────────────────────────────
# ── Train / evaluate ──────────────────────────
# ── Entry point ───────────────────────────────
```

**Main execution pattern:**
```python
if __name__ == "__main__":
    main()
```

---

*Convention analysis: 2026-03-01*
