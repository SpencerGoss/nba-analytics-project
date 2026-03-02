# Phase 1: Foundation & Outputs — Research

**Researched:** 2026-03-01
**Domain:** Python ML pipeline debugging, SQLite persistence, pipeline architecture
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FR-1.1 | Fix injury proxy join so `missing_minutes` and `star_player_out` features reach the model with non-null values | Join key format mismatch traced — see Injury Bug Root Cause section |
| FR-1.2 | Wire calibrated model artifact (`game_outcome_model_calibrated.pkl`) into `predict_cli.py` inference path | Load path traced — see Calibration Bug section |
| FR-1.3 | Validate all feature columns have <95% null rate before model training; fail loudly if exceeded | pandas null rate pattern documented in Code Examples |
| FR-6.1 | Create append-only SQLite prediction store (`predictions_history.db`) | Schema and WAL pattern documented |
| FR-6.2 | Enable WAL mode on prediction store for concurrent read/write | SQLite WAL pragma documented |
| FR-6.3 | Export daily JSON snapshot of predictions for web consumption | json.dumps pattern with dataclass documented |
| FR-6.4 | Store prediction results with timestamps for historical accuracy tracking | Schema includes timestamp and game_date columns |
| FR-6.5 | Serialize model metadata (feature importances, thresholds, training dates) as JSON alongside pickle artifacts | Metadata dict pattern from game_outcome_model.py documented |
| FR-7.1 | Each pipeline stage runs independently with documented inputs/outputs | Stage boundary analysis documented |
| FR-7.2 | External data scrapers follow existing `src/data/get_*.py` module pattern | Existing pattern documented |
| FR-7.3 | `update.py` remains thin — each capability is one-line call to its module | Current update.py structure analyzed |
| FR-7.4 | Document pipeline stage order, dependencies, and expected runtime in a pipeline reference | Documentation template provided |
| NFR-1 | Lookahead bias prevention: `.shift(1)` before `.rolling()`, assert join shape >0, log null rates | Guard patterns documented |
| NFR-2 | Daily update <15 minutes; training <15 minutes; Basketball Reference 3-second delay | Current baseline confirmed ~10-15 min; no new performance risk in Phase 1 |
| NFR-3 | All model outputs serializable to JSON; prediction history queryable via SQLite | JSON + SQLite patterns documented |
</phase_requirements>

---

## Summary

Phase 1 is a bug-fix and infrastructure phase. The two critical bugs are both confirmed by direct code inspection: (1) injury proxy features are computed and joined correctly in `team_game_features.py`, but the join silently drops all rows because `game_id` format in `player_game_logs.csv` does not match `game_id` format in `team_game_logs.csv`; (2) the calibrated model artifact (`game_outcome_model_calibrated.pkl`) is saved by `calibration.py` but `predict_cli.py` calls `predict_game()` from `game_outcome_model.py`, which hard-codes loading `game_outcome_model.pkl` (uncalibrated) — the calibrated artifact is never referenced anywhere in the inference path.

The prediction store (FR-6.x) is entirely new infrastructure. There is currently no `src/outputs/` module, no `predictions_history.db`, and `predict_cli.py` writes to stdout only. The store must be an append-only SQLite database with WAL mode enabled, and a JSON snapshot function that serializes the latest predictions to disk. This is low-complexity new code that follows existing patterns.

Pipeline organization (FR-7.x) requires formalizing what already partially exists: `update.py` is already thin and calls modules. The gap is that feature engineering (`build_team_game_features`, `build_matchup_dataset`) and model training are NOT called from `update.py` — they run separately via `train_all_models.py`. Each stage needs a documented input/output contract, and a `PIPELINE.md` reference document.

**Primary recommendation:** Fix the `game_id` join key mismatch in injury proxy first (one-line fix once the format difference is confirmed), wire the calibrated model load path in `predict_game()`, add null-rate guards to `game_outcome_model.py`, then build `src/outputs/prediction_store.py` and `src/outputs/json_export.py`, and finally write the pipeline reference document.

---

## Standard Stack

### Core (already in requirements.txt — no new installs needed for Phase 1)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | >=2.0 | DataFrame joins, null analysis, CSV I/O | Already used throughout pipeline |
| sqlite3 | stdlib | Prediction store, WAL mode, concurrent reads | No ORM needed; already used via `database/nba.db` |
| pickle | stdlib | Model artifact load/save | Already used in all model files |
| json | stdlib | JSON snapshot export, metadata serialization | Already used in predict_cli.py output |
| pathlib | stdlib | Path management | Already used in update.py |

### No New Libraries Required for Phase 1

Phase 1 is entirely bug fixes and new Python modules using the existing stack. No new `pip install` steps.

---

## Architecture Patterns

### Recommended New Directory Structure

```
src/
├── data/           # (existing) API fetchers
├── processing/     # (existing) CSV consolidation
├── features/       # (existing) feature engineering
├── models/         # (existing) training and inference
└── outputs/        # NEW: prediction store and JSON export
    ├── __init__.py
    ├── prediction_store.py   # append-only SQLite writer
    └── json_export.py        # daily JSON snapshot generator

data/
└── outputs/        # NEW: JSON snapshot destination
    └── predictions_YYYYMMDD.json

database/
├── nba.db          # (existing)
└── predictions_history.db   # NEW: append-only prediction store
```

### Pattern 1: Append-Only SQLite Prediction Store

**What:** An SQLite database with WAL mode that accumulates one row per `predict_cli.py` run.
**When to use:** Every time `predict_game()` produces a prediction.

```python
# src/outputs/prediction_store.py
import sqlite3
import json
from datetime import datetime
from pathlib import Path

STORE_PATH = "database/predictions_history.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS game_predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,       -- ISO8601 UTC timestamp of prediction
    game_date           TEXT,                -- scheduled game date (YYYY-MM-DD)
    home_team           TEXT NOT NULL,
    away_team           TEXT NOT NULL,
    home_win_prob       REAL NOT NULL,
    away_win_prob       REAL NOT NULL,
    model_name          TEXT,                -- e.g. "gradient_boosting"
    model_artifact      TEXT,               -- filename of .pkl used
    decision_threshold  REAL,
    feature_count       INTEGER,
    actual_home_win     INTEGER,            -- NULL until result known; 1/0
    notes               TEXT                -- JSON blob for extras
);
"""

WAL_PRAGMA = "PRAGMA journal_mode=WAL;"
INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_game_date ON game_predictions(game_date);"

def init_store(store_path: str = STORE_PATH) -> None:
    """Create the database and enable WAL mode if it does not exist."""
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(store_path)
    con.execute(WAL_PRAGMA)
    con.execute(CREATE_TABLE_SQL)
    con.execute(INDEX_SQL)
    con.commit()
    con.close()

def write_game_prediction(
    prediction: dict,
    store_path: str = STORE_PATH,
) -> int:
    """
    Append one game prediction record.
    prediction dict must include: home_team, away_team, home_win_prob, away_win_prob.
    Optional: game_date, model_name, model_artifact, decision_threshold, feature_count, notes.
    Returns the rowid of the inserted record.
    """
    init_store(store_path)
    now = datetime.utcnow().isoformat()
    notes = prediction.get("notes")
    if notes and not isinstance(notes, str):
        notes = json.dumps(notes)
    con = sqlite3.connect(store_path)
    cur = con.execute(
        """INSERT INTO game_predictions
           (created_at, game_date, home_team, away_team,
            home_win_prob, away_win_prob, model_name, model_artifact,
            decision_threshold, feature_count, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now,
            prediction.get("game_date"),
            prediction["home_team"],
            prediction["away_team"],
            prediction["home_win_prob"],
            prediction["away_win_prob"],
            prediction.get("model_name"),
            prediction.get("model_artifact"),
            prediction.get("decision_threshold"),
            prediction.get("feature_count"),
            notes,
        ),
    )
    rowid = cur.lastrowid
    con.commit()
    con.close()
    return rowid
```

### Pattern 2: Daily JSON Snapshot Export

**What:** Serialize all predictions from the current day (or a date range) to a JSON file.
**When to use:** After each `predict_cli.py` run.

```python
# src/outputs/json_export.py
import sqlite3
import json
from datetime import date, datetime
from pathlib import Path

STORE_PATH = "database/predictions_history.db"
OUTPUT_DIR = "data/outputs"

def export_daily_snapshot(
    game_date: str | None = None,
    store_path: str = STORE_PATH,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Export all predictions for a given date to a JSON file.
    game_date: 'YYYY-MM-DD'. Defaults to today.
    Returns the path of the written file.
    """
    if game_date is None:
        game_date = date.today().isoformat()

    con = sqlite3.connect(store_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM game_predictions WHERE game_date = ? ORDER BY created_at",
        (game_date,),
    ).fetchall()
    con.close()

    records = [dict(r) for r in rows]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"predictions_{game_date.replace('-', '')}.json"
    with open(out_path, "w") as f:
        json.dump({
            "exported_at": datetime.utcnow().isoformat(),
            "game_date": game_date,
            "count": len(records),
            "predictions": records,
        }, f, indent=2)

    print(f"JSON snapshot → {out_path}  ({len(records)} records)")
    return str(out_path)
```

### Pattern 3: Calibrated Model Load Path Fix

**What:** Load `game_outcome_model_calibrated.pkl` instead of `game_outcome_model.pkl` in `predict_game()`.
**Where:** `src/models/game_outcome_model.py`, function `predict_game()`, lines 299-305.

The current broken code:
```python
# CURRENT (broken): always loads uncalibrated model
model_path = os.path.join(artifacts_dir, "game_outcome_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)
```

The fix:
```python
# FIXED: prefer calibrated artifact, fall back to uncalibrated with warning
cal_path = os.path.join(artifacts_dir, "game_outcome_model_calibrated.pkl")
raw_path = os.path.join(artifacts_dir, "game_outcome_model.pkl")

if os.path.exists(cal_path):
    model_path = cal_path
    model_artifact_name = "game_outcome_model_calibrated.pkl"
else:
    import warnings
    warnings.warn(
        "Calibrated model not found — using uncalibrated model. "
        "Run src/models/calibration.py to generate the calibrated artifact.",
        stacklevel=2,
    )
    model_path = raw_path
    model_artifact_name = "game_outcome_model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)
```

### Pattern 4: Null Rate Guard (FR-1.3, NFR-1)

**What:** Assert that no feature column has a null rate >= 95% before training.
**Where:** Add to `get_feature_cols()` or beginning of `train_game_outcome_model()` in `game_outcome_model.py`.

```python
def validate_feature_null_rates(df: pd.DataFrame, feat_cols: list, threshold: float = 0.95) -> None:
    """
    Raise ValueError if any feature column has null rate >= threshold.
    Also logs null rates for all columns with any nulls.
    """
    null_rates = df[feat_cols].isnull().mean()
    high_null = null_rates[null_rates >= threshold]
    if not high_null.empty:
        detail = "\n".join(
            f"  {col}: {rate:.1%} null" for col, rate in high_null.items()
        )
        raise ValueError(
            f"Feature columns exceed {threshold:.0%} null threshold:\n{detail}\n"
            "Fix the upstream feature pipeline before training."
        )
    # Log columns with any nulls (for visibility, not failure)
    some_null = null_rates[(null_rates > 0) & (null_rates < threshold)]
    if not some_null.empty:
        print("  Columns with partial nulls (will be imputed):")
        for col, rate in some_null.items():
            print(f"    {col}: {rate:.1%}")
```

### Pattern 5: predict_game() with Store Write

**What:** After computing a prediction, write to `predictions_history.db` and export JSON.
**Where:** End of `predict_game()` in `game_outcome_model.py`.

```python
# Add at bottom of predict_game(), before return statement:
from src.outputs.prediction_store import write_game_prediction
from src.outputs.json_export import export_daily_snapshot

result = {
    "home_team": home_team_abbr,
    "away_team": away_team_abbr,
    "home_win_prob": round(float(prob[1]), 4),
    "away_win_prob": round(float(prob[0]), 4),
    "model_artifact": model_artifact_name,      # tracked above
    "feature_count": len(feat_cols),
}

try:
    write_game_prediction(result)
    export_daily_snapshot()
except Exception as e:
    import warnings
    warnings.warn(f"Could not write prediction to store: {e}", stacklevel=2)

return result
```

### Pattern 6: Model Metadata Serialization (FR-6.5)

**What:** Save the metadata dict from `train_game_outcome_model()` as JSON alongside the .pkl.
**Where:** `game_outcome_model.py`, already builds `metrics` dict — just add JSON serialization.

```python
# In train_game_outcome_model(), after saving the pkl:
import json
metadata = {
    **metrics,
    "trained_at": datetime.now().isoformat(),
    "train_start_season": start_season,
    "feature_list": feat_cols,
    "top_importances": importances.head(20).to_dict(),
}
meta_path = os.path.join(artifacts_dir, "game_outcome_metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved → {meta_path}")
```

### Anti-Patterns to Avoid

- **Swallowing the calibration fallback silently:** If calibrated model is missing, `predict_game()` must warn loudly — not silently fall back. An assertion that fails with a clear message is better than silent degradation.
- **Writing to the prediction store inside the feature engineering step:** Only `predict_game()` (inference path) writes to the store — not training or feature building.
- **Using the prediction store as the source of truth for features:** The store is append-only output, never read back into the feature pipeline.
- **Global except that swallows store write errors:** Store write failures should warn (not crash), but must be visible.

---

## Injury Bug Root Cause Analysis

### The Join Chain

```
player_game_logs.csv
  → injury_proxy.py: build_injury_proxy_features()
      → outputs: DataFrame keyed on (game_id, team_id)
      → saved to: data/features/injury_proxy_features.csv
  → team_game_features.py: build_team_game_features()
      → line 534: output.merge(injury_df, on=["team_id", "game_id"], how="left")
```

### Confirmed Join Keys

- `injury_proxy.py` loads `player_game_logs.csv`, which has `game_id` as it came from `player_game_logs` API endpoint.
- `team_game_features.py` loads `team_game_logs.csv`, which has `game_id` as it came from `team_game_logs` API endpoint.
- **Both tables get `game_id` from different NBA API endpoints.** The NBA API is known to return `game_id` in `"00XXXXXXXXXX"` format (10-digit string with leading zeros) from some endpoints and as a plain integer from others. If `player_game_logs` preprocesses `game_id` as `int64` while `team_game_logs` retains it as a string (or vice versa), the left join on `["team_id", "game_id"]` will produce zero matches — all `missing_minutes` values fill as NaN, then `.fillna(0)` makes them look valid but they are all zero.

### The Symptom

CONCERNS.md confirms: "Features either missing from input CSV or silently all-null and imputed to mean value." The imputer fills all-zero injury columns with column mean = 0, which renders the features invisible to the model. Feature importances do not list `home_missing_minutes`, `away_missing_minutes`, etc.

### How to Diagnose Definitively

```python
import pandas as pd
gl_player = pd.read_csv("data/processed/player_game_logs.csv", usecols=["game_id"])
gl_team   = pd.read_csv("data/processed/team_game_logs.csv",   usecols=["game_id"])
print("player_game_logs game_id dtype:", gl_player["game_id"].dtype)
print("team_game_logs   game_id dtype:", gl_team["game_id"].dtype)
print("player sample:", gl_player["game_id"].iloc[:3].tolist())
print("team sample:  ", gl_team["game_id"].iloc[:3].tolist())
```

### The Fix

Normalize `game_id` to the same type in both DataFrames before joining. The safest fix is to cast both to `str` and strip whitespace:

```python
# In injury_proxy.py: build_injury_proxy_features() — before returning result
result["game_id"] = result["game_id"].astype(str).str.strip()

# In team_game_features.py: build_team_game_features() — before the merge
output["game_id"]   = output["game_id"].astype(str).str.strip()
injury_df["game_id"] = injury_df["game_id"].astype(str).str.strip()

# Then verify the join worked:
before_merge = len(output)
output = output.merge(injury_df, on=["team_id", "game_id"], how="left")
n_matched = output["missing_minutes"].notna().sum()
print(f"  Injury join: {n_matched:,} rows matched out of {len(output):,}")
assert n_matched > 0, "Injury proxy join matched zero rows — check game_id format"
```

**Also add `team_id` type check:** If `team_id` is `int64` in one table and `object` in the other, the join similarly fails silently. Cast both to `int`.

### Also Needed: Assert After Every Join (NFR-1)

The CONCERNS.md flags this broadly: "Multiple inner joins on (game_id, team_id, season) can silently drop rows if join keys mismatch; no row count assertions after joins." Add post-join assertions throughout `injury_proxy.py` and `team_game_features.py`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Probability calibration | Custom calibration curve fitter | `sklearn.calibration.CalibratedClassifierCV` (already used) | sklearn's implementation handles edge cases; v2 model already IS a `CalibratedClassifierCV` |
| SQLite schema migration | Custom migration scripts | Add `CREATE TABLE IF NOT EXISTS` + `IF NOT EXISTS` for indexes | SQLite handles idempotent DDL correctly; no migration framework needed at this scale |
| JSON serialization of numpy types | Custom JSON encoder | `float(np.float64_val)` cast before serializing | stdlib json can't serialize numpy floats; explicit cast is simpler than custom encoder |
| Concurrent SQLite access | File locking or threading logic | `PRAGMA journal_mode=WAL` | WAL mode is the correct SQLite primitive for concurrent read-write; handles web server reads during update writes |

**Key insight:** Phase 1 has no algorithmic complexity. Every problem has an idiomatic Python/pandas/SQLite solution that already exists or is one stdlib function call.

---

## Common Pitfalls

### Pitfall 1: Silent Left Join (The Actual Injury Bug)

**What goes wrong:** `df.merge(other, on=["game_id", "team_id"], how="left")` returns all rows from `df` with NaN for all `other` columns — no error, no warning.
**Why it happens:** Type mismatch between join keys (int vs string). `"0022400001" != 22400001`.
**How to avoid:** Cast join keys to a canonical type BEFORE merging; assert `n_matched > 0` after any left join where zero matches indicates a bug.
**Warning signs:** All injury feature columns are exactly 0 or exactly 1.0 (the fillna defaults); feature importances list does not include `missing_minutes`.

### Pitfall 2: Calibrated Model Not Found, No Warning

**What goes wrong:** `calibration.py` saves the artifact, but if it hasn't been run since the last model retrain, the calibrated artifact is stale or missing. `predict_game()` must NOT silently use the uncalibrated model.
**Why it happens:** Model retraining (via `train_all_models.py`) does NOT automatically run `calibration.py` after training. The two scripts are decoupled.
**How to avoid:** In `predict_game()`, prefer calibrated artifact, but issue a loud `warnings.warn()` if falling back to uncalibrated. Better: add a call to `run_calibration_analysis()` at the end of `train_game_outcome_model()`.
**Warning signs:** `models/artifacts/game_outcome_model_calibrated.pkl` timestamp is older than `game_outcome_model.pkl` timestamp.

### Pitfall 3: Prediction Store WAL Mode Not Set at Creation

**What goes wrong:** If `predictions_history.db` is created without `PRAGMA journal_mode=WAL`, subsequent reads from a web process while `update.py` writes will cause "database is locked" errors.
**Why it happens:** SQLite default journal mode is DELETE (file-level locking).
**How to avoid:** Always set `PRAGMA journal_mode=WAL` as the first statement after `sqlite3.connect()` when creating or opening the store. WAL mode persists after the connection closes.
**Warning signs:** Intermittent `OperationalError: database is locked` when running `update.py` and querying simultaneously.

### Pitfall 4: JSON Serialization of numpy/pandas Types

**What goes wrong:** `json.dumps({"prob": np.float64(0.7234)})` raises `TypeError: Object of type float64 is not JSON serializable`.
**Why it happens:** NumPy numeric types are not Python builtins — stdlib `json` does not know them.
**How to avoid:** Explicitly cast: `float(prob[1])`, `int(n_features)`. Already done correctly in the existing `predict_game()` return dict — maintain this pattern in all new prediction records.
**Warning signs:** `TypeError` in the `json_export.py` when `sqlite3.Row` fields contain numpy types.

### Pitfall 5: update.py Does Not Call Feature Engineering or Training

**What goes wrong:** After the daily data fetch + preprocess, `predictions_history.db` and `game_matchup_features.csv` are stale — they reflect the previous model, not the fresh data. A developer might assume `update.py` keeps everything current.
**Why it happens:** `update.py` only runs fetch + preprocess (Steps 1-3). Feature engineering (`build_team_game_features`) and model training (`train_all_models.py`) are separate manual steps.
**How to avoid:** The pipeline reference document (FR-7.4) must make this explicit. The document should show exactly which commands to run and in what order for a full rebuild vs. a daily data refresh.
**Warning signs:** Predictions use features from last week's feature build even though today's data has been refreshed.

### Pitfall 6: predict_game() Loads Features File at Inference Time

**What goes wrong:** `predict_game()` in `game_outcome_model.py` reads `game_matchup_features.csv` at inference time (line 307: `df = pd.read_csv(features_path)`). If this file was built with the broken injury join, the inference feature row also has all-zero injury features.
**Why it happens:** The inference path does not rebuild features from scratch — it re-uses the last saved features CSV.
**How to avoid:** After fixing the injury join bug, rebuild the features CSV (`build_team_game_features()` + `build_matchup_dataset()`) before running any inference. The pipeline reference must document this dependency.

---

## Code Examples

### Diagnosing the game_id type mismatch

```python
# Run this to confirm the root cause before writing any fix
import pandas as pd

player_logs = pd.read_csv("data/processed/player_game_logs.csv", usecols=["game_id", "team_id"])
team_logs   = pd.read_csv("data/processed/team_game_logs.csv",   usecols=["game_id", "team_id"])

print(f"player_game_logs game_id dtype: {player_logs['game_id'].dtype}")
print(f"player_game_logs team_id dtype:  {player_logs['team_id'].dtype}")
print(f"team_game_logs   game_id dtype:  {team_logs['game_id'].dtype}")
print(f"team_game_logs   team_id dtype:  {team_logs['team_id'].dtype}")
print()
print(f"player sample game_ids: {player_logs['game_id'].iloc[:3].tolist()}")
print(f"team sample game_ids:   {team_logs['game_id'].iloc[:3].tolist()}")

# Quick join test
test_join = player_logs.drop_duplicates(["game_id", "team_id"]).merge(
    team_logs.drop_duplicates(["game_id", "team_id"]),
    on=["game_id", "team_id"],
    how="inner",
)
print(f"\nInner join rows: {len(test_join):,}  (0 = type mismatch confirmed)")
```

### SQLite WAL mode initialization

```python
import sqlite3
from pathlib import Path

def _get_connection(store_path: str) -> sqlite3.Connection:
    """Open connection with WAL mode enabled."""
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(store_path)
    con.execute("PRAGMA journal_mode=WAL;")
    return con
```

### Null rate guard for feature assembly

```python
def validate_feature_null_rates(
    df: pd.DataFrame,
    feat_cols: list,
    threshold: float = 0.95,
) -> None:
    """Raise on columns with >= threshold null rate; log partial nulls."""
    null_rates = df[feat_cols].isnull().mean()
    bad = null_rates[null_rates >= threshold]
    if not bad.empty:
        lines = [f"  {col}: {rate:.1%}" for col, rate in bad.items()]
        raise ValueError(
            f"Feature columns exceed {threshold:.0%} null threshold "
            f"(implying broken upstream join):\n" + "\n".join(lines)
        )
    partial = null_rates[(null_rates > 0) & (null_rates < threshold)]
    for col, rate in partial.items():
        print(f"  [null audit] {col}: {rate:.1%} null (will be imputed)")
```

### Loading the calibrated model with fallback warning

```python
import os, pickle, warnings

def _load_game_outcome_model(artifacts_dir: str) -> tuple:
    """
    Load game outcome model, preferring the calibrated artifact.
    Returns (model, artifact_name).
    """
    cal_path = os.path.join(artifacts_dir, "game_outcome_model_calibrated.pkl")
    raw_path = os.path.join(artifacts_dir, "game_outcome_model.pkl")

    if os.path.exists(cal_path):
        with open(cal_path, "rb") as f:
            return pickle.load(f), "game_outcome_model_calibrated.pkl"

    warnings.warn(
        "Calibrated model artifact not found. Using uncalibrated model. "
        "Run: python src/models/calibration.py",
        UserWarning, stacklevel=3,
    )
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"No model artifact found in {artifacts_dir}")
    with open(raw_path, "rb") as f:
        return pickle.load(f), "game_outcome_model.pkl"
```

### predictions_history.db schema (CREATE TABLE)

```sql
CREATE TABLE IF NOT EXISTS game_predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,        -- ISO8601 UTC, e.g. "2026-03-01T14:32:00"
    game_date           TEXT,                 -- "YYYY-MM-DD", nullable for historical backfill
    home_team           TEXT NOT NULL,        -- team abbreviation, e.g. "BOS"
    away_team           TEXT NOT NULL,
    home_win_prob       REAL NOT NULL,        -- calibrated probability [0,1]
    away_win_prob       REAL NOT NULL,
    model_name          TEXT,                 -- "gradient_boosting" | "random_forest" | "logistic"
    model_artifact      TEXT,                -- filename of .pkl used at inference
    decision_threshold  REAL,                -- threshold used for predicted label
    feature_count       INTEGER,             -- number of features in the feature vector
    actual_home_win     INTEGER,             -- NULL until result known; 1 or 0 after game
    notes               TEXT                 -- JSON blob for any extra metadata
);

CREATE INDEX IF NOT EXISTS idx_game_date
    ON game_predictions(game_date);
CREATE INDEX IF NOT EXISTS idx_created_at
    ON game_predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_teams
    ON game_predictions(home_team, away_team);
```

### Pipeline reference document structure (FR-7.4)

The pipeline reference (suggested path: `docs/PIPELINE.md`) should document:

```markdown
# NBA Analytics Pipeline Reference

## Stage Order (full rebuild)

| Stage | Command | Inputs | Outputs | Est. Runtime |
|-------|---------|--------|---------|--------------|
| 1. Fetch | `python update.py` | NBA API | `data/raw/*/` | 10-15 min |
| 2. Preprocess | (called by update.py) | `data/raw/` | `data/processed/*.csv` | 1-2 min |
| 3. Feature build | `python src/features/team_game_features.py` | `data/processed/` | `data/features/` | 5-10 min |
| 4. Train | `python src/models/train_all_models.py` | `data/features/` | `models/artifacts/` | 5-10 min |
| 5. Calibrate | `python src/models/calibration.py` | `data/features/`, `models/artifacts/` | `models/artifacts/*_calibrated.pkl` | 1-2 min |
| 6. Predict | `python src/models/predict_cli.py game --home X --away Y` | `data/features/`, `models/artifacts/` | stdout + `database/predictions_history.db` | <1 min |

## Daily Update (existing data, no retraining)
Run only Stage 1 (which includes Stage 2 automatically).

## Dependency Graph
Stages are strictly sequential. No stage can run before its predecessor completes.
```

---

## State of the Art

| Old Approach | Current Approach | Notes |
|--------------|------------------|-------|
| Predict to stdout only | Append to SQLite + JSON export | Required for web platform; no external library needed |
| Raw model probabilities | CalibratedClassifierCV (isotonic) | Already trained; just needs wiring |
| Silent join failures | Post-join row count assertions | One-line assert after every merge |
| Implicit stage dependencies | Documented pipeline reference | Explicit inputs/outputs per stage |

---

## Open Questions

1. **game_id type mismatch: exact format**
   - What we know: Both `player_game_logs` and `team_game_logs` have `game_id`; injury join is silent-failing
   - What's unclear: Whether the mismatch is int vs string, leading-zero string vs stripped string, or something else
   - Recommendation: Run the diagnostic code in Code Examples before writing the fix — do NOT assume the fix without confirming the format

2. **Should calibration.py run automatically after train_all_models.py?**
   - What we know: They are currently decoupled; the calibrated artifact is stale after any retrain
   - What's unclear: Whether the user wants calibration to always run or be optional
   - Recommendation: Add a `run_calibration_analysis()` call at the end of `train_game_outcome_model()` (or in `train_all_models.py`) with a flag to skip it. Make "calibrate after train" the default.

3. **game_date for prediction store: how to determine?**
   - What we know: `predict_game()` currently does not take a `game_date` argument; it reads from the features CSV
   - What's unclear: Whether the user will supply the game date at the CLI, or whether it should be inferred from the matchup row's date
   - Recommendation: Add an optional `--date YYYY-MM-DD` arg to `predict_cli.py`; if omitted, use today's date as the predicted game date

4. **Should predict_game() rebuild features before inference?**
   - What we know: Currently reads the last-saved `game_matchup_features.csv`, which may be stale
   - What's unclear: Whether real-time feature rebuilding is within scope for Phase 1
   - Recommendation: Do NOT rebuild in Phase 1 — keep the read-from-CSV approach but document the dependency clearly. Stale features are acceptable; silently broken features are not.

---

## Sources

### Primary (HIGH confidence — direct code inspection)

- `src/features/injury_proxy.py` — full file read; join logic on lines 163-171; output keys confirmed as `(game_id, team_id)`
- `src/features/team_game_features.py` — full file read; injury join at lines 530-542; merge keys confirmed as `["team_id", "game_id"]`
- `src/models/calibration.py` — full file read; calibrated model save at lines 385-389; saved as `game_outcome_model_calibrated.pkl`
- `src/models/predict_cli.py` — full file read; 47 lines; calls `predict_game()` from `game_outcome_model.py`; no store write; no calibrated model reference
- `src/models/game_outcome_model.py` — full file read; `predict_game()` loads `game_outcome_model.pkl` (uncalibrated); hard-coded on line 299
- `update.py` — full file read; does NOT call feature engineering or model training; only fetch + preprocess + odds
- `.planning/codebase/CONCERNS.md` — confirmed bugs; injury join bug, calibration disconnect both documented
- `.planning/codebase/ARCHITECTURE.md` — pipeline layer structure, data flow confirmed
- `.planning/config.json` — `workflow.nyquist_validation` not present; validation section omitted per instructions

### Secondary (MEDIUM confidence)

- SQLite WAL mode documentation — WAL is the standard concurrent-access mode; behavior is well-established
- sklearn `CalibratedClassifierCV` — already used in codebase; API confirmed by direct code inspection

---

## Metadata

**Confidence breakdown:**
- Bug root causes: HIGH — both bugs confirmed by direct code inspection; no speculation required
- Prediction store design: HIGH — standard SQLite + stdlib json pattern; no external dependencies
- Pipeline organization: HIGH — derived from direct reading of update.py and all stage entry points
- Type mismatch diagnosis: MEDIUM — the root cause (type mismatch) is the most likely explanation; exact format difference (int vs string vs leading-zero string) must be confirmed by running the diagnostic before writing the fix

**Research date:** 2026-03-01
**Valid until:** Stable — no fast-moving libraries; valid until code changes
