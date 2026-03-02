# Architecture Patterns

**Domain:** NBA game prediction and ATS betting analytics
**Researched:** 2026-03-01
**Confidence:** HIGH — analysis based directly on codebase inspection

---

## Current State Summary

The existing system is a well-structured five-layer ETL + ML pipeline:

```
NBA API → src/data/ → data/raw/ → src/processing/ → data/processed/
→ src/features/ → data/features/ → src/models/ → models/artifacts/
→ src/models/predict_cli.py → stdout (JSON)
```

The system works but has three structural gaps that block the desired evolution:

1. **External data has no designated home.** Basketball Reference scrapers, injury reports, and odds-as-features have nowhere clean to live — they would be wedged into `src/data/` (NBA API layer) or `src/processing/` (consolidation layer), creating conceptual confusion.

2. **The ATS model has no separate target.** The current architecture models `home_win` (binary). An ATS model needs spread coverage as a target, which requires sportsbook lines as *inputs* at feature engineering time — currently they only appear post-prediction in `data/odds/model_vs_odds.csv`.

3. **Outputs are not persistent or web-queryable.** Predictions go to stdout; the `model_vs_odds.csv` file exists but has no history, no schema versioning, and is not stored in SQLite.

---

## Recommended Architecture

### The Evolved Five-Layer Pipeline + Outputs Layer

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 0: Data Ingestion                                         │
│  (unchanged structure, expanded scope)                            │
│                                                                   │
│  src/data/                        src/data/external/             │
│  ├── get_*.py (NBA API, 20+)      ├── bref_scraper.py            │
│  └── get_odds.py (Odds API)       ├── injury_report.py           │
│                                   └── referee_data.py (future)   │
└────────────────────┬─────────────────────────┬───────────────────┘
                     │                         │
                     ▼                         ▼
              data/raw/                data/raw/external/
              (existing NBA API)       (new: bbref, injury)

┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1: Preprocessing                                          │
│  (expanded to handle external sources)                           │
│                                                                   │
│  src/processing/                                                  │
│  ├── preprocessing.py (NBA API → data/processed/)                │
│  └── external_preprocessing.py (external → data/processed/)      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
                      data/processed/
                      (all sources merged, canonical tables)

┌──────────────────────────────────────────────────────────────────┐
│  LAYER 2: Feature Engineering                                    │
│  (new: schedule, external, and ATS-specific features)            │
│                                                                   │
│  src/features/                                                    │
│  ├── team_game_features.py (existing rolling stats)              │
│  ├── player_features.py (existing player rolling)                │
│  ├── schedule_features.py (new: rest, B2B, travel)               │
│  ├── injury_features.py (new: replaces injury_proxy.py)          │
│  ├── external_features.py (new: BBref advanced, referee)         │
│  ├── odds_features.py (new: spread as input feature)             │
│  └── era_labels.py (existing)                                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
                      data/features/
                      ├── game_matchup_features.csv (existing, expanded)
                      ├── player_game_features.csv (existing)
                      └── game_ats_features.csv (new: adds spread as feature)

┌──────────────────────────────────────────────────────────────────┐
│  LAYER 3: Model Training                                         │
│  (new: ATS model alongside existing win-probability model)       │
│                                                                   │
│  src/models/                                                      │
│  ├── game_outcome_model.py (existing: predicts home_win)         │
│  ├── ats_model.py (new: predicts covers_spread binary)           │
│  ├── player_performance_model.py (existing: pts/reb/ast)         │
│  ├── playoff_odds_model.py (existing: Monte Carlo)               │
│  ├── train_all_models.py (existing orchestrator, expanded)       │
│  ├── backtesting.py (existing, extend to ATS model)              │
│  └── calibration.py (existing, wire to production path)          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
                      models/artifacts/
                      ├── game_outcome_model.pkl (existing)
                      ├── game_outcome_model_calibrated.pkl (wire up)
                      ├── ats_model.pkl (new)
                      └── *_metadata.json (existing pattern)

┌──────────────────────────────────────────────────────────────────┐
│  LAYER 4: Prediction & Value Detection                           │
│  (expanded: ATS prediction + value bet flagging)                 │
│                                                                   │
│  src/models/predict_cli.py (expanded: add --spread flag)         │
│  src/models/value_bets.py (new: identifies disagreement)         │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 5: Outputs (NEW LAYER)                                    │
│  (web-ready: persistent storage, JSON, queryable history)        │
│                                                                   │
│  src/outputs/                                                     │
│  ├── prediction_store.py (persist predictions to SQLite)         │
│  └── json_export.py (serialize daily summary as JSON)            │
│                                                                   │
│  data/outputs/                                                    │
│  ├── predictions_history.db (SQLite: all predictions + results)  │
│  ├── daily_predictions.json (web-consumable snapshot)            │
│  └── value_bets.json (flagged games, web-consumable)             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Component Boundaries

| Component | Responsibility | Inputs | Outputs | Talks To |
|-----------|----------------|--------|---------|----------|
| `src/data/get_*.py` | Fetch NBA API endpoints, retry logic, season loops | nba_api | `data/raw/*/` CSVs | update.py orchestrator |
| `src/data/external/` | Fetch Basketball Reference, NBA injury reports | HTTP/scraping | `data/raw/external/*/` CSVs | update.py orchestrator |
| `src/processing/preprocessing.py` | Consolidate NBA API raw CSVs → canonical tables | `data/raw/` | `data/processed/` CSVs | Feature engineering layer |
| `src/processing/external_preprocessing.py` | Consolidate external raw CSVs → canonical tables | `data/raw/external/` | `data/processed/` CSVs | Feature engineering layer |
| `src/features/team_game_features.py` | Rolling stats, SOS, rest, volatility, matchup diffs | `data/processed/` | `data/features/game_matchup_features.csv` | Model training |
| `src/features/schedule_features.py` | B2B detection, days rest, travel distance | `data/processed/team_game_logs.csv` | Merged into matchup features | Feature engineering |
| `src/features/injury_features.py` | Official injury report + proxy for historical | `data/processed/`, `data/raw/external/injuries/` | Injury columns in matchup features | Feature engineering |
| `src/features/odds_features.py` | Sportsbook lines as model inputs, implied probability | `data/odds/game_lines.csv` | `data/features/game_ats_features.csv` | ATS model training only |
| `src/models/game_outcome_model.py` | Train/predict home_win probability | `data/features/game_matchup_features.csv` | `models/artifacts/game_outcome_model.pkl` | predict_cli, value_bets |
| `src/models/ats_model.py` | Train/predict covers_spread binary | `data/features/game_ats_features.csv` | `models/artifacts/ats_model.pkl` | predict_cli, value_bets |
| `src/models/value_bets.py` | Compare model probability to market implied odds | `models/artifacts/`, `data/odds/game_lines.csv` | `data/outputs/value_bets.json` | Outputs layer |
| `src/outputs/prediction_store.py` | Persist predictions + eventual results to SQLite | Prediction dicts | `data/outputs/predictions_history.db` | Web layer (future) |
| `src/outputs/json_export.py` | Serialize daily predictions to web-ready JSON | Prediction dicts | `data/outputs/daily_predictions.json` | Web layer (future) |
| `update.py` | Daily orchestrator: fetch → preprocess → odds → outputs | All above | Populated DB + JSON files | Task Scheduler |

### Boundary Rules

**Rule 1 — External data stays in its own subdirectory.** NBA API fetchers live in `src/data/`. External scrapers (Basketball Reference, injury reports) live in `src/data/external/`. They share the raw CSV output contract but do not share code. This prevents the NBA API layer from accumulating scraping logic.

**Rule 2 — Odds-as-features are ATS-only.** The `odds_features.py` module and the `game_ats_features.csv` output are exclusively for the ATS model. The win-probability model (`game_outcome_model.py`) must NOT receive spread as an input — doing so would create a model that predicts against itself. Separate feature tables enforce this boundary.

**Rule 3 — Models do not call each other.** The ATS model and win-probability model are trained independently. `value_bets.py` is the only component that loads both artifacts simultaneously to compute the disagreement signal.

**Rule 4 — Outputs are a distinct layer, not a side effect.** Predictions are not written to disk from inside model training scripts. They flow through `predict_cli.py` or an equivalent orchestration call, then into `src/outputs/` modules which handle persistence. This makes the outputs layer swappable without touching model code.

---

## Data Flow

### Daily Update Flow (Expanded)

```
07:00 Task Scheduler → update.py
  │
  ├─ Step 1A: NBA API regular season fetch
  │   src/data/get_*.py → data/raw/{endpoint}/
  │
  ├─ Step 1B: External data fetch (NEW)
  │   src/data/external/injury_report.py → data/raw/external/injuries/
  │   src/data/external/bref_scraper.py  → data/raw/external/bref/
  │
  ├─ Step 1C: Odds refresh
  │   scripts/fetch_odds.py → data/odds/game_lines.csv
  │
  ├─ Step 2: Preprocess all sources
  │   src/processing/preprocessing.py → data/processed/ (NBA API tables)
  │   src/processing/external_preprocessing.py → data/processed/ (external tables)
  │
  ├─ Step 3: [Manual trigger only] Feature engineering + model training
  │   (Not run daily — training runs on demand after sufficient new data)
  │
  └─ Step 4: Generate daily outputs (NEW)
      src/models/predict_cli.py → predictions for today's games
      src/models/value_bets.py  → flags disagreements with market
      src/outputs/prediction_store.py → data/outputs/predictions_history.db
      src/outputs/json_export.py → data/outputs/daily_predictions.json
                                → data/outputs/value_bets.json
```

### Model Training Flow (Expanded)

```
Manual: python src/models/train_all_models.py --rebuild-features
  │
  ├─ Build matchup features (win-probability inputs)
  │   src/features/team_game_features.py
  │   src/features/schedule_features.py    (new)
  │   src/features/injury_features.py      (new, replaces injury_proxy.py)
  │   src/features/external_features.py    (new, if BBref data available)
  │   → data/features/game_matchup_features.csv
  │
  ├─ Build ATS features (win-prob features + spread as input)
  │   (Above matchup features) + src/features/odds_features.py
  │   → data/features/game_ats_features.csv
  │
  ├─ Train game_outcome_model (target: home_win)
  │   Input: game_matchup_features.csv
  │   Filters to modern era (2014+) if MODERN_ERA_ONLY = True
  │   Output: models/artifacts/game_outcome_model.pkl
  │
  ├─ Train ats_model (target: covers_spread)
  │   Input: game_ats_features.csv (includes spread)
  │   Output: models/artifacts/ats_model.pkl
  │
  ├─ Train player_performance_model (targets: pts/reb/ast)
  │   Input: player_game_features.csv
  │   Output: models/artifacts/player_{stat}_{algo}.pkl
  │
  └─ Simulate playoff_odds (Monte Carlo)
      Input: data/processed/standings.csv
      Output: in-memory result, optionally exported to JSON
```

### Prediction + Value Bet Flow

```
User or update.py triggers prediction step
  │
  ├─ predict_cli.py game --home BOS --away LAL
  │   Load game_outcome_model.pkl → win probability (e.g. 0.58)
  │   Load ats_model.pkl          → cover probability (e.g. 0.54)
  │   Load game_lines.csv         → current spread (e.g. BOS -3.5)
  │   Output: {
  │     "home_win_prob": 0.58,
  │     "cover_prob": 0.54,
  │     "market_spread": -3.5,
  │     "market_implied_prob": 0.52,
  │     "value_bet_flag": true,     ← model disagrees with market by >5pp
  │     "edge": "+2pp"
  │   }
  │
  └─ src/outputs/prediction_store.py
      Append to data/outputs/predictions_history.db
      Table: predictions(game_date, home_team, away_team, home_win_prob,
                          cover_prob, market_spread, value_bet_flag,
                          actual_result, actual_cover, model_version)
```

### Outputs State Management

```
data/outputs/predictions_history.db
  - SQLite, append-only during season
  - Schema includes actual_result column (backfilled by update.py next day)
  - Enables accuracy tracking over time
  - Queryable for web layer without model rerun

data/outputs/daily_predictions.json
  - Overwritten daily by json_export.py
  - Contains today's game predictions + value bets
  - Web layer reads this file; no direct DB connection required initially

data/outputs/value_bets.json
  - Subset of daily_predictions.json where value_bet_flag = true
  - Separate file for easy web consumption
```

---

## Patterns to Follow

### Pattern 1: External Scraper Module
**What:** One Python module per external data source in `src/data/external/`.
**When:** Adding Basketball Reference, NBA injury reports, referee data.
**Structure:**
```python
# src/data/external/injury_report.py
INJURIES_DIR = "data/raw/external/injuries"
OUTPUT_PREFIX = "injury_report"

def fetch_todays_injury_report() -> pd.DataFrame:
    """Fetch NBA official injury report (PDF or web)."""
    ...

def save_injury_report(df: pd.DataFrame, date_str: str) -> None:
    """Save to data/raw/external/injuries/injury_report_{date}.csv"""
    ...

def get_injury_report(date_str: str = None) -> pd.DataFrame:
    """Public API for update.py. Default: today."""
    ...
```
**Why:** Mirrors existing `src/data/get_*.py` pattern. Single responsibility per source. Easy to add/remove from update.py without touching other modules.

### Pattern 2: Separate Feature Tables Per Model Target
**What:** The ATS model reads `game_ats_features.csv`; the win-probability model reads `game_matchup_features.csv`. They share most columns, but `game_ats_features.csv` additionally contains the sportsbook spread, implied probability, and line movement.
**When:** Any time a feature should only be visible to one model.
**Why:** Prevents accidental feature leakage between tasks. Makes it impossible to accidentally feed spread into the win-probability model.
```python
# In src/features/odds_features.py
def build_ats_feature_table() -> pd.DataFrame:
    """Load game_matchup_features.csv, join spread data, output game_ats_features.csv."""
    matchup_df = pd.read_csv("data/features/game_matchup_features.csv")
    odds_df = pd.read_csv("data/odds/game_lines.csv")
    merged = matchup_df.merge(odds_df, on=["game_date", "home_team", "away_team"], how="left")
    merged["spread_implied_prob"] = _american_to_prob(merged["home_ml"])
    merged.to_csv("data/features/game_ats_features.csv", index=False)
    return merged
```

### Pattern 3: Prediction Store (Append-Only SQLite)
**What:** A dedicated module writes predictions to SQLite after each prediction run.
**When:** Every daily prediction cycle; backfill actual results next morning.
**Structure:**
```python
# src/outputs/prediction_store.py
PRED_DB = "data/outputs/predictions_history.db"

def store_prediction(prediction: dict) -> None:
    """Append one game prediction to predictions_history.db."""
    ...

def backfill_actuals(game_date: str) -> int:
    """Look up game results in team_game_logs and update actual_result column."""
    ...
```
**Why:** Makes model accuracy trackable over time. Keeps prediction history queryable without re-running the model. Decouples the web layer — it reads the DB, it does not call the model.

### Pattern 4: JSON Export as Contract with Web Layer
**What:** `json_export.py` writes a well-defined JSON schema daily that the web platform will consume.
**When:** End of each daily update cycle.
**Schema:**
```json
{
  "generated_at": "2026-03-01T07:15:00Z",
  "model_version": "game_outcome_gb_20260301",
  "games": [
    {
      "game_date": "2026-03-01",
      "home_team": "BOS",
      "away_team": "LAL",
      "home_win_prob": 0.58,
      "cover_prob": 0.54,
      "market_spread": -3.5,
      "value_bet": true,
      "confidence": "medium"
    }
  ]
}
```
**Why:** Freezes the contract now so the web milestone has a guaranteed input format. The web platform never imports Python modules — it reads files.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Odds Data as Both Input and Output in the Same Model
**What:** Using `game_lines.csv` spread as a feature inside `game_outcome_model.py`.
**Why bad:** The win-probability model's purpose is to discover signal *independent* of the market. If the market spread is an input, the model mostly learns to reproduce market consensus, and the "disagreement" value bet signal disappears. Vegas lines are highly efficient — a model trained on them mostly rediscovers the market.
**Instead:** Feed spread only into `ats_model.py`. Use `value_bets.py` to compare `game_outcome_model` probability to market implied probability externally.

### Anti-Pattern 2: Monolithic Update Orchestrator
**What:** Expanding `update.py` into a 300-line script that handles data fetching, preprocessing, feature engineering, training, prediction, and output serialization.
**Why bad:** Today's `update.py` is ~90 lines and cleanly calls separate modules. If new steps (external scraping, injury parsing, ATS feature building, prediction storage) are inlined, debugging a daily failure means tracing through a single giant script.
**Instead:** Keep `update.py` as a thin orchestrator. Each new capability lives in its own module and is imported with a one-line call from the orchestrator.

### Anti-Pattern 3: Mixing Historical and Live Injury Data
**What:** Using the same code path to populate injury features for training (historical proxy from game logs) and for prediction (today's official injury report).
**Why bad:** The injury proxy (`injury_proxy.py`) infers past absences from log gaps — this is fine for model training on historical data. But for today's game prediction, real injury reports are available. Mixing them produces wrong signal for live predictions: the proxy will always estimate 100% availability until game time.
**Instead:** `injury_features.py` should have two separate paths:
- `build_historical_injury_features(df)` — proxy-based, used during training on historical data
- `get_live_injury_features(home_team, away_team)` — reads today's official injury report, used at prediction time

### Anti-Pattern 4: Saving Model Artifacts Without Loading Them
**What:** The existing `calibration.py` saves `game_outcome_model_calibrated.pkl` but nothing ever loads it for production predictions.
**Why bad:** Work that produces no output is waste. Calibrated probabilities are meaningfully different — they are required for the value bet edge calculation to be accurate.
**Instead:** Wire the calibrated model into the prediction path. In `predict_game()`, load `game_outcome_model_calibrated.pkl` if it exists, fall back to `game_outcome_model.pkl`. Document this in a comment.

---

## Scalability Considerations

| Concern | Now (current scale) | Web milestone | Far future |
|---------|---------------------|---------------|------------|
| Prediction storage | stdout only | SQLite append-only, MB scale | SQLite → Postgres migration if needed |
| Feature computation | Regenerate all features before training (~5 min) | Same — acceptable | Incremental feature updates if full rebuild exceeds 30 min |
| External data fetching | Not implemented | Add to daily update, ~2-5 min additional | Rate limit management per source |
| Odds integration | Subprocess call to `fetch_odds.py` | Inline module call for reliability | Dedicated odds microservice if multiple sources added |
| Model retraining | Manual, ~10 min | Weekly scheduled trigger | Automated if accuracy drift detected |
| SQLite concurrency | Single writer, fine | Web reads while update writes — use WAL mode | Migrate to Postgres if concurrent writes needed |

**SQLite WAL mode note (HIGH confidence):** Enable WAL journal mode when creating `predictions_history.db` so the web layer can read while `update.py` is writing. One line at DB creation: `PRAGMA journal_mode=WAL;`

---

## Build Order and Phase Dependencies

The components above have hard dependencies that constrain what can be built when:

```
Phase 1 (Foundation — must come first):
  Fix injury_proxy.py → verify features reach model
  Add schedule_features.py (rest/B2B)
  Wire calibrated model into predict_game()
  [No external data yet — these fix existing broken signals]

Phase 2 (External Data — unlocks better features):
  src/data/external/injury_report.py
  src/data/external/bref_scraper.py
  src/processing/external_preprocessing.py
  [Phase 1 must complete first so feature table structure is stable]

Phase 3 (ATS Model — requires external data + feature table):
  src/features/odds_features.py → game_ats_features.csv
  src/models/ats_model.py (target: covers_spread)
  src/models/value_bets.py
  [Requires Phase 2 data to be in processed tables for rich features]
  [Requires at least one full season of spread data in game_lines.csv history]

Phase 4 (Persistent Outputs — can start in parallel with Phase 3):
  src/outputs/prediction_store.py
  src/outputs/json_export.py
  data/outputs/predictions_history.db schema
  [Can be built as soon as Phase 1 is complete; does not require ATS model]
  [Start now to accumulate prediction history before web milestone]
```

**Critical path:** Phase 1 → Phase 2 → Phase 3. Phase 4 can run in parallel after Phase 1.

**Reason Phase 1 is first:** External data and ATS features are wasted signal if the injury features that the current model already knows about are silently null. Fix what's broken before adding more inputs.

**Reason Phase 4 starts early:** Prediction history only accumulates while the system is running. Starting the prediction store in Phase 1 or 2 means having months of tracked predictions by the web milestone. Starting it at web milestone means starting with an empty history.

---

## Directory Layout for Evolved System

```
nba-analytics-project/
├── update.py                          # (existing, expanded by ~20 lines)
│
├── src/
│   ├── data/
│   │   ├── get_*.py                   # (existing, 20+ NBA API modules)
│   │   ├── get_odds.py                # (existing)
│   │   └── external/                  # (NEW directory)
│   │       ├── injury_report.py       # NBA official injury PDFs/web
│   │       └── bref_scraper.py        # Basketball Reference advanced stats
│   │
│   ├── processing/
│   │   ├── preprocessing.py           # (existing, unchanged)
│   │   └── external_preprocessing.py  # (NEW: consolidate external raw → processed)
│   │
│   ├── features/
│   │   ├── team_game_features.py      # (existing)
│   │   ├── player_features.py         # (existing)
│   │   ├── era_labels.py              # (existing)
│   │   ├── injury_features.py         # (NEW: replaces/extends injury_proxy.py)
│   │   ├── schedule_features.py       # (NEW: rest/B2B/travel, extracted from team_game_features)
│   │   ├── external_features.py       # (NEW: BBref advanced stats as features)
│   │   └── odds_features.py           # (NEW: spread as ATS model input)
│   │
│   ├── models/
│   │   ├── game_outcome_model.py      # (existing, use calibrated model)
│   │   ├── ats_model.py               # (NEW: covers_spread classifier)
│   │   ├── player_performance_model.py# (existing)
│   │   ├── playoff_odds_model.py      # (existing)
│   │   ├── train_all_models.py        # (existing, add ats_model call)
│   │   ├── predict_cli.py             # (existing, add spread/ATS output)
│   │   ├── value_bets.py              # (NEW: flag model vs market disagreement)
│   │   ├── backtesting.py             # (existing, extend to ATS)
│   │   └── calibration.py             # (existing, wire to production)
│   │
│   └── outputs/                       # (NEW directory)
│       ├── prediction_store.py        # Append predictions to SQLite
│       └── json_export.py             # Write daily_predictions.json
│
├── data/
│   ├── raw/
│   │   ├── (existing NBA API subdirectories)
│   │   └── external/                  # (NEW)
│   │       ├── injuries/              # injury_report_{date}.csv
│   │       └── bref/                  # bref_{stat_type}_{season}.csv
│   ├── processed/                     # (existing, expanded to include external tables)
│   ├── features/
│   │   ├── game_matchup_features.csv  # (existing: win-prob model inputs)
│   │   ├── game_ats_features.csv      # (NEW: matchup + spread = ATS model inputs)
│   │   └── player_game_features.csv   # (existing)
│   ├── odds/                          # (existing)
│   └── outputs/                       # (NEW directory)
│       ├── predictions_history.db     # SQLite: all predictions + actuals
│       ├── daily_predictions.json     # Today's predictions (web-consumable)
│       └── value_bets.json            # Today's flagged bets (web-consumable)
│
└── models/artifacts/                  # (existing, add ats_model.pkl)
```

---

## Sources

- Codebase analysis: `src/models/game_outcome_model.py`, `src/data/get_odds.py`, `src/features/team_game_features.py`, `src/features/injury_proxy.py`, `update.py`, `scripts/fetch_odds.py` — direct inspection, 2026-03-01
- Known issues: `.planning/codebase/CONCERNS.md` — calibrated model never loaded, injury features silently null
- Project requirements: `.planning/PROJECT.md` — ATS model, web-ready outputs, external data sources
- Confidence: HIGH — all claims derived from direct codebase inspection, no inferred or web-sourced claims

---

*Architecture research: 2026-03-01*
