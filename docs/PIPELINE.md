# NBA Analytics Pipeline Reference

## Overview

The pipeline has two operating modes:

- **Daily data refresh** (run by `update.py` on schedule): fetches current-season data only, no model retraining
- **Full rebuild** (manual): all six stages in order, required after schema changes or for new season model training

## Stage Order (Full Rebuild)

| Stage | Command | Inputs | Outputs | Est. Runtime |
|-------|---------|--------|---------|--------------|
| 1. Fetch | `python update.py` | NBA API (nba_api) | `data/raw/*/` | 10-15 min |
| 2. Preprocess | (called by update.py) | `data/raw/` | `data/processed/*.csv` | 1-2 min |
| 3. Feature build | `python src/features/team_game_features.py` | `data/processed/` | `data/features/` | 5-10 min |
| 4. Train | `python src/models/train_all_models.py` | `data/features/` | `models/artifacts/` | 5-10 min |
| 5. Calibrate | `python src/models/calibration.py` | `data/features/` + `models/artifacts/` | `models/artifacts/game_outcome_model_calibrated.pkl` + `reports/calibration/` | 1-2 min |
| 6. Predict | `python src/models/predict_cli.py game --home X --away Y` | `data/features/` + `models/artifacts/` | stdout + `database/predictions_history.db` + `data/outputs/predictions_YYYYMMDD.json` | <1 min |

Stages are strictly sequential. No stage can run before its predecessor completes.

## Daily Data Refresh (update.py)

`python update.py` runs **Stages 1 and 2 only**. It does NOT run feature build, training, or calibration.

After a daily refresh, the existing model artifacts continue to serve predictions from the previous training run. This is intentional — model retraining happens separately on a schedule or manually.

Refresh schedule: daily at 7:00 AM via Windows Task Scheduler (`scripts/run_update.bat`).
Estimated runtime: 2-5 min for current-season fetch + preprocessing.

## What update.py Does (and Does Not) Do

**Does:**
- Auto-detects current NBA season (month >= 10 → current calendar year, else previous year)
- Fetches current season data for all registered endpoints (~18 module imports, ~10-15 API calls)
- Conditionally fetches playoff data if month >= 4 (April)
- Conditionally fetches hustle stats if year >= 2015 (NBA tracking data starts 2015-16)
- Rebuilds ALL `data/processed/*.csv` tables from all accumulated raw seasons
- Refreshes sportsbook odds data via Pinnacle guest API (free, no key required; non-fatal on network error)
- Logs errors to `logs/pipeline_errors.log`

**Does NOT:**
- Rebuild feature tables (`data/features/*.csv`) — run Stage 3 manually
- Retrain models (`models/artifacts/`) — run Stage 4 manually
- Recalibrate models — run Stage 5 manually after every retraining run

## Dependency Graph

```
Stage 1 (Fetch)
    └── Stage 2 (Preprocess)
            └── Stage 3 (Feature Build)
                    └── Stage 4 (Train)
                            └── Stage 5 (Calibrate)
                                    └── Stage 6 (Predict)
```

Each stage depends on all preceding stages completing successfully.

## Stage Details

### Stage 1: Fetch (`update.py`)

- **Entry point:** `update.py` (project root)
- **Module pattern:** calls `get_*_all_seasons(start_year, end_year)` via one-line imports from each `src/data/get_*.py`
- **Output:** one CSV per season per table in `data/raw/{table_name}/{table_name}_{season}.csv`
- **API rate limiting:** each fetcher uses `fetch_with_retry()` with 3 retries and 10-second delays
- **Season format:** 6-digit integer (e.g., `202425` for 2024-25)
- **Reference tables:** `get_player_master()` and `get_teams()` are always refreshed (no year args)

### Stage 2: Preprocess (`src/processing/preprocessing.py`)

- **Entry point:** `run_preprocessing()` called by `update.py`
- **Input:** all raw CSVs in `data/raw/`
- **Output:** `data/processed/*.csv` — one file per table type, all seasons combined
- **Column normalization:** raw UPPERCASE → processed lowercase_underscore
- **Key column renames:** `person_id → player_id`, `display_first_last → player_name`, `from_year → from_season`; standings: `teamid → team_id`, `wins → w`, `losses → l`, `winpct → w_pct`

### Stage 3: Feature Build (`src/features/team_game_features.py`)

- **Entry point:** `build_team_game_features()` then `build_matchup_dataset()` (or `python src/features/team_game_features.py`)
- **Optional via train script:** `python src/models/train_all_models.py --rebuild-features` runs Stage 3 first
- **Input:** `data/processed/team_game_logs.csv`, `data/processed/standings.csv`, injury proxy features
- **Output:** `data/features/team_game_features.csv`, `data/features/game_matchup_features.csv`
- **Rolling windows:** 5, 10, 20 games; stats include pts, fg_pct, reb, ast, plus_minus, opp_pts (defensive signal)
- **Critical:** injury proxy join requires `game_id` and `team_id` cast to consistent types; uses `merge_asof` with `MAX_STALE_DAYS=25` for absent rotation detection
- **NFR-1 (no lookahead):** all rolling features use `.shift(1)` before `.rolling()` — prior games only

### Stage 4: Train (`src/models/train_all_models.py`)

- **Entry point:** `python src/models/train_all_models.py` (add `--rebuild-features` to run Stage 3 first)
- **Input:** `data/features/game_matchup_features.csv`, `data/features/player_game_features.csv`
- **Output:** `models/artifacts/*.pkl` + `models/artifacts/game_outcome_metadata.json`
- **Models trained (3 tasks):**
  1. Game outcome (GradientBoosting / RandomForest / Logistic — best selected automatically)
  2. Player performance (Ridge / GBM / RandomForest per target: pts, reb, ast)
  3. Playoff odds (simulation-based)
- **Validation:** season-based expanding windows; held-out test seasons defined at top of each model file

### Stage 5: Calibrate (`src/models/calibration.py`)

- **Entry point:** `python src/models/calibration.py` (or `from src.models.calibration import run_calibration_analysis`)
- **Input:** `data/features/game_matchup_features.csv` + `models/artifacts/game_outcome_model.pkl`
- **Output (artifacts):** `models/artifacts/game_outcome_model_calibrated.pkl`
- **Output (reports):** `reports/calibration/calibration_curve.png`, `reports/calibration/calibration_metrics.csv`, `reports/calibration/calibration_by_era.csv`, `reports/calibration/calibration_by_season.csv`
- **When to run:** after every Stage 4 training run — the calibrated artifact becomes stale after retraining
- **Note:** Stage 6 prefers the calibrated artifact; it warns (does not crash) if only the uncalibrated model is found

### Stage 6: Predict (`src/models/predict_cli.py`)

- **Entry point:** `python src/models/predict_cli.py game --home BOS --away LAL [--date YYYY-MM-DD]`
- **Also supports:** `python src/models/predict_cli.py player --name "LeBron James"`
- **Input:** `data/features/game_matchup_features.csv` (last-built), `models/artifacts/`
- **Output:** JSON to stdout + row in `database/predictions_history.db` (WAL mode) + `data/outputs/predictions_YYYYMMDD.json`
- **Prediction store:** non-fatal — store write failure issues UserWarning but never prevents result from being returned
- **Note:** uses last-built features CSV, not rebuilt at inference time — ensure Stage 3 was run after the latest data refresh

## External Data Scrapers (FR-7.2)

External data modules live in `src/data/external/` and follow the `src/data/get_*.py` callable pattern:

| Module | Function | Data Source | Output | Runtime |
|--------|----------|-------------|--------|---------|
| `bref_scraper.py` | `get_referee_crew_assignments(start_date, end_date)` | Basketball Reference box scores | `data/raw/external/referee_crew/` | ~3 sec/game (rate limited) |
| `injury_report.py` | `get_todays_nba_injury_report()` | NBA API + PDF fallback | `data/raw/external/injury_reports/` | <5 sec |

### Basketball Reference Referee Scraper

- **Entry point:** `from src.data.external.bref_scraper import get_referee_crew_assignments`
- **Usage:** `get_referee_crew_assignments("2025-01-01", "2025-03-31")` -- on-demand only
- **Rate limit:** 3-second delay between requests (robots.txt + NFR-2). Maximum 20 requests/minute.
- **NOT in daily pipeline:** The scraper takes ~3 sec/game. A full season (~1,230 games) takes ~1 hour. Run manually for historical backfill.
- **Output:** CSV files in `data/raw/external/referee_crew/` with columns: game_date, game_id_bref, home_team, away_team, referee_1, referee_2, referee_3

### NBA Injury Report Fetcher

- **Entry point:** `from src.data.external.injury_report import get_todays_nba_injury_report`
- **Usage:** `get_todays_nba_injury_report()` -- fetches current day's report
- **Primary source:** `nba_api` `LeagueInjuryReport` endpoint (no URL guessing needed)
- **Fallback:** PDF from `ak-static.cms.nba.com` if API is unavailable
- **Code path:** INFERENCE ONLY -- never use for training. Training uses `build_injury_proxy_features()` from `src/features/injury_proxy.py`
- **Output:** CSV snapshots in `data/raw/external/injury_reports/` for archival

### Training vs Inference Injury Paths (FR-4.4)

| Path | Module | Data Source | When Used |
|------|--------|-------------|-----------|
| Training | `src/features/injury_proxy.py` -> `build_injury_proxy_features()` | Historical game logs (merge_asof proxy) | Feature build (Stage 3) |
| Inference | `src/data/external/injury_report.py` -> `get_todays_nba_injury_report()` | Live NBA injury report (API + PDF) | Prediction (Stage 6) |

These two paths MUST NOT share inputs or call each other. See FR-4.4.

**NBA API modules (src/data/get_*.py):**

- **Callable as module:** `from src.data.get_player_stats import get_player_stats_all_seasons`
- **Function signature:** `get_*(start_year, end_year)` or `get_*()` for dimension tables
- **Output:** CSVs to `data/raw/{table_name}/` (raw), then `data/processed/` after Stage 2
- **Rate limiting:** `fetch_with_retry()` with 3 retries, appropriate delays per source

## Common Operations

### After installing on a new machine (first-time setup)

```bash
python backfill.py                                      # ~30-45 min: fetch all historical data (Stages 1+2)
python src/features/team_game_features.py               # Stage 3
python src/models/train_all_models.py                   # Stage 4
python src/models/calibration.py                        # Stage 5
```

### After a data refresh (daily, no retraining)

```bash
python update.py            # Stages 1+2 only
# Model continues to use last-trained artifacts
```

### Full rebuild (new season data, retrain model)

```bash
python update.py                                        # Stages 1+2
python src/features/team_game_features.py               # Stage 3
python src/models/train_all_models.py                   # Stage 4
python src/models/calibration.py                        # Stage 5
```

### Shortcut: rebuild features + train in one command

```bash
python src/models/train_all_models.py --rebuild-features    # Stages 3+4
python src/models/calibration.py                            # Stage 5
```

### Run a prediction

```bash
python src/models/predict_cli.py game --home BOS --away LAL --date 2026-03-15
python src/models/predict_cli.py player --name "LeBron James"
```

### Backfill referee data (one-time, on-demand)

```bash
python -c "from src.data.external.bref_scraper import get_referee_crew_assignments; get_referee_crew_assignments('2013-10-01', '2025-06-30')"
# ~3,700 games at 3 sec each = ~3 hours. Run overnight.
```

---

*Pipeline reference: Phase 3 -- External Data Layer (2026-03-02)*
*Requirements covered: FR-4.4, FR-7.1, FR-7.2, FR-7.3, FR-7.4, NFR-2*
