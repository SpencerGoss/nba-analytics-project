---
phase: "06"
plan: "01"
subsystem: production-fixes-injury-data
tags: [model-calibration, injury-data, fetch-odds, update-pipeline]
dependency_graph:
  requires: [models/artifacts/game_outcome_model_calibrated.pkl, scripts/fetch_odds.py, update.py]
  provides: [src/data/get_injury_data.py, data/raw/injuries/]
  affects: [update.py, scripts/fetch_odds.py]
tech_stack:
  added: [pdfplumber (optional, PDF fallback)]
  patterns: [inference-path-boundary, calibrated-model-fallback, non-fatal-pipeline-step]
key_files:
  created: [src/data/get_injury_data.py]
  modified: [scripts/fetch_odds.py, update.py]
decisions:
  - "Use try/except FileNotFoundError pattern in fetch_odds.py to prefer calibrated model with graceful fallback"
  - "New injury fetcher (get_injury_data.py) wraps NBA PDF source with nba_api as primary; saves dated CSVs to data/raw/injuries/"
  - "Injury step in update.py is non-fatal (try/except) so a missing pdfplumber or API failure never breaks the daily pipeline"
  - "Respected FR-4.4 training/inference code path boundary; injury fetcher is INFERENCE PATH only"
metrics:
  duration: "25 min"
  completed_date: "2026-03-04"
---

# Phase 6 Plan 1: Production Fixes & Injury Data Integration Summary

**One-liner:** Wired calibrated model into fetch_odds.py with FileNotFoundError fallback, added daily injury report fetcher saving dated CSVs to data/raw/injuries/ via NBA PDF source.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Wire calibrated model into fetch_odds.py | f8170ca | scripts/fetch_odds.py |
| 2 | Create src/data/get_injury_data.py | 74760be | src/data/get_injury_data.py |
| 3 | Integrate get_injury_report() into update.py | 74760be | update.py |

## What Was Built

### FIX-01/02: Calibrated Model in fetch_odds.py

`load_model_game_projections()` in `scripts/fetch_odds.py` now:
- First attempts to load `models/artifacts/game_outcome_model_calibrated.pkl`
- If `FileNotFoundError`, logs a `log.warning()` and falls back to `game_outcome_model.pkl`
- Outer `except Exception` still falls back to the feature-based proxy as before
- Matches the exact pattern from `src/models/game_outcome_model.py`

### DATA-01: Injury Data Fetcher

New module `src/data/get_injury_data.py` provides:

**`get_injury_report(season_year=None)`**
- Primary source: `nba_api` `LeagueInjuryReport` endpoint (gracefully skips if unavailable)
- Fallback: NBA official PDF reports from `ak-static.cms.nba.com` — tries 5 time slots in reverse order (most recent first)
- Saves to `data/raw/injuries/injury_report_YYYY-MM-DD.csv` (one file per day, overwritten on re-run)
- Output schema: `date, player_name, player_id, team_abbr, status, injury_type`
- Status values normalized: `out | questionable | probable | day-to-day`

**`load_historical_injuries(injuries_dir=None)`**
- Globs all `injury_report_*.csv` files from `data/raw/injuries/`
- Concatenates and returns as a single DataFrame

**update.py Step 4:**
- Calls `get_injury_report(season_year=current_year)` in the daily pipeline
- Wrapped in try/except — failure is non-fatal, pipeline continues

## Deviations from Plan

### Auto-fixed Issues

None — plan executed as written with one implementation note:

**Note: nba_api LeagueInjuryReport not available in installed version**
- Found during Task 2: `LeagueInjuryReport` is not exposed in the installed `nba_api` version
- The script handles this via try/except ImportError and falls back to the PDF approach
- This matches the behavior already present in `src/data/external/injury_report.py`
- The new script still exposes the `get_injury_report()` interface as required

## Self-Check: PASSED

Files created/modified:
- `scripts/fetch_odds.py` — calibrated model loading at line 320-335
- `src/data/get_injury_data.py` — 260-line module with get_injury_report() and load_historical_injuries()
- `update.py` — Step 4 injury report call at line 130-138

Commits:
- f8170ca: fix(v2/phase6): wire calibrated model into fetch_odds.py
- 74760be: feat(v2/phase6): add injury data fetcher and integrate into update pipeline

Test suite: 59/59 passed after all changes.
