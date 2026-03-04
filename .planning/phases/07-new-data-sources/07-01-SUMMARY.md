---
phase: "07"
plan: "01"
subsystem: new-data-sources
tags: [lineup-data, balldontlie, nba-api, TeamDashLineups, data-ingestion]
dependency_graph:
  requires: [src/data/api_client.py, nba_api]
  provides: [data/raw/lineups/, src/data/get_lineup_data.py, src/data/get_balldontlie.py]
  affects: [src/processing/preprocessing.py]
tech_stack:
  added: [nba_api.stats.endpoints.TeamDashLineups, requests (BallDontLie)]
  patterns: [fetch_with_retry, cursor-based pagination, optional SEASONAL_TABLE]
key_files:
  created:
    - src/data/get_lineup_data.py
    - src/data/get_balldontlie.py
    - data/raw/lineups/lineup_data_202324.csv
    - data/raw/lineups/lineup_data_202425.csv
  modified:
    - src/processing/preprocessing.py
decisions:
  - Used TeamDashLineups frame[1] (not frame[0]) for 5-man lineup data
  - BallDontLie v1 (no-auth) is sunset; v2 requires API key — implemented graceful fallback
  - GP >= 5 filter applied to remove noise from rarely-used lineup combinations
  - lineup_data added to SEASONAL_TABLES as optional=True (graceful skip when no files)
metrics:
  duration_minutes: 15
  completed_date: "2026-03-04"
  tasks_completed: 2
  files_created: 4
  files_modified: 1
---

# Phase 7 Plan 1: New Data Sources Summary

**One-liner:** 5-man lineup efficiency data (net/off/def rating) via TeamDashLineups for all 30 teams, plus BallDontLie v2 API client with graceful no-key fallback.

## What Was Built

### DATA-02: Lineup Efficiency Data (TeamDashLineups)

**src/data/get_lineup_data.py** — Fetches 5-man lineup efficiency stats from nba_api for all 30 NBA teams per season.

- Uses `TeamDashLineups` endpoint with `measure_type_detailed_defense='Advanced'` and `per_mode_detailed='Per100Possessions'`
- Fetches `frame[1]` from the endpoint response (frame[0] is team-level summary, frame[1] is per-lineup data)
- Applies `GP >= 5` filter to remove noise from rarely-used lineups
- Rate limits at 1 second between API calls (30 calls per season)
- Saves to `data/raw/lineups/lineup_data_{season_code}.csv` (one file per season)
- Function signature: `get_lineup_data(start_year: int = 2015, end_year: int = 2024) -> None`

**Output columns:** `season`, `team_id`, `team_abbreviation`, `group_name`, `gp`, `min`, `net_rating`, `off_rating`, `def_rating`, `ts_pct`, `ast_ratio`, `oreb_pct`, `dreb_pct`, `reb_pct`

**src/processing/preprocessing.py** — Added `_transform_lineup_data()` transform function and registered `lineup_data` in the `SEASONAL_TABLES` registry with `optional=True` so incremental preprocessing skips gracefully if no files exist.

### DATA-03: BallDontLie API Client

**src/data/get_balldontlie.py** — Client for BallDontLie NBA data API.

- `get_balldontlie_injuries()`: Returns empty DataFrame with warning. BallDontLie provides NO injuries endpoint in any API version. Injury data comes from NBA official PDF reports via `get_injury_data.py`.
- `get_balldontlie_stats(season)`: Fetches game results using cursor-based pagination. Requires `BALLDONTLIE_API_KEY` environment variable.
- `get_balldontlie_teams()`: Fetches NBA team directory.
- Returns empty DataFrames gracefully when API key is absent or request fails.

## Sample Row Counts

| Season | Teams | Lineups (GP >= 5) |
|--------|-------|-------------------|
| 2023-24 | 30 | 1,673 |
| 2024-25 | 30 | 1,566 |

Notable variation by team reflects coaching stability and rotation depth:
- GSW 2023-24: 83 lineups (deep rotation, many combinations)
- MEM 2023-24: 12 lineups (injuries decimated roster — few stable lineups)
- PHI 2023-24: 38 lineups (fewer due to Embiid injury absence)

## BallDontLie API Status

| Endpoint | Status | Notes |
|----------|--------|-------|
| v1 (balldontlie.io/api/v1/) | DEAD (404) | Sunset in 2024 |
| v2 (api.balldontlie.io/v1/) | Requires API key | 401 without key |
| /injuries | Does not exist | Use get_injury_data.py instead |
| /games | Available with key | Cursor-based pagination |
| /teams | Available with key | |

To use: Add `BALLDONTLIE_API_KEY=your_key` to `.env` file. Free keys available at https://www.balldontlie.io.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Discovery] BallDontLie v1 API sunset**

- **Found during:** Task research/Step 5 implementation
- **Issue:** The plan specified BallDontLie v1 (`balldontlie.io/api/v1/`) as "free, no-key required" but the v1 API returns 404 (it was sunset in 2024). The current API requires authentication.
- **Fix:** Implemented the client for v2 (api.balldontlie.io/v1) with API key support and graceful fallback when key is absent. Documented the API key requirement clearly in the module docstring.
- **Files modified:** src/data/get_balldontlie.py

**2. [Rule 1 - Discovery] TeamDashLineups endpoint parameter names differ from docs**

- **Found during:** Task 2/Step 2 implementation
- **Issue:** The plan suggested `measure_type_simple` and `per_mode_simple` parameters, but the actual endpoint uses `measure_type_detailed_defense` and `per_mode_detailed`.
- **Fix:** Inspected `inspect.signature()` at runtime to discover correct parameter names. Used `'Advanced'` for `measure_type_detailed_defense` and `'Per100Possessions'` for `per_mode_detailed`.
- **Files modified:** src/data/get_lineup_data.py

## Test Results

All 59 existing tests continue to pass after changes to preprocessing.py:
```
59 passed in 1.29s
```

## Commits

| Hash | Description |
|------|-------------|
| 653ee49 | feat(v2/phase7): add lineup data fetcher (TeamDashLineups, all teams) |
| f6acde3 | feat(v2/phase7): add BallDontLie API client |

## Self-Check: PASSED
