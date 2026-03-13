# Handoff — NBA Analytics Project

_Last updated: 2026-03-13 Session 15 (Config wiring + dedup + test coverage)_

## What Was Done This Session

### Config Centralization (continued from Session 14)
- Wired `src/config.py` into 14+ builder scripts — all hardcoded `CURRENT_SEASON = 202526` replaced
- Added `get_current_season_id()` for player_game_logs format
- Added `EAST_DIVISIONS`/`WEST_DIVISIONS` to config — replaced 60+ lines of duplicate dicts in standings/playoff_odds
- Fixed 3 remaining hardcoded seasons: `tune_hyperparams.py`, `fetch_historical_players.py`, `fetch_player_positions.py`
- Fixed string-based season comparisons (`.astype(str) >= "202122"`) to integer comparisons in 4 files — string comparison is lexicographic, not numeric

### Duplicate Helper Extraction
- Created `scripts/builder_helpers.py` with: `load_team_names()`, `record_str()`, `games_behind()`, `safe_float()`, `write_json()`, `load_json()`
- Wired 8 builder scripts to delegate to shared helpers (was 4x `_load_team_names`, 3x `_record_str`, 2x `_games_behind`)

### Test Coverage
- Added `tests/test_builder_helpers.py` (20 tests) — all helper functions
- Added `tests/test_fetch_odds.py` (19 tests) — implied prob, team mapping, game lines (mocked API), model_vs_odds assembly
- Added `tests/test_pipeline_runner.py` (18 tests) — registry, modes, dry-run, state
- Added `tests/test_build_elo_timeline.py` (7 tests) + `tests/test_build_game_detail.py` (9 tests)
- Test baseline: 1582 -> 1621 (+39 tests)

## What's Next

### Remaining Work
- **Dashboard tiered loading** (Plan D Task 4): deferred — high-risk for 9K-line file
- **Full model retrain** (Plan B Task 8): Huber GBM candidate + ensemble optimizer ready
- Add tests for: `sync_to_sqlserver.py`, `predict_cli.py`, `retrain_all.py`

### Known Issues
- CLV closing_spread always NULL (data-over-time issue — pipeline needs daily runs to accumulate)
- Lineup features missing for 2025-26 season
- ATS features stop at 2024-25

## Test Baseline
- 1621 tests passing (0 failures)

## Session Intent 2026-03-13
Goal: Continue project quality improvements — config wiring, dedup, test coverage
Outcome: achieved
