# Handoff -- NBA Analytics Project

_Last updated: 2026-03-13 Session 16 (Bug fixes, extended Elo, test coverage, dashboard refactor)_

## What Was Done This Session

### Bug Fixes
- **cv_utils.py off-by-one**: First CV fold was using `min_train_seasons - 1` training seasons, not `min_train_seasons` as documented. Also fixed same off-by-one in `tune_hyperparams.py`
- **Confidence tier suppression**: All games without market odds were getting "Skip" tier. Added probability-based fallback tiering (>=70% Best Bet, >=62% Solid Pick, >=55% Lean)
- **Dashboard tierPill CSS mismatch**: CSS class was derived from edge magnitude but label was downgraded by model disagreement. Added `_confLevelFromTier()` to derive class from label
- **predict_cli.py**: Fixed `.astype(str)` season comparison bug
- **Eliminated ALL remaining `.astype(str)` season comparisons**: 7 files fixed (ats_model.py, ats_backtest.py, player_performance_model.py, playoff_odds_model.py, build_picks.py, tune_hyperparams.py, predict_cli.py). ATS model constants changed from strings to ints.

### Extended Elo API (Phase 3 TODO resolved)
- `get_current_elos(extended=True)` returns `{team: {elo, elo_fast, momentum}}` per team
- `margin_model.py` now injects fast Elo + momentum signals in prediction path
- Backward-compatible: `get_current_elos()` still returns simple `{team: elo}` dict

### Dashboard Refactor
- **Promise.allSettled fragile array pattern resolved**: Replaced 3 parallel arrays (20 fetch calls, 20 fallbacks, 20-variable destructuring) with named config objects `{k, url, fb}`. Adding/removing fetches now requires editing only one place.

### Test Coverage (+47 new tests)
- `tests/test_predict_cli.py` (12 tests) -- arg parsing, missing matchup, no history
- `tests/test_retrain_all.py` (7 tests) -- step runner, pipeline orchestration, step ordering
- `tests/test_sync_to_sqlserver.py` (28 tests) -- clean_value, coerce_dates, sql_type, table naming, conn_string, create SQL
- Extended Elo tests: 2 new tests in test_elo.py
- Test baseline: 1621 -> 1668

## What's Next

### Remaining Work
- **Full model retrain**: Huber GBM candidate + ensemble optimizer ready but not yet executed
- **Dashboard tiered loading** (Plan D Task 4): deferred -- high-risk for 9K-line file
- **Additional test coverage**: nba_api data fetchers (`get_player_stats.py`, `get_team_stats.py`) still lack unit tests

### Known Issues
- CLV closing_spread always NULL (data-over-time issue -- pipeline needs daily runs to accumulate)
- Lineup features missing for 2025-26 season
- ATS features stop at 2024-25
- Zero TODOs remaining in src/ and scripts/

## Test Baseline
- 1668 tests passing (0 failures)

## Git State
- Branch: main
- Recent commits:
  - `4a83dcb` fix: eliminate remaining .astype(str) season comparisons across 7 files
  - `cace168` feat: extended Elo API, named fetch config, +47 tests, predict_cli season fix
  - `5dc7804` fix: cv_utils off-by-one, confidence tier fallback, tierPill CSS alignment
