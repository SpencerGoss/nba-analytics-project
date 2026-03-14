# Handoff -- NBA Analytics Project

_Last updated: 2026-03-13 Session 17 (Critical pipeline fixes, test coverage expansion)_

## What Was Done This Session

### Critical Pipeline Fixes (Session 17)
- **CalibrationUnpickler**: Fixed model wrapper deserialization across 6 loading sites (game_outcome_model, fetch_odds, ensemble, model_explainability, value_bet_detector, playoff_odds_model)
- **sys.path ordering**: Fixed ModuleNotFoundError in fetch_odds.py, build_game_context.py, build_meta.py, builder_helpers.py
- **Confidence tier inflation**: Capped no-odds tier at "Solid Pick" (was giving "Best Bet" without market data)
- **Duplicate prediction prevention**: Added SELECT-before-INSERT guard in prediction_store.py
- **Props edge saturation**: IQR floor raised from 0.1 to 3.0 in betting_router.props()
- **numpy bool serialization**: Wrapped numpy.bool_ with bool() in build_props.py
- **datetime.utcnow() deprecation**: Replaced in prediction_store.py and json_export.py

### Test Coverage Expansion (Session 17)
- **test_calibration.py** (17 tests): CalibratedWrapper, PlattWrapper, ECE, bin stats, model loading
- **test_prediction_store.py** (15 tests): init, write, duplicate prevention, notes serialization
- **test_ats_model.py** (21 tests): feature selection, null validation, pipeline cloning, predict_ats
- Test baseline: 1675 -> 1739

### Previous Session (Session 16)
- cv_utils.py off-by-one fix, confidence tier suppression, dashboard tierPill CSS fix
- predict_cli.py season comparison fix, eliminated .astype(str) season comparisons across 7 files

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
- 1739 tests passing (0 failures)

## Git State
- Branch: main
- Recent commits:
  - `4a83dcb` fix: eliminate remaining .astype(str) season comparisons across 7 files
  - `cace168` feat: extended Elo API, named fetch config, +47 tests, predict_cli season fix
  - `5dc7804` fix: cv_utils off-by-one, confidence tier fallback, tierPill CSS alignment
