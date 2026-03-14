# Handoff -- NBA Analytics Project

_Last updated: 2026-03-13 Session 18 (Model retrain with pace/four_factors, test expansion)_

## What Was Done This Session

### Model Retrain (Session 18)
- **Pace + Four Factors features**: Added diff_pace_game_roll20 (#16 importance) and diff_four_factors_roll20 (#55) to production model
- **Accuracy: 67.5% -> 67.9%**, AUC: 0.7422 -> 0.7455, Brier: 0.2052 -> 0.2038
- Model auto-pruned from 100 to 67 features (was 71 before — leaner AND better)
- Calibration re-run: Platt scaling selected, base Brier 0.2038
- Dashboard JSON rebuilt with new model, about.html updated

### Test Coverage Expansion (Session 18)
- **test_get_player_stats.py** (6 tests): season loop, failure skipping, output directory
- **test_get_team_stats.py** (6 tests): fetch/save, multi-season, format, directory
- **test_fetch_player_positions.py** (41 tests): height parsing, position mapping for all NBA labels, edge cases
- Test baseline: 1739 -> 1792

### Previous Session: Critical Pipeline Fixes (Session 17)
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
- **Dashboard tiered loading** (Plan D Task 4): deferred -- high-risk for 9K-line file
- **Daily pipeline automation**: Task Scheduler + PowerShell — deferred until dashboard changes finalize

### Known Issues
- CLV closing_spread always NULL (data-over-time issue -- pipeline needs daily runs to accumulate)
- Lineup features missing for 2025-26 season
- ATS features stop at 2024-25
- Zero TODOs remaining in src/ and scripts/

## Test Baseline
- 1800 tests passing (0 failures)

## Git State
- Branch: main
- Recent commits:
  - `4a83dcb` fix: eliminate remaining .astype(str) season comparisons across 7 files
  - `cace168` feat: extended Elo API, named fetch config, +47 tests, predict_cli season fix
  - `5dc7804` fix: cv_utils off-by-one, confidence tier fallback, tierPill CSS alignment
