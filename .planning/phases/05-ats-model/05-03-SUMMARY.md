---
phase: 05-ats-model
plan: 03
subsystem: models
tags: [value-bet, odds, calibration, detection, probability]

# Dependency graph
requires:
  - phase: 05-ats-model plan: 01
    provides: game_ats_features.csv with home_implied_prob (no-vig)
  - scripts/fetch_odds.py: american_odds_to_implied_prob(), fetch_game_lines(), QuotaError
  - models/artifacts/game_outcome_model_calibrated.pkl: calibrated win probabilities

provides:
  - src/models/value_bet_detector.py: detect_value_bets(), no_vig_prob(),
    check_remaining_quota(), run_value_bet_scan()
  - models/artifacts/game_outcome_model_calibrated.pkl: UPDATED -- now reflects
    Phase 4 retrained model (regenerated via calibration.py)

affects:
  - 05-ats-model (plans 04+): ATS backtest uses value-bet detector output
  - daily inference: run_value_bet_scan(use_live_odds=True) for upcoming games

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-tier odds sourcing: Kaggle historical (offline) vs Odds API (current-season)"
    - "Multiplicative no-vig removal: raw_home/total, raw_away/total"
    - "QuotaError guard: checks x-requests-remaining before batch API calls"
    - "Graceful API key fallback: returns -1 with UserWarning when ODDS_API_KEY unset"

key-files:
  created:
    - src/models/value_bet_detector.py
  modified:
    - src/models/calibration.py
    - models/artifacts/game_outcome_model_calibrated.pkl (regenerated artifact)

key-decisions:
  - "game_ats_features.csv already has home_implied_prob (no-vig); historical mode
    uses this directly rather than recomputing from raw moneylines"
  - "calibration.py had Unicode box-drawing chars (U+2500) -- replaced with ASCII
    dashes for Windows cp1252 compatibility (Rule 1 auto-fix)"
  - "check_remaining_quota() returns -1 (non-fatal) when ODDS_API_KEY is unset;
    QuotaError raised only when key is present but quota is below min_remaining"
  - "run_value_bet_scan defaults to historical mode in __main__ (safe offline use)"

patterns-established:
  - "Pattern: no_vig_prob() as single source for vig removal in value-bet comparison"
  - "Pattern: QuotaError custom exception for API credit exhaustion signal"
  - "Pattern: Graceful API fallback -- ODDS_API_KEY absent triggers historical mode"

requirements-completed: [FR-5.3, FR-5.4, NFR-1, NFR-3]

# Metrics
duration: 41min
completed: 2026-03-02
---

# Phase 5 Plan 03: Value-Bet Detector Summary

Calibrated model regenerated (Phase 4 baseline) and value-bet detector built: no-vig probability comparison between model win probabilities and market odds, with quota guard and historical/live dual-mode scanning.

## Performance

- **Duration:** 41 min
- **Started:** 2026-03-02T19:09:13Z
- **Completed:** 2026-03-02T19:50:16Z
- **Tasks:** 2
- **Files modified:** 2 created/modified + 1 artifact regenerated

## Accomplishments

- Regenerated `game_outcome_model_calibrated.pkl` (Mar 2 14:34) -- now reflects Phase 4 retrained model (was dated Mar 1 21:30, predating the 04-02 retrain)
- Built `src/models/value_bet_detector.py` with 4 public exports:
  - `no_vig_prob(home_ml, away_ml)`: multiplicative vig removal, sums to exactly 1.0
  - `detect_value_bets(games_df, threshold)`: flags games where |edge| > threshold, adds edge/is_value_bet/bet_side columns
  - `check_remaining_quota(min_remaining)`: reads x-requests-remaining header; raises QuotaError if below minimum; returns -1 with UserWarning if no API key (non-fatal)
  - `run_value_bet_scan(use_live_odds)`: high-level daily scan; historical mode uses game_ats_features.csv, live mode uses The Odds API
- Historical scan verified: 1,894 games scanned, results JSON-serializable (NFR-3 compliant)
- Calibration.py Unicode fix: box-drawing chars (U+2500 x982 instances) caused cp1252 crash -- replaced with ASCII dashes (Rule 1 auto-fix)
- Module docstring documents two-tier sourcing strategy and Pitfall 1 (Odds API historical is paid-only)

## Task Commits

1. **Task 1: Regenerate calibrated model and build value-bet detector** - `25da6fc` (feat)
2. **Task 2: Document historical odds sourcing strategy and quota guard** - `124ab34` (feat)

## Files Created/Modified

- `src/models/value_bet_detector.py` - 6 exports: detect_value_bets, no_vig_prob, check_remaining_quota, run_value_bet_scan, QuotaError, module constants
- `src/models/calibration.py` - ASCII-only print statements; no logic changes
- `models/artifacts/game_outcome_model_calibrated.pkl` - Regenerated artifact (non-git, updated in place)

## Decisions Made

- **Home_implied_prob already no-vig in ATS features:** `game_ats_features.csv` stores pre-computed no-vig probabilities from `ats_features.py`. Historical mode uses `home_implied_prob` directly rather than recomputing from raw moneylines (more efficient; consistent with how the feature was built).

- **Graceful quota guard:** `check_remaining_quota()` returns -1 (not raises) when API key is absent. Only raises `QuotaError` when key IS set but remaining credits fall below threshold. This makes the function safe for offline environments.

- **Historical scan defaults to recent 4 seasons:** `run_value_bet_scan(use_live_odds=False)` filters to the most recent 4 seasons (from season column) to keep the sample representative without loading all 18K rows for a quick scan.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] calibration.py crashes on Windows with Unicode print statements**

- **Found during:** Task 1 (running `python src/models/calibration.py`)
- **Issue:** 982 occurrences of U+2500 box-drawing char (`─`), plus em-dash, en-dash, arrow, and approximately-equal Unicode chars. Windows cp1252 terminal raises UnicodeEncodeError on any of these.
- **Fix:** Replaced all non-ASCII characters with ASCII equivalents: `─` -> `-`, `—` -> `--`, `→` -> `->`, `≈` -> `~`. No logic changes.
- **Files modified:** `src/models/calibration.py`
- **Verification:** `python src/models/calibration.py` runs successfully; calibrated model regenerated.
- **Committed in:** 25da6fc (Task 1 commit, alongside value_bet_detector.py)

**2. [Rule 1 - Bug] run_value_bet_scan() referenced wrong column in historical mode**

- **Found during:** Task 2 (running plan verification test)
- **Issue:** Historical mode referenced `home_moneyline` column which doesn't exist in `game_ats_features.csv`. The file has `home_implied_prob` (already no-vig) instead.
- **Fix:** Added column detection: if `home_implied_prob` present, use directly as `market_implied_prob`. Fallback to computing from `home_moneyline`/`away_moneyline` if raw moneylines available instead.
- **Files modified:** `src/models/value_bet_detector.py`
- **Verification:** `run_value_bet_scan(use_live_odds=False)` returns 1,894 games, JSON-serializable.
- **Committed in:** 124ab34 (Task 2 commit)

Total deviations: 2 auto-fixed (1 encoding bug, 1 column name mismatch)

## Issues Encountered

None beyond the auto-fixed deviations above.

## User Setup Required

None. Historical mode works without any API key. Live mode (`use_live_odds=True`) requires `ODDS_API_KEY` in `.env` -- already configured in this environment (500 credits remaining).

## Next Phase Readiness

- Value-bet detector ready for ATS backtest integration (Plan 05-04/05-05)
- `run_value_bet_scan(use_live_odds=True)` ready for daily use with existing ODDS_API_KEY
- Calibrated model is now current -- safe to use for production win-probability inference

## Self-Check: PASSED

All created/modified files verified:

- FOUND: src/models/value_bet_detector.py
- FOUND: src/models/calibration.py
- FOUND: .planning/phases/05-ats-model/05-03-SUMMARY.md
- FOUND: models/artifacts/game_outcome_model_calibrated.pkl (timestamp > game_outcome_model.pkl)

All commits verified:

- FOUND: 25da6fc (Task 1: value_bet_detector + calibration fix)
- FOUND: 124ab34 (Task 2: historical mode odds sourcing fix)

---
*Phase: 05-ats-model*
*Completed: 2026-03-02*
