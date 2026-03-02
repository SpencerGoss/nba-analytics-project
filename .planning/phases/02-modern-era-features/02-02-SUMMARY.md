---
phase: 02-modern-era-features
plan: "02"
subsystem: features
tags: [pandas, numpy, rolling-features, four-factors, dean-oliver, advanced-metrics, nba-analytics]

# Dependency graph
requires:
  - phase: 02-modern-era-features
    plan: "01"
    provides: ADV_ROLL_STATS rolling columns (efg_game_roll20, tov_pct_game_roll20, oreb_pct_game_roll20, ft_rate_game_roll20) in team_game_features.csv
provides:
  - FOUR_FACTORS_WEIGHTS constant with Dean Oliver canonical weights (eFG% +0.40, TOV% -0.25, OREB% +0.20, FT rate +0.15)
  - _four_factors_composite() helper computing weighted home-minus-away efficiency differential
  - diff_four_factors_composite column in game_matchup_features.csv (68,165 non-null values)
  - Advanced metric differential columns (diff_off_rtg_game_roll5/10/20, diff_def_rtg_game_roll5/10/20, diff_net_rtg_game_roll5/10/20, diff_pace_game_roll20, diff_efg_game_roll20, diff_ts_game_roll20, diff_tov_poss_game_roll20)
  - 48 total diff_ columns in matchup dataset
affects:
  - game_outcome_model.py (get_feature_cols() auto-picks up all new diff_ columns via startswith("diff_") pattern)
  - model training (Four Factors composite + pace-normalized diffs available as features)
  - Phase 3 onwards (matchup feature set expanded with efficiency composite signal)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "FOUR_FACTORS_WEIGHTS module-level dict separating weight configuration from computation logic"
    - "Helper function before its caller: _four_factors_composite() defined before build_matchup_dataset() to avoid NameError"
    - "diff_ prefix convention: all matchup features use diff_ prefix for automatic model pickup without modifying get_feature_cols()"
    - "Selective roll windows: ORtg/DRtg/net_rtg get 5/10/20 windows (primary signals); rate stats (eFG%, TS%, pace, tov_poss) get roll20 only (shorter windows are noisy)"

key-files:
  created: []
  modified:
    - src/features/team_game_features.py

key-decisions:
  - "Helper function before caller: FOUR_FACTORS_WEIGHTS and _four_factors_composite() placed before build_matchup_dataset() so they are defined when called — Python NameError if placed after"
  - "diff_ prefix on composite: naming diff_four_factors_composite (not four_factors_composite_diff) ensures automatic pickup by get_feature_cols() startswith('diff_') filter, no model code changes needed"
  - "Selective roll windows for rate stats: only roll20 for eFG%, TS%, pace, tov_poss — shorter windows are too noisy for rate stats; ORtg/DRtg/net_rtg get all three windows as primary efficiency signals"
  - "fillna(0) inside composite: missing data treated as no differential contribution rather than propagating NaN, prevents sparse data from zeroing out the composite entirely"

patterns-established:
  - "Four Factors composite definition order: constant dict → helper function → usage site (all in same file, defined before use)"
  - "Advanced metric diff windows: roll20-only for rate stats, multi-window for count-based efficiency ratings"

requirements-completed: [FR-2.4, NFR-1]

# Metrics
duration: 10min
completed: 2026-03-02
---

# Phase 2 Plan 02: Four Factors Composite and Advanced Metric Diffs Summary

**Four Factors differential composite (Dean Oliver weights: eFG% +0.40, TOV% -0.25, OREB% +0.20, FT rate +0.15) added to game_matchup_features.csv alongside 13 advanced metric diff columns; 48 total diff_ columns auto-picked by model feature selection**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-02T03:42:49Z
- **Completed:** 2026-03-02T03:52:40Z
- **Tasks:** 2 of 2
- **Files modified:** 1

## Accomplishments
- Added FOUR_FACTORS_WEIGHTS constant at module level with Dean Oliver's canonical efficiency weights
- Implemented _four_factors_composite() helper before build_matchup_dataset() that loops over weight dict, computes home-minus-away diff per factor, and returns a weighted composite Series
- Wired 13 advanced metric rolling columns into diff_stats (ORtg/DRtg/net_rtg across 5/10/20 windows, pace/eFG%/TS%/tov_poss at roll20 only)
- Computed diff_four_factors_composite after the diff_stats loop with 68,165 non-null values in the output CSV
- Verified all 48 diff_ columns auto-picked by get_feature_cols() via startswith("diff_") pattern — no changes to game_outcome_model.py needed
- Full regression pass: existing features (three_style_mismatch, rebounding_edge, revenge_game, pts_roll20, win_pct_roll20) all preserved, no _x/_y merge artifacts

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Four Factors composite and advanced metric diffs** - `584ff2c` (feat)
2. **Task 2: Rebuild feature files and verify end-to-end pipeline integrity** - verification only, no new code (CSV files are gitignored, no commit needed)

## Files Created/Modified
- `src/features/team_game_features.py` - Added FOUR_FACTORS_WEIGHTS constant, _four_factors_composite() helper, advanced metric diff entries in diff_stats list, Four Factors composite computation after diff loop

## Decisions Made
- **Helper function placement:** FOUR_FACTORS_WEIGHTS and _four_factors_composite() placed before build_matchup_dataset() to avoid Python NameError at runtime. Plan said "before build_matchup_dataset()" but initial edit accidentally placed them after — fixed immediately.
- **diff_ prefix maintained:** Composite named diff_four_factors_composite so get_feature_cols() in game_outcome_model.py picks it up automatically via the existing startswith("diff_") filter. No model code changes required.
- **Selective roll windows:** ORtg/DRtg/net_rtg get 5/10/20 windows (primary efficiency signals with meaningful short-window variation); eFG%, TS%, pace, tov_poss get roll20 only (rate stats are too noisy at shorter windows).
- **fillna(0) in composite:** Missing factor contributions treated as zero differential rather than NaN propagation, preventing sparse early-season data from producing null composites.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed helper function placement after build_matchup_dataset()**
- **Found during:** Task 1 code review after edit
- **Issue:** Initial edit inserted FOUR_FACTORS_WEIGHTS and _four_factors_composite() after build_matchup_dataset() instead of before it — would cause NameError at runtime
- **Fix:** Removed the misplaced block and re-inserted before build_matchup_dataset() using a separate edit
- **Files modified:** src/features/team_game_features.py
- **Verification:** Pipeline ran successfully with no NameError; Four Factors composite appeared in output with 68,165 non-null values
- **Committed in:** 584ff2c (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug: incorrect insertion position)
**Impact on plan:** Fix was necessary for correctness. No scope creep.

## Issues Encountered
- PerformanceWarnings about DataFrame fragmentation during build (pre-existing issue from many sequential column assignments). These are out-of-scope pre-existing warnings that don't affect correctness. The existing `df = df.copy()` defragmentation call in build_team_game_features() partially addresses this.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- diff_four_factors_composite and 13 advanced metric diff columns available in game_matchup_features.csv for model training
- 48 total diff_ columns automatically included in get_feature_cols() feature selection
- No blockers for Phase 2 Plan 03

---
*Phase: 02-modern-era-features*
*Completed: 2026-03-02*

## Self-Check: PASSED

- src/features/team_game_features.py: FOUND
- .planning/phases/02-modern-era-features/02-02-SUMMARY.md: FOUND
- Commit 584ff2c: FOUND
- FOUR_FACTORS_WEIGHTS in code: PASS
- _four_factors_composite() helper defined before build_matchup_dataset(): PASS
- diff_four_factors_composite in build_matchup_dataset(): PASS
- Advanced diff columns (off_rtg_game_roll5, efg_game_roll20) in diff_stats: PASS
- Python syntax valid: PASS
