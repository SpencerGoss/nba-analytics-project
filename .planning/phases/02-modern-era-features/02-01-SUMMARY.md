---
phase: 02-modern-era-features
plan: "01"
subsystem: features
tags: [pandas, numpy, rolling-features, advanced-metrics, pace-normalized, nba-analytics]

# Dependency graph
requires:
  - phase: 01-foundation-outputs
    provides: team_game_features.py with opponent self-join pattern and _rolling_mean_shift helper
provides:
  - Per-game ORtg, DRtg, net_rtg, eFG%, TS%, pace, tov_poss, tov_pct, oreb_pct, ft_rate computed via Oliver possession formula
  - ADV_ROLL_STATS rolling loop (10 metrics x 3 windows = 30 new columns) using shift-1 leakage prevention
  - Expanded opponent self-join (opp_box) replacing narrow opp_style join
  - team_game_features.csv with 30 new rolling columns (121 columns total)
affects:
  - 02-modern-era-features (plan 02 Four Factors composite uses these rolling columns)
  - model training (ORtg/DRtg pace-normalized signals ready for feature selection)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ADV_ROLL_STATS module-level list separating advanced metrics from ROLL_STATS (pre-join vs post-join)"
    - "Single expanded opp_box self-join providing all opponent box score columns in one merge"
    - "Oliver possession formula: poss_est = FGA - OREB + TOV + 0.44*FTA"
    - "Default arg capture in lambda: lambda g, s=stat: to prevent late-binding closure bug in loops"

key-files:
  created: []
  modified:
    - src/features/team_game_features.py

key-decisions:
  - "ADV_ROLL_STATS defined at module level separate from ROLL_STATS — advanced metrics require opponent self-join unavailable at initial ROLL_STATS computation time"
  - "Single opp_box merge replaces narrow opp_style join — extends existing pattern with all needed columns rather than a second separate merge (prevents duplicate column naming conflicts)"
  - "opp_dreb_game renamed to opp_dreb in the expanded join — column name standardized, downstream opp_dreb_roll20 reference updated accordingly"

patterns-established:
  - "Post-join rolling block: advanced metrics requiring opponent data go in separate ADV_ROLL_STATS loop after the self-join, not in ROLL_STATS"
  - "Assert guard after self-join: assert df['opp_fga'].notna().sum() > 0 to catch alignment failures at build time"
  - "Null-rate diagnostics: print non-null percentage for each roll20 column after rolling computation"

requirements-completed: [FR-2.1, FR-2.2, FR-2.3, NFR-1]

# Metrics
duration: 14min
completed: 2026-03-02
---

# Phase 2 Plan 01: Modern Era Features - Advanced Metrics Summary

**10 Oliver-formula pace-normalized metrics (ORtg, DRtg, net_rtg, eFG%, TS%, pace, tov_per_poss, tov_pct, oreb_pct, ft_rate) rolled across 5/10/20 game windows via expanded opponent self-join; 30 new columns in team_game_features.csv, ORtg mean 108.1, pace mean 100.7**

## Performance

- **Duration:** 14 min
- **Started:** 2026-03-02T03:24:38Z
- **Completed:** 2026-03-02T03:39:30Z
- **Tasks:** 2 of 2
- **Files modified:** 1

## Accomplishments
- Replaced narrow 4-column opp_style self-join with broader 11-column opp_box join providing all box score data needed for advanced metric computation
- Computed Oliver-formula possession estimates (poss_est, opp_poss_est, avg_poss) as intermediate columns for metric derivation
- Added all 10 per-game advanced metrics: off_rtg_game, def_rtg_game, net_rtg_game, pace_game, efg_game, ts_game, tov_poss_game, tov_pct_game, oreb_pct_game, ft_rate_game
- Rolled all 10 metrics across windows [5, 10, 20] using ADV_ROLL_STATS loop with _rolling_mean_shift (shift-1 enforced, no lookahead bias)
- Verified 30 new rolling columns in output CSV with sane ranges: ORtg=108.1, pace=100.7 for modern era data

## Task Commits

Each task was committed atomically:

1. **Task 1 + 2: Extend opponent self-join, compute per-game advanced metrics, roll with ADV_ROLL_STATS** - `8f2b9b6` (feat)

## Files Created/Modified
- `src/features/team_game_features.py` - Added ADV_ROLL_STATS constant, replaced opp_style with opp_box join, added possession estimates + 10 per-game advanced metrics, added ADV_ROLL_STATS rolling loop with null-rate diagnostics

## Decisions Made
- **ADV_ROLL_STATS separate from ROLL_STATS:** Advanced metrics need opponent data from the self-join; they cannot be in ROLL_STATS which runs before the join. Separate module-level constant and loop maintains clean separation.
- **Single expanded merge:** Replaced the narrow opp_style join rather than adding a second join. A second self-join with overlapping column names would create _x/_y suffix conflicts. One merge with all needed columns is cleaner.
- **opp_dreb renamed:** The expanded join renames `dreb` to `opp_dreb` (without `_game` suffix). Downstream `opp_dreb_roll20` computation updated to reference `opp_dreb` instead of `opp_dreb_game`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- PerformanceWarnings about DataFrame fragmentation appeared during build (many sequential column assignments). These are pre-existing warnings in the codebase unrelated to this plan's changes. They do not affect correctness - the existing `df = df.copy()` defragmentation call before the motivation section already partially addresses this. No action taken (out-of-scope pre-existing issue).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- 30 new rolling columns available in team_game_features.csv for Plan 02-02 (Four Factors composite)
- off_rtg_game_roll20, def_rtg_game_roll20, efg_game_roll20, tov_pct_game_roll20, oreb_pct_game_roll20, ft_rate_game_roll20 are the key columns for Four Factors composite assembly in build_matchup_dataset()
- No blockers for Phase 2 Plan 02

---
*Phase: 02-modern-era-features*
*Completed: 2026-03-02*
