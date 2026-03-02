---
phase: 01-foundation-outputs
plan: 01
subsystem: features
tags: [pandas, merge_asof, null-guard, injury-proxy, sklearn]

# Dependency graph
requires: []
provides:
  - "build_injury_proxy_features() returns non-zero absent rotation rows with correct key types"
  - "injury proxy join keys normalized to str(game_id) + int(team_id) in both producer and consumer"
  - "post-merge assertion in team_game_features.py detects silent join failures (n_matched > 0)"
  - "validate_feature_null_rates() in game_outcome_model.py raises ValueError for >=95% null columns"
affects: [02-foundation-outputs, model-training, feature-engineering]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "merge_asof for projecting player rotation status onto all team-game dates without O(n^2) loop"
    - "anti-join pattern: left merge + filter on NaN played column to find absent players"
    - "normalize join keys before any merge: .astype(str).str.strip() for game_id, .astype(int) for team_id"
    - "post-merge assertion: assert n_matched > 0 immediately after left join to catch silent mismatches"
    - "null-rate guard in training: validate_feature_null_rates() before any model.fit() call"

key-files:
  created: []
  modified:
    - src/features/injury_proxy.py
    - src/features/team_game_features.py
    - src/models/game_outcome_model.py

key-decisions:
  - "Root cause of absent=0: player game logs only contain rows for games played, so anti-joining the same df always returns 0 absent rows — requires merge_asof across team-game schedule"
  - "merge_asof strategy: per (team, player), project last known in_rotation status onto all team-game dates; days_since > 0 filter ensures only prior appearances count"
  - "MAX_STALE_DAYS = ROLL_WINDOW * 5 = 25 days: beyond this, player's rotation status is treated as stale (handles trades, long-term injuries, suspensions)"
  - "total_expected_minutes denominator: concat absent expected minutes with played rotation minutes (in_rotation=True rows from df) to get full team rotation baseline"

patterns-established:
  - "merge_asof absent detection: use this pattern for any 'expected but not present' feature across sorted time series"
  - "always assert n_matched > 0 after any left join that would silently drop all rows on key mismatch"

requirements-completed: [FR-1.1, FR-1.3, NFR-1]

# Metrics
duration: 90min
completed: 2026-03-01
---

# Phase 1 Plan 01: Injury Proxy Fix and Null-Rate Guard Summary

**Fixed silent zero-absent-rotation bug via merge_asof across team-game schedules; added post-merge assertion and 95%-null training guard**

## Performance

- **Duration:** ~90 min
- **Started:** 2026-03-01
- **Completed:** 2026-03-01
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Fixed fundamental logic bug: injury proxy now detects 137,736 absent rotation instances (was 0), with 60.9% of team-games showing some missing minutes and average rotation availability 0.893
- Added join-key type normalization (game_id to str.strip(), team_id to int) in both injury_proxy.py return path and team_game_features.py before merge, plus post-merge assertion that fires on zero-match joins
- Added validate_feature_null_rates() to game_outcome_model.py — raises ValueError with column name(s) when any feature column hits >= 95% null rate, preventing silent model training on broken features

## Task Commits

1. **Task 1: Diagnose and fix game_id type mismatch in injury proxy join** - `2208360` (fix)
2. **Task 2: Add null-rate guard to game_outcome_model training** - `c5c038e` (feat)

## Files Created/Modified

- `src/features/injury_proxy.py` - Rewrote absent rotation detection using merge_asof; added type normalization before return
- `src/features/team_game_features.py` - Added type normalization on both sides before injury merge; added post-merge assertion
- `src/models/game_outcome_model.py` - Added validate_feature_null_rates() function and call inside train_game_outcome_model()

## Decisions Made

- **Root cause was not a type mismatch** (both CSVs have int64 game_id, inner join gives 136K rows). The actual bug: game logs only contain rows for games where a player appeared. Anti-joining the same df always produces 0 absent rows because rotation players ARE in the played set by definition.
- **merge_asof strategy chosen** over a loop-based approach (102s for full dataset in testing): per (team, player), find each player's last known rotation status before each team-game date. Stale threshold of 25 days prevents false positives from traded/long-injured players.
- **Separate denominator computation**: total_expected_minutes = absent expected minutes + played rotation minutes (players with in_rotation=True who DID play). Necessary because expected_df only contains absent players by construction.
- **Type normalization retained** even though it wasn't the root cause: prevents future CSV format drift from reintroducing silent join failures.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Rewrote entire absent rotation detection algorithm**
- **Found during:** Task 1 diagnostic
- **Issue:** Plan assumed the bug was a game_id type mismatch (int vs str) between player_game_logs and team_game_logs. Diagnostic confirmed both files have int64 game_ids and inner join produces 136,348 rows. The actual bug: game logs only contain rows for played games, so anti-joining the rotation subset against the same df produces 0 absent rows — rotation players appear in actual_played by definition.
- **Fix:** Replaced anti-join on same df with merge_asof-based projection: for each (team, player), find last known rotation status before each team-game date via pd.merge_asof, filter to days_since > 0 and <= 25, then anti-join against actual_played. Also rebuilt total_expected_minutes denominator to include played rotation players.
- **Files modified:** src/features/injury_proxy.py
- **Verification:** Running build_injury_proxy_features() now produces 137,736 absent instances (was 0), rotation_availability mean = 0.893 (was 0.000), 60.9% of team-games show some missing minutes.
- **Committed in:** 2208360 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - logic bug more fundamental than diagnosed)
**Impact on plan:** The fix scope was larger than planned (algorithm rewrite vs. one-line type cast) but all changes are within the same files and serve the same purpose. No scope creep.

## Issues Encountered

- The research diagnostic assumed type mismatch but the actual CSV data has consistent int64 types. The join was "failing" at a higher level — the anti-join algorithm itself was structurally incapable of detecting absent players. Required three rounds of debugging (diagnostic run, loop test, merge_asof test) to confirm the correct root cause.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Injury proxy features are now correct and meaningful; team_game_features.py can be rebuilt to pick up real missing_minutes and star_player_out values
- After rebuilding team_game_features.csv and game_matchup_features.csv, the model's injury proxy feature importances should become non-zero
- validate_feature_null_rates() will now catch if the next feature rebuild produces broken columns
- Known: team_game_features.py needs to be run end-to-end to regenerate features CSV with fixed injury data before retraining

---
*Phase: 01-foundation-outputs*
*Completed: 2026-03-01*

## Self-Check: PASSED

- src/features/injury_proxy.py: FOUND
- src/features/team_game_features.py: FOUND
- src/models/game_outcome_model.py: FOUND
- .planning/phases/01-foundation-outputs/01-01-SUMMARY.md: FOUND
- Commit 2208360 (Task 1): FOUND
- Commit c5c038e (Task 2): FOUND
- validate_feature_null_rates() callable and raises ValueError for columns at >=95% null: VERIFIED
- build_injury_proxy_features() produces non-zero absent rows (137,736) and correct dtypes: VERIFIED
