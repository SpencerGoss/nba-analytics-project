---
phase: 04-rest-schedule-features
plan: 02
subsystem: features, models
tags: [season-month, schedule-features, model-retrain, feature-engineering, gradient-boosting, random-forest]

# Dependency graph
requires:
  - phase: 04-01
    provides: travel_miles, cross_country_travel, diff_is_back_to_back in matchup CSV
  - phase: 02-modern-era-features
    provides: build_matchup_dataset() structure and get_feature_cols() pattern
provides:
  - season_month feature (1-12) computed from game_date in build_matchup_dataset()
  - Updated get_feature_cols() schedule_cols with all Phase 4 features
  - Retrained game outcome model with 68 features including 10 Phase 4 features
  - Updated game_matchup_features.csv with 272 columns (added season_month)
affects: [05-model-update-ats, calibration.py needs regeneration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "season_month = pd.to_datetime(matchup['game_date']).dt.month after merge, not per-team"
    - "schedule_cols set in get_feature_cols() for explicit home_/away_ feature inclusion"
    - "diff_ variants auto-included via startswith('diff_') filter -- no duplication in schedule_cols needed"

key-files:
  created: []
  modified:
    - src/features/team_game_features.py
    - src/models/game_outcome_model.py

key-decisions:
  - "season_month added to build_matchup_dataset() AFTER the matchup merge: game-level feature (same for both teams), not a per-team feature -- adding per-team would duplicate it into home_/away_ variants unnecessarily"
  - "schedule_cols updated to include home_travel_miles, away_travel_miles, home_cross_country_travel, away_cross_country_travel, season_month: diff_ variants already auto-picked up by startswith filter"
  - "Calibrated model (game_outcome_model_calibrated.pkl) predates retrained model and needs regeneration via calibration.py -- noted but not auto-run per plan spec"
  - "random_forest selected over gradient_boosting: mean val acc 0.6560 vs 0.6544 across 4 expanding validation splits"

patterns-established:
  - "Pattern: game-level context features go directly on matchup DataFrame after merge; do NOT add to build_team_game_features() to avoid home_/away_ duplication"

requirements-completed: [FR-3.4, NFR-1, NFR-2]

# Metrics
duration: 17min
completed: 2026-03-02
---

# Phase 4 Plan 02: Season Month + Model Retrain Summary

season_month (1-12) added as game-level context feature to matchup dataset; schedule_cols updated with all Phase 4 travel features; game outcome model retrained with 68 features including 10 Phase 4 schedule features achieving 66.8% test accuracy

## Performance

- **Duration:** ~17 min
- **Started:** 2026-03-02T08:32:26Z
- **Completed:** 2026-03-02T08:49:36Z
- **Tasks:** 2
- **Files modified:** 2 (src/features/team_game_features.py, src/models/game_outcome_model.py)

## Accomplishments

- Added `season_month = pd.to_datetime(matchup["game_date"]).dt.month` in `build_matchup_dataset()` AFTER the matchup merge (correct position per plan spec -- game-level feature, not per-team)
- Updated `get_feature_cols()` schedule_cols to include: home_travel_miles, away_travel_miles, home_cross_country_travel, away_cross_country_travel, season_month
- Rebuilt game_matchup_features.csv: 272 columns (added season_month), 68,165 matchup rows, season_month range 1-12 across 10 unique months (Oct-May + July/August for bubble/pre-season)
- Retrained game outcome model: random_forest selected, 68 total features, 10 Phase 4 features, 66.8% test accuracy, 0.7256 ROC-AUC
- All Phase 4 features (FR-3.1 through FR-3.4) now reflected in both training data and model feature list

## Task Commits

Each task committed atomically:

1. **Task 1: Add season_month and update schedule_cols** - `5d0c603` (feat)
2. **Task 2: Retrain game outcome model** - No code changes (model artifacts gitignored); documented in final metadata commit

## Files Created/Modified

- `src/features/team_game_features.py` - Added season_month computation block in build_matchup_dataset() after the matchup merge, before differential features
- `src/models/game_outcome_model.py` - Updated schedule_cols set in get_feature_cols() with Phase 4 travel and season_month features

## Model Retrain Results

- **Model selected:** random_forest (mean val acc=0.6560, mean val auc=0.6868)
- **Test accuracy:** 66.8% (0.6680)
- **Test ROC-AUC:** 0.7256
- **Total features:** 68 (up from ~58 pre-Phase 4)
- **Phase 4 features in model:** 10
  - away_cross_country_travel, away_is_back_to_back, away_travel_miles
  - diff_cross_country_travel, diff_is_back_to_back, diff_travel_miles
  - home_cross_country_travel, home_is_back_to_back, home_travel_miles
  - season_month

## Decisions Made

- season_month placed directly on matchup DataFrame (not per-team): avoids unnecessary home_/away_ duplication for a game-level feature
- schedule_cols updated with explicit home_/away_ travel variants; diff_ variants auto-included via startswith filter (no duplication needed).
- random_forest selected over gradient_boosting by narrow margin (0.6560 vs 0.6544 mean val acc across 4 expanding splits)
- Calibrated model needs regeneration: game_outcome_model_calibrated.pkl (Mar 1 21:30) predates retrained model (Mar 2 03:48). Run `python src/models/calibration.py` before using calibrated inference.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- PerformanceWarning (DataFrame fragmentation) during feature rebuild: pre-existing issue from incremental column insertions; out of scope per deviation rules
- Calibrated model artifact predates retrained model: noted per plan spec; do NOT run calibration automatically

## User Setup Required

None - no external service configuration required.

## Phase 4 Completion Status

All Phase 4 requirements satisfied:

- FR-3.1: days_rest and is_back_to_back -- implemented and verified (04-01)
- FR-3.2: travel_miles (haversine, 0-2704 mi) -- implemented and in model (04-01)
- FR-3.3: cross_country_travel (binary timezone flag) -- implemented and in model (04-01)
- FR-3.4: season_month (1-12 from game_date) -- implemented and in model (04-02)
- NFR-1: All features use shift(1) / prior-game data only -- verified
- NFR-2: Pipeline runtime under 15 minutes -- feature build ~5 min, model train ~5 min

## Next Phase Readiness

- Phase 5 (ATS Model) can proceed with 68-feature game outcome model as baseline
- Calibrated model should be regenerated before production inference use
- All Phase 4 features are available in game_matchup_features.csv for ATS feature development

## Self-Check: PASSED

- FOUND: src/features/team_game_features.py
- FOUND: src/models/game_outcome_model.py
- FOUND: data/features/game_matchup_features.csv
- FOUND: models/artifacts/game_outcome_metadata.json
- FOUND: .planning/phases/04-rest-schedule-features/04-02-SUMMARY.md
- FOUND: commit 5d0c603 (feat: add season_month feature and update schedule_cols)

---
*Phase: 04-rest-schedule-features*
*Completed: 2026-03-02*
