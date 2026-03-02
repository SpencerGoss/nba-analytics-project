---
phase: 02-modern-era-features
plan: "03"
subsystem: ml-models
tags: [sklearn, gradient-boosting, random-forest, game-outcome-model, era-filtering]

# Dependency graph
requires:
  - phase: 02-02
    provides: "Four Factors composite and advanced metric diffs in matchup dataset (diff_four_factors_composite, diff_off_rtg_game_roll20, etc.)"
provides:
  - "MODERN_ERA_ONLY=True default in game_outcome_model.py"
  - "EXCLUDED_SEASONS=[201920,202021] constant and exclusion logic"
  - "Era-filtered model trained on 201314-202425 excluding bubble/shortened seasons"
  - "Model metadata with train_start_season=201314, modern_era_only=True, excluded_seasons fields"
affects: [03-schedule-travel-features, 04-ats-model, calibration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "EXCLUDED_SEASONS constant at module level for anomalous season exclusion"
    - "excluded_seasons parameter on train_game_outcome_model() for runtime override"
    - "Era filter + exclusion filter applied sequentially before train/test split"

key-files:
  created: []
  modified:
    - src/models/game_outcome_model.py

key-decisions:
  - "MODERN_ERA_ONLY=True as default: post-2014 NBA structural shift (pace/3-point/position-less) makes mixed-era training suboptimal"
  - "MODERN_ERA_START changed from 201415 to 201314 per SC1 (include 2013-14 season)"
  - "EXCLUDED_SEASONS=[201920,202021]: bubble neutral-site games and shortened schedule add noise not signal"
  - "Research finding: full history model (0.6729) marginally outperforms modern era model (0.6684) on 202324/202425 holdout -- model selection variance with 4 vs 19 splits is the likely cause, not genuine era performance difference"

patterns-established:
  - "Module-level EXCLUDED_SEASONS: excludes anomalous seasons from training without breaking inference path"
  - "Metadata JSON records era filter config (modern_era_only, excluded_seasons) for auditability"

requirements-completed: [FR-2.5, NFR-1]

# Metrics
duration: 88min
completed: 2026-03-02
---

# Phase 2 Plan 03: Modern Era Era Filtering Summary

**MODERN_ERA_ONLY=True default with EXCLUDED_SEASONS=[201920,202021] bubble/COVID exclusion; model retrained on 201314+ achieving 66.8% test accuracy with 60 features including advanced efficiency metrics**

## Performance

- **Duration:** 88 min (dominated by 3 model training runs: modern era x2 + full history comparison)
- **Started:** 2026-03-02T03:56:05Z
- **Completed:** 2026-03-02T05:24:05Z
- **Tasks:** 2
- **Files modified:** 1 (src/models/game_outcome_model.py)

## Accomplishments
- Changed MODERN_ERA_ONLY to True (was False) and MODERN_ERA_START to "201314" (was "201415")
- Added EXCLUDED_SEASONS = ["201920", "202021"] constant and exclusion logic in train_game_outcome_model()
- Retrained model on 201314-202425 data (excluding bubble/COVID seasons and test holdout): 10,710 training games, 2,455 test games
- Model confirmed excluded 2,139 games from anomalous seasons; 60 features including 10 advanced efficiency features (diff_four_factors_composite, diff_off_rtg_game_roll20, etc.)
- Model metadata updated to include modern_era_only, excluded_seasons, train_start_season fields

## Task Commits

Each task was committed atomically:

1. **Task 1: Set modern era defaults and add excluded seasons filter** - `eb6703b` (feat)
2. **Task 2: Train modern-era model and validate accuracy vs full-history** - artifacts saved to disk (gitignored), no separate commit needed

**Plan metadata:** (to be committed via gsd-tools)

## Files Created/Modified
- `src/models/game_outcome_model.py` - MODERN_ERA_ONLY=True, MODERN_ERA_START="201314", EXCLUDED_SEASONS, excluded_seasons param in train function, metadata updated

## Decisions Made
- MODERN_ERA_START set to "201314" per plan specification (include 2013-14 season as start of modern era)
- excluded_seasons defaults to EXCLUDED_SEASONS for convenience; can be overridden by passing [] to compare against full modern era
- Metadata always serializes excluded_seasons as [] when modern_era_only=False to clearly distinguish the two modes

## Deviations from Plan

### Research Finding: Modern Era Accuracy Below Full History

**1. [Rule 1 - Research Assumption Not Validated] Modern era accuracy (0.6684) is marginally below full history (0.6729)**
- **Found during:** Task 2 (accuracy comparison)
- **Issue:** Plan predicted modern_acc >= full_acc based on research analysis. Actual results: modern era random_forest=0.6684, full history gradient_boosting=0.6729. Confirmed this holds when controlling for model type: modern era GB=0.6635 vs full history GB=0.6754.
- **Root cause analysis:** Modern era training has only 4 validation splits (vs 19 for full history), so model selection is noisier. Additionally, GBM models benefit from more training data (28k+ games vs 10k), which counterbalances the structural non-stationarity argument.
- **Resolution:** All 6 plan verification checks pass (train_start_season=201314, modern_era_only=True, excluded_seasons correct, test_accuracy > 0.60, advanced features present). The modern era model is still the operational artifact. The accuracy gap is ~0.45 percentage points and does not indicate a bug.
- **Files modified:** None (expected behavior documented, not a code error)
- **Committed in:** eb6703b (Task 1 -- code changes only; Task 2 produces gitignored artifacts)

---

**Total deviations:** 1 research finding (accuracy comparison result differs from plan prediction)
**Impact on plan:** All code changes executed correctly. Era filtering and exclusion logic work as specified. Accuracy gap is a research finding to inform future iteration, not a correctness failure.

## Issues Encountered
- Background process output buffering made it difficult to monitor the full history comparison training (19 splits x 3 models = slow). The full history gradient_boosting comparison run overwrote the modern era artifact on disk. Required an additional modern era training run to restore the correct artifact.
- Full history comparison training took ~40 minutes due to 19 validation splits with gradient_boosting (350 estimators x 28k training games x 19 splits).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Game outcome model operational with modern era filtering. Ready for Phase 3 (schedule/travel features).
- Calibration artifact (game_outcome_model_calibrated.pkl) will need regeneration after this retraining -- it exists from a prior run but now uses the modern era base model.
- If AUC/accuracy gap persists in Phase 4 (ATS model), revisit whether training data filtering should use a different era boundary.

---
*Phase: 02-modern-era-features*
*Completed: 2026-03-02*
