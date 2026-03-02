---
phase: 01-foundation-outputs
plan: 02
subsystem: models
tags: [sklearn, pickle, json, calibration, inference, game-outcome]

# Dependency graph
requires: []
provides:
  - _load_game_outcome_model() helper that prefers calibrated artifact in models/game_outcome_model.py
  - predict_game() return dict includes model_artifact (str) and feature_count (int)
  - train_game_outcome_model() writes game_outcome_metadata.json with required fields
affects: [01-03, 01-04, api, web-outputs]

# Tech tracking
tech-stack:
  added: [json (stdlib), datetime (stdlib)]
  patterns: [prefer-calibrated-artifact, model-metadata-json, builtins-only-json]

key-files:
  created: []
  modified: [src/models/game_outcome_model.py]

key-decisions:
  - "Added _load_game_outcome_model() to centralize model loading logic and enforce calibrated-first preference"
  - "UserWarning (not silent fallback, not hard error) when calibrated model missing — allows workflow to continue with visible warning"
  - "Metadata JSON uses only Python builtins (float/int/str/list/dict) — numpy types never written to JSON to avoid TypeError"
  - "top_importances built inline with best-effort try/except — metadata never crashes training"

patterns-established:
  - "Prefer-calibrated artifact: _load_game_outcome_model() checks for _calibrated.pkl first, UserWarning on fallback"
  - "Model metadata JSON: all training runs write game_outcome_metadata.json alongside pkl artifacts"
  - "Return dict enrichment: predict_game() returns model_artifact and feature_count for downstream use"

requirements-completed: [FR-1.2, FR-6.5, NFR-3]

# Metrics
duration: 2min
completed: 2026-03-02
---

# Phase 1 Plan 02: Calibrated Inference Path and Model Metadata JSON Summary

**Inference path now loads calibrated model (isotonic CalibratedClassifierCV) when available, with UserWarning fallback; training serializes game_outcome_metadata.json with feature_list, accuracy, AUC, and top importances for web consumption**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-02T00:37:02Z
- **Completed:** 2026-03-02T00:39:05Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Added `_load_game_outcome_model()` helper that checks for `game_outcome_model_calibrated.pkl` first, falls back to uncalibrated with a `UserWarning` that includes remediation instructions (`python src/models/calibration.py`)
- Raises `FileNotFoundError` with clear message when neither artifact exists
- Updated `predict_game()` to use the new helper and extended return dict with `model_artifact` (artifact filename) and `feature_count` (int)
- Added JSON + datetime imports at module level; inserted metadata serialization block in `train_game_outcome_model()` that writes `game_outcome_metadata.json` with all required fields using Python builtins only

## Task Commits

Each task was committed atomically:

1. **Task 1: Extract _load_game_outcome_model() helper and fix predict_game() load path** - `afe9eee` (feat)
2. **Task 2: Serialize model metadata as JSON in train_game_outcome_model()** - `33360cb` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified
- `src/models/game_outcome_model.py` - Added `_load_game_outcome_model()` helper, updated `predict_game()` call and return dict, added metadata JSON serialization in `train_game_outcome_model()`

## Decisions Made
- Used `UserWarning` (not `RuntimeError`) for missing calibrated model: inference should still work with uncalibrated model, just with a loud, visible warning so the operator knows to run calibration
- `top_importances` extraction wrapped in `try/except` — metadata is informational; a missing named_step should never crash a training run
- All numeric values explicitly cast to `float()` or `int()` before JSON serialization — prevents `TypeError: Object of type float32 is not JSON serializable` at runtime

## Deviations from Plan

None - plan executed exactly as written. The metadata keys used directly reference local variables (`best_name`, `test_acc`, `test_auc`, `best_threshold`, `start_season`, `feat_cols`) which were already in scope — matching the plan's intent despite slightly different key names in the `metrics` dict.

## Issues Encountered
- The plan's overall verification imports `validate_feature_null_rates` (added in plan 01), but plan 01 has not yet been executed. The three plan-02 functions (`predict_game`, `train_game_outcome_model`, `_load_game_outcome_model`) all import cleanly. `validate_feature_null_rates` will be available after plan 01 executes.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `_load_game_outcome_model()` is ready for use by plan 03 (prediction store) — `predict_game()` now returns `model_artifact` that plan 03 can store alongside each prediction
- `game_outcome_metadata.json` will be written on next training run and is ready for web consumption (FR-6.5)
- Plan 01 (injury proxy fix + null-rate guard) can be executed independently — no dependency on plan 02's changes

## Self-Check: PASSED

- FOUND: src/models/game_outcome_model.py
- FOUND: .planning/phases/01-foundation-outputs/01-02-SUMMARY.md
- FOUND commit: afe9eee (Task 1)
- FOUND commit: 33360cb (Task 2)

---
*Phase: 01-foundation-outputs*
*Completed: 2026-03-02*
