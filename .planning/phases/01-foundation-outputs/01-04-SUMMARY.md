---
phase: 01-foundation-outputs
plan: 04
subsystem: documentation
tags: [pipeline, documentation, update.py, orchestrator, reference]

# Dependency graph
requires:
  - phase: 01-01
    provides: injury proxy via merge_asof, team_game_features.py fixed
  - phase: 01-02
    provides: _load_game_outcome_model, calibrated model loading, metadata JSON
  - phase: 01-03
    provides: prediction store (predictions_history.db), JSON export (data/outputs/)
provides:
  - "docs/PIPELINE.md: single authoritative reference for all 6 pipeline stages"
  - "update.py: confirmed thin orchestrator using only one-line module imports"
  - "Pipeline documentation distinguishing daily refresh vs full rebuild operating modes"
affects:
  - phase-02
  - phase-03
  - phase-04
  - phase-05

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pipeline documentation standard: commands + inputs + outputs + runtimes per stage"
    - "Two operating modes: daily refresh (Stages 1+2 only) vs full rebuild (all 6 stages)"
    - "update.py thin orchestrator: one-line module imports, no subprocess calls to Python scripts"

key-files:
  created:
    - docs/PIPELINE.md
  modified: []

key-decisions:
  - "PIPELINE.md explicitly states what update.py does NOT run — feature build, training, calibration — to prevent confusion about pipeline contract"
  - "Stage 5 output includes both models/artifacts/game_outcome_model_calibrated.pkl and reports/calibration/ diagnostic reports"
  - "Stage 3 can be triggered via train_all_models.py --rebuild-features shortcut (documented)"

patterns-established:
  - "Stage documentation format: Entry point + input paths + output paths + conditional behavior + caveats"
  - "Common operations section: copy-paste recipes for first-time setup, daily refresh, full rebuild, prediction run"

requirements-completed: [FR-7.1, FR-7.2, FR-7.3, FR-7.4, NFR-2]

# Metrics
duration: 3min
completed: 2026-03-02
---

# Phase 1 Plan 04: Pipeline Reference Document Summary

**6-stage NBA analytics pipeline documented in docs/PIPELINE.md with daily-vs-rebuild operating modes, stage-by-stage input/output paths, and update.py confirmed as thin orchestrator with 18 module imports and zero subprocess calls**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-02T01:06:00Z
- **Completed:** 2026-03-02T01:09:22Z
- **Tasks:** 2
- **Files modified:** 1 (docs/PIPELINE.md created)

## Accomplishments

- Audited update.py: 18 `src.*` module imports confirmed, zero subprocess calls to Python scripts — thin orchestrator pattern verified
- Created docs/PIPELINE.md covering all 6 stages with commands, inputs, outputs, and estimated runtimes
- Documented the explicit daily-refresh vs full-rebuild operating modes distinction
- Captured Stage 5 calibration's dual output: model artifact + diagnostic reports in reports/calibration/
- Added --rebuild-features shortcut documentation for train_all_models.py

## Task Commits

Each task was committed atomically:

1. **Task 1: Audit update.py for subprocess calls** - `23d9b73` (read-only confirmation, no code changes)
2. **Task 2: Write docs/PIPELINE.md** - `23d9b73` (docs — committed together since Task 1 produced no file changes)

**Plan metadata:** (final metadata commit — see below)

## Files Created/Modified

- `docs/PIPELINE.md` - Pipeline reference: 6-stage table, daily vs full rebuild distinction, per-stage detail with entry points, input/output paths, runtimes, calibration output paths, common operations recipes

## Decisions Made

- PIPELINE.md explicitly calls out what update.py does NOT run (feature build, training, calibration) — prevents future Claude sessions from thinking daily refresh covers all stages
- Stage 5 outputs documented as two separate destinations: `models/artifacts/game_outcome_model_calibrated.pkl` (inference artifact) and `reports/calibration/` (diagnostic CSVs + PNG) — they serve different consumers
- `--rebuild-features` shortcut documented: allows Stages 3+4 in a single command via `train_all_models.py`

## Deviations from Plan

None - plan executed exactly as written. update.py already followed the thin orchestrator pattern; Task 1 was confirmation only.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 1 is now complete. All four plans delivered:
- 01-01: Injury proxy fixed (merge_asof, MAX_STALE_DAYS=25)
- 01-02: Calibrated model loading, null-rate validation, metadata JSON serialization
- 01-03: Prediction store (predictions_history.db WAL), JSON export (data/outputs/)
- 01-04: Pipeline reference document (docs/PIPELINE.md)

Phase 2 can proceed. The pipeline is documented, functional, and all Phase 1 modules import without errors.

---
*Phase: 01-foundation-outputs*
*Completed: 2026-03-02*
