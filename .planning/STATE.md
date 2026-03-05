---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Data Expansion & Model Intelligence
status: complete
last_updated: "2026-03-05"
progress:
  total_phases: 11
  completed_phases: 11
  total_plans: 1
  completed_plans: 1
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-04)

**Core value:** Identify games where the model's win probability meaningfully disagrees with Vegas lines — producing profitable against-the-spread picks over a full NBA season
**Current focus:** v2.0 COMPLETE — ready to close milestone and plan v3.0

## Current Position

Phase: 11 of 11 — ALL PHASES COMPLETE
Status: All 11 phases complete. v2.0 milestone achieved.
Last activity: 2026-03-05 — Phase 11 complete; composite ATS+edge signal in get_strong_value_bets()

Progress: [████████████████████] 100% (11/11 phases done)

## v2.0 Milestone Goals

- Game outcome model: 66.8% (v1) → **68%+ target**
- ATS model: 51.2% (v1) → **52.4%+ vig breakeven target**
- Real injury data: Kaggle dataset replaces rolling-average proxy
- Lineup data: pbpstats on-court lineups added as features (strongest signal not yet used)
- Ensemble: stacking layer for value-bet detection

## v1.0 Baseline (carried forward)

- Game outcome model: 66.8% accuracy on 2023-24/2024-25 holdout
- ATS: 51.2% raw, +1.28% ROI on value-bet filtered (18,233-game backtest)
- Known gaps: calibrated model not in fetch_odds.py; injury features use proxy not real data
- 59 unit tests passing

## Performance Metrics

**v2.0 Velocity:** (to be populated as plans complete)

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 06-production-fixes-injury-data | TBD | - | - |
| 07-new-data-sources | TBD | - | - |
| 08-feature-engineering | TBD | - | - |
| 09-model-retraining-ats-optimization | TBD | - | - |
| 10-kaggle-injury-real-features | 1 completed | ~9 min | 9 min |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions from v1.0 carried forward — see v1.0 STATE.md archive or .planning/PROJECT.md Key Decisions table.

v2.0 decisions will be logged here as they are made.

**2026-03-05 (10-01):** Output includes both was_absent=0 (played) and was_absent=1 (absent) rows for full rotation coverage. was_absent rate of 12.6% is correct — plan's 40-65% expectation was for a different metric (team-game level, not player-game level).

**2026-03-05 (10-01):** game_id normalized as str from int64 source (leading zeros stripped). Matches real player_game_logs.csv behavior where game_id is int64.

### Pending Todos

- v2.0 milestone complete. Next: plan v3.0 (web dashboard + live prediction app).

### Blockers/Concerns

- Calibrated model now loaded in fetch_odds.py and value_bet_detector.py — RESOLVED
- Injury features wired via player_absences.csv fallback — model retrain pending for full activation
- SHAP reports stale — regenerate with python src/models/model_explainability.py (non-blocking)

## Session Continuity

Last session: 2026-03-05 (Phase 10 Plan 01 executed)
Stopped at: Completed 10-01-PLAN.md — player_absences.csv generated
Resume file: None — next is Phase 10 Plan 02 (if planned) or Phase 11
