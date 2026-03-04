---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Data Expansion & Model Intelligence
status: not started
last_updated: "2026-03-04"
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-04)

**Core value:** Identify games where the model's win probability meaningfully disagrees with Vegas lines — producing profitable against-the-spread picks over a full NBA season
**Current focus:** Phase 6 — Production Fixes & Injury Data (not started)

## Current Position

Phase: 6 of 9 (Production Fixes & Injury Data) — NOT STARTED
Plan: None started
Status: v2.0 roadmap created; ready to begin Phase 6 planning
Last activity: 2026-03-04 — v2.0 roadmap initialized

Progress: [____________________] 0%

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

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions from v1.0 carried forward — see v1.0 STATE.md archive or .planning/PROJECT.md Key Decisions table.

v2.0 decisions will be logged here as they are made.

### Pending Todos

- Begin Phase 6 planning: `/gsd:plan-phase 6`

### Blockers/Concerns

- Calibrated model not in fetch_odds.py (FIX-01) — known gap from v1.0, Phase 6 resolves
- Injury features all-null risk in production — verify after Phase 6 Kaggle integration
- pbpstats API rate limits and data availability: unknown until Phase 7 research

## Session Continuity

Last session: 2026-03-04 (v2.0 roadmap created)
Stopped at: Roadmap written; no plans started
Resume file: None — start with `/gsd:plan-phase 6`
