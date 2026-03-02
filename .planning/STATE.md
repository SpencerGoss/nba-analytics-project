---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: in_progress
last_updated: "2026-03-02T01:06:00Z"
progress:
  total_phases: 1
  completed_phases: 0
  total_plans: 4
  completed_plans: 3
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-01)

**Core value:** Identify games where the model's win probability meaningfully disagrees with Vegas lines — producing profitable against-the-spread picks over a full NBA season
**Current focus:** Phase 1 — Foundation & Outputs

## Current Position

Phase: 1 of 5 (Foundation & Outputs)
Plan: 3 of 4 in current phase
Status: In progress
Last activity: 2026-03-02 — Completed 01-03 (prediction store and JSON export wired into predict_game)

Progress: [███░░░░░░░] 15%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 2 min
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation-outputs | 3 | 5 min | 2 min |

**Recent Trend:**
- Last 5 plans: 01-01 (90s), 01-02 (2 min), 01-03 (1 min)
- Trend: Consistent

*Updated after each plan completion*

| Phase 01-foundation-outputs P01 | 90s | 2 tasks | 3 files |
| Phase 01-foundation-outputs P02 | 2 min | 2 tasks | 4 files |
| Phase 01-foundation-outputs P03 | 1 min | 2 tasks | 5 files |

## Accumulated Context

### Decisions

Decisions logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Pre-phase]: Modern era only (2014+) for game outcome model training — post-2014 NBA plays fundamentally differently; mixed-era training hurts accuracy
- [Pre-phase]: ATS model uses separate feature table from win-probability model — spread as input is ATS-only; conflating them destroys value-bet signal
- [Pre-phase]: Vegas lines are model inputs for ATS model, not just comparison targets — market consensus is the strongest public predictor
- [Pre-phase]: Start prediction store in Phase 1, not at web milestone — every day without it is lost calibration history
- [01-02]: UserWarning (not hard error) for missing calibrated model — inference still works with uncalibrated, operator sees loud warning to run calibration
- [01-02]: Metadata JSON uses only Python builtins — numpy types never written to JSON to avoid TypeError at runtime
- [Phase 01-foundation-outputs]: Injury proxy root cause was logic bug (anti-join on same df), not type mismatch — requires merge_asof strategy across team-game schedule
- [Phase 01-foundation-outputs]: merge_asof with MAX_STALE_DAYS=25 chosen for absent rotation detection — handles trades and long-term injuries without O(n^2) loop
- [Phase 01-03]: Non-fatal store write: store failure issues UserWarning but never prevents inference result from being returned
- [Phase 01-03]: game_date field included in result dict so consumers can inspect date stored alongside probabilities

### Pending Todos

None yet.

### Blockers/Concerns

- Injury proxy logic bug RESOLVED (01-01): absent rotation detection now works via merge_asof; 137K absent instances found
- The Odds API historical depth: unknown free-tier range — audit before ATS backfill design (Phase 5)
- Basketball Reference HTML selectors: need spike test of 10 games before committing to referee feature (Phase 3)
- geopy 2.4.x API shape: verify with quick test before building schedule_features.py (Phase 4)

## Session Continuity

Last session: 2026-03-02
Stopped at: Completed 01-03-PLAN.md (prediction store and JSON export wired into predict_game)
Resume file: None
