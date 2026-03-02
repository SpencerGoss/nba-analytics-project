---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
last_updated: "2026-03-02T01:01:37.928Z"
progress:
  total_phases: 1
  completed_phases: 0
  total_plans: 4
  completed_plans: 2
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-01)

**Core value:** Identify games where the model's win probability meaningfully disagrees with Vegas lines — producing profitable against-the-spread picks over a full NBA season
**Current focus:** Phase 1 — Foundation & Outputs

## Current Position

Phase: 1 of 5 (Foundation & Outputs)
Plan: 2 of 4 in current phase
Status: In progress
Last activity: 2026-03-02 — Completed 01-02 (calibrated inference path + model metadata JSON)

Progress: [██░░░░░░░░] 10%

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 2 min
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation-outputs | 1 | 2 min | 2 min |

**Recent Trend:**
- Last 5 plans: 01-02 (2 min)
- Trend: -

*Updated after each plan completion*
| Phase 01-foundation-outputs P01 | 90 | 2 tasks | 3 files |

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

### Pending Todos

None yet.

### Blockers/Concerns

- Injury proxy logic bug RESOLVED (01-01): absent rotation detection now works via merge_asof; 137K absent instances found
- The Odds API historical depth: unknown free-tier range — audit before ATS backfill design (Phase 5)
- Basketball Reference HTML selectors: need spike test of 10 games before committing to referee feature (Phase 3)
- geopy 2.4.x API shape: verify with quick test before building schedule_features.py (Phase 4)

## Session Continuity

Last session: 2026-03-01
Stopped at: Completed 01-01-PLAN.md (injury proxy logic fix and null-rate guard)
Resume file: None
