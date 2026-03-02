# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-01)

**Core value:** Identify games where the model's win probability meaningfully disagrees with Vegas lines — producing profitable against-the-spread picks over a full NBA season
**Current focus:** Phase 1 — Foundation & Outputs

## Current Position

Phase: 1 of 5 (Foundation & Outputs)
Plan: 0 of 4 in current phase
Status: Ready to plan
Last activity: 2026-03-01 — Roadmap created, phase structure finalized

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: none yet
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Pre-phase]: Modern era only (2014+) for game outcome model training — post-2014 NBA plays fundamentally differently; mixed-era training hurts accuracy
- [Pre-phase]: ATS model uses separate feature table from win-probability model — spread as input is ATS-only; conflating them destroys value-bet signal
- [Pre-phase]: Vegas lines are model inputs for ATS model, not just comparison targets — market consensus is the strongest public predictor
- [Pre-phase]: Start prediction store in Phase 1, not at web milestone — every day without it is lost calibration history

### Pending Todos

None yet.

### Blockers/Concerns

- Injury proxy join key mismatch: exact cause unknown — trace `injury_proxy.py` before writing new injury code (Phase 1)
- The Odds API historical depth: unknown free-tier range — audit before ATS backfill design (Phase 5)
- Basketball Reference HTML selectors: need spike test of 10 games before committing to referee feature (Phase 3)
- geopy 2.4.x API shape: verify with quick test before building schedule_features.py (Phase 4)

## Session Continuity

Last session: 2026-03-01
Stopped at: Roadmap written. All 5 phases defined. Ready to plan Phase 1.
Resume file: None
