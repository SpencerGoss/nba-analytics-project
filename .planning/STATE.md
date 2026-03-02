---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: in_progress
last_updated: "2026-03-02T03:39:30Z"
progress:
  total_phases: 5
  completed_phases: 1
  total_plans: 9
  completed_plans: 5
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-01)

**Core value:** Identify games where the model's win probability meaningfully disagrees with Vegas lines — producing profitable against-the-spread picks over a full NBA season
**Current focus:** Phase 2 — Modern Era Features

## Current Position

Phase: 2 of 5 (Modern Era Features) — IN PROGRESS
Plan: 1 of 4 in current phase (02-01 complete)
Status: Phase 2 started — 02-01 advanced metrics complete, 02-02 Four Factors composite next
Last activity: 2026-03-02 — 02-01 pace-normalized advanced metrics wired into feature pipeline

Progress: [████▓░░░░░] 22%

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 2 min
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation-outputs | 4 | 8 min | 2 min |
| 02-modern-era-features | 1 | 14 min | 14 min |

**Recent Trend:**
- Last 5 plans: 01-01 (90s), 01-02 (2 min), 01-03 (1 min), 01-04 (3 min), 02-01 (14 min)
- Trend: Longer for complex feature engineering plans

*Updated after each plan completion*

| Phase 01-foundation-outputs P01 | 90s | 2 tasks | 3 files |
| Phase 01-foundation-outputs P02 | 2 min | 2 tasks | 4 files |
| Phase 01-foundation-outputs P03 | 1 min | 2 tasks | 5 files |
| Phase 01-foundation-outputs P04 | 3 min | 2 tasks | 1 file |
| Phase 02-modern-era-features P01 | 14 min | 2 tasks | 1 file |

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
- [01-04]: PIPELINE.md explicitly calls out what update.py does NOT run — prevents future sessions from assuming daily refresh covers all stages
- [01-04]: Stage 5 calibration outputs two destinations: models/artifacts/ (inference artifact) and reports/calibration/ (diagnostic reports)
- [Gap closure]: Unicode print statements (arrows, em-dashes) crash on Windows cp1252 — replaced with ASCII
- [Gap closure]: sklearn 1.8 removed CalibratedClassifierCV(cv='prefit') — replaced with _CalibratedWrapper using IsotonicRegression directly
- [02-01]: ADV_ROLL_STATS defined at module level separate from ROLL_STATS — advanced metrics require opponent self-join unavailable at initial ROLL_STATS computation time
- [02-01]: Single opp_box merge replaces narrow opp_style join — prevents duplicate column naming conflicts from a second self-join
- [02-01]: opp_dreb_game renamed to opp_dreb in expanded join — column name standardized, downstream reference updated

### Pending Todos

None yet.

### Blockers/Concerns

- Injury proxy logic bug RESOLVED (01-01): absent rotation detection now works via merge_asof; 137K absent instances found
- sklearn cv='prefit' removal RESOLVED: _CalibratedWrapper replaces deprecated API
- The Odds API historical depth: unknown free-tier range — audit before ATS backfill design (Phase 5)
- Basketball Reference HTML selectors: need spike test of 10 games before committing to referee feature (Phase 3)
- geopy 2.4.x API shape: verify with quick test before building schedule_features.py (Phase 4)

## Session Continuity

Last session: 2026-03-02
Stopped at: Completed 02-01 — advanced metrics rolling features in team_game_features.csv; next is 02-02 Four Factors composite
Resume file: None
