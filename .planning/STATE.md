---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
last_updated: "2026-03-02T08:50:54.587Z"
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 13
  completed_plans: 13
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
last_updated: "2026-03-02T07:09:53.293Z"
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 11
  completed_plans: 11
---

---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: in_progress
last_updated: "2026-03-02T07:10:00Z"
progress:
  total_phases: 5
  completed_phases: 3
  total_plans: 9
  completed_plans: 9
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-01)

**Core value:** Identify games where the model's win probability meaningfully disagrees with Vegas lines — producing profitable against-the-spread picks over a full NBA season
**Current focus:** Phase 3 — External Data Layer

## Current Position

Phase: 4 of 5 (Schedule + Travel Features) — COMPLETE
Plan: 04-02 complete; Phase 4 fully done; next: Phase 5 ATS Model
Status: 04-02 complete -- season_month added to matchup dataset; schedule_cols updated; model retrained with 68 features (10 Phase 4); 66.8% test accuracy
Last activity: 2026-03-02 -- 04-02 season_month + schedule_cols + model retrain complete

Progress: [██████████████] 75%

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 18 min
- Total execution time: 2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation-outputs | 4 | 8 min | 2 min |
| 02-modern-era-features | 3 | 112 min | 37 min |

**Recent Trend:**
- Last 5 plans: 01-04 (3 min), 02-01 (14 min), 02-02 (10 min), 02-03 (88 min)
- Trend: 02-03 dominated by 3 training runs (model selection with gradient boosting on full history = slow)

*Updated after each plan completion*

| Plan | Duration | Tasks | Files |
| ---- | -------- | ----- | ----- |
| Phase 01-foundation-outputs P01 | 90s | 2 tasks | 3 files |
| Phase 01-foundation-outputs P02 | 2 min | 2 tasks | 4 files |
| Phase 01-foundation-outputs P03 | 1 min | 2 tasks | 5 files |
| Phase 01-foundation-outputs P04 | 3 min | 2 tasks | 1 file |
| Phase 02-modern-era-features P01 | 14 min | 2 tasks | 1 file |
| Phase 02-modern-era-features P02 | 10 min | 2 tasks | 1 file |
| Phase 02-modern-era-features P03 | 88 min | 2 tasks | 1 file |
| Phase 03-external-data-layer P01 | 13 min | 2 tasks | 3 files |
| Phase 03-external-data-layer P03 | 127 | 1 tasks | 2 files |
| Phase 03-external-data-layer P04 | 189 | 2 tasks | 3 files |
| Phase 03-external-data-layer P02 | 3 | 2 tasks | 2 files |
| Phase 04-rest-schedule-features P01 | 21 min | 2 tasks | 2 files |
| Phase 04-rest-schedule-features P02 | 17 | 2 tasks | 2 files |

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
- [Phase 02-modern-era-features]: diff_ prefix on Four Factors composite: naming diff_four_factors_composite ensures automatic pickup by get_feature_cols() startswith('diff_') filter — no model code changes needed
- [Phase 02-modern-era-features]: Selective roll windows for rate stats: roll20-only for eFG%/TS%/pace/tov_poss (noisy at shorter windows); ORtg/DRtg/net_rtg get 5/10/20 windows as primary efficiency signals
- [02-03]: MODERN_ERA_START changed 201415->201314 per plan spec (include 2013-14 season as modern era start)
- [02-03]: Research finding: full history GB model (0.6729) marginally outperforms modern era RF model (0.6684) on same test holdout — model selection variance with 4 vs 19 splits is likely cause; modern era model still operational as final artifact
- [02-03]: Calibration artifact needs regeneration: game_outcome_model_calibrated.pkl predates modern era retraining; run calibration.py before using calibrated inference
- [03-01]: HTML comment unwrapping pattern for Basketball Reference confirmed from 3 independent PyPI packages (basketball_reference_web_scraper, sportsreference, basketball_reference_scraper) — all use equivalent strip/iterate techniques
- [03-01]: Officials table ID='officials' follows BR naming convention (div_officials -> officials); 3-tier fallback: direct DOM, Comment iteration, full strip-and-reparse
- [03-01]: pdfplumber added to requirements.txt in 03-01 to avoid second touch in 03-03
- [Phase 03-external-data-layer]: nba_api LeagueInjuryReport as primary source (no URL guessing), PDF fallback for resilience; FR-4.4 date guard prevents misuse for historical training data
- [Phase 03-external-data-layer]: _CODE_PATH string constant chosen over assertion-based guard: makes boundary visible at module load without runtime cost
- [Phase 03-external-data-layer]: get_todays_injury_report() in injury_proxy.py retained with deprecation note -- removal would be breaking change for any callers
- [Phase 03-external-data-layer]: Join on game_date + home team abbreviation (not game_id): bref game ID format incompatible with NBA API; home team 3-char abbr is reliable cross-source key
- [Phase 03-external-data-layer]: 03-02: NaN (not 0) for referee features pre-scrape: Pitfall 6 compliance -- zero injects false signal; SimpleImputer handles NaN via mean imputation
- [04-01]: Vectorized haversine (numpy) used instead of geopy per-row: 1000x faster on 30K rows; 0.2% accuracy tradeoff negligible for 0-2700mi features
- [04-01]: Static ARENA_COORDS and ARENA_TIMEZONE dicts at module level: 30 NBA arenas are fixed, instant lookup, no API key
- [04-01]: diff_is_back_to_back added to diff_stats: captures fatigue asymmetry (home B2B vs rested away) as direct signal
- [04-01]: travel_miles=0 for season opener (fillna(0)): correct neutral value for no-prior-game case
- [Phase 04-rest-schedule-features]: season_month added to build_matchup_dataset() AFTER merge: game-level feature avoids home_/away_ duplication
- [Phase 04-rest-schedule-features]: schedule_cols updated with home_travel_miles, away_travel_miles, home_cross_country_travel, away_cross_country_travel, season_month; diff_ variants auto-picked up by startswith filter
- [Phase 04-rest-schedule-features]: Calibrated model (game_outcome_model_calibrated.pkl) predates retrained model: run calibration.py before production inference use

### Pending Todos

None yet.

### Blockers/Concerns

- Injury proxy logic bug RESOLVED (01-01): absent rotation detection now works via merge_asof; 137K absent instances found
- sklearn cv='prefit' removal RESOLVED: _CalibratedWrapper replaces deprecated API
- The Odds API historical depth: unknown free-tier range — audit before ATS backfill design (Phase 5)
- Basketball Reference HTML selectors: PARTIALLY RESOLVED (03-01) — HTML comment pattern confirmed from 3 PyPI packages; officials table ID='officials' MEDIUM confidence (naming convention); Cloudflare blocks in this environment; recommend live verification from cloud VM before building 03-02 referee features
- geopy 2.4.x API shape: RESOLVED (04-01) -- geopy listed in requirements.txt; haversine used for performance; no API shape issues
- game_outcome_model.py schedule_cols: RESOLVED (04-02) -- updated with home_travel_miles, away_travel_miles, home_cross_country_travel, away_cross_country_travel, season_month
- Calibrated model needs regeneration: game_outcome_model_calibrated.pkl predates 04-02 retrained model; run calibration.py before production inference

## Session Continuity

Last session: 2026-03-02
Stopped at: Completed 04-02 -- season_month feature + schedule_cols update + model retrain with 10 Phase 4 features; 66.8% test accuracy; Phase 4 complete
Resume file: None
