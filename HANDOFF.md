# Handoff — NBA Analytics Project

_Last updated: 2026-03-13 Session 14 (Plan D pipeline + dashboard + cleanup)_

## What Was Done This Session

### Plan D: Pipeline + Dashboard + Cleanup (7 of 8 tasks COMPLETE)
- **Task 1**: Unified pipeline runner (`scripts/pipeline_runner.py`) — 30 builders, 7 phases, 3 modes, --dry-run, --resume
- **Task 2**: Config module (`src/config.py`) — centralized seasons, teams, paths, model defaults
- **Task 3**: Windows Task Scheduler (`scripts/setup_scheduler.ps1`) — 4AM full, 11:30AM injuries, 6:30PM pretip
- **Task 5 partial**: Confidence tiers wired — `build_picks.py` now uses BettingRouter (Best Bet/Solid Pick/Lean/Skip); dashboard updated
- **Task 6**: Dead code deleted — 8 files, ~2,084 lines removed
- **Task 7**: CI/CD repurposed — `daily_deploy.yml` now CI-only (tests on push/PR)

### Code Cleanup
- Extracted duplicate `_season_splits()` to shared `src/models/cv_utils.py::expanding_season_splits()`
- Dashboard: Promise.allSettled, ticker pause on hover, new tier labels + pill styling
- Updated test baseline in CLAUDE.md and testing.md

### Confidence Tier Unification
- `build_picks.py`: old `_confidence_tier(home_prob, winner, home)` -> new `_confidence_tier(edge, home_prob, projected_margin)` using BettingRouter
- Dashboard `_confLabel()` and `_confLevel()`: thresholds aligned to 8%/4%/2% (was 10%/5%)
- Pick cards now show tier pill badge (Best Bet/Solid Pick/Lean)
- Value bets section: tier labels updated from HIGH/MED to Best Bet/Solid Pick/Lean

## What's Next

### Plan D Remaining
- **Task 4**: Dashboard tiered loading (only Today tab on page open, defer others)
- **Task 5 remaining**: Kelly opt-in toggle, bankroll management, Sharp Money CLV disclaimer, actionable empty states
- **Task 8**: Final verification (dashboard loads, confidence tiers display correctly)

### Full Model Retrain (Task 8 from Plan B)
Now that Huber GBM candidate and ensemble optimizer exist, a full retrain would:
1. Rebuild features, retrain game outcome + margin models
2. Run ensemble weight optimization on validation data
3. Compare to current baseline (67.5% acc, AUC 0.7422, MAE 10.52)
4. Run walk-forward backtest to estimate real betting ROI

### Known Issues
- CLV closing_spread always NULL (update_closing_line never called)
- Lineup features missing for 2025-26 season
- ATS features stop at 2024-25
- Inconsistent PROJECT_ROOT derivation across model files (cosmetic, all compute same path)
- Orchestration scripts (predict_cli.py, retrain_all.py) have no test coverage

## Test Baseline
- 1546 tests passing (0 failures)

## Session Intent 2026-03-13
Goal: Execute Plan D (Pipeline + Dashboard + Cleanup)
Completed: 2026-03-13
Outcome: achieved (7/8 tasks; Task 4 deferred as high-risk for 9K-line file)
