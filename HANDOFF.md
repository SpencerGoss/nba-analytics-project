# Handoff — NBA Analytics Project

_Last updated: 2026-03-13 Session 13 (Plan B model improvements + code review)_

## What Was Done This Session

### Plan B: Model Improvements (6 of 8 tasks COMPLETE)
- Walk-forward betting backtest (`src/models/backtest.py`) — flat-bet + Kelly, realistic vig
- Statistical significance (`src/models/significance.py`) — binomial, bootstrap, McNemar's
- Ensemble weight optimizer (`optimize_ensemble_weights()` in ensemble.py) — Brier grid search
- SHAP analysis (`src/models/shap_analysis.py`) — TreeExplainer with calibration unwrapping
- Margin model upgrade — Huber GBM candidate, residual_std JSON artifact, segmented MAE
- Skipped: temperature scaling (marginal value), orthogonal features (deferred)

### Code Review Fixes
- over_prob formula: `spread_width * 2` -> `spread_width` (was compressing estimates)
- XGBoost eval_set: was using test labels for early stopping (leakage)
- False positive dismissed: season_game_num via cumcount()+1 is public info, not leakage

### Key Analysis Results
- SHAP: diff_elo #1 (0.264), lineup features #4/#13/#15, injury features #6/#9
- Significance: 67.5% game outcome p=3.5e-29; 55% ATS NOT significant vs 52.4% breakeven (p=0.053)
- Need 2,276 bets to confirm ATS beats vig — validates ATS weight=0 decision

### Codebase Health
- Missing `src/processing/__init__.py` — fixed
- Identified: duplicate `_season_splits()`, inconsistent PROJECT_ROOT definitions (Plan D cleanup)

## What's Next

### Plan D: Pipeline + Dashboard + Cleanup (not started)
8 tasks: pipeline runner with retry, config module, dashboard performance, betting UX, dead code removal. Spec at `docs/superpowers/plans/2026-03-13-pipeline-dashboard-cleanup.md`.

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
- Duplicate `_season_splits()` in game_outcome_model.py and margin_model.py
- Inconsistent PROJECT_ROOT derivation across model files
- Orchestration scripts (predict_cli.py, retrain_all.py) have no test coverage

## Test Baseline
- 1580 tests passing (0 failures)

## Session Intent 2026-03-13
Goal: Execute Plan B (Model Improvements) + comprehensive project health check
Completed: 2026-03-13
Outcome: achieved (6/8 tasks, 2 intentionally skipped as low-value)
