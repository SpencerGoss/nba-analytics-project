# Handoff — NBA Analytics Project

_Last updated: 2026-03-11 Session 8 (model tuning + dashboard aurora redesign)_

## What Was Done This Session

### Model Upgrades (6 improvements)
- **Fast Elo (K=40)** — `src/features/elo.py` now computes both standard (K=20) and fast (K=40) Elo; `elo_momentum = fast - standard` captures surging/slumping teams; 9 new tests
- **Elo x context interactions** — `elo_x_rest`, `elo_x_b2b`, `elo_x_streak` added to `team_game_features.py`
- **GBM regularization** — `min_samples_leaf=20`, `max_features=0.7`, early stopping (`n_iter_no_change=15`, `n_estimators=500`)
- **Auto feature pruning** — drops features with importance < 0.001; retains pruned set only if AUC improves
- **Platt scaling calibration** — `calibration.py` auto-selects Platt (2-param sigmoid) vs Isotonic by Brier score
- **Confidence-dependent ensemble** — ATS weight=0.0 (near-random AUC removed); dynamic weights: high-conf 0.75/0.25, default 0.65/0.35, uncertain 0.55/0.45; 12 new ensemble tests

### Dashboard Visual Redesign
- **Aurora backgrounds** — elongated skewed color bands (green/gold/blue/purple) replacing circular blobs; 50-60s animation cycles
- **Gradient nav border** — green-to-blue gradient via background-clip technique
- **Card upgrades** — gradient top-border on hover, 32px backdrop blur, inner glow
- **Spotlight card** — animated gradient sweep, triple-layer glow, gradient border (green/blue/purple)
- **Organic mesh** — deep indigo mid-tone (`--bg-mid: #0D1326`), purple shimmer layer
- **about.html** — updated stats (67.5%, 349+ features, 1407 tests), added Margin + Ensemble model specs, aurora background

### Infrastructure
- Season history builder uses dynamic season codes (not hardcoded)
- JSON builders use compact output (no indent)
- Test baseline: **1407 passing** (+21 new)
- All docs updated (CLAUDE.md, testing.md, MEMORY.md, PROJECT_JOURNAL.md, WORKING_NOTES.md)

## Commits
- `f8c96e0` — feat: model upgrades, dashboard aurora redesign, infrastructure (50 files, +3030/-335)
- `1cc5b11` — docs: journal + working notes for session 8

## Push Status
- **NOT YET PUSHED** — git credential manager needs interactive auth
- Run `git push origin main` in terminal to deploy

## Next Steps (priority order)

1. **Push to GitHub** — `git push origin main` (needs terminal auth)
2. **Retrain models** — run `python update.py` to retrain with new features + hyperparameters
3. **Debug empty game_lines.csv** — Pinnacle API calls need debugging (fetch_odds.py)
4. **Wire CLV closing line fetch** — `update_closing_line()` exists but is never called
5. **Hustle stats features** — team_hustle_stats.csv exists but unused; high-value add for model

## Critical Gotchas
- Ensemble weights are now **dynamic** (not fixed) — code in `ensemble.py` selects regime per-prediction
- Calibration uses `_PlattWrapper` or `_CalibratedWrapper` — both implement `predict_proba()` identically
- Feature pruning runs automatically during training — check logs for pruned features
- `_setHtml(el, html)` for ALL dashboard DOM writes — security hook blocks innerHTML
- `game_lines.csv` at `data/odds/` (NOT `data/processed/`)
- `dashboard/data/*.json` must be committed after each `update.py` run
