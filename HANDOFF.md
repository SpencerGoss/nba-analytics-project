# Handoff — NBA Analytics Project

_Last updated: 2026-03-06_

## What Was Built

**Model Improvement Phase 1 — COMPLETE**

Two critical bugs fixed:

### CF-1: ATS Model Calibration Fix (DONE)
- `src/models/ats_model.py` now selects model on **Brier score** (min), not accuracy (max)
- `CALIBRATION_SEASON = "202122"` held out from expanding-window CV
- Calibration split saved to `models/artifacts/ats_calibration_split.json`
- `src/models/calibration.py` uses held-out 2021-22 season for isotonic fit (no in-sample leakage)
- ATS model retrained: **logistic_l1** selected, test acc=54.9%, AUC=0.5571 (up from 53.5%)
- Metadata includes `validation_mean_brier=0.2498` and `calibration_season=202122`

### CF-2: Injury Features Wired (DONE)
- `src/data/get_historical_absences.py` fixed: added `format="mixed"` to game_date parsing
- `data/processed/player_absences.csv` generated: **1,098,538 rows**, 12.6% absence rate
- `src/features/injury_proxy.py` now uses **primary path** (126,818 team-game rows, 60.9% with missing minutes)
- `data/features/game_matchup_features.csv` rebuilt: 68,216 rows x 291 cols, home_missing_minutes non-zero in 55.8% of games
- Game outcome model retrained: **11 injury features** now appear in importances (home_rotation_availability rank #5!)
- AUC improved: 0.7256 → **0.7419**

## Current State

- Raw data: fresh (Mar 5-6 2026)
- Game outcome model: retrained, AUC=0.7419, test acc=67.4%
- ATS model: retrained (Brier-optimized), test acc=54.9%, AUC=0.5571
- Calibrated model: rebuilt with held-out 2021-22 season
- Tests: **145 passing**, 0 failing
- Spec: `docs/specs/2026-03-06-model-improvement-phase1.md` — COMPLETE
- Branch: `main` (uncommitted changes — commit next)

## Pinnacle API Details (for reference)
- Base URL: `https://guest.api.arcadia.pinnacle.com/0.1`
- No auth required
- NBA league ID: 487

## What's Next

### Phase 1 Remaining (from research plan)
Per `docs/plans/2026-03-06-model-improvement-research.md`:
1. Add **LightGBM** as candidate model in `game_outcome_model.py` (+0.5-1.5% acc, 10-30x faster)
2. Add **Pythagorean win%** rolling feature to `team_game_features.py`
3. Implement **Fractional Kelly** sizing in `value_bet_detector.py` (0.5x Kelly, +3% EV filter)
4. Implement **CLV (Closing Line Value)** tracking — proves whether 54.9% ATS is real edge

### Phase 2 (higher effort)
- Optuna HPO on LightGBM, XGBoost + blending, SBRO historical odds, margin regression model

### Known Stubs
- `fetch_player_props()` is a no-op stub
- `database/nba.db` — empty legacy artifact; pipeline is CSV-based

## Key Decisions
- Brier score (not accuracy) for ATS model selection — University of Bath research: +34.69% ROI vs -35.17%
- 2021-22 as calibration holdout — largest season available that's not test data, ~1,200 games
- `format="mixed"` is required for ALL `pd.to_datetime()` on game_date columns
