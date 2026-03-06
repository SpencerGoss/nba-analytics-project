# Spec: Model Improvement Phase 1
Date: 2026-03-06
Status: COMPLETE

## Goal
Fix two critical bugs that are currently degrading both models before any architecture work: (1) injury features are silently zeroed out in training data, and (2) the ATS model selects on accuracy instead of calibration, which research shows destroys ROI.

## Non-Goals
- Dashboard changes
- LightGBM/XGBoost architecture upgrade (Phase 2)
- Optuna hyperparameter tuning (Phase 2)
- SBRO odds data integration (Phase 2)
- Any frontend work

## Problem Summary

### CF-2: Injury Features Zero-Filled
`player_absences.csv` does not exist at `data/processed/player_absences.csv`. `injury_proxy.py` falls back to the slow merge_asof path which produces incomplete data. The CONCERNS.md confirms: `home_missing_minutes`, `away_missing_minutes`, `home_star_player_out` etc. are not appearing in model importances — meaning they're imputed to zero. Player availability is the single strongest predictor of game outcome; this needs to work.

Fix path:
1. Run `src/data/get_historical_absences.py` to generate `player_absences.csv`
2. Rebuild `injury_proxy_features.csv` via the primary path
3. Rebuild matchup dataset and retrain

### CF-1: ATS Model Selects on Wrong Metric
`src/models/ats_model.py` line 351 selects the best model via `max(model_scores, key=lambda k: model_scores[k]["mean_val_acc"])`. University of Bath research on NBA data: calibration-optimized = +34.69% ROI vs accuracy-optimized = -35.17% ROI. The ATS model needs to select on Brier score (lower = better calibration = higher ROI).

Also: `calibration.py` fits the isotonic calibrator in-sample (on training data) — a known V1 limitation noted in CONCERNS.md. Fix: reserve 2021-22 as a held-out calibration season.

## Acceptance Criteria

### CF-2 Injury Pipeline
- [ ] `data/processed/player_absences.csv` exists with >100K rows
- [ ] `data/features/injury_proxy_features.csv` rebuilt using primary path (not fallback)
- [ ] `data/features/game_matchup_features.csv` rebuilt with populated injury columns (not all-zero)
- [ ] After retraining, `models/artifacts/game_outcome_importances.csv` lists `diff_missing_usg_pct` or `home_missing_minutes` in the top 30 features
- [ ] All 145 existing tests still pass

### CF-1 ATS Calibration Fix
- [ ] `src/models/ats_model.py` model selection uses `min(model_scores, key=lambda k: model_scores[k]["mean_val_brier"])` (not accuracy)
- [ ] Each validation split computes and stores `brier_score_loss` alongside accuracy
- [ ] Metadata JSON includes `validation_mean_brier` field
- [ ] 2021-22 season is held out as calibration split (not in training, not in test)
- [ ] `src/models/calibration.py` updated to use dedicated calibration split when available
- [ ] ATS model retrained and calibration.py re-run
- [ ] All 145 existing tests still pass

## Implementation Tasks

### Phase A: Generate player_absences.csv (CF-2, Step 1)
- [x] Read `src/data/get_historical_absences.py` to understand what it does and how long it takes
- [x] Fixed: `format="mixed"` added to `pd.to_datetime(df["game_date"])` — was crashing without it
- [x] Run `get_historical_absences.py` — generated player_absences.csv successfully
- [x] Verify row count: 1,098,538 rows (>100K ✓), all columns correct, 12.6% absence rate ✓

### Phase B: Rebuild injury features and matchup dataset (CF-2, Step 2-3)
- [x] Run `src/features/injury_proxy.py` — uses PRIMARY path (126,818 rows, 60.9% games with missing mins)
- [x] Run `src/features/team_game_features.py` (both functions) — 68,216 matchup rows x 291 cols
- [x] Spot-check `game_matchup_features.csv`: home_missing_minutes non-zero in 55.8% of rows (>10% ✓)
- [x] Run tests: 145 passed

### Phase C: Fix ATS model selection metric (CF-1)
- [x] Read full `src/models/ats_model.py` to understand the model loop
- [x] Add `brier_score_loss` import from sklearn.metrics
- [x] In the per-split loop, compute and store `brier_score_loss(y_val, val_proba)` as `split_brier`
- [x] Change `model_scores[name]` dict to include `mean_val_brier`
- [x] Change line 351 from `max(...mean_val_acc)` to `min(...mean_val_brier)` for model selection
- [x] Update metadata JSON to include `validation_mean_brier`

### Phase D: Add dedicated calibration split (CF-1)
- [x] In `ats_model.py`, reserve season `202122` as calibration split (exclude from train, not in test)
- [x] Update `calibration.py` to accept an optional `calibration_season` parameter
- [x] When calibration_season provided: fit isotonic regression on that season's OOF predictions instead of in-sample
- [x] Update CONCERNS.md: mark "Minor V1 Model Calibration Data Leakage" as RESOLVED

### Phase E: Retrain and verify
- [x] Run `src/models/ats_model.py` — logistic_l1 selected, test acc=54.9%, AUC=0.5571
- [x] Run `src/models/game_outcome_model.py` — gradient_boosting, test acc=67.4%, AUC=0.7419
- [x] Run `src/models/calibration.py` — held-out 202122 calibration (1,230 games, no leakage)
- [x] Run `python -m pytest tests/ -q` — 145 passed ✓
- [x] Check `game_outcome_importances.csv` — 11 injury features appear (home_rotation_availability rank #5!)
- [x] Check ATS metadata JSON — validation_mean_brier=0.2498, calibration_season=202122 ✓

## Key Files
- `src/data/get_historical_absences.py` — generates player_absences.csv
- `src/features/injury_proxy.py` — builds injury_proxy_features.csv
- `src/features/team_game_features.py` — builds team_game_features.csv + matchup dataset
- `src/models/ats_model.py` — ATS model training (fix model selection metric)
- `src/models/game_outcome_model.py` — game outcome model training
- `src/models/calibration.py` — calibration (add held-out split)
- `data/processed/player_absences.csv` — generated by get_historical_absences.py
- `data/features/injury_proxy_features.csv` — output of injury_proxy.py
- `data/features/game_matchup_features.csv` — final matchup dataset

## Open Questions
- How long does `get_historical_absences.py` take to run? (answer before starting)
- Does 2021-22 reservation reduce ATS training data too much? (check: how many games is 2021-22?)
