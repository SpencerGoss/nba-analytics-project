---
phase: 05-ats-model
plan: 02
subsystem: models
tags: [ats, classification, sklearn, expanding-window, logistic-regression, betting, spread]

# Dependency graph
requires:
  - phase: 05-ats-model
    plan: 01
    provides: data/features/game_ats_features.csv (18,496 rows, 276 cols, covers_spread target)
provides:
  - src/models/ats_model.py: ATS classifier with train_ats_model() and predict_ats()
  - models/artifacts/ats_model.pkl: Trained logistic regression pipeline (gitignored)
  - models/artifacts/ats_model_features.pkl: 71 feature names including spread/implied_prob (gitignored)
  - models/artifacts/ats_model_metadata.json: Model metadata with test_accuracy, model_type, etc. (gitignored)
affects:
  - future-inference: predict_ats() integrated into value-bet detector
  - phase-06-inference: ATS model prediction alongside win-probability model

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Expanding-window season splits (MIN_TRAIN_SEASONS=4, 11 validation splits)"
    - "sklearn Pipeline: SimpleImputer -> StandardScaler -> Classifier"
    - "Three-candidate model selection: logistic vs gradient_boosting vs random_forest"
    - "Decision threshold tuning via grid search (0.35..0.65 step 0.01)"
    - "ATS-specific market signals: spread + home_implied_prob + away_implied_prob alongside diff_ features"
    - "Metadata JSON with only Python builtins (no numpy types)"

key-files:
  created:
    - src/models/ats_model.py
  modified: []

key-decisions:
  - "Logistic regression selected over gradient boosting and random forest: mean val acc 0.5287 vs 0.5228 (GB) vs 0.5287 (RF slightly worse on AUC tiebreak)"
  - "51.2% test accuracy expected for ATS -- Vegas sets efficient lines; beating random by 1-2% is meaningful signal"
  - "16.6% null rate on implied_prob columns accepted: moneyline data missing for ESPN-sourced rows (Jan 2023+); SimpleImputer handles via mean imputation"
  - "Model artifacts gitignored (large binaries): ats_model.pkl, ats_model_features.pkl in models/artifacts/"
  - "EXCLUDED_SEASONS had 0 effect: game_ats_features.csv already excluded 201920/202021 at ATS feature build time (inner join on Kaggle odds data which had no games those seasons)"

# Metrics
duration: 35min
completed: 2026-03-02
---

# Phase 5 Plan 02: ATS Classifier Training Summary

Logistic regression ATS classifier trained on 18,233 non-push games with 71 features (51 diff_ matchup differentials + schedule + injury + spread/implied_prob market signals), achieving 51.2% accuracy on 2023-24 and 2024-25 holdout seasons via expanding-window validation (11 splits)

## Performance

- **Duration:** 35 min
- **Started:** 2026-03-02T19:09:15Z
- **Completed:** 2026-03-02T19:44:00Z
- **Tasks:** 2
- **Files modified:** 1 (src/models/ats_model.py -- artifacts gitignored)

## Accomplishments

- Created `src/models/ats_model.py` mirroring `game_outcome_model.py` architecture exactly
- `get_ats_feature_cols()`: 71 features -- 51 diff_ matchup differentials + 8 schedule/context + 3 ATS market signals (spread, home_implied_prob, away_implied_prob)
- `_ats_season_splits()`: 11 expanding-window validation splits (seasons 2010-11 through 2022-23)
- Logistic regression beat gradient boosting and random forest on mean validation accuracy (0.5287 vs 0.5228 vs 0.5284)
- Test accuracy 51.2% on 2,455 holdout games (2023-24/2024-25); ROC-AUC 0.5077
- `predict_ats()` function verified: returns `covers_spread_prob` (0-1) and `covers_spread_pred` (binary) columns
- 263 push rows correctly excluded (NaN covers_spread dropped before training)
- All 3 artifact files verified: ats_model.pkl (loadable pipeline), ats_model_features.pkl (71 features including spread), ats_model_metadata.json (all required fields)

## Task Commits

1. **Task 1: Build ATS classifier with expanding-window validation** - `170d856` (feat)
2. **Task 2: Verify ATS model artifacts and prediction function** - (no new files; verified via existing Task 1 artifacts)

## Files Created/Modified

- `src/models/ats_model.py` - ATS classifier: get_ats_feature_cols(), _ats_season_splits(), train_ats_model(), predict_ats()
- `models/artifacts/ats_model.pkl` - Trained logistic regression pipeline (gitignored)
- `models/artifacts/ats_model_features.pkl` - 71 feature names (gitignored)
- `models/artifacts/ats_model_metadata.json` - Metadata: model_type, test_accuracy, test_auc, threshold, training_date, n_train_rows, n_test_rows, test_seasons, excluded_seasons (gitignored)

## Decisions Made

- **Logistic regression selected:** Model selection ran 3 candidates x 11 splits (33 fits). Logistic regression (0.5287 mean val acc) edged gradient boosting (0.5228) and random forest (0.5284) -- margin is ~0.5%, reflecting the inherent difficulty of ATS prediction.
- **51.2% test accuracy is meaningful:** Vegas lines are set to be near-50/50 for bettors. Any systematic edge above 52.4% (vig breakeven) is exploitable. Current model at 51.2% is below vig breakeven but establishes the baseline; combining with win-probability disagreement signal (Phase 6) is the value-bet strategy.
- **16.6% null rate on implied_prob accepted:** 3,298 ESPN-sourced rows (post-Jan 2023) lack moneyline data. SimpleImputer fills with mean implied prob (~0.5). This is intentional -- missing market data should not exclude recent games from training.
- **EXCLUDED_SEASONS had 0 effect:** The ATS feature table was built from an inner join with Kaggle betting data. Kaggle had no games for 201920/202021 (COVID seasons), so they were already excluded at join time. The filter in train_ats_model() is defensive.

## Deviations from Plan

None - plan executed exactly as written. All task acceptance criteria met on first attempt.

## Model Performance Summary

| Metric | Value |
|--------|-------|
| Model type | Logistic Regression |
| Training rows | 15,778 |
| Test rows | 2,455 |
| Features | 71 |
| Mean val accuracy (11 splits) | 52.87% |
| Test accuracy | 51.20% |
| Test ROC-AUC | 0.5077 |
| Decision threshold | 0.51 |

## Top Features (by logistic regression coefficient magnitude)

1. diff_tov_poss_game_roll20
2. diff_tov_roll20
3. diff_pace_game_roll20
4. diff_off_rtg_game_roll10
5. diff_pts_roll20

Turnover differential (rolling 20 games) is the strongest ATS predictor, followed by pace and offensive rating differentials. Market signals (spread, implied_prob) rank lower -- they are meaningful but their information is partially subsumed by the diff_ features.

## Next Phase Readiness

- `predict_ats()` is ready for Phase 6 inference integration
- ATS model artifacts in `models/artifacts/` alongside win-probability model
- Value-bet detector (Phase 6) will compare `home_win_prob` (game_outcome_model) vs `home_implied_prob` (from odds) to identify disagreement games, then use ATS model for spread prediction

## Self-Check: PASSED

All created files verified:
- FOUND: src/models/ats_model.py
- FOUND: models/artifacts/ats_model.pkl
- FOUND: models/artifacts/ats_model_features.pkl
- FOUND: models/artifacts/ats_model_metadata.json

All commits verified:
- FOUND: 170d856 (Task 1 - ATS classifier)

---
*Phase: 05-ats-model*
*Completed: 2026-03-02*
