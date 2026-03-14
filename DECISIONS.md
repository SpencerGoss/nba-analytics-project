# Project Decisions

Architectural and design decisions for the NBA analytics project.
Consult before re-opening settled questions.

---

## Primary Evaluation Metric: Brier Score
Date: 2026-03-08
Status: Active

### Decision
Use brier score as primary model evaluation metric, not AUC.

### Rationale
Calibration-optimized model (min brier_score_loss) gave +34.69% ROI vs accuracy-optimized model at -35.17% ROI on NBA data (University of Bath research replicated here). AUC measures discrimination only; brier measures calibration which directly translates to betting edge. Switching from accuracy to brier improved ATS from 53.5% to 54.9% and changed selected model from logistic to logistic_l1.

### Revisit If
Moving to a non-probability output model (e.g., pure classification without probability).

---

## Feature Column Management: Explicit List in ml_signal.py
Date: 2026-03-08
Status: Active

### Decision
Maintain explicit _ML_FEATURE_COLS list in ml_signal.py matching train scripts exactly.

### Rationale
LightGBM uses column position, not name. Mismatch = silent wrong predictions with no error. Any feature col with `_roll` in name is auto-captured by `roll_cols` in build_matchup_dataset(); never also add to `context_cols` — duplicates cause ValueError.

### Revisit If
Switching to a model type that uses named features (e.g., sklearn with feature_names_in_).

---

## ATS Calibration Season Hold-Out: 202122
Date: 2026-03-08
Status: Active

### Decision
CALIBRATION_SEASON="202122" is permanently held out from cross-validation for the ATS model.

### Rationale
Using the same data for CV and calibration causes optimistic calibration estimates. The 2021-22 season is held out as a clean out-of-sample calibration set that never influences model selection.

### Revisit If
Accumulating enough post-2022 seasons to use a more recent hold-out year.

---

## Data Source: Pinnacle Guest API (No Auth)
Date: 2026-03-08
Status: Active

### Decision
Use Pinnacle guest API (league 487, no auth) as the primary odds source, not The Odds API.

### Rationale
The Odds API key expired (401). Pinnacle guest API confirmed free and keyless as of 2026-03-06. Team name mapping (ODDS_TEAM_TO_ABB) reused unchanged. Player props available via separate endpoint (GET /matchups/{id}/markets/straight).

### Revisit If
Pinnacle restricts guest access, or a paid API with better coverage is justified.

---

## Ensemble Weights: win=0.5 / ats=0.3 / margin=0.2
Date: 2026-03-08
Status: Active

### Decision
NBAEnsemble blends three models with fixed weights: win_prob=0.5, ats_prob=0.3, margin_signal=0.2.

### Rationale
Win probability model is most reliable (67.9% acc, AUC 0.7455). ATS model adds calibrated spread edge. Margin model (Ridge, MAE 10.574) is the weakest signal. Weights reflect relative confidence, not equal blending.

### Revisit If
ATS model accuracy consistently exceeds win model, or margin MAE drops below 9.0.
