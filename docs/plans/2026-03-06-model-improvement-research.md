# NBA Model Improvement Research Synthesis
*Generated: 2026-03-06 | 5 parallel research agents + synthesis*

## Executive Summary

The project is already at the theoretical ceiling for pre-game team-level prediction (68% / AUC 0.726, versus a published ceiling of 67-70%). Marginal accuracy gains require either new data types (player-level RAPM, real injury activation) or architecture upgrades (LightGBM/XGBoost replacing Random Forest). The single highest-leverage move is implementing CLV (Closing Line Value) tracking — positive CLV is the only durable proof of edge that survives sample noise. The ATS model's metric selection is wrong: training to maximize accuracy rather than Brier score / calibration is what University of Bath research showed destroys ROI (-35% vs +34% depending on which metric you optimize). Two of five research agents independently confirmed this from different research angles.

---

## Critical Findings

**CF-1: ATS model optimization metric is wrong — fix before any other model work**

The ATS training loop uses `accuracy_score` for model selection. University of Bath research on NBA data (confirmed by Agent 4 and Agent 5 independently) shows:
- Calibration-optimized selection: **+34.69% ROI**
- Accuracy-optimized selection: **-35.17% ROI**

Fix: switch ATS model selection CV metric to Brier score. Add a dedicated held-out calibration season (e.g., 2021-22) between train and test, separate from the expanding-window validation.

**CF-2: Injury features are still zero-filled in training data**

`data/processed/player_absences.csv` was generated in Phase 10 but is NOT wired into `injury_proxy.py` or `team_game_features.py`. The model trains with `home_missing_minutes`, `home_star_player_out`, etc. silently imputed to zero. Player availability is the single strongest known predictor of game outcomes. Fix this before retraining anything else.

---

## Phase 1 — Do First (High Impact, Low/Medium Effort)

**1. Wire player_absences.csv into injury pipeline**
- Fix bare `except Exception` in `build_team_game_features()` that silently drops injury columns
- Merge `data/processed/player_absences.csv` into `src/features/injury_proxy.py`
- Regenerate `game_matchup_features.csv`, retrain both models, run `calibration.py`
- Expected impact: +1-3% accuracy
- Files: `src/features/injury_proxy.py`, `src/features/team_game_features.py`

**2. Switch ATS model selection to Brier score + fix calibration split**
- Replace `accuracy_score` with `brier_score_loss` for ATS model selection
- Reserve 2021-22 as a held-out calibration split (separate from expanding-window)
- Resolves "Minor V1 Model Calibration Data Leakage" in CONCERNS.md
- Expected impact: Most significant single ATS ROI improvement
- Files: `src/models/ats_model.py`, `src/models/calibration.py`

**3. Implement CLV (Closing Line Value) tracking**
- At prediction time, log Pinnacle opening spread into `predictions_history.db`
- Add nightly job to fetch closing line and compute `CLV = opening_line - closing_line`
- Expected impact: Zero accuracy change; proves whether 53.5% ATS is real edge or variance
- Files: `scripts/fetch_odds.py`, `database/predictions_history.db` (schema migration), new `src/models/clv_tracker.py`

**4. Add LightGBM as a candidate model**
- Add `lightgbm.LGBMClassifier` to the candidate dict in `game_outcome_model.py`
- Run through existing expanding-window CV loop (no HPO yet)
- Expected impact: +0.5-1.5% accuracy; 10-30x faster training than sklearn GBM
- `pip install lightgbm`
- Files: `src/models/game_outcome_model.py`, `requirements.txt`

**5. Verify / add Four Factors as explicit rolling differentials**
- Verify `diff_efg_game`, `diff_tov_poss_game`, `diff_oreb_pct_game`, `diff_ft_rate_game` exist in `game_matchup_features.csv`
- Four Factors account for 96% of variance in team wins
- Expected impact: +0.5-1% if gap found; if already present, no work needed
- Files: `src/features/team_game_features.py`

**6. Add Pythagorean win% as a rolling feature**
- Compute `pts^14.3 / (pts^14.3 + opp_pts^14.3)` on rolling 10-game window with `shift(1)`
- Add `diff_pythagorean_win_pct` to matchup dataset
- Expected impact: +0.3-0.5% accuracy; more predictive than raw W/L
- Files: `src/features/team_game_features.py`

**7. Implement Fractional Kelly position sizing**
- Add `kelly_fraction` field to `get_strong_value_bets()` output
- Use 0.5x Kelly, minimum +3% EV threshold filter
- Expected impact: Improved ROI with current model, zero model changes needed
- Files: `src/models/value_bet_detector.py`

---

## Phase 2 — Do Second (High Impact, Higher Effort)

**8. Hyperparameter optimization with Optuna on LightGBM**
- 100 trials, custom expanding-window CV (season boundaries, not random splits)
- Tune: `num_leaves`, `learning_rate`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- Wall clock: ~15-30 minutes
- Expected impact: +0.3-0.8% accuracy
- `pip install optuna`
- Files: New `src/models/hpo_lgbm.py`

**9. Add XGBoost + model blending**
- Add `XGBClassifier` as fifth candidate; run Optuna HPO (50 trials)
- Blend: `(lgbm_prob + xgb_prob + lr_prob) / 3`
- Expected impact: +0.3-0.8% from blending; XGBoost alone +0.5-1.5%
- Files: `src/models/game_outcome_model.py`, `requirements.txt`

**10. SBRO historical odds data integration**
- Download free XLS from sportsbookreviewsonline.com (2007-2025, opening + closing lines)
- Compute line movement delta, RLM flag, enable CLV backtesting
- Expected impact: +0.5-1.5% ATS win rate
- Files: New `src/data/get_sbro_odds.py`, new `src/features/line_movement_features.py`

**11. Add margin regression model**
- Train `XGBRegressor` to predict `home_pts - away_pts`
- Compare predicted margin to spread; use gap as parallel ATS signal
- Expected impact: +0.5-1% ATS win rate
- Files: New `src/models/margin_model.py`, update `src/models/value_bet_detector.py`

**12. Activate nba_api tracking endpoints**
- Add fetchers for `LeagueDashPtDefend`, `LeagueHustleStatsTeam`, `BoxScorePlayerTrackV3`
- Roll into 10-game team-level averages with `shift(1)`
- Free, zero new dependencies, exposes Second Spectrum defensive/hustle metrics
- Expected impact: +0.3-0.8% accuracy
- Files: New `src/data/get_tracking_stats.py`, update `src/features/team_game_features.py`

**13. Add travel direction feature**
- Use existing `ARENA_TIMEZONE` to add `westward_travel` binary flag
- Eastward vs. westward win rate: 44.51% vs 40.83%, p=0.024
- Expected impact: +0.2-0.4% accuracy
- Files: `src/features/team_game_features.py`

---

## Phase 3 — Do Later (Medium Impact / Research-Level)

14. **EWMA feature variants** — `pd.Series.ewm(span=5/10).mean()` with `shift(1)` for volatile roster periods
15. **SHAP-based feature pruning** — `shap.TreeExplainer` on LightGBM, drop bottom-quartile by mean |SHAP|
16. **Situational edges** — B2B road flag, coaching change flag (52.78% ATS fade rate), popular team bias
17. **L2M referee bias features** — free CSV from `atlhawksfanatic.github.io/L2M/`, per-crew home call bias
18. **OOF stacking ensemble** — LightGBM + XGBoost + LR base, LR meta-learner (+0.5-1% over blending)
19. **Incremental preprocessing** — seasonal chunking to reduce daily rebuild from ~10 min to ~1-2 min

---

## New Data Pipelines to Build

| Source | URL | Cost | Features Enabled | Priority |
|--------|-----|------|-----------------|----------|
| SBRO Historical Odds | sportsbookreviewsonline.com | Free (XLS download) | Opening/closing lines, line movement, CLV | HIGH |
| nba_api Tracking Stats | nba_api already installed | Free | Contested shots, deflections, paint pts | HIGH |
| L2M Referee Reports | atlhawksfanatic.github.io/L2M/ | Free (CSV) | Referee home bias per crew | MEDIUM |
| TheRundown API | therundown.io | Free (20K pts/day) | DK+FanDuel+Pinnacle backup | Contingency |
| nbarapm.com EPM | nbarapm.com | Free (manual CSV) | Player RAPM for active roster | MEDIUM |

---

## What NOT to Do

1. **No neural networks** — NeurIPS 2023 (176 datasets): tree methods beat NNs on tabular data consistently
2. **No SMOTE** — destroys probability calibration, actively harms ATS ROI
3. **No DraftKings/FanDuel direct scraping** — explicit ToS prohibition, civil liability risk
4. **No accuracy optimization for ATS** — use Brier score (University of Bath research is unambiguous)
5. **No shot chart pipeline in daily update** — 3-4 hour runtime; make monthly scheduler if needed
6. **No Basketball Reference scraper** — Cloudflare-blocked on Windows; use nba_api tracking instead
7. **No Action Network subscription yet** — wait until CLV tracking confirms genuine edge first
8. **No GCN (Graph Neural Network)** — research-level complexity for ~0.5-1% lift over LightGBM

---

## Key Packages for Future Sessions

| Package | Purpose | Phase |
|---------|---------|-------|
| `lightgbm` | Primary gradient boosting candidate | 1 |
| `xgboost.XGBRegressor` | Margin regression model | 2 |
| `optuna` | HPO with custom expanding-window CV | 2 |
| `shap.TreeExplainer` | Feature pruning (already in model_explainability.py) | 3 |
| `sklearn.calibration.CalibratedClassifierCV(cv='prefit')` | Calibrate on held-out split | 1 |
| `pandas.Series.ewm(span=N)` | EWMA rolling features | 3 |

---

## Realistic Ceilings

- **Game outcome model**: 68% → 70-71% realistic with Phase 1-2 upgrades (theoretical max ~72%)
- **ATS model**: 53.5% → 55% realistic ceiling (professional syndicates operate at 54-55%)
- Current 68% accuracy and 53.5% ATS are **already at/near the upper bound** for pre-game team-level stats; the research confirms this explicitly
