# Roadmap: NBA Analytics — Model Enhancement & ATS Betting

## Milestones

- [x] **v1.0 — Foundation & ATS Model** — Phases 1-5 (shipped 2026-03-04) — [archive](milestones/v1.0-ROADMAP.md)
- [x] **v2.0 — Data Expansion & Model Intelligence** — Phases 6-11 (complete 2026-03-05)

## Phases

<details>
<summary>✅ v1.0 — Foundation & ATS Model (Phases 1-5) — SHIPPED 2026-03-04</summary>

- [x] Phase 1: Foundation & Outputs (4/4 plans) — completed 2026-03-02
- [x] Phase 2: Modern Era Features (3/3 plans) — completed 2026-03-02
- [x] Phase 3: External Data Layer (4/4 plans) — completed 2026-03-02
- [x] Phase 4: Rest & Schedule Features (2/2 plans) — completed 2026-03-02
- [x] Phase 5: ATS Model (4/4 plans) — completed 2026-03-02

See [v1.0 archive](milestones/v1.0-ROADMAP.md) for full phase details.

</details>

### v2.0 — Data Expansion & Model Intelligence

- [x] **Phase 6: Production Fixes & Injury Data** - Wire calibrated model into all inference paths; add daily injury fetcher — completed 2026-03-04
- [x] **Phase 7: New Data Sources** - TeamDashLineups lineup data + BallDontLie API client — completed 2026-03-04
- [x] **Phase 8: Feature Engineering** - Lineup net rating differential features (+19 cols), matchup CSV 272→291 cols — completed 2026-03-04
- [x] **Phase 9: Model Retraining & ATS Optimization** - Game outcome 68%, ATS 53.5% (+2.2% ROI) with lineup features — completed 2026-03-04
- [x] **Phase 10: Kaggle Injury Data & Real Absence Features** - Built historical absence dataset (build_player_absences()); player_absences.csv from 1.38M game log rows — completed 2026-03-04
- [x] **Phase 11: Ensemble Detector, Metadata & Calibration Fix** - get_strong_value_bets() verified, game_outcome_metadata.json at v2 (68%, v2.0), 70 tests passing — completed 2026-03-05

## Phase Details

### Phase 6: Production Fixes & Injury Data
**Goal**: The production inference pipeline uses calibrated probabilities and real historical injury records are available as a structured dataset
**Depends on**: Nothing (first v2 phase — builds on v1 shipped system)
**Requirements**: FIX-01, FIX-02, DATA-01
**Success Criteria** (what must be TRUE):
  1. Running `fetch_odds.py` loads `game_outcome_model_calibrated.pkl` — no uncalibrated model warnings in output
  2. All inference code paths (fetch_odds, predict scripts) emit calibrated win probabilities, verified by spot-checking a prediction against the calibration curve
  3. Kaggle NBA injury dataset (2016-2025) is downloaded, parsed, and stored as a structured table with player_id, game_date, and absence_flag columns — queryable from project data directory
**Plans**: TBD

### Phase 7: New Data Sources
**Goal**: pbpstats lineup combinations with net ratings and BallDontLie injury/stats data are fetched, stored, and available for feature engineering
**Depends on**: Phase 6
**Requirements**: DATA-02, DATA-03
**Success Criteria** (what must be TRUE):
  1. pbpstats lineup data is fetched for target seasons and stored — lineup combinations with net ratings readable from the data pipeline
  2. BallDontLie API is integrated as a callable data source — injury status and team stats retrievable without error
  3. Both new data sources are accessible in a consistent format that feature engineering scripts can join against existing game logs
**Plans**: TBD

### Phase 8: Feature Engineering
**Goal**: game_matchup_features.csv contains real injury absence features (replacing proxy) and lineup net rating differential columns — ready for model training
**Depends on**: Phase 7
**Requirements**: FEAT-01, FEAT-02, FEAT-03
**Success Criteria** (what must be TRUE):
  1. Injury absence features in the matchup CSV are derived from Kaggle records — not rolling-average proxy; SHAP/explainability report shows injury features sourced from real absence data
  2. `home_lineup_net_rtg`, `away_lineup_net_rtg`, and `diff_lineup_net_rtg` columns are present and non-null in game_matchup_features.csv for games where lineup data exists
  3. game_matchup_features.csv is regenerated with all new columns and passes data integrity validation with no schema errors
**Plans**: TBD

### Phase 9: Model Retraining & ATS Optimization
**Goal**: Game outcome model hits 68%+ holdout accuracy, ATS model exceeds 52.4% vig breakeven, ensemble value-bet detector is operational, and all backtest reports are updated
**Depends on**: Phase 8
**Requirements**: MODEL-01, MODEL-02, MODEL-03, VALID-01, VALID-02, VALID-03
**Success Criteria** (what must be TRUE):
  1. Retrained game outcome model achieves >= 68% accuracy on 2023-24/2024-25 holdout — result recorded in reports/backtest_game_outcome.csv
  2. ATS model achieves >= 52.4% accuracy on holdout test set — result recorded in reports/ats_backtest.csv with v2 label
  3. Ensemble/stacking layer combining game outcome and ATS signals produces a value-bet score — callable and returning ranked bet recommendations
  4. Calibration report for both new models is regenerated and present in reports/calibration/
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation & Outputs | v1.0 | 4/4 | Complete | 2026-03-02 |
| 2. Modern Era Features | v1.0 | 3/3 | Complete | 2026-03-02 |
| 3. External Data Layer | v1.0 | 4/4 | Complete | 2026-03-02 |
| 4. Rest & Schedule Features | v1.0 | 2/2 | Complete | 2026-03-02 |
| 5. ATS Model | v1.0 | 4/4 | Complete | 2026-03-02 |
| 6. Production Fixes & Injury Data | v2.0 | done | Complete | 2026-03-04 |
| 7. New Data Sources | v2.0 | done | Complete | 2026-03-04 |
| 8. Feature Engineering | v2.0 | done | Complete | 2026-03-04 |
| 9. Model Retraining & ATS Optimization | v2.0 | done | Complete | 2026-03-04 |
| 10. Real Absence Features & Pipeline Fix | v2.0 | 1/1 | Complete | 2026-03-04 |
| 11. Ensemble Detector, Metadata & Calibration Fix | v2.0 | 1/1 | Complete | 2026-03-05 |

---
*v2.0 roadmap created: 2026-03-04*
