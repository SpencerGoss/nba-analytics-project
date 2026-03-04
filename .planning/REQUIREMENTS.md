# Requirements: NBA Analytics v2.0

**Defined:** 2026-03-04
**Milestone:** v2.0 — Data Expansion & Model Intelligence
**Core Value:** Identify games where the model's win probability meaningfully disagrees with Vegas lines — producing profitable against-the-spread picks over a full NBA season

## v2 Requirements

### Data Sources

- [ ] **DATA-01**: Kaggle NBA injury dataset (2016-2025) downloaded and integrated as structured absence records per player per game
- [ ] **DATA-02**: pbpstats lineup tracking data fetched and stored — on-court lineup combinations with net ratings per game
- [ ] **DATA-03**: BallDontLie API integrated as secondary data source for injury status and team stats

### Production Fixes

- [ ] **FIX-01**: Calibrated model (`game_outcome_model_calibrated.pkl`) loaded in `fetch_odds.py` instead of uncalibrated model
- [ ] **FIX-02**: Calibrated probabilities used in all production inference paths

### Feature Engineering

- [ ] **FEAT-01**: Real injury absence features replace proxy — derived from Kaggle injury records (not rolling-average proxy)
- [ ] **FEAT-02**: Lineup net rating differential added as matchup feature — `home_lineup_net_rtg`, `away_lineup_net_rtg`, `diff_lineup_net_rtg`
- [ ] **FEAT-03**: game_matchup_features.csv regenerated with all new features included

### Model Training

- [ ] **MODEL-01**: Game outcome model retrained with new injury + lineup features — target >= 68% holdout accuracy
- [ ] **MODEL-02**: ATS model retrained or tuned above 52.4% vig breakeven on holdout test set
- [ ] **MODEL-03**: Ensemble/stacking layer combining game outcome and ATS signals for value-bet detection

### Backtesting & Validation

- [ ] **VALID-01**: Game outcome backtest rerun with new model — reports/backtest_game_outcome.csv updated
- [ ] **VALID-02**: ATS backtest rerun — reports/ats_backtest.csv updated with v2 model results
- [ ] **VALID-03**: Calibration report regenerated for new models

## v3 Requirements (Deferred)

### Web Platform
- **WEB-01**: Live prediction feed in browser (daily games + probabilities)
- **WEB-02**: Historical accuracy tracking UI
- **WEB-03**: Model transparency / feature importance display

### Player Props
- **PROP-01**: Player prop betting model (pts/reb/ast over/under)
- **PROP-02**: Confidence intervals on player projections

## Out of Scope

| Feature | Reason |
|---------|--------|
| Shot chart ingestion | 3-4 hour runtime, low ROI vs. effort |
| Real-time in-game predictions | Pregame-only scope |
| Mobile app | Future milestone |
| Basketball Reference scraper | Cloudflare-blocked in Windows dev; deprioritized |
| Unit test coverage expansion | Separate quality effort, not model improvement |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FIX-01 | Phase 6 | Pending |
| FIX-02 | Phase 6 | Pending |
| DATA-01 | Phase 6 | Pending |
| DATA-02 | Phase 7 | Pending |
| DATA-03 | Phase 7 | Pending |
| FEAT-01 | Phase 8 | Pending |
| FEAT-02 | Phase 8 | Pending |
| FEAT-03 | Phase 8 | Pending |
| MODEL-01 | Phase 9 | Pending |
| MODEL-02 | Phase 9 | Pending |
| MODEL-03 | Phase 9 | Pending |
| VALID-01 | Phase 9 | Pending |
| VALID-02 | Phase 9 | Pending |
| VALID-03 | Phase 9 | Pending |

**Coverage:**
- v2 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0

---
*Requirements defined: 2026-03-04*
