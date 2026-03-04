# Milestones

## v1.0 — Foundation & ATS Model

**Shipped:** 2026-03-04
**Phases:** 5 | **Plans:** 17 | **All complete**

### What Was Delivered

End-to-end NBA analytics pipeline: data ingestion → feature engineering → game outcome prediction → ATS betting model → prediction store → web dashboard.

### Key Results
- Game outcome model: **66.8% accuracy** on 2023-24/2024-25 holdout
- ATS backtest: 18,233 games, **+1.28% ROI** on value-bet filtered subset (51.2% raw accuracy — below 52.4% vig breakeven, improvement target for v2)
- Prediction store: WAL-mode SQLite + daily JSON snapshots
- 59 unit tests passing
- Static web dashboard (Chart.js, no build step)

### Accomplishments by Phase
1. **Phase 1 — Foundation & Outputs**: Fixed injury proxy join (merge_asof), wired calibrated model into inference path, built prediction store and JSON export, documented 6-stage pipeline
2. **Phase 2 — Modern Era Features**: Added ORtg/DRtg/pace/eFG%/TS% rolling features, Four Factors composite, restricted training to modern era (2013-14+)
3. **Phase 3 — External Data Layer**: Basketball Reference referee scraper (Cloudflare-blocked in Windows dev, wired for cloud use), live NBA injury report fetcher, training/inference path separation
4. **Phase 4 — Rest & Schedule Features**: Haversine travel distance, back-to-back flags, cross-country travel, season_month features for all 30 arenas
5. **Phase 5 — ATS Model**: ATS classifier (logistic regression, expanding-window validation), value-bet detector, 18,233-game backtest harness, Kaggle historical odds integration

### v2 work completed (outside GSD framework, committed to main)
- Shared API client refactoring (18 duplicate retry patterns → single `src/data/api_client.py`)
- Incremental preprocessing (mtime-based, skips unchanged seasons)
- ATS model v2 experiments (GBM, MLP tested; results in `reports/ats_model_v2_experiments.txt`)
- Data integrity validation framework (`src/validation/data_integrity.py`)
- Player backtest boundary fix (now runs through all available seasons)
- Regenerated SHAP/explainability reports

### Known Gaps (carried to v2)
- ATS accuracy 51.2% — below 52.4% vig breakeven
- Basketball Reference scraper Cloudflare-blocked in Windows dev environment
- Calibrated model not wired into `fetch_odds.py` (uses uncalibrated probabilities)
- Injury features may still be all-null in production model (verify with `reports/explainability/`)

### Archives
- `.planning/milestones/v1.0-ROADMAP.md` — full phase details
- `.planning/milestones/v1.0-REQUIREMENTS.md` — all requirements with outcomes
- `.planning/milestones/v1.0-MILESTONE-AUDIT.md` — audit results
