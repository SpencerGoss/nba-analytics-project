# NBA Analytics Project — Overview

End-to-end NBA analytics system: data ingestion → feature engineering (Elo ratings, EWMA, streaks, cross-matchup interactions, 370+ features) → game outcome prediction (67.9%, AUC 0.7455) → margin regression (Ridge, MAE 10.52) → NBAEnsemble (confidence-dependent weights) → **player prop predictions** (two-stage: minutes model → per-stat models with quantile regression + conformal intervals) → BettingRouter (confidence tiers: Best Bet/Solid Pick/Lean/Skip) → prediction store → dashboard v3 (9 tabs) + CLV tracking + SQL Server warehouse.

## v3.0 Results (March 2026)
- Game outcome model: **67.9% accuracy, AUC 0.7455** — gradient_boosting, 67 features incl. pace + four factors, auto feature pruning
- Margin model: **Ridge, CV MAE 10.52** — with live Elo refresh (diff_elo = 31.9% feature importance)
- ATS model: **55.0% accuracy** — weight set to 0.0 in ensemble (near-random, disabled)
- Ensemble: confidence-dependent weights (high: 0.75/0/0.25, default: 0.65/0/0.35, uncertain: 0.55/0/0.45)
- Player props: **two-stage pipeline** — minutes model (GBM, Huber loss, MAE 5.03) → per-stat models (PTS/REB/AST/3PM) with quantile regression (p25/p50/p75) and conformal prediction intervals (90% coverage)
- BettingRouter: market-specific outputs with confidence tiers (edge ≥8% = Best Bet, ≥4% = Solid Pick, ≥2% = Lean, <2% or models disagree = Skip)
- Feature matrix: **370+ columns** including Elo (standard + fast + momentum), EWMA, pace, four factors, streaks, cross-matchup interactions
- Odds: Pinnacle guest API (free, keyless) — centralized devigging in odds_utils.py
- Tests: **1800 passing** across 25+ test files
- Pipeline: `python update.py` runs full daily refresh; weekly prop model retrain (Monday)
- Dashboard: live at GitHub Pages; 9 tabs (Today, Players, Teams, H2H, Standings, Injuries, Rankings, Season History, Betting Tools)

## Key Links
- **Architecture & full description:** [`docs/project_overview.md`](docs/project_overview.md)
- **Pipeline stage reference:** [`docs/PIPELINE.md`](docs/PIPELINE.md)
- **Known bugs & tech debt:** [`.planning/codebase/CONCERNS.md`](.planning/codebase/CONCERNS.md)
- **Session log:** [`PROJECT_JOURNAL.md`](PROJECT_JOURNAL.md)
- **Architectural decisions:** [`DECISIONS.md`](DECISIONS.md)

## Quick Commands
```bash
pytest -v                                           # run tests
python update.py                                    # daily data refresh
python backfill.py                                  # full historical rebuild
python -m http.server 8080 --directory dashboard    # serve dashboard
```

## Stack
Python 3.14+, pandas, scikit-learn, SQLite, SQL Server 2019 (SSMS), Chart.js dashboard, Node.js (dashboard optimizer). Windows 11.

## What's Next
- Plan B: Model improvements (SHAP analysis, Huber loss margin model, temperature scaling, ensemble weight optimization, walk-forward backtest)
- Plan D: Pipeline runner, config module, dashboard performance, betting UX, dead code removal
