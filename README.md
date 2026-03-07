# Baseline Analytics

> **[Live Dashboard ->](https://spencergoss.github.io/nba-analytics-project/)**

End-to-end NBA analytics pipeline: historical data ingestion, feature engineering, ML-based game outcome prediction, against-the-spread modeling, and a fully live web dashboard. Free to use during beta.

---

## What It Does

- **Game Outcome Prediction** — 67.1% accuracy, AUC 0.7406 (GradientBoosting, calibrated)
- **ATS Betting Model** — 54.9% accuracy, Brier-score optimized, logistic L1 selected
- **Value Bet Detection** — Identifies spread-covering opportunities where model edge exceeds vig
- **Fractional Kelly Sizing** — Conservative (0.5x) bet sizing based on confirmed edge
- **CLV Tracking** — Logs opening and closing lines; measures if you beat the closing line
- **Live Web Dashboard** — Static Chart.js dashboard auto-deployed to GitHub Pages on every push
- **Data Integrity Validation** — Stage-by-stage validators catch silent failures before they propagate
- **Automated Pipeline** — Daily refresh from NBA API; hourly dashboard JSON rebuild via scheduler

---

## Live Dashboard

**URL:** https://spencergoss.github.io/nba-analytics-project/

![Dashboard](docs/screenshot.png)

**What you'll find:**

- Today's picks with model vs. Pinnacle line comparison
- Value bet cards with plain-English explanations and edge percentages
- Team standings (SU / ATS / O/U), sortable by any column
- Injury report with player absence impact on win probability
- Head-to-head history, team trends, power rankings
- Line movement / sharp money tracker
- Performance tab: rolling accuracy, CLV summary, ROI by market
- Odds converter and Kelly sizing tool (opt-in, disclaimer required)

Dashboard is rebuilt hourly by `scripts/scheduler.py` and served as static JSON files via GitHub Pages — no backend required.

---

## Model Performance

| Model | Accuracy | AUC | Notes |
|-------|----------|-----|-------|
| Game Outcome | 67.1% | 0.7406 | GradientBoosting; calibrated with isotonic regression |
| ATS | 54.9% | 0.5571 | Logistic L1; Brier-score optimized; CALIBRATION_SEASON=202122 held out |

- Expanding-window cross-validation only — no look-ahead bias
- Feature matrix: 296 columns, 68,216 training rows
- No data leakage: all rolling features use `shift(1)` before `.rolling()`
- LightGBM added as a candidate model; GradientBoosting still wins without HPO (Phase 2 target)

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate          # Git Bash on Windows
# .venv\Scripts\Activate.ps1           # PowerShell alternative

# 2. Install dependencies
pip install -r requirements.txt

# 3. First-time: fetch all historical data (~30-60 min)
python backfill.py

# 4. Daily update (fetch + preprocess + features + predict)
python update.py

# 5. Serve dashboard locally
python -m http.server 8080 --directory dashboard
# Open http://localhost:8080
```

---

## Project Structure

```
nba-analytics-project/
├── src/
│   ├── data/             # NBA API fetchers (game logs, odds, injuries, lineups)
│   ├── features/         # Feature engineering — team rolling stats, matchup matrix
│   ├── models/           # ML models, calibration, backtesting, CLV tracker
│   ├── processing/       # Preprocessing pipeline (raw -> processed CSVs)
│   └── validation/       # Stage-by-stage data integrity validators
├── scripts/
│   ├── build_*.py        # 14 dedicated JSON data builders (one per dashboard section)
│   ├── fetch_odds.py     # Pinnacle guest API odds fetcher
│   └── scheduler.py      # Hourly rebuild orchestrator
├── dashboard/
│   └── index.html        # Static Chart.js dashboard (no build step)
├── data/
│   ├── raw/              # Source-of-truth CSV pulls from NBA API (never modified)
│   ├── processed/        # Cleaned, combined CSVs
│   ├── features/         # Model-ready feature tables (296-col matchup matrix)
│   └── odds/             # Sportsbook lines from Pinnacle
├── tests/                # 25 test modules, 456+ passing
├── models/artifacts/     # Trained model PKLs (gitignored)
├── database/
│   └── predictions_history.db  # WAL-mode SQLite prediction store (active)
├── .github/workflows/
│   └── deploy-pages.yml  # GitHub Pages auto-deploy on push to main
├── update.py             # Daily pipeline orchestrator
├── backfill.py           # Full historical data rebuild
└── requirements.txt
```

---

## Data Pipeline

```
nba_api
  -> data/raw/           (one CSV per season per endpoint, never modified)
  -> data/processed/     (cleaned, combined across seasons)
  -> data/features/      (296-col matchup matrix: rolling stats, injury proxy, lineup features)
  -> models/artifacts/   (trained PKLs: game_outcome, ATS, calibrated variants)
  -> database/predictions_history.db  (WAL-mode SQLite)
  -> dashboard/data/*.json            (16 JSON files, rebuilt hourly)
  -> GitHub Pages                     (auto-deployed on push to main)
```

**Data scale:**

- 136,452+ historical game rows across all NBA seasons
- 1,098,538 player-game absence rows (injury proxy)
- 296 features per matchup row
- Odds sourced from Pinnacle guest API (free, no key required)

---

## Dashboard Sections

### Free Data

| Tab | What It Shows |
|-----|---------------|
| Today's Picks | Best bets of the day — model vs. Pinnacle, edge %, situational badges |
| Standings | SU / ATS / O/U records, last-10 form, sortable columns |
| Team Trends | Rolling 10-game stats: ATS%, O/U%, avg margin, home/away splits |
| Head-to-Head | Last 10 meetings, series record, scoring trends |
| Injury Report | Questionable/out players with spread impact estimates |
| Power Rankings | Model-derived rankings vs. media consensus |
| Matchup Analysis | Pace, ORtg/DRtg, 3PT matchup radar charts |

### Betting Tools (free during beta)

| Tab | What It Shows |
|-----|---------------|
| Value Bets | Cards with model% vs Pinnacle%, edge %, plain-English bullets |
| Player Props | Points/Rebounds/Assists model projections vs book lines |
| Totals | Over/under model picks with edge vs market |
| Line Movement | Opening vs. current line, direction interpretation |
| Performance / ROI | Rolling accuracy, CLV summary, ROI by market, calibration table |
| Odds Converter | American / Decimal / Implied% live cross-conversion |
| Bet Tracker | Local-only personal P&L tracker (localStorage, no server) |

> Bet sizing (Kelly) is opt-in only, hidden by default. A disclaimer modal is required before enabling. Terms agreement gate required before accessing any betting section.

---

## Development

### Run Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
# Expected: 456+ passing, 0 failing
```

### Rebuild Dashboard JSON Manually

```bash
# Rebuild all 16 JSON data files
python scripts/build_dashboard.py

# Or run a specific builder
python scripts/build_performance.py
python scripts/build_power_rankings.py
```

### Run the Hourly Scheduler

```bash
python scripts/scheduler.py
# Runs all builders in dependency order; repeats every 60 minutes
```

### Backfill Historical Data

```bash
python backfill.py
# Full historical rebuild — takes 30-60 min; nba_api rate-limited at 1 req/sec
```

### Retrain Models

```bash
# After retraining, always run calibration immediately after
python src/models/retrain_all.py
python src/models/calibration.py
```

---

## Roadmap

- [x] Phase 1 — LightGBM candidate, Pythagorean win%, Fractional Kelly, CLV tracking (v2.2)
- [x] Dashboard v2.3 — all ~35 sections wired to real data, dark/light theme, Value Bets tab
- [ ] Phase 2 — Optuna HPO on LightGBM/XGBoost; model blending
- [ ] Dashboard v3 — Linear/Coinbase aesthetic full redesign; 16 JSON data files; hourly scheduler
- [ ] Margin regression model (predict point differential)
- [ ] Historical player comparison tool (80 seasons of data)
- [ ] Player props model (currently a stub)
- [ ] SBRO historical odds for CLV backtesting
- [ ] Premium tier (future — subscriber accounts, advanced features)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12+ |
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn, LightGBM |
| Hyperparameter tuning | Optuna (Phase 2) |
| Database | SQLite (WAL mode) |
| NBA data | nba_api |
| Odds data | Pinnacle guest API (free, no key) |
| Dashboard | Chart.js, plain HTML/CSS/JS (no npm, no Node) |
| Deployment | GitHub Pages (auto-deploy via GitHub Actions) |
| Tests | pytest (456+ passing) |
| Platform | Windows 11, Git Bash |

---

## Environment Setup

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required variables (see `.env.example` for full list):

- `BALLDONTLIE_API_KEY` — free tier at balldontlie.io

The Pinnacle odds API requires no key.

---

## License

MIT
