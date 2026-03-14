# NBA Analytics Project

## What This Is
Python NBA analytics pipeline: data ingestion → feature engineering (Elo ratings, EWMA, streaks, cross-matchup interactions) → game outcome prediction (67.7%, AUC 0.7427) → ATS betting model (54.9%, Brier-optimized) → margin regression model (Ridge, MAE 10.52) → NBAEnsemble (3-model blend, confidence-dependent weights, ATS=0) → prediction store → dashboard v3 (9 tabs: Today, Players, Teams, H2H, Standings, Injuries, Rankings, Season History, Betting Tools [Picks/Value Bets/Props/Sharp Money/Performance/Bet Tracker]) + CLV tracking + SQL Server warehouse. Dashboard live at GitHub Pages. Optuna HPO complete: gradient_boosting confirmed best. Odds: Pinnacle guest API (free, keyless).

## Stack
Python 3.14+, pandas, scikit-learn, SQLite, SQL Server 2019 (SSMS), Chart.js dashboard, Node.js (dashboard optimizer).
Runs on Windows 11. Shell: Git Bash. Use forward slashes in paths. Activate venv: `source .venv/Scripts/activate` (Git Bash) or `.venv\Scripts\Activate.ps1` (PowerShell).

## Commands
- `.venv/Scripts/python.exe -m pytest tests/ -q` — run tests (1675 passing, current baseline as of 2026-03-13)
- `python update.py` — daily pipeline
- `python backfill.py` — full historical rebuild
- `python -m http.server 8080 --directory dashboard` — serve dashboard

## Key Paths
- `src/data/` — fetchers | `src/features/` — engineering | `src/models/` — models+calibration | `src/processing/` — preprocessing | `src/validation/` — integrity
- `src/features/elo.py` — Elo rating system (K=20 standard + K=40 fast Elo, MOV multiplier, season regression, elo_momentum feature)
- `src/models/value_bet_detector.py` — kelly_fraction | `src/models/clv_tracker.py` — CLV | `src/models/ensemble.py` — NBAEnsemble (dynamic weights: high-conf 0.75/0/0.25, default 0.65/0/0.35, uncertain 0.55/0/0.45)
- `src/models/betting_router.py` — BettingRouter: confidence tiers (Best Bet/Solid Pick/Lean/Skip), market-specific routing (moneyline/spread/props)
- `src/models/odds_utils.py` — centralized devigging, American-to-decimal, EV calc
- `src/models/player_minutes_model.py` — Stage 1 prop model: GBM Huber loss, blowout adjustment | `src/models/player_stat_models.py` — Stage 2: per-stat GBMs (PTS/REB/AST/3PM) + quantile regression
- `src/models/conformal.py` — conformal prediction intervals (90% coverage) | `src/features/player_features.py` — build_player_prop_features()
- `scripts/sync_to_sqlserver.py` — syncs CSV+SQLite data to SQL Server `nba_analytics` DB | `scripts/optimize_dashboard.js` — Node.js dashboard minifier
- `scripts/build_value_bets.py` — reads `data/odds/game_lines.csv` (NOT data/processed/); columns: game_date, home_market_prob
- `dashboard/data/*.json` — COMMITTED to git (GitHub Pages has no build step); push after update.py
- `data/raw/`, `data/processed/`, `data/features/` — pipeline stages | `models/artifacts/` — PKLs (gitignored)
- `.planning/codebase/CONCERNS.md` — known bugs | `DECISIONS.md` — architectural decisions (consult before re-opening settled questions)

## Hard Rules (never violate)
- Never commit `.env`
- `shift(1)` before ALL rolling features — no data leakage
- Expanding-window validation only — never train on future data
- Never modify `data/raw/` files — source of truth
- After retraining any model → run `src/models/calibration.py` immediately; `fetch_odds.py` must always load `game_outcome_model_calibrated.pkl`
- NBA API (nba_api): throttle at 1 req/sec minimum; never loop without sleep; shot chart fetch is 3-4h — never run in daily pipeline
- `pd.to_datetime()` on game_date must use `format="mixed"` — NBA API sends "YYYY-MM-DD 00:00:00" for current season, plain dates for history; `player_game_logs.csv` uses `season_id=22025` for 202526 (all other CSVs use `season=202526`)
- `update.py` step 3: call both `build_team_game_features()` AND `build_matchup_dataset()`; step 6: `generate_today_predictions()` writes to predictions_history.db
- If injury cols missing from matchup CSV — `player_absences.csv` may be missing; run `get_historical_absences.py` first, then rebuild injury_proxy + matchup
- ATS model selection uses `min(brier_score_loss)` NOT accuracy — never revert to accuracy; CALIBRATION_SEASON=202122 (int) is permanently held out from CV
- Season codes are 6-digit INTEGERS — ALWAYS use `.astype(int)` for comparisons, NEVER `.astype(str)`; string comparison is lexicographic (bug found in 10+ files)
- Never use `fillna(0)` in prediction/inference paths — sklearn Pipeline has imputer as first step that handles NaN with mean strategy matching training; fillna(0) causes train/inference skew
- Never use Unicode → in print() — Windows cp1252 raises UnicodeEncodeError; use -> instead
- Any feature col with `_roll` in name is auto-captured by `roll_cols` in build_matchup_dataset(); never also add to `context_cols` -- duplicates cause ValueError
- CLV formula: `clv = opening_spread - closing_spread` (positive = better line than closing); do NOT invert; `closing_spread` is NULL in DB until game closes — always guard with `pd.isna()` before `float()` cast
- Dashboard: always use `.venv/Scripts/python.exe` for ML scripts (lacks optuna/lightgbm); HPO flag is `--trials N`; `calibration.py`/`ats_model.py` need sys.path set — use python -c workaround
- After any debug session or non-obvious fix → invoke `working-memory` skill to extract insight
- Dashboard JS: data-dependent UI must be in Promise.all loader (not tab-click handlers); all dynamic DOM writes use `_setHtml(el,html)` — for tbody/thead/tfoot it uses createElement('table')+innerHTML (NOT createContextualFragment which strips `<tr>`/`<td>` tags); security hook blocks Edit when replacement contains "innerHTML"
- Always guard `g.ats||''` before `.includes()`/`.startsWith()` in gameCard — ats field can be absent; CSP `frame-ancestors` is ignored in meta tags (HTTP header only)
- `game_lines.csv` at `data/odds/` (NOT data/processed/); fetch_odds.py writes columns `date`+`home_moneyline`; build_value_bets.py converts to `game_date`+`home_market_prob`
- `player_stats.csv` stores season TOTALS — divide by `gp` before projections; Step 7 calls all 29 builders; deploy: `python update.py` then `git add dashboard/data/ && git push`
- CLV: `backfill_closing_lines()` runs as Step 3b BEFORE `refresh_odds_data()` — captures yesterday's spreads as closing lines; do NOT reorder these steps
- Dashboard helpers: `_confMeterHtml()`, `_whyThisPickHtml()`, `_factorBadgeHtml()`, `_emptyStateHtml()`, `_sparklineHtml()` — use these for new pick/game UI
- XGBoost is an optional model candidate (requires xgboost package); `_build_fit_params()` handles eval_set for early stopping
- Loading calibrated model: ALL code that loads `game_outcome_model_calibrated.pkl` MUST use `_CalibrationUnpickler` (from `src.models.game_outcome_model`); raw deserialization will fail with `__main__._PlattWrapper` error because calibration.py was run as `__main__`
- Scripts in `scripts/` that import `from src.*` MUST have `sys.path.insert(0, str(PROJECT_ROOT))` BEFORE the import — subprocess.run from update.py does not set PYTHONPATH

## Skill Routing (auto-trigger — no prompting needed)

### Session / Workflow

| Situation | Skill |
|-----------|-------|
| Start of any work session | `session-kickoff` |
| Resume after context compaction | `continue` |
| End of session | `session-wrap-up` → logs `project-journal` + runs `git-workflow` |
| Context at 70%+ | `context-budget-checkpoint` (proactive) / `context-rescue` (emergency) |
| Switching to/from Claude.ai | `handoff-bridge` |
| Starting session with a specific goal | `session-intent-setter` |
| Capture non-obvious insight / debug finding | `working-memory` |

### NBA-Specific

| Situation | Skill |
|-----------|-------|
| Adding features, scripts, or models | `nba-feature-dev` |
| Backfilling or fetching historical data | `nba-backfill` |
| Evaluating or comparing model performance | `nba-model-evaluation` |
| Modifying dashboard/index.html or wiring new JSON | `nba-dashboard-dev` |
| Adding/modifying Plotly charts | `plotly-charts` |
| Querying or extending predictions_history.db | `sqlite-analytics` |
| Picks, value bets, CLV, Kelly, ATS analysis | `nba-betting-analysis` |
| Large data job needs chunking / retry | `batch-data-processing` |
| After pipeline change or model retrain | `regression-test-automation` |
| Setting up pipeline monitoring / alerting | `alert-configuration` |

### Development

| Situation | Skill |
|-----------|-------|
| Before building any feature or fix (plan first) | `spec-driven-dev` |
| Implementing any feature or fix (TDD) | `tdd-workflow` |
| Something is broken — read CONCERNS.md first | `debug-session` |
| Code review after writing code | `code-review-session` |
| Testing the dashboard / UI | `webapp-testing` |
| Making an architectural decision | `decision-log` → writes to `DECISIONS.md` |
| After completing a non-trivial workaround | `skill-extractor` |
| Wrong Claude output / bad response | `prompt-autopsy` |
| Brainstorm / parallel agents / verify complete | `superpowers:brainstorming` / `superpowers:dispatching-parallel-agents` / `superpowers:verification-before-completion` |

### Git / DevOps / Maintenance

| Situation | Skill |
|-----------|-------|
| Any git commit / push / PR / merge conflict | `git-workflow` |
| Before pushing to GitHub (secret scan) | `security-audit` |
| Adding a new API key or secret | `env-config` |
| Adding or upgrading Python packages | `dependency-management` |
| CLAUDE.md feeling stale or bloated | `rule-freshness-audit` |
| Weekly meta-review | `weekly-review` |

## Plugin Integration

| Workflow | Plugin / Skill | Trigger |
|----------|---------------|---------|
| ML model training, retrain, HPO | `machine-learning-ops` + `scientific-skills:scikit-learn` | Model work in `src/models/` |
| Statistical analysis, calibration | `scientific-skills:statsmodels` | Regression, Brier scores, significance tests |
| Data pipeline, backfill, ETL | `data-engineering` | Pipeline changes in `update.py`, `backfill.py`, `src/data/` |
| Dashboard charts, visualizations | `scientific-skills:plotly` + `playground` | Chart work, data exploration prototyping |
| Betting analysis, Kelly, CLV, ATS | `quantitative-trading` | Odds, spreads, value bets, risk metrics |
| Forecasting (game totals, player props) | `scientific-skills:timesfm-forecasting` | Time-series prediction tasks |
| PR review | `code-review` (5 parallel agents) or `pr-review-toolkit` | `/code-review` or PR creation |
| Git commit, push, PR | `commit-commands` | `/commit`, `/push`, `/commit-push-pr` |
| Python type checking | `pyright-lsp` | Auto-activates on `.py` file edits |
| Security monitoring | `security-guidance` | Passive hook — runs automatically |

Existing custom skills (nba-feature-dev, nba-model-evaluation, etc.) remain primary for NBA-specific workflows. Plugins provide supplementary tooling — use both together when applicable.

## See Also
`AI_INDEX.md` (task routing) | `ARCHITECTURE.md` | `AGENTS.md` | `CONTEXT.md` | `DECISIONS.md` | `HANDOFF.md` | `WORKING_NOTES.md` | `PROJECT_JOURNAL.md` | Global rules: `~/.claude/rules/common/`
