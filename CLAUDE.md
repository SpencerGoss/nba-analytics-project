# NBA Analytics Project

## What This Is
Python NBA analytics pipeline: data ingestion → feature engineering → game outcome prediction (67.1%, AUC 0.7406) → ATS betting model (54.9%, Brier-optimized) → margin regression model (Ridge, MAE 10.574) → NBAEnsemble (3-model blend) → prediction store → dashboard v3 (9 tabs: Today, Players, Teams, H2H, Standings, Injuries, Rankings, Season History, Betting Tools [Picks/Value Bets/Props/Sharp Money/Performance/Bet Tracker]) + CLV tracking. Dashboard live at GitHub Pages. Optuna HPO complete: gradient_boosting confirmed best. Odds: Pinnacle guest API (free, keyless).

## Stack
Python 3.14+, pandas, scikit-learn, SQLite, Chart.js dashboard. No npm/Node.
Runs on Windows 11. Shell: Git Bash. Use forward slashes in paths. Activate venv: `source .venv/Scripts/activate` (Git Bash) or `.venv\Scripts\Activate.ps1` (PowerShell).

## Commands
- `.venv/Scripts/python.exe -m pytest tests/ -q` — run tests (598+ passing, current baseline as of 2026-03-08)
- `python update.py` — daily pipeline
- `python backfill.py` — full historical rebuild
- `python -m http.server 8080 --directory dashboard` — serve dashboard

## Key Paths
- `src/data/` — fetchers | `src/features/` — engineering | `src/models/` — models+calibration | `src/processing/` — preprocessing | `src/validation/` — integrity
- `src/models/value_bet_detector.py` — kelly_fraction | `src/models/clv_tracker.py` — CLV | `src/models/ensemble.py` — NBAEnsemble (win=0.5/ats=0.3/margin=0.2)
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
- ATS model selection uses `min(brier_score_loss)` NOT accuracy — never revert to accuracy; CALIBRATION_SEASON="202122" is permanently held out from CV
- Never use Unicode → in print() — Windows cp1252 raises UnicodeEncodeError; use -> instead
- Any feature col with `_roll` in name is auto-captured by `roll_cols` in build_matchup_dataset(); never also add to `context_cols` -- duplicates cause ValueError
- CLV formula: `clv = opening_spread - closing_spread` (positive = better line than closing); do NOT invert; `closing_spread` is NULL in DB until game closes — always guard with `pd.isna()` before `float()` cast
- Dashboard: always use `.venv/Scripts/python.exe` for ML scripts (lacks optuna/lightgbm); HPO flag is `--trials N`; `calibration.py`/`ats_model.py` need sys.path set — use python -c workaround
- After any debug session or non-obvious fix → invoke `working-memory` skill to extract insight
- Dashboard JS: data-dependent UI must be in Promise.all loader (not tab-click handlers); all dynamic DOM writes use `_setHtml(el,html)` — security hook blocks Edit when replacement contains "innerHTML"
- Always guard `g.ats||''` before `.includes()`/`.startsWith()` in gameCard — ats field can be absent; CSP `frame-ancestors` is ignored in meta tags (HTTP header only)
- `game_lines.csv` at `data/odds/` (NOT data/processed/); fetch_odds.py writes columns `date`+`home_moneyline`; build_value_bets.py converts to `game_date`+`home_market_prob`
- `player_stats.csv` stores season TOTALS — divide by `gp` before projections; Step 7 calls all 24 builders; deploy: `python update.py` then `git add dashboard/data/ && git push`

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

## See Also
`AI_INDEX.md` (task routing) | `ARCHITECTURE.md` | `AGENTS.md` | `CONTEXT.md` | `DECISIONS.md` | `HANDOFF.md` | `WORKING_NOTES.md` | `PROJECT_JOURNAL.md` | Global rules: `~/.claude/rules/common/`
