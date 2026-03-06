# NBA Analytics Project

## What This Is
Python NBA analytics pipeline: data ingestion → feature engineering → game outcome prediction (67.1%, AUC 0.7406) → ATS betting model (54.9%, Brier-optimized, calibration_season=202122) → prediction store → web dashboard (fully live data, dark/light theme, Value Bets tab, CLV card, sortable standings, real accuracy chart) + CLV tracking. v2.3 complete. Odds: Pinnacle guest API (free, keyless). Next: Phase 2 — Optuna HPO on LightGBM/XGBoost, model blending.

## Stack
Python 3.14+, pandas, scikit-learn, SQLite, Chart.js dashboard. No npm/Node.
Runs on Windows 11. Shell: Git Bash. Use forward slashes in paths. Activate venv: `source .venv/Scripts/activate` (Git Bash) or `.venv\Scripts\Activate.ps1` (PowerShell).

## Commands
- `.venv/Scripts/python.exe -m pytest tests/ -q` — run tests (145 passing baseline)
- `python update.py` — daily pipeline
- `python backfill.py` — full historical rebuild
- `python -m http.server 8080 --directory dashboard` — serve dashboard

## Key Paths
- `src/data/` — NBA API fetchers | `src/features/` — feature engineering
- `src/models/` — models + calibration | `src/processing/` — preprocessing
- `src/validation/` — data integrity validation
- `src/models/value_bet_detector.py` — value bet detection (kelly_fraction field)
- `src/models/clv_tracker.py` — CLV tracking (opening/closing line, edge flag)
- `scripts/build_dashboard.py` — rebuilds dashboard/index.html from nba1.html template with real CSV data
- `data/raw/`, `data/processed/`, `data/features/` — pipeline stages
- `models/artifacts/` — trained model PKLs (gitignored)
- `.planning/STATE.md` — phase tracker | `.planning/codebase/CONCERNS.md` — known bugs

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
- CLV formula: `clv = opening_spread - closing_spread` (positive = better line than closing); do NOT invert
- `calibration.py`/`ats_model.py` need sys.path set before running as scripts; use python -c workaround
- After any debug session or non-obvious fix → invoke `working-memory` skill to extract insight
- Dashboard JS: data-dependent UI (CLV card, value bets) must be populated from the Promise.all data loader, not only from tab-click handlers (tab never fires if user doesn't click that tab)

## Skill Routing (auto-trigger — no prompting needed)

### Session / Workflow

| Situation | Skill |
|-----------|-------|
| Start of any work session | `session-kickoff` |
| Resume after context compaction | `continue` |
| End of session | `session-wrap-up` → logs `project-journal` + runs `git-workflow` |
| Context window filling up | `context-rescue` |
| Switching to/from Claude.ai | `handoff-bridge` |
| Capture non-obvious insight / debug finding | `working-memory` |

### NBA-Specific

| Situation | Skill |
|-----------|-------|
| Adding features, scripts, or models | `nba-feature-dev` |
| Backfilling or fetching historical data | `nba-backfill` |
| Evaluating or comparing model performance | `nba-model-evaluation` |

### Development

| Situation | Skill |
|-----------|-------|
| Before building any feature or fix (plan first) | `spec-driven-dev` |
| Implementing any feature or fix (TDD) | `tdd-workflow` |
| Something is broken — read CONCERNS.md first | `debug-session` |
| Code review after writing code | `code-review-session` |
| Requesting a code review (superpowers) | `superpowers:requesting-code-review` |
| Testing the dashboard / UI | `webapp-testing` |

### Planning / Reasoning

| Situation | Skill |
|-----------|-------|
| Brainstorming approach / architecture | `superpowers:brainstorming` |
| Before claiming work is complete | `superpowers:verification-before-completion` |
| Independent tasks that can run in parallel | `superpowers:dispatching-parallel-agents` |

### Git / DevOps

| Situation | Skill |
|-----------|-------|
| Any git commit / push / PR / merge conflict | `git-workflow` |
| Before pushing to GitHub (secret scan) | `security-audit` |
| Adding a new API key or secret | `env-config` |
| Adding or upgrading Python packages | `dependency-management` |

## See Also
- `AI_INDEX.md` — task routing | `ARCHITECTURE.md` — system structure
- `AGENTS.md` — agent roles and all skills | `CONTEXT.md` — constraints and gotchas
- `HANDOFF.md` — session state | `WORKING_NOTES.md` — persistent insights (create if missing)
- `PROJECT_JOURNAL.md` / `PROJECT_OVERVIEW.md` — history and current state
- Global rules: `~/.claude/rules/common/` — code-style, testing, security (auto-loaded globally)
