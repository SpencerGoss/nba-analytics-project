<!-- TL;DR: Maps tasks to files. Read this before reading anything else. -->
# AI Index — NBA Analytics Project

## How to Use
Find your task below. Read only the listed files. Skip everything else.

## Tasks

### Start a session
Read: `.planning/STATE.md` → `PROJECT_JOURNAL.md` → `CLAUDE.md`
Or: type `continue` to trigger session-kickoff automatically.

### Add/fix a data fetcher
Read: `ARCHITECTURE.md` → `src/data/<relevant>.py` → `.claude/rules/nba-domain.md`
Hard rule: never modify `data/raw/` files.

### Add/fix a feature
Read: `ARCHITECTURE.md` → `src/features/<relevant>.py` → `.claude/rules/nba-domain.md`
Hard rule: all rolling features must use `shift(1)` — no data leakage.

### Retrain a model
Read: `ARCHITECTURE.md` → `src/models/` → `.planning/STATE.md`
Hard rule: expanding-window validation only. After retraining → run calibration.py.

### Fix a bug
Read: `.planning/codebase/CONCERNS.md` → `CONTEXT.md` → failing file
Then: invoke `debug-session` skill

### Review code
Invoke: `code-review-session` skill after writing code
PR review: `superpowers:requesting-code-review`

### Plan a new phase
Invoke: `spec-driven-dev` (spec first) → `.planning/STATE.md` for position

### Implement a feature or fix (TDD)
Invoke: `superpowers:test-driven-development` → `tdd-workflow`

### Before claiming work is done
Invoke: `superpowers:verification-before-completion`

### Write a multi-step plan
Invoke: `superpowers:writing-plans`

### Brainstorm approach or architecture
Invoke: `superpowers:brainstorming`

### Check project progress / what's next
Read: `.planning/STATE.md` → `PROJECT_JOURNAL.md`

### Commit changes
Invoke: `git-workflow` skill → `/wrap-up` command

### Security check before push
Invoke: `security-audit` skill

### Update journal / log progress
Invoke: `project-journal` skill → `/wrap-up` command → `session-wrap-up`

### Something is broken
Read: `.planning/codebase/CONCERNS.md` first
Then: invoke `debug-session` skill

### Pipeline is slow or using too much memory
Invoke: `performance-tuning` skill

### Add or upgrade packages
Invoke: `dependency-management` skill

### Context window filling up
Invoke: `context-budget-checkpoint` skill (proactive at 70%) or `context-rescue` skill (emergency)

### Add a new data provider or API client
Invoke: `nba-backfill` skill (data ingestion) + `env-config` skill (secrets)

### Work with player absence / injury features
Read: `src/data/get_historical_absences.py` (generates player_absences.csv from game logs — Phase 10)
Read: `src/data/get_injury_data.py` (injury report normalization from PDF and nba_api)
Read: `src/data/get_balldontlie.py` (BallDontLie API client — injuries, stats, teams)
Read: `src/data/get_lineup_data.py` (on-court lineup data from nba_py)

### Detect or analyze value bets
Read: `src/models/value_bet_detector.py` (value bet detection using calibrated model + edge filtering; `get_strong_value_bets()` is main entry point)

### Explain model predictions or feature importance
Read: `src/models/model_explainability.py` (SHAP-based feature importance analysis)

### Validate data integrity
Read: `src/validation/data_integrity.py` (data validation utilities — v2 addition)

### Isolate feature work from main branch
Use a separate git branch via `git-workflow` skill

## Plugin Routing

| Task | Plugin | Notes |
|------|--------|-------|
| Retrain/evaluate ML models | `machine-learning-ops` + `scientific-skills:scikit-learn` | HPO, experiment tracking, pipeline management |
| Statistical calibration, Brier scores | `scientific-skills:statsmodels` | Regression, significance testing |
| Dashboard chart creation/updates | `scientific-skills:plotly` + `playground` | Interactive prototyping |
| Betting analysis, Kelly criterion, CLV | `quantitative-trading` | Risk metrics, portfolio optimization |
| Time-series forecasting | `scientific-skills:timesfm-forecasting` | Player props, game totals |
| Data backfill, ETL pipeline work | `data-engineering` | Pairs with batch-data-processing skill |
| PR or pre-commit code review | `code-review` or `pr-review-toolkit` | 5-6 parallel review agents |
| Git operations | `commit-commands` | /commit, /push, /commit-push-pr |
