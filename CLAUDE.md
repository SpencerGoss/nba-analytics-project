# NBA Analytics Project (nba-analytics-project)

## What This Is
Python NBA analytics pipeline: data ingestion → feature engineering → game outcome prediction (66.8%) → ATS betting model → prediction store → web dashboard. Currently in v2.0 development.

## Stack
Python 3.12+, pandas, scikit-learn, SQLite, Chart.js dashboard. No npm/Node.

## Key Paths
- `src/data/` — NBA API fetchers (20+ scripts)
- `src/features/` — feature engineering (rolling windows, injury proxy)
- `src/models/` — game outcome + ATS models
- `src/processing/` — preprocessing pipeline
- `src/validation/` — data integrity validators (v2)
- `database/predictions_history.db` — SQLite predictions store
- `data/raw/`, `data/processed/`, `data/features/` — data pipeline stages
- `.planning/STATE.md` — current progress tracker
- `.planning/codebase/CONCERNS.md` — known bugs and tech debt

## Commands
- `pytest -v` — run tests
- `python update.py` — run daily pipeline
- `python backfill.py` — full historical rebuild
- `python -m http.server 8080 --directory dashboard` — serve dashboard

## Hard Rules
- Never commit `.env` — it contains real API keys
- No data leakage in features — all rolling/shift features must use `shift(1)` so row N only sees rows 0..N-1
- Expanding-window validation only — never train on future data
- Never modify raw data files in `data/raw/` — they are source of truth

## Skill Routing (use these automatically — no prompting needed)

| Situation | Skill to use |
|-----------|-------------|
| Any git commit, branch, or push | `gsd:commit` / git-workflow |
| Adding a new API key or .env var | add to `.env` + `.env.example` |
| Something is broken / unexpected output | `gsd:debug` — read CONCERNS.md first |
| Before pushing to GitHub | `security-audit` skill |
| Start of a new session | `PROJECT_JOURNAL.md` + `.planning/STATE.md` |
| Switching to/from Claude.ai | package `PROJECT_JOURNAL.md` + `STATE.md` as context |
| CLAUDE.md getting long or stale | trim to under 100 lines, move detail to `docs/` |
| Planning a new feature or phase | `gsd:plan-phase` skill |

**Hard rules reminder**: no `.env` commits, `shift(1)` before all rolling features, never modify `data/raw/`.
