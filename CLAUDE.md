# NBA Analytics Project

## What This Is
Python NBA analytics pipeline: data ingestion → feature engineering → game outcome prediction (66.8%) → ATS betting model → prediction store → web dashboard. Currently in v2.0 (Phase 9).

## Stack
Python 3.12+, pandas, scikit-learn, SQLite, Chart.js dashboard. No npm/Node.

## Commands
- `pytest -v` — run tests (59 passing baseline)
- `python update.py` — daily pipeline
- `python backfill.py` — full historical rebuild
- `python -m http.server 8080 --directory dashboard` — serve dashboard

## Key Paths
- `src/data/` — NBA API fetchers | `src/features/` — feature engineering
- `src/models/` — models + calibration | `src/processing/` — preprocessing
- `data/raw/`, `data/processed/`, `data/features/` — pipeline stages
- `models/artifacts/` — trained model PKLs (gitignored)
- `.planning/STATE.md` — phase tracker | `.planning/codebase/CONCERNS.md` — known bugs

## Hard Rules (never violate)
- Never commit `.env`
- `shift(1)` before ALL rolling features — no data leakage
- Expanding-window validation only — never train on future data
- Never modify `data/raw/` files — source of truth
- After retraining any model → run `calibration.py` immediately
- `fetch_odds.py` must load calibrated model first

## Skill Routing (auto-trigger — no prompting needed)

| Situation | Skill / Command |
|-----------|----------------|
| Start of session / "continue" | `session-kickoff` → `/continue` |
| End of session | `/wrap-up` → `project-journal` + `git-workflow` |
| Planning a new phase | `gsd:plan-phase` |
| Executing a phase | `gsd:execute-phase` |
| Something is broken | `debug-session` — read CONCERNS.md first |
| Structured GSD debugging | `gsd:debug` |
| New feature / task | `/new-task` |
| Code review | `/review` → `code-reviewer` agent |
| Any git commit / push | `git-workflow` |
| Before pushing to GitHub | `security-audit` |
| Adding a new API key or data source | `env-config` + `api-integration` |
| Testing the dashboard / UI | `webapp-testing` |
| CLAUDE.md too long / add a rule | `context-file-maintainer` → `/learn-rule` |
| Architecture change / new agent | `vscode-ai-project-scaffolder` |
| Switching to/from Claude.ai | `handoff-bridge` |

## See Also
- `AI_INDEX.md` — task routing | `ARCHITECTURE.md` — system structure
- `AGENTS.md` — agent roles and all skills | `CONTEXT.md` — constraints and gotchas
- `.claude/rules/` — nba-domain, code-style, testing (auto-loaded)
