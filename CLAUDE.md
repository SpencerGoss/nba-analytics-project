# NBA Analytics Project

## What This Is
Python NBA analytics pipeline: data ingestion → feature engineering → game outcome prediction (68%) → ATS betting model (53.5%, +2.2% ROI) → prediction store → web dashboard. Currently in v2.0 (Phase 10 — gap closure).

## Stack
Python 3.14+, pandas, scikit-learn, SQLite, Chart.js dashboard. No npm/Node.
Runs on Windows 11. Shell: Git Bash. Use forward slashes in paths. Activate venv: `source .venv/Scripts/activate` (Git Bash) or `.venv\Scripts\Activate.ps1` (PowerShell).

## Commands
- `pytest -v` — run tests (115 passing baseline)
- `python update.py` — daily pipeline
- `python backfill.py` — full historical rebuild
- `python -m http.server 8080 --directory dashboard` — serve dashboard
- Start each session: invoke `session-kickoff` skill before any work

## Key Paths
- `src/data/` — NBA API fetchers | `src/features/` — feature engineering
- `src/models/` — models + calibration | `src/processing/` — preprocessing
- `src/validation/` — data integrity validation
- `src/models/value_bet_detector.py` — value bet detection (recent addition)
- `src/models/model_explainability.py` — SHAP-based feature importance (recent addition)
- `data/raw/`, `data/processed/`, `data/features/` — pipeline stages
- `models/artifacts/` — trained model PKLs (gitignored)
- `.planning/STATE.md` — phase tracker | `.planning/codebase/CONCERNS.md` — known bugs

## Hard Rules (never violate)
- Never commit `.env`
- `shift(1)` before ALL rolling features — no data leakage
- Expanding-window validation only — never train on future data
- Never modify `data/raw/` files — source of truth
- After retraining any model → run `src/models/calibration.py` immediately
- `scripts/fetch_odds.py` must load calibrated model first
- NBA API (nba_api): throttle at 1 req/sec minimum; never loop without sleep; shot chart fetch is 3-4h — never run in daily pipeline
- After any debug session or non-obvious fix → invoke `working-memory` skill to extract insight

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
| Cleaning up code structure | `refactor-session` |
| Pipeline is slow / memory issues | `performance-tuning` |
| Testing the dashboard / UI | `webapp-testing` |
| Dashboard UI design / production polish | `frontend-design` |

### Planning / Reasoning

| Situation | Skill |
|-----------|-------|
| Brainstorming approach / architecture | `superpowers:brainstorming` |
| Writing a multi-step implementation plan | `superpowers:writing-plans` |
| Before implementing (TDD superpowers) | `superpowers:test-driven-development` |
| Before claiming work is complete | `superpowers:verification-before-completion` |
| Independent tasks that can run in parallel | `superpowers:dispatching-parallel-agents` |

### Git / DevOps

| Situation | Skill |
|-----------|-------|
| Any git commit / push / PR / merge conflict | `git-workflow` |
| Before pushing to GitHub (secret scan) | `security-audit` |
| Adding a new API key or secret | `env-config` |
| Adding or upgrading Python packages | `dependency-management` |
| Setting up or updating CI pipelines | `ci-cd-setup` |

### Python

| Situation | Skill |
|-----------|-------|
| Python testing patterns (pytest, fixtures, mocks) | `python-development:python-testing-patterns` |
| Typed settings / env var configuration | `python-development:python-configuration` |
| KISS / SRP design patterns | `python-development:python-design-patterns` |
| Managing packages with uv | `python-development:uv-package-manager` |

### Project Meta

| Situation | Skill |
|-----------|-------|
| CLAUDE.md too long / add a rule | `context-file-maintainer` |
| Architecture change / new project scaffolding | `vscode-ai-project-scaffolder` |

## Working Memory Loop
After debug/TDD sessions, `working-memory` skill extracts insights into `WORKING_NOTES.md`. `session-kickoff` loads Core Insights from that file each session. `session-wrap-up` synthesizes and updates it. Check `WORKING_NOTES.md` before touching any domain you've worked in before.

## See Also
- `AI_INDEX.md` — task routing | `ARCHITECTURE.md` — system structure
- `AGENTS.md` — agent roles and all skills | `CONTEXT.md` — constraints and gotchas
- `HANDOFF.md` — session state | `WORKING_NOTES.md` — persistent insights (create if missing)
- `PROJECT_JOURNAL.md` / `PROJECT_OVERVIEW.md` — history and current state
- Global rules: `~/.claude/rules/common/` — code-style, testing, security (auto-loaded globally)
