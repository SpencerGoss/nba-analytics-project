# NBA Analytics Project — Session Journal

Append a dated entry at the start of each session. Keep entries brief — just what was done and what's next.

---

## 2026-03-05 — Bug fix: calibrated model not loading in fetch_odds.py

**Done:**
- Diagnosed fetch_odds.py silently falling back to a feature-based proxy instead of the trained 68% game outcome model
- Root cause: PROJECT_ROOT was not on sys.path, so the deserializer could not find src.models.calibration._CalibratedWrapper
- Fix: Added sys.path.insert guard after PROJECT_ROOT is resolved in scripts/fetch_odds.py
- Verified: now logs "Loaded calibrated game outcome model"; win probs generated for 870 current-season games
- All 59 tests still passing

**Next:**
- Audit other scripts/ scripts for the same missing sys.path guard

---

## 2026-03-04 — Project restructure

**Done:**
- v1.0 milestone complete (all 5 phases, 17/17 plans). See `.planning/STATE.md`.
- Added `.env.example`, `PROJECT_OVERVIEW.md`, `PROJECT_JOURNAL.md` (this file)
- Moved `CLAUDE_CODE_TASKS.md` → `docs/plans/v2-multi-agent-task-prompt.md`
- Collapsed stale duplicate blocks in `.planning/STATE.md`
- Updated `CLAUDE.md` skill routing to match current skills
- Updated `.gitignore` to cover dashboard generated data

**v1.0 status summary:**
- Game outcome: 66.8% accuracy, calibrated model saved
- ATS model: 51.2% (below 52.4% vig breakeven — v2 improvement target)
- Prediction store: operational (WAL-mode SQLite + JSON snapshots)
- Dashboard: built (`dashboard/index.html` + `scripts/export_dashboard_data.py`)
- Tests: 3 test files covering preprocessing, injury proxy, team game features

**Known open issues:**
- ATS accuracy below breakeven — v2 feature engineering needed
- Injury features may still be all-null in production model (verify with `reports/explainability/`)
- Basketball Reference scraper blocked by Cloudflare in Windows dev environment
- Calibrated model not loaded in `fetch_odds.py` (uses uncalibrated probabilities)

**Next session should:**
- Decide v2 focus: ATS model improvement vs injury feature debugging vs test coverage
- Check `.planning/codebase/CONCERNS.md` for full issue list before starting any work

---

## Template for new entries

```
## YYYY-MM-DD — <one-line topic>

**Done:**
- ...

**Next:**
- ...
```
