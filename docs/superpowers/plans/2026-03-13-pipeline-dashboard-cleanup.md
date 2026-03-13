# Pipeline, Dashboard & Cleanup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automate the pipeline via Windows Task Scheduler (3x daily), improve dashboard performance and betting UX, clean up dead code and infrastructure.

**Architecture:** Pipeline automation with health reporting; dashboard tiered loading + Promise.allSettled for resilience; betting tools with strict confidence tiers and Kelly opt-in; dead code deletion; config consolidation; CI/CD cleanup.

**Tech Stack:** Python 3.14, Windows Task Scheduler, HTML/CSS/JS (Chart.js), GitHub Actions, pytest

**Spec:** `docs/superpowers/specs/2026-03-13-project-overhaul-design.md` (Phases 5-7)

**Depends on:** Plans A-C should be complete (pipeline automation deploys the improved models; dashboard shows new confidence tiers and prop predictions).

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `scripts/pipeline_runner.py` | Unified pipeline runner with health reporting, resume, logging |
| `src/config.py` | Shared constants (seasons, teams, conference mappings) |
| `scripts/setup_scheduler.ps1` | PowerShell script to configure Windows Task Scheduler |
| `tests/test_config.py` | Config module tests |

### Modified Files
| File | Changes |
|------|---------|
| `dashboard/index.html` | Tiered loading, Promise.allSettled, confidence tiers, performance fixes |
| `update.py` | Single builder registry, health reporting |
| `.github/workflows/daily_deploy.yml` | Repurpose as CI-only (tests + pyright on PR) |

### Deleted Files
| File | Reason |
|------|--------|
| `scripts/build_dashboard.py` | Dead code (never called) |
| `scripts/export_dashboard_data.py` | Dead code |
| `scripts/generate_sample_dashboard_data.py` | Dead code |
| `scripts/package_for_colab.py` | Dead code |
| `scripts/build_player_props.py` | Deprecated duplicate of build_props.py |
| `src/data/get_player_bio_stats.py` | Never called in update.py |
| `tests/test_build_player_props.py` | Tests for deleted script |

---

## Chunk 1: Pipeline Automation

### Task 1: Create Unified Pipeline Runner

Replace the fragmented update.py + scheduler.py + GitHub Actions with a single runner.

**Files:**
- Create: `scripts/pipeline_runner.py`
- Test: manual (Task Scheduler integration)

- [ ] **Step 1: Implement pipeline_runner.py**

Create `scripts/pipeline_runner.py` with:
- `PipelineRunner` class with modes: `full` (4AM), `injuries_odds` (11:30AM), `pretip` (6PM)
- Single builder registry (source of truth for all 29 builders)
- 7-phase dependency graph with parallel execution within phases
- Per-step resume capability (tracks completed steps in `logs/pipeline_state.json`)
- Structured JSON logging to `logs/pipeline_YYYYMMDD_HHMMSS.log`
- Health report output to `dashboard/data/pipeline_report.json`
- Auto git add + push dashboard/data/ after successful run

- [ ] **Step 2: Test with --dry-run**

```bash
.venv/Scripts/python.exe scripts/pipeline_runner.py --mode full --dry-run
```
Expected: Prints execution plan without running builders

- [ ] **Step 3: Test with single builder**

```bash
.venv/Scripts/python.exe scripts/pipeline_runner.py --builder build_standings
```

- [ ] **Step 4: Commit**

```bash
git add scripts/pipeline_runner.py
git commit -m "feat: unified pipeline runner with health reporting and resume

Replaces fragmented update.py/scheduler.py/GitHub Actions.
Single builder registry, 7-phase dependency graph, structured logging.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Create Config Module

Consolidate hardcoded constants shared across files.

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
def test_current_season_is_integer():
    from src.config import get_current_season
    season = get_current_season()
    assert isinstance(season, int)
    assert season > 200000  # e.g., 202526

def test_conference_teams():
    from src.config import EAST_TEAMS, WEST_TEAMS
    assert len(EAST_TEAMS) == 15
    assert len(WEST_TEAMS) == 15
    assert "BOS" in EAST_TEAMS
    assert "LAL" in WEST_TEAMS

def test_modern_era_start():
    from src.config import MODERN_ERA_START
    assert isinstance(MODERN_ERA_START, int)
    assert MODERN_ERA_START == 201314
```

- [ ] **Step 2: Implement src/config.py**

```python
# src/config.py
"""Shared constants for the NBA analytics project."""
from datetime import date

MODERN_ERA_START = 201314
TEST_SEASONS = [202324, 202425]
EXCLUDED_SEASONS = []
CALIBRATION_SEASON = 202122

EAST_TEAMS = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND",
              "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"]
WEST_TEAMS = ["DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN",
              "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"]
ALL_TEAMS = sorted(EAST_TEAMS + WEST_TEAMS)

def get_current_season() -> int:
    """Derive current season from date. Oct+ = new season."""
    today = date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    end_year = start_year + 1
    return int(f"{start_year}{str(end_year)[-2:]}")
```

- [ ] **Step 3: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_config.py -v
git add src/config.py tests/test_config.py
git commit -m "feat: shared config module for seasons, teams, constants

Replaces hardcoded values across build_standings, build_playoff_odds,
game_outcome_model, margin_model, etc.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Set Up Windows Task Scheduler

**Files:**
- Create: `scripts/setup_scheduler.ps1`

- [ ] **Step 1: Create PowerShell setup script**

```powershell
# scripts/setup_scheduler.ps1
# Run as Administrator to set up NBA pipeline scheduled tasks

$projectDir = "C:\Users\Spencer\OneDrive\Desktop\GIT\nba-analytics-project"
$python = "$projectDir\.venv\Scripts\python.exe"
$runner = "$projectDir\scripts\pipeline_runner.py"

# 4:00 AM ET — Full pipeline (post-game)
$action1 = New-ScheduledTaskAction -Execute $python -Argument "$runner --mode full" -WorkingDirectory $projectDir
$trigger1 = New-ScheduledTaskTrigger -Daily -At "4:00AM"
Register-ScheduledTask -TaskName "NBA-Pipeline-Full" -Action $action1 -Trigger $trigger1 -Description "Full NBA pipeline: fetch, features, predict, deploy"

# 11:30 AM ET — Injuries + odds
$action2 = New-ScheduledTaskAction -Execute $python -Argument "$runner --mode injuries_odds" -WorkingDirectory $projectDir
$trigger2 = New-ScheduledTaskTrigger -Daily -At "11:30AM"
Register-ScheduledTask -TaskName "NBA-Pipeline-Injuries" -Action $action2 -Trigger $trigger2 -Description "Injuries + odds refresh"

# 6:30 PM ET — Pre-tip final
$action3 = New-ScheduledTaskAction -Execute $python -Argument "$runner --mode pretip" -WorkingDirectory $projectDir
$trigger3 = New-ScheduledTaskTrigger -Daily -At "6:30PM"
Register-ScheduledTask -TaskName "NBA-Pipeline-Pretip" -Action $action3 -Trigger $trigger3 -Description "Pre-tip final predictions + deploy"
```

- [ ] **Step 2: Commit (do NOT run yet — requires admin and testing)**

```bash
git add scripts/setup_scheduler.ps1
git commit -m "feat: Windows Task Scheduler setup script for 3x daily pipeline

4AM full, 11:30AM injuries/odds, 6:30PM pre-tip. Requires admin to install.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 2: Dashboard Performance + Betting UX

### Task 4: Dashboard Performance Fixes

**Files:**
- Modify: `dashboard/index.html`

- [ ] **Step 1: Switch Promise.all to Promise.allSettled**

Find the main data loading Promise.all and replace with Promise.allSettled. Add error handling per JSON file — show error state for failed tabs, render the rest.

- [ ] **Step 2: Implement tiered JSON loading**

Only load Today tab data on page open (todays_picks.json, meta.json, game_context.json, live_scores.json). Defer all other JSONs until their tab is activated.

- [ ] **Step 3: Add loading progress indicator**

Show "Loading data... 3/20" counter during initial multi-second load.

- [ ] **Step 4: Add cache-busting**

Append `?v=YYYYMMDD` to all JSON fetch URLs based on meta.json timestamp.

- [ ] **Step 5: Add Plotly.purge() on tab switch**

When leaving a chart tab, call `Plotly.purge()` on the chart container to free memory.

- [ ] **Step 6: Add loading="lazy" to headshot images**

All dynamically generated `<img>` tags for player headshots get `loading="lazy"`.

- [ ] **Step 7: Paginate Players table**

Add pagination at 50 rows per page instead of rendering 450+ at once.

- [ ] **Step 8: Commit**

```bash
git add dashboard/index.html
git commit -m "perf: dashboard — tiered loading, Promise.allSettled, pagination

Cuts initial payload ~60-70%. Prevents total failure on JSON 404.
Players table paginated at 50 rows.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Betting Tools UX Improvements

**Files:**
- Modify: `dashboard/index.html`

- [ ] **Step 1: Add strict confidence tiers to Picks tab**

Replace raw numeric scores with Best Bet / Solid Pick / Lean / Skip labels. Use `_factorBadgeHtml()` helper pattern. Color code: gold for Best Bet, green for Solid Pick, blue for Lean, gray for Skip.

- [ ] **Step 2: Add "Best Bets" summary at top of Betting Tools**

Show top 3 highest-edge plays prominently above the main picks list.

- [ ] **Step 3: Add Kelly opt-in toggle**

Hidden by default. When enabled, shows Kelly fraction and suggested unit size next to each pick. Uses localStorage to persist preference.

- [ ] **Step 4: Add bankroll management to Bet Tracker**

Add starting bankroll field, current bankroll calculation, ROI% display.

- [ ] **Step 5: Add Bet Tracker import/export**

JSON import/export buttons alongside existing CSV export. Persists across devices.

- [ ] **Step 6: Fix Sharp Money tab**

Add disclaimer when CLV data is NULL: "Closing line data not yet available. CLV tracking begins once pipeline runs daily."

- [ ] **Step 7: Pause ticker on hover**

Add CSS: `.game-ticker:hover { animation-play-state: paused; }`

- [ ] **Step 8: Actionable empty states**

Replace "No picks available" with "Predictions generated daily at 10am ET. Check back soon."

- [ ] **Step 9: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: betting tools — confidence tiers, Best Bets summary, Kelly opt-in

Strict confidence labels (Best Bet/Solid Pick/Lean/Skip).
Kelly bet sizing hidden by default, opt-in toggle.
Bankroll management + import/export in Bet Tracker.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 3: Code Cleanup + CI/CD

### Task 6: Delete Dead Code

**Files:**
- Delete: 6 dead scripts + 1 test file

- [ ] **Step 1: Verify scripts are truly unused**

```bash
grep -r "build_dashboard\|export_dashboard_data\|generate_sample_dashboard\|package_for_colab\|build_player_props\|get_player_bio_stats" update.py scripts/ src/ --include="*.py" -l
```
Expected: No references (or only self-references)

- [ ] **Step 2: Delete files**

```bash
rm scripts/build_dashboard.py
rm scripts/export_dashboard_data.py
rm scripts/generate_sample_dashboard_data.py
rm scripts/package_for_colab.py
rm scripts/build_player_props.py
rm src/data/get_player_bio_stats.py
rm tests/test_build_player_props.py
```

- [ ] **Step 3: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q --tb=short`
Expected: All pass (test count decreases by deleted test file count)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: delete 6 dead scripts + 1 stale test file (~2,370 lines)

Removed: build_dashboard, export_dashboard_data, generate_sample_dashboard_data,
package_for_colab, build_player_props (deprecated dup), get_player_bio_stats.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Repurpose CI/CD

**Files:**
- Modify: `.github/workflows/daily_deploy.yml`

- [ ] **Step 1: Repurpose as CI-only**

Replace `daily_deploy.yml` content: remove builders/commit/push steps, keep only:
- Trigger on: push to main, pull_request to main
- Steps: checkout, setup Python, install deps, run pytest, run pyright (if available)
- Remove the schedule trigger (Task Scheduler handles daily runs locally)

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/daily_deploy.yml
git commit -m "ci: repurpose daily_deploy as CI-only (tests on push/PR)

Removed builders/commit/push (Task Scheduler handles locally).
Added PR trigger so tests run on pull requests.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All tests pass

- [ ] **Step 2: Verify dashboard loads locally**

```bash
.venv/Scripts/python.exe -m http.server 8080 --directory dashboard
```
Open http://localhost:8080 and verify:
- Today tab loads first (tiered loading)
- No console errors on tab switch
- Confidence tiers display correctly
- Kelly toggle works

- [ ] **Step 3: Verify dead code is gone**

```bash
ls scripts/build_dashboard.py 2>/dev/null && echo "STILL EXISTS" || echo "Deleted OK"
ls scripts/build_player_props.py 2>/dev/null && echo "STILL EXISTS" || echo "Deleted OK"
```
Expected: Both "Deleted OK"
