---
phase: 03-external-data-layer
plan: 03
subsystem: external-data
tags: [injury-report, nba-api, pdfplumber, inference-path, fr-4.4]
dependency_graph:
  requires:
    - nba_api>=1.4.0 (leagueinjuryreport endpoint)
    - pdfplumber>=0.10.0 (PDF fallback)
    - requests>=2.28.0 (HTTP client)
  provides:
    - src/data/external/injury_report.py (get_todays_nba_injury_report)
    - data/raw/external/injury_reports/ (CSV snapshot archive)
  affects:
    - src/features/injury_proxy.py (inference path complement — never call from training)
tech_stack:
  added:
    - pdfplumber>=0.10.0 (NBA injury report PDF parsing, in-memory via io.BytesIO)
  patterns:
    - nba_api LeagueInjuryReport as primary, PDF fallback pattern
    - FR-4.4 date guard: ValueError for historical date misuse
    - CSV snapshot archiving with timestamp (YYYYMMDD_HHMM)
key_files:
  created:
    - src/data/external/__init__.py
    - src/data/external/injury_report.py
  modified: []
decisions:
  - "nba_api LeagueInjuryReport as primary source (no URL guessing) per RESEARCH.md Open Question 3"
  - "PDF fallback tries time slots in reverse order (06_00PM -> 05_00PM -> 01_30PM -> 11_00AM -> 06_00AM)"
  - "Includes 'Probable' in RELEVANT_STATUSES (plan spec says Out/Questionable/Probable)"
  - "Module docstring and function docstring both carry FR-4.4 INFERENCE PATH ONLY warning"
  - "date/raw/external/ gitignored by design — .gitkeep created on disk, not committed"
metrics:
  duration_seconds: 127
  completed_date: "2026-03-02"
  tasks_completed: 1
  tasks_total: 1
  files_created: 2
  files_modified: 0
---

# Phase 03 Plan 03: NBA Injury Report Fetcher Summary

NBA pre-game injury report fetcher using nba_api LeagueInjuryReport as primary source with pdfplumber PDF fallback, enforcing FR-4.4 inference-path-only separation via date guard.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Build injury report fetcher with nba_api primary and PDF fallback | 2b57880 | src/data/external/__init__.py, src/data/external/injury_report.py |

## What Was Built

### src/data/external/injury_report.py

Standalone module providing `get_todays_nba_injury_report()` for the inference path only.

**Strategy (waterfall):**
1. Call `nba_api.stats.endpoints.leagueinjuryreport.LeagueInjuryReport` (primary)
2. If API returns empty, fall back to PDF parsing via `pdfplumber`
3. PDF fallback tries time slots in reverse order: `06_00PM`, `05_00PM`, `01_30PM`, `11_00AM`, `06_00AM`
4. Returns empty DataFrame with RuntimeWarning if all sources fail

**FR-4.4 date guard (`_assert_recent_date`):**
Raises `ValueError` if a date more than 2 days in the past is passed explicitly. Prevents this inference-path module from being misused for historical training data.

**Snapshot archiving:**
If `save_snapshot=True` (default), saves a CSV to `data/raw/external/injury_reports/injury_report_YYYYMMDD_HHMM.csv` for each fetch.

**Returns:**
DataFrame with columns: `player_name`, `team`, `player_status`, `reason`, `game_date` — filtered to Out/Questionable/Probable.

### src/data/external/__init__.py

Empty package init (required for `from src.data.external.injury_report import ...` to work per Pitfall 5 in RESEARCH.md).

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written.

**Note on .gitkeep:** The plan spec required `data/raw/external/injury_reports/.gitkeep`. This file was created on disk, but `data/raw/` is in `.gitignore` (by project design — raw data is never committed). The directory exists locally and git-ignored content is acceptable here. The `__init__.py` was also created for the external package (per Pitfall 5 in RESEARCH.md — `src/data/external/` needs this for imports to work).

**Note on `build_injury_proxy_features` in source:** The function name appears in three docstring/error-message strings within `injury_report.py` as guidance to developers. These are not imports or calls — the plan verification check "No import of build_injury_proxy_features" passes (no functional reference).

**Note on pdfplumber in requirements.txt:** Already present (added by 03-01 which ran in parallel). No duplicate added.

## Verification Results

All plan verification steps passed:
1. `python -c "from src.data.external.injury_report import get_todays_nba_injury_report; print('ok')"` - PASSED
2. Module docstring contains "INFERENCE PATH ONLY" and "FR-4.4" - PASSED
3. Date guard raises ValueError for dates > 2 days old - PASSED
4. `data/raw/external/injury_reports/` directory exists on disk - PASSED
5. No functional import of `build_injury_proxy_features` in injury_report.py - PASSED

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| src/data/external/injury_report.py exists | FOUND |
| src/data/external/__init__.py exists | FOUND |
| data/raw/external/injury_reports/ exists | FOUND |
| commit 2b57880 exists | FOUND |
