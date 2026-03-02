---
phase: 03-external-data-layer
plan: 04
subsystem: external-data
tags: [injury-proxy, injury-report, fr-4.4, pipeline-docs, code-path-separation]
dependency_graph:
  requires:
    - src/features/injury_proxy.py (from Phase 01 — training path)
    - src/data/external/injury_report.py (from 03-03 — inference path)
  provides:
    - src/features/injury_proxy.py (hardened with TRAINING PATH boundary)
    - src/data/external/injury_report.py (hardened with INFERENCE PATH boundary)
    - docs/PIPELINE.md (external scraper documentation complete)
  affects:
    - FR-4.4 invariant: zero cross-imports verified by grep
    - FR-7.2: PIPELINE.md now documents both new external modules
tech_stack:
  added: []
  patterns:
    - _CODE_PATH module-level constant pattern for code path declaration
    - Module docstring boundary warning pattern (TRAINING vs INFERENCE)
    - Deprecation note pattern: retain function, mark as superseded
key_files:
  created: []
  modified:
    - src/features/injury_proxy.py
    - src/data/external/injury_report.py
    - docs/PIPELINE.md
decisions:
  - "_CODE_PATH string constant chosen over assertion-based guard: docstring + constant makes boundary visible at module load without runtime cost"
  - "get_todays_injury_report() in injury_proxy.py retained (not removed) with deprecation note -- removal would be a breaking change for any callers"
  - "Usage section docstring updated to remove functional-looking import line (grep check requires zero cross-import matches)"
metrics:
  duration_seconds: 189
  completed_date: "2026-03-02"
  tasks_completed: 2
  tasks_total: 2
  files_created: 0
  files_modified: 3
---

# Phase 03 Plan 04: Code Path Separation and Pipeline Documentation Summary

Hardened training/inference injury code path separation with _CODE_PATH constants and docstring boundary guards, verified zero cross-imports (FR-4.4), and documented both new external scrapers in PIPELINE.md (FR-7.2).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Harden training/inference code path separation with explicit guards | c59f750 | src/features/injury_proxy.py, src/data/external/injury_report.py |
| 2 | Update PIPELINE.md with external data scraper documentation | 06b676d | docs/PIPELINE.md |

## What Was Built

### src/features/injury_proxy.py (modified)

**Module docstring update:**
- Added "Code path boundary (FR-4.4)" section with explicit warning: this module is TRAINING PATH ONLY and must never import `src.data.external.injury_report`
- Updated Usage section to reference canonical inference module as a comment (not importable line) to keep grep check clean

**New constant:**
```python
_CODE_PATH = "TRAINING"  # FR-4.4 code path declaration
```

**get_todays_injury_report() marked superseded:**
- Added deprecation note: "Prefer src.data.external.injury_report.get_todays_nba_injury_report() which includes PDF fallback and snapshot saving. This function is retained for backward compatibility."
- Function body unchanged -- retained for backward compatibility

**apply_live_injuries() code path note added:**
- Added "Code path: INFERENCE. This function adjusts features for live prediction. It is NOT called during training."

### src/data/external/injury_report.py (modified)

**New constant:**
```python
_CODE_PATH = "INFERENCE"  # FR-4.4 code path declaration
```

**Verified existing guards (from 03-03):**
- Module docstring already contains "INFERENCE PATH ONLY -- never import or call this module from build_injury_proxy_features()"
- `_assert_recent_date()` date guard already present and raises ValueError for dates >2 days old
- `get_todays_nba_injury_report()` function docstring already carries "INFERENCE PATH ONLY" warning

### docs/PIPELINE.md (modified)

**External Data Scrapers section completely rewritten:**
- Summary table with both modules: `bref_scraper.py` and `injury_report.py`
- Dedicated subsections for each with entry point, usage, rate limits, output paths
- FR-4.4 Training vs Inference table showing side-by-side path comparison
- Updated "NBA API modules" bullets retained for existing `src/data/get_*.py` modules

**Common Operations: referee backfill recipe added:**
```bash
python -c "from src.data.external.bref_scraper import get_referee_crew_assignments; get_referee_crew_assignments('2013-10-01', '2025-06-30')"
# ~3,700 games at 3 sec each = ~3 hours. Run overnight.
```

**Footer updated:** Phase 3 (2026-03-02), added FR-4.4 to requirements covered.

## Verification Results

All plan verification steps passed:

1. `grep -c "TRAINING" src/features/injury_proxy.py` = 3 (boundary comments present)
2. `grep -c "INFERENCE" src/data/external/injury_report.py` = 4 (boundary comments present)
3. Zero cross-imports: `grep "from src.data.external.injury_report" src/features/injury_proxy.py` = 0 matches
4. Zero cross-imports: `grep "from src.features.injury_proxy import build_injury_proxy" src/data/external/injury_report.py` = 0 matches
5. PIPELINE.md documents both external scrapers (8 occurrences of bref_scraper/injury_report)
6. PIPELINE.md has FR-4.4 content (3 occurrences)
7. `get_todays_injury_report()` in injury_proxy.py has deprecation note (1 occurrence)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed functional import appearance from docstring**
- **Found during:** Task 1 verification grep
- **Issue:** The plan updated the Usage section to reference `from src.data.external.injury_report import get_todays_nba_injury_report` as an example. This appeared as a functional import to the grep verification check (which requires 0 matches).
- **Fix:** Changed the Usage example to a comment line (`#   src.data.external.injury_report.get_todays_nba_injury_report()`) rather than a code-formatted import. Preserves documentation value, satisfies grep invariant.
- **Files modified:** src/features/injury_proxy.py
- **Impact:** None -- docstring-only change, no behavioral difference

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| src/features/injury_proxy.py has _CODE_PATH = "TRAINING" | FOUND |
| src/data/external/injury_report.py has _CODE_PATH = "INFERENCE" | FOUND |
| Zero cross-imports (FR-4.4 invariant) | VERIFIED |
| docs/PIPELINE.md documents bref_scraper.py | FOUND |
| docs/PIPELINE.md documents injury_report.py | FOUND |
| docs/PIPELINE.md has FR-4.4 training vs inference table | FOUND |
| commit c59f750 exists | FOUND |
| commit 06b676d exists | FOUND |
