# NBA Analytics — Working Notes

## Core Insights (loaded by session-kickoff)

- `shift(1)` before ALL rolling features — data leakage is the #1 silent killer of model validity
- Expanding-window validation only — never train on future data
- sys.path must include PROJECT_ROOT before any model load referencing `src.*` modules (calibrated model deserializer needs dotted class path importable)
- ALL inference paths must load calibrated model — game_outcome_model_calibrated.pkl, not base .pkl
- update.py step 3: call BOTH `build_team_game_features()` AND `build_matchup_dataset()`; step 6: `generate_today_predictions()` via ScoreboardV2 writes to predictions_history.db
- Injury proxy join in `build_team_game_features()` uses bare `except Exception` — silent failure leaves matchup CSV without injury cols; fix: merge injury_proxy_features.csv directly and rebuild matchup
- NBA API game_date is "YYYY-MM-DD 00:00:00" for current season — use `format="mixed"` in ALL pd.to_datetime() for game_date cols
- Never use Unicode → in print() on Windows — cp1252 raises UnicodeEncodeError; use `->` instead
- Season codes are integers (e.g., `202425`), not strings like "2024-25"
- 145 tests passing (2026-03-05); run with `.venv/Scripts/python.exe -m pytest tests/ -q`

## Domain Notes

### [model]

[2026-03-05] [model] INSIGHT: calibrated model not loading in fetch_odds.py was due to PROJECT_ROOT missing from sys.path — deserializer could not find src.models.calibration._CalibratedWrapper
[2026-03-05] [model] WHY: Python serialization requires the full dotted class path to be importable; adding sys.path.insert guard after PROJECT_ROOT resolution fixed it

[2026-03-05] [model] INSIGHT: ATS jumped from 51.4% to 53.5% (+2.2% holdout ROI) when lineup net rating features were added
[2026-03-05] [model] WHY: lineup features were present in game_matchup_features.csv (291 cols) but were not being passed to ATS training; wiring them in was the fix

### [pipeline]

[2026-03-05] [pipeline] INSIGHT: update.py and backfill.py were missing feature rebuild step — preprocessing alone leaves game_matchup_features.csv stale; fixed by adding build_team_game_features() call after preprocessing in both scripts
[2026-03-05] [pipeline] WHY: fetch_odds.py loads features from data/features/game_matchup_features.csv; if that CSV is stale, all predictions use outdated team stats

[2026-03-05] [pipeline] INSIGHT: build_matchup_dataset() was also missing from update.py — only build_team_game_features() was called, leaving game_matchup_features.csv (used by fetch_odds.py) perpetually stale after every daily run
[2026-03-05] [pipeline] WHY: build_matchup_dataset() is only in the __main__ block of team_game_features.py, not called by build_team_game_features(); update.py must import and call both explicitly

[2026-03-05] [pipeline] INSIGHT: Basketball Reference scraper is blocked by Cloudflare in Windows dev environment
[2026-03-05] [pipeline] WHY: BallDontLie stub also blocked without BALLDONTLIE_API_KEY; nba_api remains the primary data source

[2026-03-05] [pipeline] INSIGHT: player_absences.csv generated (1.1M rows) from Kaggle data; was_absent rate ~12.6% is correct — the 40-65% expectation in early plans was for team-game level aggregates, not player-game rows
[2026-03-05] [pipeline] WHY: plan 10-01 complete; game_id normalized as str from int64 (leading zeros stripped, consistent with player_game_logs.csv)

### [testing]

[2026-03-04] [testing] INSIGHT: ~~59 tests passing as of v2.0 baseline~~ — superseded
[2026-03-05] [testing] INSIGHT: 145 tests passing (2026-03-05); 4 test files added in audit session (BallDontLie, injury data, lineup data, lineup features)
[2026-03-05] [testing] WHY: run with `python -m pytest tests/ -q`; test_ats_model_missing_falls_back expects ats_prob=None (not 0.5) when model file missing

### [features]

[2026-03-05] [features] INSIGHT: NBA API returns game_date as "YYYY-MM-DD 00:00:00" for current season but "YYYY-MM-DD" for historical seasons — mixed formats in one CSV column
[2026-03-05] [features] WHY: pandas infers format="%Y-%m-%d" from historical rows first, then raises ValueError on the time suffix; fix is format="mixed" in all pd.to_datetime() calls in team_game_features.py (lines 340, 849) and injury_proxy.py (lines 161, 327, 362, 673)

[2026-03-05] [features] INSIGHT: Unicode arrows (→) in Python print() raise UnicodeEncodeError on Windows cp1252 terminals — use ASCII -> instead
[2026-03-05] [features] WHY: Windows default console encoding is cp1252 which cannot encode \u2192; affects any print() with non-ASCII chars regardless of source file encoding

### [injury]

[2026-03-05] [injury] INSIGHT: build_team_game_features() silently drops injury proxy columns when build_injury_proxy_features() raises any exception — bare `except Exception` swallows it, CSV is written without those columns, and fetch_odds.py falls back to proxy win-prob model.
[2026-03-05] [injury] WHY: Fix: merge injury_proxy_features.csv directly into team_game_features.csv then rebuild matchup dataset. Root cause: except block should only catch ImportError, not all exceptions.

### [skills]

[2026-03-05] [skills] INSIGHT: gsd:* skills removed from this project — replace with: `spec-driven-dev` (planning), `nba-feature-dev` (executing), `session-wrap-up` (milestone close)
[2026-03-05] [skills] WHY: gsd skill family deprecated; workflow now uses spec-driven-dev -> tdd-workflow -> code-review-session -> session-wrap-up pipeline
