# NBA Analytics — Working Notes

## Core Insights (loaded by session-kickoff)

- `shift(1)` before ALL rolling features — data leakage is the #1 silent killer of model validity
- Expanding-window validation only — never train on future data
- sys.path must include PROJECT_ROOT before any model load that references `src.*` modules
- ALL inference paths must load calibrated model — check playoff_odds_model, model_explainability, fetch_odds (not just fetch_odds)
- update.py + backfill.py must call `build_team_game_features()` after preprocessing — otherwise features are stale when fetch_odds runs
- injury_proxy.py now has 3-tier fallback: real absences CSV → legacy CSV → rolling proxy; player_absences.csv is now wired in
- `get_strong_value_bets()` now loads ats_model.pkl and applies ATS filter (0.53 threshold); ats_prob=None when model missing (not 0.5)
- Season codes are integers (e.g., `202425`), not strings like "2024-25"
- 115 tests passing as of 2026-03-05 audit (was 59 baseline)
- numpy_gbm.py deleted — was 700 lines of dead code, never imported anywhere

## Domain Notes

### [model]

[2026-03-05] [model] INSIGHT: calibrated model not loading in fetch_odds.py was due to PROJECT_ROOT missing from sys.path — deserializer could not find src.models.calibration._CalibratedWrapper
[2026-03-05] [model] WHY: Python serialization requires the full dotted class path to be importable; adding sys.path.insert guard after PROJECT_ROOT resolution fixed it

[2026-03-05] [model] INSIGHT: ATS jumped from 51.4% to 53.5% (+2.2% holdout ROI) when lineup net rating features were added
[2026-03-05] [model] WHY: lineup features were present in game_matchup_features.csv (291 cols) but were not being passed to ATS training; wiring them in was the fix

### [pipeline]

[2026-03-05] [pipeline] INSIGHT: update.py and backfill.py were missing feature rebuild step — preprocessing alone leaves game_matchup_features.csv stale; fixed by adding build_team_game_features() call after preprocessing in both scripts
[2026-03-05] [pipeline] WHY: fetch_odds.py loads features from data/features/game_matchup_features.csv; if that CSV is stale, all predictions use outdated team stats

[2026-03-05] [pipeline] INSIGHT: Basketball Reference scraper is blocked by Cloudflare in Windows dev environment
[2026-03-05] [pipeline] WHY: BallDontLie stub also blocked without BALLDONTLIE_API_KEY; nba_api remains the primary data source

[2026-03-05] [pipeline] INSIGHT: player_absences.csv generated (1.1M rows) from Kaggle data; was_absent rate ~12.6% is correct — the 40-65% expectation in early plans was for team-game level aggregates, not player-game rows
[2026-03-05] [pipeline] WHY: plan 10-01 complete; game_id normalized as str from int64 (leading zeros stripped, consistent with player_game_logs.csv)

### [testing]

[2026-03-04] [testing] INSIGHT: ~~59 tests passing as of v2.0 baseline~~ — superseded
[2026-03-05] [testing] INSIGHT: 115 tests passing after full audit session; 4 new test files committed (BallDontLie, injury data, lineup data, lineup features)
[2026-03-05] [testing] WHY: run with `python -m pytest tests/ -q`; test_ats_model_missing_falls_back expects ats_prob=None (not 0.5) when model file missing

### [skills]

[2026-03-05] [skills] INSIGHT: gsd:* skills removed from this project — replace with: `spec-driven-dev` (planning), `nba-feature-dev` (executing), `session-wrap-up` (milestone close)
[2026-03-05] [skills] WHY: gsd skill family deprecated; workflow now uses spec-driven-dev -> tdd-workflow -> code-review-session -> session-wrap-up pipeline
