# NBA Analytics — Working Notes

## Core Insights (loaded by session-kickoff)

- `shift(1)` before ALL rolling features -- data leakage is the #1 silent killer of model validity
- Expanding-window validation only; ALL inference paths load `game_outcome_model_calibrated.pkl`, not base .pkl
- sys.path must include PROJECT_ROOT before any model load (src.* class path must be importable during deserialization)
- update.py step 3: call BOTH `build_team_game_features()` AND `build_matchup_dataset()`; step 6: writes 9 predictions/night to `predictions_history.db`
- ATS model selection uses `min(brier_score_loss)` NOT `max(accuracy)` -- University of Bath: accuracy-opt = -35% ROI, calibration-opt = +35% ROI; CALIBRATION_SEASON="202122" held out from CV
- NBA API game_date: use `format="mixed"` in ALL pd.to_datetime() calls; never use Unicode arrow in print() (cp1252); season codes are integers (`202425`)
- Pinnacle guest API (https://guest.api.arcadia.pinnacle.com/0.1, league 487) -- no auth, no quota; filter matchups to parentId=None + alignment=home/away; ODDS_API_KEY removed
- `database/nba.db` is empty/legacy; pipeline is CSV-based; only `predictions_history.db` is active
- 145 tests passing; run with `.venv/Scripts/python.exe -m pytest tests/ -q`
- injury_proxy.py primary path requires `data/processed/player_absences.csv`; if missing, fallback produces all-zero injury cols silently; regenerate with `python src/data/get_historical_absences.py`

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

[2026-03-05] [injury] INSIGHT: build_team_game_features() silently drops injury proxy columns when build_injury_proxy_features() raises any exception -- bare `except Exception` swallows it, CSV is written without those columns, and fetch_odds.py falls back to proxy win-prob model.
[2026-03-05] [injury] WHY: Fix: merge injury_proxy_features.csv directly into team_game_features.csv then rebuild matchup dataset. Root cause: except block should only catch ImportError, not all exceptions.

[2026-03-06] [injury] INSIGHT: get_historical_absences.py crashed with ValueError on current-season game_dates due to time suffix ("YYYY-MM-DD 00:00:00") -- same format="mixed" issue as other files.
[2026-03-06] [injury] WHY: Added format="mixed" to line 102. After fix: 1,098,538 rows generated (75 seasons), 12.6% absence rate. injury_proxy.py primary path now works -- 60.9% of games have missing_minutes > 0, 9.2% have star_player_out.

### [ats]

[2026-03-06] [ats] INSIGHT: ATS model was selecting on max(accuracy) -- University of Bath research shows calibration-optimized = +34.69% ROI vs accuracy-optimized = -35.17% ROI on NBA data. Switching to min(brier_score_loss) changed selected model from logistic to logistic_l1 and improved test accuracy from 53.5% to 54.9%.
[2026-03-06] [ats] WHY: Brier score measures probability calibration quality, not just correct/wrong. A well-calibrated model assigns higher probability to home-covers bets that actually cover, enabling more reliable Kelly sizing. Accuracy metric ignores probability magnitude entirely.

### [api]

[2026-03-06] [api] INSIGHT: Pinnacle guest API works without auth — GET /leagues/487/matchups + GET /leagues/487/markets/straight; filter matchups to parentId=None with alignment=home/away (neutral=futures); join on matchupId; prices use designation: home/away
[2026-03-06] [api] WHY: The Odds API key was expired (401); Pinnacle guest API confirmed free+keyless 2026-03-06; team names are identical to Odds API full names so ODDS_TEAM_TO_ABB mapping reused unchanged; player props stubbed (Pinnacle props need different endpoint)

### [skills]

[2026-03-05] [skills] INSIGHT: gsd:* skills removed from this project — replace with: `spec-driven-dev` (planning), `nba-feature-dev` (executing), `session-wrap-up` (milestone close)
[2026-03-05] [skills] WHY: gsd skill family deprecated; workflow now uses spec-driven-dev -> tdd-workflow -> code-review-session -> session-wrap-up pipeline
