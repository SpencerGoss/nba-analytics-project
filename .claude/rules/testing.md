<!-- TL;DR: Test conventions and coverage expectations. Auto-loaded every session. -->

# Testing — NBA Analytics Project

## Framework
pytest. Run with: `pytest -v` or `python -m pytest tests/ -q`
Current baseline: 1621 tests passing (as of 2026-03-13).

## Coverage Target
80%+ for any new module added in v2.0.

## Test Locations
- `tests/test_preprocessing.py` — preprocessing pipeline
- `tests/test_injury_proxy.py` — injury feature generation
- `tests/test_team_game_features.py` — feature matrix builder
- `tests/test_value_bet_detector.py` — value bet detection
- `tests/test_get_balldontlie.py` — BallDontLie API client
- `tests/test_get_injury_data.py` — injury report fetcher
- `tests/test_get_lineup_data.py` — lineup data fetcher
- `tests/test_lineup_features.py` — lineup feature engineering
- `tests/test_build_picks.py` — picks builder (margin, kelly, confidence)
- `tests/test_build_game_context.py` — situational flags, rest/streaks
- `tests/test_build_line_movement.py` — spread movement, sharp classification
- `tests/test_build_player_comparison.py` — player comparison JSON
- `tests/test_power_rankings.py` — power rankings algorithm
- New tests go in `tests/` with `test_<module_name>.py` naming

## What to Test
1. Data leakage: assert that no row N uses data from rows >= N (shift validation)
2. Shape contracts: output DataFrame has expected columns and no nulls in key cols
3. Season format: output uses integer season codes
4. Model artifacts: after retraining, assert calibrated model loads and produces probabilities in [0,1]

## TDD Approach (mandatory for new features)
1. Write test first (RED) — run it, confirm it fails
2. Write minimal implementation (GREEN) — run it, confirm it passes
3. Refactor — keep tests green

## What NOT to Test
- `data/raw/` file contents (source of truth, not generated)
- Third-party API response shapes (mock at the boundary instead)
