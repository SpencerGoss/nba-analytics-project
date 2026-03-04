<!-- TL;DR: Test conventions and coverage expectations. Auto-loaded every session. -->

# Testing — NBA Analytics Project

## Framework
pytest. Run with: `pytest -v` or `python -m pytest tests/ -q`
Current baseline: 59 tests passing.

## Coverage Target
80%+ for any new module added in v2.0.

## Test Locations
- `tests/test_preprocessing.py` — preprocessing pipeline
- `tests/test_injury_proxy.py` — injury feature generation
- `tests/test_team_game_features.py` — feature matrix builder
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
