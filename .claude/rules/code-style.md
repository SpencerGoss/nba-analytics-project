<!-- TL;DR: Python style conventions. Auto-loaded every session. -->

# Code Style — NBA Analytics Project

## Python Version
Python 3.12+. Use modern syntax (match statements, walrus operator where appropriate).

## Immutability
- Prefer creating new DataFrames over mutating existing ones
- Use `.copy()` when modifying a slice of a DataFrame
- Avoid `inplace=True` on pandas operations

## File Size
- 200-400 lines typical, 800 max
- Extract utilities to separate modules if a file exceeds 600 lines

## Functions
- Keep functions under 50 lines
- One responsibility per function
- No deep nesting (>4 levels) — extract to helper functions

## Naming
- snake_case for all variables, functions, modules
- UPPER_SNAKE for constants
- Descriptive names — `home_team_rolling_win_pct` not `hwp`

## Error Handling
- Always handle errors explicitly — never silently swallow exceptions
- Log context on failure: which team, season, game_id
- Validate at script entry points (CLI args, API responses)

## No Hardcoded Values
- Season years, team counts, feature column names → use constants or config
- API endpoints → use env vars or config files
