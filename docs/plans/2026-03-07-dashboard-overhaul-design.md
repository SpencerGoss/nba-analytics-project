# Dashboard Overhaul Design — 2026-03-07

## Goals
1. Fix all broken data (value_bets empty, player_props book_line null, stale picks)
2. Remove hardcoded JS constants (ADV, accuracy %, timestamps)
3. Build out 3 placeholder tabs: Sharp Money Tracker, Bet Tracker, Season History
4. Security hardening (SRI, CSP, remove fake account UI)
5. UI polish where appropriate

## Root Causes Identified

### value_bets.json empty
- Script works; was just stale data at last commit. Re-run after fresh fetch_odds.py.

### player_props book_line: null
- `fetch_player_props()` is implemented but `player_props_lines.csv` was never populated.
- Fix: run `fetch_odds.py` which writes `data/processed/player_props_lines.csv`, then `build_props.py`.

### Hardcoded ADV constant (line 2181)
- 17 current players with hand-typed PER/TS%/USG%/WS/BPM/VORP.
- `advanced_stats.json` has 504 live players with ts, usg, off_rtg, def_rtg, net_rtg, efg.
- Fix: Remove current-player entries from ADV; keep only legends (Jordan/Bird/Magic/Kareem/Kobe/Wilt) as LEGENDS_ADV.

## Architecture

### Phase 1 — Python agents (parallel)
- **P1**: Fetch fresh odds (`fetch_odds.py`) -> populate player_props_lines.csv -> rebuild props/value_bets
- **P2**: Build `scripts/build_season_history.py` using `data/processed/team_game_logs.csv`
  - Output: `dashboard/data/season_history.json` — array of {season, team, w, l, games: [...]}
  - Limit to last 10 seasons (2014-15 to 2024-25) to keep JSON small

### Phase 2 — HTML edits (sequential in main context)
- **H1**: Remove ADV current-player entries; keep LEGENDS_ADV; _mergeAdv uses ADVANCED_STATS primary
- **H2**: Sharp Money Tracker — replace placeholder with line_movement.json-driven table + move chart
- **H3**: Bet Tracker — localStorage-based form + history table + ROI stats
- **H4**: Security — add Plotly SRI hash, CSP meta tag, remove fake account dropdown
- **H5**: Fix static accuracy % in OG tags to read from performance.json on load
- **H6**: Wire season_history tab (page-history) with season selector + game table

## Data Shapes

### season_history.json
```json
[
  {
    "season": "2024-25",
    "teams": [
      {"abbr": "BOS", "name": "Boston Celtics", "w": 61, "l": 21},
      ...
    ],
    "games": [
      {"date": "2024-10-22", "home": "BOS", "away": "NYK", "home_pts": 108, "away_pts": 99, "wl": "W"},
      ...
    ]
  }
]
```

### Sharp Money Tracker UI
- Table: Game | Opening Spread | Current Spread | Move | Direction | Steam?
- Steam = |move| >= 1.5 pts, highlighted green/red
- Data source: `line_movement.json` (already built)

### Bet Tracker UI
- localStorage key: `baseline_bets_v1`
- Fields: date, team, opponent, market (ML/ATS/OU), line, stake, result (W/L/push/pending)
- Stats bar: total bets, W-L, ROI%, total profit/loss
- Add-bet modal with form validation

## Security Changes
- Add `integrity` SHA-384 attribute to Plotly CDN script
- Add `<meta http-equiv="Content-Security-Policy">` restricting script-src
- Remove fake account dropdown entirely (no auth system exists)
- Audit all innerHTML assignments for XSS — use esc() helper consistently

## Approach
Parallel: Python agents P1 + P2 run concurrently.
Sequential: HTML edits H1-H6 run in main context while Python agents work.
Integration: After agents complete, wire season_history tab and push all.
