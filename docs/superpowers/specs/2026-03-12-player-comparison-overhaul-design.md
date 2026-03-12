# Player Comparison Tool Overhaul — Design Spec

## Overview

Redesign the Players tab comparison tool from a basic stat table into an immersive Trading Card experience with player headshots, team-branded gradients, tug-of-war stat bars, radar/trend charts, and a corrected era adjustment system. The goal is a visually rich, data-dense comparison that handles both modern and historical players seamlessly.

## Selected Design: Trading Card Style (Option B)

Two side-by-side player cards with team color gradients, headshots (from NBA CDN), stat highlights, career milestones, and accolades. Stats use tug-of-war bars (wider side = higher value) for instant visual comparison.

## Sections (7 total)

### 1. Search + Controls Bar
- Fuzzy search (autocomplete from `player_index.json`) for Player A and Player B
- Quick-swap preset buttons: "MJ vs LeBron", "Bird vs Magic", "Curry vs Durant"
- Era Adjustment toggle (on/off) — when on, all stat displays use era-adjusted values
- Swap button (flip A ↔ B)

### 2. Trading Cards (Hero Section)
- Two cards side by side, each with:
  - Player headshot from `https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png`
  - Fallback: generic silhouette for legend players with negative IDs or failed loads
  - Team gradient background (primary -> secondary color from team palette)
  - Player name, jersey number, position(s)
  - Career span (e.g., "1996-2016")
  - Career GP count
  - Top 3 career averages (PTS, REB, AST) in large font
  - Milestone badges derived from career totals thresholds (e.g., career PTS > 25000 -> "25K Club", career GP > 1000 -> "1K Games")
  - Best season callout (season string + PPG that season)

**`best_season` normalization**: The build script must standardize this field. Regular players store a string (`"2005-06"`); legends store a dict. Normalize to always be a string in `build_player_comparison.py`. Client code reads `best_season` as a string and looks up the PPG from the `seasons` array.

### 3. Tabbed Stat Comparison (Tug-of-War Bars)
Three sub-tabs (merged Shooting into Advanced to avoid overlap):
- **Per Game**: PTS, REB, AST, STL, BLK, FGM, FGA, FTM, FTA
- **Efficiency**: TS%, FG%, 3P%, FT%, PER36 PTS/REB/AST
- **Career Totals**: Total PTS, REB, AST, STL, BLK, GP, Minutes

Each stat row: `[Player A value] ——[bar]—— [Player B value]`
Bar width proportional to relative values. Winning side highlighted with team color. When era adjustment is ON, counting stats (PTS, REB, AST, STL, BLK) use adjusted values; percentages and rate stats remain raw.

### 4. Visualizations (Side by Side)
- **Radar Chart** (Chart.js): 6 axes — PTS, REB, AST, STL, BLK, TS%. Overlapping polygons in team colors.
- **Career Scoring Trend** (line chart): X = season, Y = PPG. Both players on same chart. Shows era-adjusted line when toggle is on.
- **Legend players with empty `seasons` arrays**: Show radar chart only (from career averages). Trend chart shows "Limited historical data" message instead of an empty chart. Only 1 of 6 legends (MJ) has partial season data.

### 5. Head-to-Head Record — DEFERRED TO V2
Pre-computing h2h for all player pairs is expensive and game logs are not loaded client-side. This section will show a placeholder: "Head-to-head data coming soon" with the overlapping active years noted. Implementation deferred to a future build script addition.

### 6. Season-by-Season Table (Collapsible)
- Expandable accordion per player
- Columns: Season, Team, GP, MIN, PTS, REB, AST, STL, BLK, FG%, 3P%, FT%, TS%
- Era-adjusted column appears when toggle is ON (showing adjusted PTS, REB, AST alongside raw)
- Sortable by any column
- **Legend handling**: Players with empty seasons arrays show "Career averages only — detailed season data not available" instead of an empty table

### 7. Similar Players + Share/Export
- "Players similar to [Player A]": top 3 by career stat similarity
  - **Algorithm**: z-score normalize career averages across all players for PTS, REB, AST, STL, BLK, TS% (6 dimensions). Compute Euclidean distance. Return 3 nearest neighbors excluding the player themselves.
  - Position is NOT a filter — let the stats speak (a versatile big who passes like a guard should match passers)
- Share button: copies comparison URL with both player IDs as query params
- Export: download comparison as CSV. PNG export deferred (html2canvas has known issues with Chart.js canvases).

## Era Adjustment — Corrected Implementation

### Problem with Current System
The existing era adjustment uses hardcoded decade-level league averages (`ERA_LEAGUE_AVG`) which:
1. Groups by decade instead of per-season
2. Uses team PPG (not per-player averages)
3. Only adjusts stat bar widths, not displayed values
4. Ignores `league_by_season` data already in `player_comparison.json`

### Definition of "League Average"
The `league_by_season` data in `player_comparison.json` stores per-season means computed across ALL player-season rows with >= 5 minutes per game. This produces values around 6-11 PPG (since it includes all rostered players, not just starters). This is fine — the formula uses RATIOS between eras, so the absolute scale cancels out. What matters is that the same methodology is used consistently across all seasons.

### Correct Formula
```
adjusted_stat = raw_stat * (modern_league_avg / player_season_league_avg)
```

Where:
- `raw_stat` = player's per-game stat for that season
- `player_season_league_avg` = league average for that stat in that specific season (from `league_by_season`)
- `modern_league_avg` = league average for the most recent complete season in the dataset

### Example (using actual data scale)
- 2024-25 league avg_pts: ~9.5 (modern reference)
- 1996-97 league avg_pts: ~8.2
- Jordan 1996-97: 29.6 PPG raw
- Adjusted: 29.6 * (9.5 / 8.2) = 34.3 PPG (modern context — higher because 90s had lower scoring environment in our metric)

The ratio-based approach means the absolute value of league averages doesn't matter — only relative differences between eras.

### Implementation
1. **Data source**: `player_comparison.json` already contains `league_by_season` with per-season `avg_pts`, `avg_reb`, `avg_ast`, `avg_stl`, `avg_blk`
2. **Modern reference**: use the last entry in `league_by_season` array (most recent complete season)
3. **Per-season adjustment**: for each player-season row, compute `adjusted = raw * (modern_avg / season_avg)` for PTS, REB, AST, STL, BLK
4. **Career adjusted averages**: GP-weighted mean of adjusted season values
5. **Career adjusted totals**: sum of (adjusted_per_game * GP) across seasons
6. **What stays raw**: FG%, 3P%, FT%, TS%, PER36 — percentages and rate stats are already era-neutral
7. **Toggle behavior**: when OFF, all displays show raw stats. When ON, counting stats reflect adjusted values. Labels change to include "(era-adj)" suffix.
8. **Missing data fallback**: if `league_by_season` entry missing for a season (pre-1969 seasons not covered), fall back to raw value (no adjustment). UI shows a subtle indicator "(unadj)" next to those values.
9. **Legend players with no seasons**: Apply era adjustment to career averages using the league average from their career midpoint season (e.g., Wilt's midpoint ~1966 -> use earliest available season average as approximation). If no matching season exists, show raw with "(unadj)" label.

### Coverage Gap
`league_by_season` starts at 1968-69. Players active before that (Wilt 1959-73, Russell 1956-69) will have partial adjustment — later seasons adjusted, earlier ones shown raw. This is clearly communicated in the UI.

## Data Requirements

### Existing (no changes needed to JSON structure)
- `player_comparison.json`: players array with seasons, career_avgs, league_by_season
- `player_index.json`: lightweight autocomplete index
- `single_game_records`: already computed and in JSON

### Build Script Changes (`build_player_comparison.py`)
1. **Normalize `best_season`**: Ensure all players (including legends) store `best_season` as a string, not a dict
2. **Inject legends into `player_index.json`**: Currently legends are added after the index is built. Move legend injection before index generation so MJ, Wilt, etc. appear in autocomplete
3. Milestone badges: computed client-side from career totals — no build change needed

### New Client-Side Data
- **Player headshots**: NBA CDN URLs from `player_id`. Legend players (negative IDs) use silhouette fallback.
- **Team color palette**: hardcoded JS map of team abbreviation -> [primary, secondary] hex colors

## UI/UX Details

### Responsive Layout
- Desktop: cards side-by-side (50/50 split)
- Tablet: cards side-by-side but narrower
- Mobile: cards stacked vertically, swipe to switch

### Animations
- Card entrance: slide in from sides
- Stat bars: animate width on load
- Radar chart: draw animation
- Tab transitions: fade

### Accessibility
- All charts have aria-labels with stat values
- Color coding supplemented with text labels for colorblind users
- Keyboard navigable tabs and search

## Features Excluded (per user request)
- "Who Wins?" AI verdict (feature #5 from original list) — not included
- PNG export — deferred due to html2canvas/canvas compatibility issues
- Head-to-head detailed stats — deferred to v2

## Files to Modify

1. **`dashboard/index.html`** — Major: new comparison UI (replace existing comparison section), era adjustment rewrite, team color palette, headshot loading, Chart.js radar/line charts, tug-of-war bars, similar players algorithm, share/CSV export
2. **`scripts/build_player_comparison.py`** — Minor: normalize `best_season` type, inject legends before index build
3. **`dashboard/sw.js`** — Minor: cache NBA CDN headshot URLs (LRU, max 200 entries)

## Testing Plan

1. Compare modern players (LeBron vs Curry): headshots load, stats correct, radar renders
2. Compare historical vs modern (MJ vs LeBron): legend data displays, era adjustment produces reasonable values, MJ's partial seasons show in trend chart
3. Compare pre-1996 legends (Wilt vs Kareem): fallback headshots, career averages display, trend chart shows "limited data", era adjustment applies to Kareem's post-1968 seasons
4. Era toggle: flip on/off, verify counting stats update, verify percentages stay raw, verify "(era-adj)" labels appear
5. Mobile: verify stacked layout, touch interactions work
6. Edge cases: player with no seasons data, player not in index, same player vs self, legend with negative player_id
7. Similar players: verify z-score similarity returns sensible matches (guards match guards, etc. by stat profile)
8. Autocomplete: verify legends appear in search (MJ, Wilt, Bird, Magic, Kareem, Kobe)
