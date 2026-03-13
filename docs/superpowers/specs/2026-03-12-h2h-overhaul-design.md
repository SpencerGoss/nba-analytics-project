# H2H Section Overhaul — Design Spec

## Summary

Full overhaul of the Head-to-Head sub-tab within Standings: expand data coverage to all 435 team pairs, add tiered history (This Season / Last 3 Seasons / All-Time), venue splits, scoring trends, streaks, quick-pick buttons for today's games, and a redesigned radar chart. Layout follows a linear top-to-bottom flow.

## Data Layer

### `scripts/build_h2h.py` Changes

**Current:** Only builds H2H for today's matchups via `todays_picks.json`.

**New:** Precompute all 435 team pairs from `data/processed/team_game_logs.csv`.

**Output format:** Dictionary keyed by canonical pair (alphabetical order, pipe-separated):

```json
{
  "DET|PHI": {
    "team_a": "DET",
    "team_b": "PHI",
    "this_season": {
      "series_record": "3-1",
      "a_wins": 3, "b_wins": 1,
      "total_meetings": 4,
      "avg_total": 218.5,
      "avg_margin": 8.2,
      "a_home_wins": 2, "a_home_losses": 0,
      "a_away_wins": 1, "a_away_losses": 1,
      "a_avg_pts": 112.3, "b_avg_pts": 106.1,
      "streak": {"team": "DET", "count": 2},
      "meetings": [...]
    },
    "last_3": { "...same shape..." },
    "all_time": { "...same shape..." }
  }
}
```

**Key design decisions:**

- **Dictionary keying** by `"TEAM_A|TEAM_B"` (alphabetical) for O(1) lookup. JS lookup: `data[canonicalKey(teamA, teamB)]`.
- **`total_meetings`** per tier: actual count of all games in that tier. Meetings array is capped at 20 for JSON size, but `total_meetings` reflects the true count so UI can show "Showing 20 of 87 all-time meetings."
- **Averages** (`avg_total`, `avg_margin`, `a_avg_pts`, `b_avg_pts`) are computed from ALL meetings in the tier, not just the capped 20.
- **Tier boundaries** use the `season` column (integer):
  - `this_season`: `season == 202526`
  - `last_3`: `season >= 202324` (current season + 2 prior — includes this_season as a superset)
  - `all_time`: all seasons in the CSV
  - Tiers are overlapping by design: `this_season` data is a subset of `last_3`, which is a subset of `all_time`.
- Each meeting: `{date, home_team, away_team, home_score, away_score, winner, margin}`
- **Franchise relocations** (SEA, NJN, etc.): treat as separate entities — only current 30 team abbreviations are paired. Historical games under old abbreviations are excluded.
- Output: `dashboard/data/head_to_head.json` (estimated ~2-3MB with compact JSON separators)
- Lazy-loaded on H2H tab open with cache-busting `?v='+Date.now()` (same pattern as `player_comparison.json`)

### Builder Strategy

Group all games by canonical pair first (one pass through the DataFrame), then split by tier per pair. Avoids O(435 * N) nested loops.

### Venue Split Calculation

For a given pair (A, B) in a tier:
- `a_home_wins` / `a_home_losses`: A's record when A is home team
- `a_away_wins` / `a_away_losses`: A's record when A is away team
- B's venue split is the inverse (B home = A away, and vice versa) — derived in JS, not stored.

### Streak Calculation

Walk meetings newest-first. Count consecutive wins by the same team. Store `{team, count}`. If 0 meetings, streak is `null`.

## UI Components (Linear Flow, Top to Bottom)

### 1. Quick Picks Bar

- Row of pill buttons for today's matchups
- Each pill: both team logos (18px) + abbreviations (e.g., "DET vs PHI")
- Click sets both dropdowns and triggers render
- Hidden if no games today or `DATA.picks` not yet loaded
- Source: `DATA.picks` (from Promise.all) — guarded with null check

### 2. Team Selectors

- Two dropdowns (`h2h-team-a`, `h2h-team-b`) with all 30 teams
- Existing `sel-wrap` styling
- Default: first today's matchup, or first alphabetical pair if no games
- JS lookup: `function _canonKey(a,b){ return [a,b].sort().join('|'); }` then `_h2hData[_canonKey(teamA, teamB)]`

### 3. Banner Card

- Team logos (40px) with abbreviations and full names
- Series record text, dynamically scoped to active tier
- Win count boxes (large font, green/red)
- Win dominance bar with logo end-caps
- Matchup streak badge in corner (e.g., "DET 3-game win streak")

### 4. Toggle Pills

- Three pills: **This Season** | **Last 3 Seasons** | **All-Time**
- Default: This Season
- Switching re-renders: stats cards, radar, results strip, meetings table
- Same visual style as Today's Games filter pills (`fb`/`fc` classes)
- If selected tier has 0 meetings, show empty state: "No meetings [this season / in last 3 seasons]"

### 5. Stats Cards (2-column grid)

**Card 1 — Matchup Summary:**
- Series record (W-L)
- Venue split: "A at home: 2-0 · A on road: 1-1"
- Current matchup streak
- Total meetings in tier

**Card 2 — Scoring Trends:**
- Avg combined total
- Avg winning margin (with leader label)
- Team A avg pts scored/allowed
- Team B avg pts scored/allowed

Side-by-side on desktop (`class="g2"`), stacked on mobile.

### 6. Radar Chart (redesigned)

**Data source:** `DATA.east` and `DATA.west` from standings (already loaded in Promise.all). Look up both teams to get Win%, Home Record, Away Record, L10, and derive Offense/Defense from `TEAM_TRENDS` (already loaded as `window.TEAM_TRENDS`).

- 6 axes: Win%, Home Record, Away Record, L10, Offense, Defense
- **Win%** from standings `w/(w+l)`
- **Home/Away Record** parsed from standings `home_record`/`away_record` strings
- **L10** parsed from standings `last10` string
- **Offense/Defense** from `TEAM_TRENDS[abbr].last10_avg_scored` / `last10_avg_allowed`
- Team primary colors from `TEAM_COLORS`
- Larger axis labels (13px), no overlap
- Transparent backgrounds matching dashboard theme
- Team legend below chart (horizontal)
- Contained in a card with header "Team Comparison"
- Uses `_ensurePlotly` / `_plotlyNewPlot` for lazy Plotly loading

### 7. Last 5 Results Strip

- W/L colored pills with scores, scoped to active tier
- Green = Team A won, Red = Team A lost
- Shows last 5 meetings in the tier (or fewer if tier has < 5)

### 8. Meetings Table

- Shows last 10 games in selected tier
- Columns: Date, Score, Winner, Margin, Total
- Winner colored green (A) or red (B)
- Total highlighted blue if above tier avg_total
- "Show all X games" button at bottom:
  - If `total_meetings <= 20`: expands to show all meetings from JSON
  - If `total_meetings > 20`: button says "Showing 20 of X meetings"
- Date formatted as "Mon DD, YYYY"

## Files Modified

| File | Change |
|------|--------|
| `scripts/build_h2h.py` | Rewrite: all 435 pairs, tiered history, venue splits, streaks, scoring stats |
| `dashboard/index.html` | Rewrite H2H rendering: quick picks, toggle pills, stats cards, redesigned radar, expandable table. Update `window.H2H_DATA` consumers (game detail modal) to use new keyed format. |
| `dashboard/data/head_to_head.json` | New format: dictionary of all pairs with tiered data |

## Consumers of `window.H2H_DATA`

The game detail modal references `window.H2H_DATA` for matchup context. Must update its lookup to use the new `_canonKey()` function and tiered structure (use `this_season` tier for modal display).

## No Changes To

- `update.py` (already calls `build_h2h` in Step 7)
- Data pipeline or model code

## Constraints

- `_setHtml()` for all dynamic DOM writes (security hook)
- `_ensurePlotly` / `_plotlyNewPlot` for Plotly charts
- `pd.to_datetime(format="mixed")` for date parsing
- No `innerHTML` directly (security hook blocks it)
- Lazy-load JSON only on tab open, not in Promise.all
- Cache-busting query param on fetch
- `time.sleep(0.6)` not needed (no API calls, reading local CSV)
