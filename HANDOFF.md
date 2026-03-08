# Handoff — NBA Analytics Project

_Last updated: 2026-03-07 Session 6_

## What Was Done This Session (Session 6)

### Dashboard — Polish (commit ce8c230)

**Player Modal**
- Season table: added FT% column + Career Totals row (GP-weighted averages for all stats)
- Career avg cards: now shows FG%, FT%, TS% via `_wAvgPct()` from season data
- Team history stints: team logos added to each badge
- Escape key closes the modal

**Standings Table**
- Added L10 (Last 10 games) column with color coding (green>=7, red<5)
- Full team names shown instead of abbreviations
- Migrated `renderStandings` to `_setHtml()` (was direct DOM mutation)

**Players Table**
- Team logos added next to team abbreviation pill
- "Legend" badge renamed to "Retired"

**Season History Game Log**
- Team logos via `home_abbr`/`away_abbr` fields
- Winner scores in green, loser muted
- Margin column added; header changed to "Final Standings"

**Other**
- Era-adjusted footnote updated to accurate 111.8 PPG description
- 560 tests passing; committed ce8c230 and pushed

## What Was Done — Session 5

### Dashboard — Major Upgrade

**Players Tab**
- "All-Time Legends" renamed to "All-Time Players" — now shows all 883 historical players (not just 6)
- `mapJsonPlayer` retired detection fixed: uses `seasons_span` check (`!includes('2024') && !includes('2025')`) instead of `!!p._legend`
- Clickable player rows: `onclick="showPlayerDetail(name)"` — opens a detail modal
- Player detail modal: headshot + team logo (NBA CDN), career avg cards, team history stints, season-by-season table with BEST highlight
- `window._FULL_PLAYER_DATA` stores raw playerJson.players for modal season data lookup
- Compare picker updated: optgroup label "All-Time Players" (was "All-Time Legends")

**Compare Feature — Team Colors**
- `TEAM_COLORS` expanded to all 30 NBA teams with `[primary, secondary]` arrays (was 2 teams)
- `TEAM_IDS` map + `teamLogoByAbbr()` using NBA CDN (already CSP-whitelisted)
- `getPlayerPrimaryTeam(p)` computes most-played-for team for historical players from raw season data
- `renderBars(id, rows, nA, nB, colA, colB)` — added colA/colB params, uses `_setHtml`
- `renderRadar(pA, pB, aA, aB, colA, colB)` — added params + `hexToRgba()` helper for fill

**Era-Adjusted Fix**
- `eraFactor()` formula fixed: was `ERA_BASELINE/perPlayerLeague` (inflated everything); now `modernAvg/eraAvg` (111.8 / era_avg) — modern players get 1.0, historical low-scoring eras get slight boost

**Playoff Picture**
- Play-In zone properly 7-10 (was labeled up to seed 8 only)
- Clinched badge (checkmark) when pct >= 97%
- Dashed visual dividers after seed 6 and seed 10

**CLV Summary Card**
- New `scripts/build_clv.py` → `dashboard/data/clv_summary.json`
- 15th fetch added to Promise.all
- `updateCLVSummary()` rewritten to use `window.CLV_DATA` (was using value_bets as proxy)
- `update.py` Step 7: `build_clv` added to builder list (now 24 total)

**Season History**
- Game log now shows full team names ("Boston Celtics" not "BOS")
- `TEAM_NAMES` dict covers 30 current + 6 historical franchises

### Automation
- `.github/workflows/daily_deploy.yml`: runs `python update.py` at 9AM EST daily, commits + pushes dashboard data
- `scripts/deploy.sh`: manual local deploy wrapper
- `.github/SETUP.md`: new PC setup instructions (clone + venv + pip + .env copy)

### Skills Added
- `nba-dashboard-dev` — dashboard patterns, _setHtml rule, Promise.all, team colors/logos
- `plotly-charts` — Plotly trace patterns, color palette, lazy rendering
- `sqlite-analytics` — DB schema, common queries, JSON export pattern
- `nba-betting-analysis` — picks pipeline, CLV formula, Kelly criterion, ATS model

### CLAUDE.md Updated
- Skill routing table: 4 new NBA-specific skills added

## Pending at Session End

**Nothing critical** — all committed and pushed to main (commit ce8c230).

## Next Steps (priority order)

1. **Wire daily_deploy.yml** — add `BALLDONTLIE_API_KEY` as a GitHub Actions secret: repo Settings > Secrets > Actions > New secret. Workflow won't run without it.
2. **PC migration** — when switching PCs: clone repo, `python -m venv .venv && .venv/Scripts/pip install -r requirements.txt`, copy `.env` from old machine, run pytest to verify (560 tests should pass)
3. **CLV data population** — currently outputs zeros (closing_spread NULL until games close); will populate naturally as clv_tracker.py runs via fetch_odds.py after game closings
4. **Verify GitHub Pages** — after push check: player modal FT%+career totals, team logos in players list, L10 in standings, Escape-to-close modal

## Key Files Changed This Session

- `dashboard/index.html` — 16 targeted edits (5143 lines now)
- `scripts/build_clv.py` — NEW
- `dashboard/data/clv_summary.json` — NEW
- `scripts/build_season_history.py` — full team names
- `dashboard/data/season_history.json` — rebuilt
- `update.py` — build_clv added
- `.github/workflows/daily_deploy.yml` — NEW
- `scripts/deploy.sh` — NEW
- `.github/SETUP.md` — NEW
- `CLAUDE.md` — 4 new skill routes

## Critical Gotchas
- Promise.all is now **15 fetches** (not 14) — adding a 16th: append to destructure list AND fetch array
- `TEAM_COLORS` now returns `[primary, secondary]` array — callers that expect a string need `colors[0]`
- `getPlayerColors(p)` returns an array or null — `updateComparison` destructures with `const [colA]=(colors||[NEUTRAL_COLOR])`
- Security hook blocks Edit when replacement contains "innerHTML" — use `_setHtml(el, html)`
- `game_lines.csv` is at `data/odds/` not `data/processed/`
- `dashboard/data/*.json` must be committed after each `update.py` run
- `player_stats.csv` has season TOTALS — always divide by `gp` before per-game projections
