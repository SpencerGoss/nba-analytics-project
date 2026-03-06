# NBA Analytics Dashboard v3 — Design Document
_Created: 2026-03-06_

## Overview

Full redesign and feature expansion of the NBA Analytics dashboard. Goal: a product-quality tool that helps common bettors understand why to choose a bet and how our predictions compare to sportsbooks. Built for personal use now, subscriber-ready later.

**Core value proposition:** A bettor should look at any pick and immediately understand — here's what the books say, here's what we say, here's why, and here's the edge.

---

## Decisions Made

| Question | Decision |
|----------|----------|
| Target user | Personal + future free subscribers |
| Hero section | Today's best bets (value bets + picks) |
| Betting markets | All — ML, ATS, totals, player props, futures |
| Data refresh | Hourly via scheduler (static JSON, no backend) |
| Visual style | Linear/Coinbase — clean fintech, big numbers, smooth animations |
| Bet sizing | Opt-in only — hidden by default, disclaimer required |
| Legal gate | Terms agreement required before any betting/modeling section |
| Live URL | https://spencergoss.github.io/nba-analytics-project/ |

---

## Section 1: Data Layer

Every value on the dashboard is built from a dedicated script and JSON file. Nothing hardcoded.

| JSON File | Builder Script | Powers |
|-----------|---------------|--------|
| `todays_picks.json` | `build_picks.py` | Hero picks grid |
| `value_bets.json` | `build_value_bets.py` | Value bet cards |
| `game_context.json` | `build_game_context.py` | B2B, rest days, travel, streaks |
| `explainers.json` | `build_explainers.py` | Plain-English bet bullets |
| `head_to_head.json` | `build_h2h.py` | H2H tab |
| `player_props.json` | `build_props.py` | Props tab |
| `totals.json` | `build_totals.py` | Over/under tab |
| `performance.json` | `build_performance.py` | ROI, CLV, accuracy |
| `standings.json` | `build_standings.py` | Standings tab |
| `injuries.json` | `build_injuries.py` | Injury report tab |
| `line_movement.json` | `build_line_movement.py` | Sharp money tab |
| `trends.json` | `build_trends.py` | Team trends tab |
| `matchup_analysis.json` | `build_matchup_analysis.py` | Matchup tab |
| `power_rankings.json` | `build_power_rankings.py` | Power rankings tab |
| `live_scores.json` | `fetch_live_scores.py` | Live ticker |
| `meta.json` | `build_dashboard.py` | Last updated, model version |

`scripts/scheduler.py` runs all builders hourly in dependency order.

---

## Section 2: Visual Design System

**Theme:** Linear/Vercel aesthetic — ultra-dark, glass-morphism cards, neon-green signals.

### Color Tokens
| Token | Value | Use |
|-------|-------|-----|
| `--bg` | `#08090E` | Page background |
| `--bg2` | `#0F1117` | Section backgrounds |
| `--bg3` | `#161B27` | Input backgrounds |
| `--card` | `rgba(255,255,255,0.06)` | Card surfaces |
| `--green` | `#00C896` | Win / positive / value |
| `--red` | `#FF5C5C` | Loss / negative |
| `--gold` | `#F5A623` | Edge / highlight |
| `--blue` | `#4F9EFF` | Neutral picks, model prob |
| `--purple` | `#A78BFA` | Props / futures |

### Components
- **Confidence pills:** `HIGH` (green) / `MED` (gold) / `LOW` (muted red)
- **Market pills:** `ML` / `ATS` / `OU` / `PROP` — color-coded
- **Edge bar:** model% vs market% side-by-side horizontal bars
- **Situational badges:** `B2B`, `REST 3`, `AWAY STREAK`, `KEY INJ`
- **Kelly chip:** opt-in only, monospace, `2.3u` format
- **Trend arrows:** `+2.3 pts` green up / `-1.1 pts` red down

---

## Section 3: Hero — Today's Best Bets

**Top:** Full-width spotlight card — top value bet of the day.
- Teams, time, spread
- WE SAY vs BOOKS SAY vs EDGE — visual bar comparison
- Plain-English "Why We Like It" bullets (3 max, auto-generated)
- Bet sizing chip (opt-in, behind settings toggle)
- Situational badges

**Below:** 3-column pick grid (1-col mobile).
- Filter pills: `All` · `Value Bets` · `ML` · `ATS` · `O/U` · `Props`
- Cards with value bets glow green; no-edge cards dimmed
- Each card: teams, win prob, ATS pick, spread, edge pill, 1-line explainer, badges

---

## Section 4: Value Bet Cards

Full explainability layer — common bettors must understand every pick.

Each card contains:
1. Market pill + teams + spread + confidence pill
2. Model% vs Pinnacle% with side-by-side bars
3. Edge percentage highlighted
4. "Why We Like It" — 2-3 auto-generated bullets from features
5. Implied odds row: American / Decimal / Implied% (auto-converts)
6. Bet sizing (opt-in, hidden by default)

**Legal protections:**
- Bet sizing hidden by default; opt-in toggle in Settings
- Disclaimer modal on first enable: "For reference only. Not financial advice."
- `ⓘ` disclaimer link always visible when sizing shown
- Terms gate required before accessing any betting section (see below)

---

## Section 5: Supporting Tabs

### Head-to-Head
- Two-team dropdown, instant JSON lookup
- Last 10 meetings: score, winner, ATS result, total
- Series record, scoring trends, home/away splits
- Model's historical accuracy on this exact matchup

### Player Props
- Points / Rebounds / Assists / 3PM lines vs model projection
- Sparkline of last 5 game results
- Filter by team, stat, value bets only
- Edge calculation: model projection vs book line

### Performance / ROI
- Rolling accuracy chart (existing, kept)
- ROI by market: ML / ATS / Totals / Props
- CLV summary (mean CLV, positive rate, edge flag)
- Model calibration table: "when we say 65%+, actual hit rate = 71%"
- Streak tracker: current W/L streak, season best

### Standings
- SU record + ATS record + O/U record
- Last 10 form pills (W/L/W/W/L...)
- East/West conference color tinting
- Sortable columns

### Injury Report
- Today's questionable/out players
- Season stats for each player
- Spread impact: "Embiid OUT -> model adjusts PHI win prob -8.2%"
- Sourced from `player_absences.csv` (1M+ rows, already in pipeline)

### Sharp Money / Line Movement
- Opening line vs current line per game
- Direction of movement + interpretation: "sharp action on OKC"
- Sourced from CLV tracker (opening/closing lines already logged)

### Team Trends
- Rolling 10-game cards per team
- ATS record, O/U record, avg margin, home/away splits
- Pace rank, offensive/defensive rating trend

### Matchup Analysis
- Pace vs pace, ORtg vs opponent DRtg
- 3PT rate vs opponent 3PT defense
- Radar charts for visual comparison

### Power Rankings
- Model-derived rankings updated daily
- Side-by-side: Our Rank vs ESPN/AP Rank
- Divergence callout: "We rank MEM 8th, media ranks 14th — undervalued"

### Bet Tracker (opt-in, localStorage only)
- User logs their own picks manually
- Tracks personal P&L, win%, ROI
- Compares to "if you followed model exactly"
- No server — all local storage, no liability

---

## Section 6: Betting Tools

**Slide-out drawer** (accessible from toolbar icon, any tab):

1. **Odds Converter** — American / Decimal / Implied% — live cross-conversion
2. **Line Comparison** — Our model vs Pinnacle, plain-English edge explanation
3. **"What Does This Mean?"** — auto-generated plain English for every bet
4. **Bet Sizing** (opt-in, behind toggle + disclaimer)

### Legal Gate (Terms Agreement)
- Triggers first time user accesses any betting/modeling section
- 5 manually-checked boxes (no "select all"):
  - Age 18+
  - Betting legal in my jurisdiction
  - Not financial advice
  - Past accuracy does not guarantee future results
  - I accept responsibility for my betting decisions
- Stored in localStorage with version + timestamp
- Version bump re-triggers gate for all users
- Declining redirects to Standings (safe content)

---

## Agent Deployment Plan

### Data Agents (parallel — independent files)
| Agent | Builds |
|-------|--------|
| Data-Context | `build_game_context.py` — B2B, rest, travel, streaks |
| Data-Explainers | `build_explainers.py` — plain-English bullets |
| Data-Props | `build_props.py` + `build_totals.py` |
| Data-Trends | `build_trends.py` + `build_h2h.py` + `build_matchup_analysis.py` |
| Data-Rankings | `build_power_rankings.py` + `build_injuries.py` + `build_line_movement.py` |
| Data-Scheduler | `scripts/scheduler.py` + `build_performance.py` |

### UI Agents (sequential — same file)
| Agent | Builds |
|-------|--------|
| UI-Theme | Full Linear/Coinbase theme overhaul, new color tokens |
| UI-Hero | Spotlight card + pick grid + filter pills |
| UI-ValueBets | Redesigned cards + explainer bullets + odds converter |
| UI-Tabs | All supporting tabs (H2H, Props, Performance, Standings, etc.) |
| UI-Tools | Betting tools drawer + terms gate + settings panel + Kelly toggle |

---

## Non-Goals (explicitly out of scope for v3)
- Backend server / user accounts (future subscriber phase)
- Real-time WebSocket updates (upgrade path from hourly to live)
- Mobile app
- Paid features / paywalls
- External sportsbook integrations / affiliate links
