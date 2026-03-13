# Baseline Black — Full Visual Overhaul Design Spec

**Date:** 2026-03-13
**Direction:** B — "Baseline Black" (Premium Sports Brand)
**Status:** Approved — spec finalized with dynamic features
**Tagline:** *"Serious analytics brand that also happens to make betting fun."*

---

## Design Goals

- Full visual overhaul of dashboard/index.html (~8,900 lines)
- Green + gold "money" palette — premium, distinctive, not a sportsbook clone
- Smooth, branded feel that sets Baseline apart from competitors
- Target audience: sports enthusiasts and bettors wanting data-driven edges
- Desktop-first, mobile passable
- Animated gradient mesh background (restrained, not cheesy)
- Hybrid navigation (macro tabs + sub-tabs)

---

## 1. Color System

### Dark Mode (Primary)

| Token | Value | Usage |
|-------|-------|-------|
| `--bg` | `#08090E` | Page background |
| `--bg2` | `#0F1117` | Elevated surfaces |
| `--bg3` | `#161B27` | Tertiary panels |
| `--card` | `rgba(255,255,255,0.04)` | Card background |
| `--card-h` | `rgba(255,255,255,0.07)` | Card hover |
| `--card-border` | `rgba(255,255,255,0.06)` | Default card borders |
| `--card-border-h` | `rgba(255,255,255,0.12)` | Hover card borders |
| `--green` | `#00C896` | Primary accent — wins, positive, picks, confidence |
| `--gold` | `#E8C547` | Money accent — edge, Kelly, value bets, POTD |
| `--gold-dark` | `#D4A745` | Gold muted for smaller elements |
| `--red` | `#FF5C5C` | Losses, negative values |
| `--blue` | `#4F9EFF` | Informational, neutral stats |
| `--purple` | `#A78BFA` | Secondary accent (rankings, historical) |
| `--t0` | `#F1F5FF` | Primary text (headings, important values) |
| `--t1` | `#8892AA` | Secondary text (descriptions, meta) |
| `--t2` | `#6B7A99` | Muted labels, tertiary text |

### Gradient System

| Name | Value | Usage |
|------|-------|-------|
| Brand gradient | `linear-gradient(135deg, #00C896, #E8C547)` | Logo, POTD borders, active nav underlines, hero text |
| Card top-bar | `linear-gradient(90deg, #00C896, #E8C547)` | 3px bar at top of pick cards |
| POTD bar | `linear-gradient(90deg, #E8C547, #D4A745)` | 4px bar on Pick of the Day (gold-dominant) |
| Confidence high | `linear-gradient(90deg, #00C896, #00E8A8)` | Confidence bar fill (high) |
| Confidence med | `linear-gradient(90deg, #E8C547, #D4A745)` | Confidence bar fill (medium) |
| Edge badge green | `rgba(0,200,150,0.12)` bg + `#00C896` text | Positive edge indicators |
| Edge badge gold | `rgba(232,197,71,0.12)` bg + `#E8C547` text | Money/value indicators |

### Light Mode (to be designed)

Light mode will invert the palette while preserving the green/gold identity. Deferred to Phase 2 of implementation — dark mode is the primary experience.

---

## 2. Typography

| Element | Font | Weight | Size | Notes |
|---------|------|--------|------|-------|
| Body text | Plus Jakarta Sans | 400-600 | 13-15px | |
| Section headings | Plus Jakarta Sans | 800 | 18-20px | letter-spacing: -0.3px |
| Hero stats | Plus Jakarta Sans | 900 | 28-36px | letter-spacing: -0.5px |
| All data/numbers | JetBrains Mono | 600-800 | 12-28px | **Every number uses mono** |
| Labels (uppercase) | Plus Jakarta Sans | 700 | 9-11px | uppercase, letter-spacing: 0.6-1.0px |
| Logo "Baseline" | Plus Jakarta Sans | 900 | 20px | Green-to-gold gradient fill |
| Badges/pills | JetBrains Mono | 700 | 10-12px | Inside pill-shaped containers |

**Hard rule:** Every number that represents data uses JetBrains Mono. All prose/labels use Jakarta Sans. No exceptions.

---

## 3. Background

Animated gradient mesh — 3 large soft blobs that drift slowly:

```css
.mesh-blob.g1 {
  width: 600px; height: 600px;
  background: rgba(0,200,150,0.07);
  top: -100px; right: -100px;
  animation: drift1 25s ease-in-out infinite;
}
.mesh-blob.g2 {
  width: 500px; height: 500px;
  background: rgba(232,197,71,0.05);
  bottom: -150px; left: -100px;
  animation: drift2 30s ease-in-out infinite;
}
.mesh-blob.g3 {
  width: 400px; height: 400px;
  background: rgba(0,200,150,0.04);
  top: 50%; left: 40%;
  animation: drift3 20s ease-in-out infinite;
}
```

- All blobs use `filter: blur(120px)` and low opacity
- Positioned behind content with `pointer-events: none; z-index: 0`
- Respects `prefers-reduced-motion` — disable animations

---

## 4. Navigation — Hybrid System

### Top Nav Bar

```
┌─────────────────────────────────────────────────────────┐
│ Baseline Analytics    | Analytics | Teams | Betting |  🌙│
│                       └──────────┬───────┘             │
└──────────────────────────────────┼─────────────────────┘
                                   │
┌──────────────────────────────────┼─────────────────────┐
│  Today's Picks │ Standings │ Rankings │ History │ Injuries │
└────────────────────────────────────────────────────────┘
```

**Macro tabs (3):**
1. **Analytics** — Today's Picks, Standings, Power Rankings, Season History, Injuries
2. **Teams & Players** — Team Trends, H2H, Players, Matchup Analysis
3. **Betting** — Value Bets, Player Props, Totals, Line Movement, Sharp Money, Performance, Odds Converter, Bet Tracker

**Macro tab style:**
- Padding: 20px 24px
- Font: 13px weight 600
- Inactive: rgba(255,255,255,0.5)
- Active: #fff with gradient underline (green→gold via border-image)

**Sub-tabs:**
- Row beneath macro tabs
- Pill-shaped (8px border-radius)
- Active: green text + green-tinted background `rgba(0,200,150,0.08)`
- Background: rgba(8,9,14,0.6) with backdrop-filter blur(12px)

---

## 5. Card System

### Standard Pick Card

```
┌─ 3px gradient top bar (green→gold) ─────────────────┐
│                                                       │
│  LAL  vs  BOS            +5.8% Edge (green pill)     │
│                                                       │
│  Spread      Model %      Market %      Kelly Size   │
│  BOS -4.5    68.2%        62.4%         2.1%         │
│                                                       │
│  Confidence  ████████████░░░░░  82%                  │
└───────────────────────────────────────────────────────┘
```

- Background: `rgba(255,255,255, 0.04)`, border: `rgba(255,255,255, 0.06)`
- Hover: border lightens, `translateY(-1px)`, deeper shadow
- Border-radius: 16px
- Pick stats in 4-column grid

### Pick of the Day Card

- Gold border: `rgba(232,197,71, 0.3)` + outer glow
- Gold top-bar (4px instead of 3px)
- "⭐ Pick of the Day" label: gold pill with gold-tinted background
- Edge badge uses gold instead of green
- Subtle pulse animation on the gold glow (optional, respects reduced-motion)

### Hero Stat Cards (top strip)

- 4-column grid at top of Analytics tab
- Each card: label (uppercase muted) + large mono number + sparkline
- Sparklines: 7 thin bars showing recent trend, last bar highlighted
- Gold sparklines for money-related stats, green for performance stats

### General Card Rules

- All cards: 16px border-radius, 1px border, subtle shadow
- Hover transition: 250ms ease, border-color + translateY + box-shadow
- No heavy drop shadows — keep it light
- Green top-border on "hot" / high-confidence picks
- Gold accents only on money-related elements

---

## 6. Key UI Components

### Confidence Meter

- 4px tall bar with rounded ends
- Background: `rgba(255,255,255, 0.06)`
- Fill: green gradient (high 70%+), gold gradient (medium 50-70%), muted gray (low <50%)
- Numeric percentage displayed to the right in mono font

### Edge Badges

- Pill-shaped (border-radius: 100px)
- Green variant: rgba green bg + green text
- Gold variant: rgba gold bg + gold text
- Font: JetBrains Mono 11px bold
- Prefix with arrow or icon for positive edge

### Section Titles

- 18px weight 800, white
- Optional badge to the right (e.g., "3 Games" in a small green pill)
- Flex row with align-items: center

### POTD Label

- Inline pill: "⭐ Pick of the Day"
- Gold text, gold-tinted background, gold border
- Uppercase, 10px, weight 800, letter-spacing 0.6px

### Sparkline Micro-Charts

- 7 thin vertical bars (4px wide, 2px border-radius)
- Height proportional to value
- Last bar uses full accent color, previous bars are muted (30% opacity)
- Placed inside hero stat cards below the main number

---

## 7. Page-by-Page Layout

### Analytics > Today's Picks (Home)

1. Hero stat strip (4 cols): Accuracy, Today's Edge, ATS Record, Value Bets
2. "Today's Picks" section title with game count badge
3. POTD card (if one exists) — gold treatment
4. Remaining pick cards — standard green treatment
5. Each card: teams, spread, model%, market%, Kelly, confidence bar

### Analytics > Standings

- Table layout with sortable columns
- Team name in bold white, numbers in mono
- Green for winning records, red for losing
- Playoff seed separators (dashed lines after 6 and 10)
- Conference toggle (East/West)

### Analytics > Power Rankings

- Numbered list with Elo ratings
- Team logo + name + rank change indicator (↑↓)
- Elo value in mono, rank change in green/red pill

### Analytics > Season History

- Season selector dropdown
- Key stats comparison grid
- Charts (Elo over time, win% rolling)

### Analytics > Injuries

- Cards per team with injured players listed
- Impact badges (High/Medium/Low) in appropriate colors
- Spread impact estimate in gold mono

### Teams & Players > Team Trends

- Rolling 10-game stats: ATS%, O/U%, avg margin
- Home/away splits
- Sparkline trend indicators

### Teams & Players > H2H

- Team picker (two dropdowns or search)
- Last 10 meetings table
- Series record, scoring trends

### Teams & Players > Players

- Search + filter
- Player cards with headshots (NBA CDN)
- Key stats in mono grid
- Era-adjusted toggle

### Teams & Players > Matchup Analysis

- Radar charts (Plotly) for pace, ORtg, DRtg, 3PT
- Side-by-side team comparison

### Betting > Value Bets

- Cards with model% vs market%, edge%, Kelly sizing
- Green/gold edge badges
- "Why this bet?" expandable section
- Kelly row with gold accent

### Betting > Player Props

- Points/Rebounds/Assists projections vs book lines
- Over/Under indicators

### Betting > Line Movement

- Opening vs current spread
- Direction arrows
- "Steam move" badge (gold pulse) for sharp money indicators

### Betting > Performance

- Rolling accuracy chart
- CLV summary
- ROI by market type
- Calibration table

### Betting > Bet Tracker

- Local-only (localStorage)
- Add/edit/delete bets
- P&L summary, ROI calculation

### Betting > Odds Converter

- American / Decimal / Implied% cross-conversion
- Real-time as you type

---

## 8. Animation & Motion

### Base Animations

| Element | Animation | Duration | Notes |
|---------|-----------|----------|-------|
| Mesh blobs | Gentle drift (translateX/Y) | 20-30s | Infinite, `will-change: transform` for GPU |
| Card hover | translateY(-1px) + border-color + shadow + spotlight intensify | 250ms | ease timing |
| POTD glow | Gold box-shadow pulse | 2.4s | ease-in-out, infinite |
| Confidence bars | Width transition | 700ms | cubic-bezier(0.16,1,0.3,1) |
| Sparkline bars | Height on load | 300ms | Staggered: `animation-delay: calc(var(--i) * 50ms)` |
| Tab transitions | Opacity + translateY(4px) | 150ms | Content fade-in on tab switch |
| Hero numbers | Slot machine roll | 800ms | Each digit column scrolls independently |

### Dynamic Interactive Effects

| Effect | Implementation | Scope |
|--------|---------------|-------|
| **Mouse-tracking spotlight** | `mousemove` -> CSS vars `--x, --y` -> `radial-gradient(circle at var(--x) var(--y), rgba(0,200,150,0.15), transparent 60%)` as `::before` overlay | All cards |
| **3D tilt on hover** | `mousemove` -> calculate rotateX/rotateY from cursor offset (2-4 deg max) -> `perspective(800px)` on parent, `transform: rotateX(var(--rx)) rotateY(var(--ry))` on card. ~25 lines vanilla JS. | Pick cards, value bet cards, hero stat cards |
| **Shimmer button sweep** | CSS `@keyframes shimmer { 0% { background-position: -200% } 100% { background-position: 200% } }` on `linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent)` overlay, 3s infinite | "View All Picks", "See Value Bets" CTAs |
| **Animated notification feed** | Vertical list, items slide in from right with stagger delay, auto-cycle every 4s, shows recent prediction results (Hit/Miss + edge) | Home page sidebar or below hero strip |
| **Meteor particles** | 5-8 CSS pseudo-elements with randomized `animation-delay`, diagonal translateX+translateY keyframes, 0.05 opacity, thin 1px streaks | Hero stat strip background only |
| **Text gradient reveal on scroll** | `background: linear-gradient(135deg, #00C896, #E8C547)`, `background-clip: text`, scroll-driven `background-position` animation via `animation-timeline: view()` | Section titles ("Today's Picks", "Value Bets", etc.) |
| **Sparkles on POTD** | 4-6 tiny star pseudo-elements with random `animation-delay`, scale(0)->scale(1)->scale(0) + opacity keyframes, positioned around card border | Pick of the Day card only |
| **Skeleton loading** | Pulsing rectangles matching card layout, `@keyframes skeleton-pulse` (already defined), shown during Promise.all fetch, hidden on data load | All sections during initial load |
| **Scroll-driven card reveals** | `animation: fadeSlideUp 0.6s ease both; animation-timeline: view(); animation-range: entry 0% entry 30%` — cards fade+slide-up as viewport reaches them | All cards and sections |
| **Hover data dimming** | On chart hover, non-selected series dim to 30% opacity via CSS `transition: opacity 200ms`. Hovered element brightens + thickens | All Chart.js and Plotly charts |

### Slot Machine Number Ticker (Detail)

Hero stat numbers (67.5%, 3 Value Bets, etc.) use a slot-machine roll effect:

```
Structure per number:
<span class="ticker">
  <span class="ticker-col" style="--target: 6">
    <span class="ticker-strip">0 1 2 3 4 5 6 7 8 9</span>
  </span>
  <span class="ticker-col" style="--target: 7">...</span>
  <span class="ticker-sep">.</span>
  <span class="ticker-col" style="--target: 5">...</span>
  <span class="ticker-col-unit">%</span>
</span>

CSS: .ticker-strip animates translateY to -(var(--target) * digit-height)
     Each column has increasing delay: calc(var(--i) * 80ms)
     Duration: 800ms, cubic-bezier(0.16, 1, 0.3, 1) (spring feel)
```

### Glassmorphic Nav Bar (Detail)

```css
.nav {
  background: rgba(8, 9, 14, 0.7);
  backdrop-filter: blur(16px) saturate(1.2);
  -webkit-backdrop-filter: blur(16px) saturate(1.2);
  border-bottom: 1px solid rgba(255,255,255,0.08);
  position: sticky;
  top: 0;
  z-index: 100;
}
```
Content scrolls behind the frosted nav. Active macro tab has gradient underline visible through the glass.

### Motion Rules

- **ALL animations** respect `prefers-reduced-motion: reduce` — disable or set duration to 0.01ms
- **Performance**: spotlight, tilt, and scroll reveals use `transform` and `opacity` only (compositor thread, never triggers layout)
- **No emojis as icons**: Replace star emoji in POTD label with inline SVG star icon
- **Focus-visible**: All interactive elements get `:focus-visible { outline: 2px solid var(--green); outline-offset: 2px }`
- **Touch devices**: 3D tilt and spotlight disabled on `(hover: none)` media query
- **Continuous animations**: Only mesh blobs and shimmer buttons loop infinitely — all others fire once

---

## 9. Implementation Notes

### Hard Rules (from CLAUDE.md)
- Use `_setHtml(el, html)` for all DOM writes — never raw innerHTML
- Data-dependent UI in Promise.all callback, not tab-click handlers
- Guard `g.ats||''` before string methods
- Dashboard JSON files are committed to git
- Use `.venv/Scripts/python.exe` for builder scripts

### Migration Strategy (8 Phases)

1. **Foundation** — CSS variable swap (colors, gradients, gold token), typography audit (mono on all numbers), glassmorphic nav bar
2. **Background & Atmosphere** — Mesh blob upgrade (will-change, GPU), meteor particles on hero strip, skeleton loading states
3. **Card System** — New card component (16px radius, gradient top-bar), mouse-tracking spotlight, 3D tilt hover, scroll-driven fade-in reveals
4. **Navigation Restructure** — Hybrid macro/sub-tab system, pill sub-tabs, tab content fade transition
5. **Hero Section** — Slot machine number ticker, sparkline stagger animation, animated notification feed, shimmer CTA buttons
6. **Dynamic Effects** — Text gradient reveal on scroll, POTD sparkles, hover data dimming on charts, stacking cards (if 4+ value bets)
7. **Tab-by-Tab Content Refresh** — Apply new card/component system to all remaining tabs (Players, Teams, H2H, Standings, Injuries, History, all Betting tabs)
8. **Polish & Accessibility** — Light mode update, focus-visible rings, touch-device fallbacks, reduced-motion gates, contrast audit, about.html refresh

### Files Affected
- `dashboard/index.html` — primary (CSS + JS + HTML structure, ~8900 lines)
- `dashboard/about.html` — update to match new design language
- No Python builder changes needed (JSON shape stays the same)
- No new dependencies — everything is vanilla CSS/JS

### Accessibility Checklist
- [ ] All text meets 4.5:1 contrast ratio (especially --t2 on --bg)
- [ ] All interactive elements have :focus-visible ring
- [ ] Touch targets minimum 44x44px
- [ ] prefers-reduced-motion gates ALL animations
- [ ] 3D tilt + spotlight disabled on (hover: none)
- [ ] No emojis used as icons — SVG only
- [ ] Every Plotly/Chart.js chart has data-table fallback
- [ ] Tab order matches visual order
- [ ] font-display: swap on Google Fonts import
- [ ] Keyboard navigable through all macro tabs and sub-tabs

---

## 10. Design Preview

Mockup file: `dashboard/design-preview.html` (served at localhost:8082)
Direction B is the approved direction.

---

## 11. Inspirations & References

| Effect | Source | URL |
|--------|--------|-----|
| Mouse spotlight on cards | Aceternity UI Spotlight | https://ui.aceternity.com/components/spotlight |
| 3D tilt hover | Vanilla Tilt.js | https://micku7zu.github.io/vanilla-tilt.js/ |
| Shimmer button | Magic UI Shimmer Button | https://magicui.design/docs/components/shimmer-button |
| Animated notification list | Magic UI Animated List | https://magicui.design/docs/components/animated-list |
| Meteor particles | Aceternity UI Meteors | https://ui.aceternity.com/components/meteors |
| Sparkles | Aceternity UI Sparkles | https://ui.aceternity.com/components/sparkles |
| Scroll-driven card reveals | Chrome Scroll-Driven Animations | https://scroll-driven-animations.style/ |
| @property number animation | CSS-Tricks Counter Animation | https://css-tricks.com/animating-number-counters/ |
| Stacking cards | Scroll-Driven Demos | https://scroll-driven-animations.style/demos/stacking-cards/css/ |
| Text gradient reveal | CSS background-clip: text | Native CSS |

All effects are re-implemented in vanilla JS/CSS — no React, no npm, no build step. Inspired by but not copied from these libraries.

---

## Approved By

- [x] User approved Direction B on 2026-03-13
- [x] Dynamic features approved on 2026-03-13
- [x] Full spec finalized
- [x] Implementation plan complete (`docs/superpowers/plans/2026-03-13-baseline-black-redesign.md`)
