# Baseline Black — Full Visual Redesign Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the NBA Analytics dashboard from its current aurora/dark theme into the "Baseline Black" premium sports brand aesthetic with green+gold identity, hybrid navigation, dynamic interactive effects, and polished motion design.

**Architecture:** Single-file dashboard (`dashboard/index.html`, ~8,889 lines) containing CSS + HTML + JS. All changes are CSS variable swaps, new CSS rules, HTML structure edits, and vanilla JS additions. No build step, no npm, no React. Secondary file `dashboard/about.html` (~991 lines) updated to match. No Python builder changes needed — JSON data shape stays identical.

**Tech Stack:** Vanilla CSS (custom properties, `@keyframes`, `animation-timeline: view()`, `@property`), vanilla JS (mousemove listeners, IntersectionObserver fallback), Chart.js, Plotly.js (lazy-loaded), Google Fonts (Plus Jakarta Sans, JetBrains Mono).

**Spec:** `docs/superpowers/specs/2026-03-13-baseline-black-design.md`

---

## File Map

All changes happen in these two files:

| File | Lines | Role |
|------|-------|------|
| `dashboard/index.html` | ~8,889 | Primary — CSS (lines 30-1160), HTML structure (1160-2060), JS (2060-8889) |
| `dashboard/about.html` | ~991 | Secondary — match new design language |

**Key locations in `dashboard/index.html`:**

| Section | Lines | What's there |
|---------|-------|-------------|
| CSS `:root` vars | 31-68 | Color tokens, fonts, radii, shadows |
| CSS `prefers-reduced-motion` | 69-70 | Already exists, will extend |
| CSS Light mode | 78-82 | `body.light` overrides |
| CSS Background | 219-235 | `.bgl`, `.blob` |
| CSS Card entrance | 237-244 | `@keyframes cardIn` |
| CSS Nav | 247-296 | `.nav`, `.nb`, `.nav-tabs` |
| CSS Inner tabs | 337-348 | `.tab-bar`, `.tab` |
| CSS Layout | 350-361 | `.wrap`, `.page` |
| CSS Cards | 364-373 | `.card` |
| CSS Stat tiles | 376-399 | `.sgrid`, `.sc` |
| HTML Nav | 1165-1203 | `<nav>` with flat tabs |
| HTML Betting dropdown | 1206-1216 | Betting pill tabs |
| HTML Game ticker | 1218-1224 | Scrolling ticker |
| HTML Page sections | 1231-1870 | All `<section id="page-*">` |
| JS Nav functions | 2132-2171 | `showTab`, `showPage`, `switchTab` |
| JS Betting nav | 2806-2888 | `showBettingTab`, `toggleBettingDropdown` |
| JS `_setHtml` | 6047-6066 | Safe DOM setter (MUST use) |
| JS Promise.all | 7108-7131 | Data fetch + all rendering |

**Hard rules (from CLAUDE.md):**
- Use `_setHtml(el, html)` for ALL DOM writes — never raw innerHTML
- Data-dependent UI goes in Promise.all callback, NOT tab-click handlers
- Guard `g.ats||''` before `.includes()`/`.startsWith()`
- `prefers-reduced-motion` gates ALL animations
- No emojis as icons — SVG only
- Security hook blocks Edit when replacement contains literal "innerHTML"

---

## Chunk 1: Foundation (CSS Variables + Typography + Glassmorphic Nav)

### Task 1.1: Update CSS Color Tokens

**Files:**
- Modify: `dashboard/index.html:31-68` (`:root` block)

- [ ] **Step 1: Replace gold token value**

Change `--gold: #F5A623` to `--gold: #E8C547` and `--goldf: #FFCC5C` to `--goldf: #E8C547`. Add `--gold-dark: #D4A745`.

```css
/* OLD */
--gold:     #F5A623;
--goldf:    #FFCC5C;
--gold-d:   rgba(245,166,35,0.10);

/* NEW */
--gold:     #E8C547;
--goldf:    #E8C547;
--gold-dark:#D4A745;
--gold-d:   rgba(232,197,71,0.10);
```

- [ ] **Step 2: Update gold references throughout CSS**

Search for all `rgba(245,166,35,` and replace with `rgba(232,197,71,`. These appear in:
- `.lm-steam-badge` border (line ~123)
- `@keyframes steam-pulse` (line ~121)
- `.drop-badge` (line ~327)
- Any other hardcoded gold rgba values

Run: Search `245,166,35` → replace with `232,197,71`

- [ ] **Step 3: Add gradient tokens to `:root`**

Add these new tokens inside the `:root` block:

```css
--grad-brand: linear-gradient(135deg, #00C896, #E8C547);
--grad-bar:   linear-gradient(90deg, #00C896, #E8C547);
--grad-potd:  linear-gradient(90deg, #E8C547, #D4A745);
--grad-conf-hi: linear-gradient(90deg, #00C896, #00E8A8);
--grad-conf-md: linear-gradient(90deg, #E8C547, #D4A745);
```

- [ ] **Step 4: Update `--card` opacity from 0.055 to 0.04**

```css
/* OLD */
--card:     rgba(255,255,255,0.055);
/* NEW */
--card:     rgba(255,255,255,0.04);
```

- [ ] **Step 5: Verify in browser**

Run: `python -m http.server 8080 --directory dashboard`
Open: `http://localhost:8080`
Expected: Gold elements now warmer/richer (#E8C547), cards slightly more transparent, no broken colors.

- [ ] **Step 6: Commit**

```bash
git add dashboard/index.html
git commit -m "style: update gold token to #E8C547, add gradient vars"
```

---

### Task 1.2: Typography Audit — Mono on All Numbers

**Files:**
- Modify: `dashboard/index.html` (CSS section, scattered through HTML/JS)

- [ ] **Step 1: Add JetBrains Mono weight 700 and 800 to font import**

Update the Google Fonts `<link>` (line ~28):

```html
<!-- OLD -->
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet"/>

<!-- NEW -->
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
```

- [ ] **Step 2: Add global number-styling utility class**

Add after the `:root` block (around line 69):

```css
/* ── TYPOGRAPHY UTILITY ── */
.mono{font-family:var(--fm)!important}
.label-up{font-size:10px;font-weight:700;letter-spacing:.7px;text-transform:uppercase;color:var(--t2)}
```

- [ ] **Step 3: Update `.sv` (stat value) font-weight**

```css
/* OLD */
.sv{font-family:var(--fm);font-size:34px;font-weight:700;...}
/* NEW */
.sv{font-family:var(--fm);font-size:34px;font-weight:800;...}
```

- [ ] **Step 4: Verify in browser**

Check: All stat numbers in hero tiles use JetBrains Mono. Labels use Jakarta Sans.

- [ ] **Step 5: Commit**

```bash
git add dashboard/index.html
git commit -m "style: add font weights 900/700/800, mono utility class"
```

---

### Task 1.3: Glassmorphic Nav Bar

**Files:**
- Modify: `dashboard/index.html:247-255` (`.nav` CSS)

- [ ] **Step 1: Update `.nav` CSS**

```css
/* OLD */
.nav{
  position:fixed;top:0;left:0;right:0;z-index:100;
  display:flex;align-items:center;gap:0;
  height:48px;
  background:rgba(8,9,14,.95);
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid var(--b0);
  padding:0 20px;
}

/* NEW */
.nav{
  position:fixed;top:0;left:0;right:0;z-index:100;
  display:flex;align-items:center;gap:0;
  height:48px;
  background:rgba(8,9,14,0.7);
  backdrop-filter:blur(16px) saturate(1.2);-webkit-backdrop-filter:blur(16px) saturate(1.2);
  border-bottom:1px solid rgba(255,255,255,0.08);
  padding:0 20px;
}
```

Key changes: lower background opacity (0.95→0.7), add `saturate(1.2)`, explicit border color.

- [ ] **Step 2: Update light mode nav override**

Update `body.light .nav` (line ~79) to use lower opacity too:

```css
body.light .nav{background:rgba(232,236,244,.75);backdrop-filter:blur(16px) saturate(1.2);-webkit-backdrop-filter:blur(16px) saturate(1.2)}
```

- [ ] **Step 3: Update logo text to gradient fill**

Replace the `.nav-logo-txt` styling and update the HTML logo:

```css
/* ADD - gradient logo text */
.nav-logo-txt b{
  font-weight:900;
  background:var(--grad-brand);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
}
```

Remove the old `.nav-logo-txt b{font-weight:800;color:var(--t0)}` rule.

- [ ] **Step 4: Update active tab indicator to gradient**

```css
/* OLD */
.nb.active::after{
  content:'';position:absolute;bottom:0;left:14px;right:14px;height:2px;
  background:var(--green);border-radius:2px 2px 0 0
}

/* NEW */
.nb.active::after{
  content:'';position:absolute;bottom:0;left:14px;right:14px;height:2px;
  background:var(--grad-bar);border-radius:2px 2px 0 0
}
```

- [ ] **Step 5: Verify glassmorphic effect in browser**

Scroll the page — content should be visible through the frosted nav. The "B" logo text should show green→gold gradient. Active tab underline should be gradient.

- [ ] **Step 6: Commit**

```bash
git add dashboard/index.html
git commit -m "style: glassmorphic nav bar with gradient logo and tab indicator"
```

---

### Task 1.4: Card System — Gradient Top Bar + Updated Borders

**Files:**
- Modify: `dashboard/index.html:364-373` (`.card` CSS)

- [ ] **Step 1: Update card `::before` to green→gold gradient**

```css
/* OLD */
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,rgba(0,200,150,.4),rgba(79,158,255,.3),transparent);opacity:0;transition:opacity 240ms ease;border-radius:var(--r-card) var(--r-card) 0 0}

/* NEW */
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--grad-bar);opacity:0;transition:opacity 240ms ease;border-radius:var(--r-card) var(--r-card) 0 0}
```

Changes: height 2px→3px, gradient is now green→gold (not green→blue→transparent).

- [ ] **Step 2: Update card border-radius from 18px to 16px**

```css
/* In :root */
--r-card: 16px;
```

- [ ] **Step 3: Add POTD card variant**

Add after the `.card:hover::before` rule:

```css
/* ── PICK OF THE DAY CARD ── */
.card.potd{border-color:rgba(232,197,71,0.3);box-shadow:var(--shadow),0 0 20px rgba(232,197,71,0.08)}
.card.potd::before{height:4px;background:var(--grad-potd);opacity:1}
.card.potd:hover{box-shadow:var(--shadow-h),0 0 28px rgba(232,197,71,0.12)}
```

- [ ] **Step 4: Add stat tile gradient top bar update**

Update `.sc::after` gradient colors to match brand:

```css
.sc.accent-green::after{background:var(--green)}
.sc.accent-gold::after{background:var(--gold)}
```

(These already exist — just verify they use `var()` not hardcoded hex.)

- [ ] **Step 5: Verify cards in browser**

Check: Hover on any card shows 3px green→gold bar. POTD class shows permanent gold bar.

- [ ] **Step 6: Commit**

```bash
git add dashboard/index.html
git commit -m "style: card system with gradient top-bar, POTD gold variant"
```

---

## Chunk 2: Background & Atmosphere (Mesh Blobs + Meteors + Skeletons)

### Task 2.1: Animated Mesh Blobs

**Files:**
- Modify: `dashboard/index.html:219-235` (`.bgl`, `.blob` CSS)

- [ ] **Step 1: Add blob drift keyframes**

Add after the existing `@keyframes skeleton-pulse` (line ~107):

```css
/* ── MESH BLOB DRIFT ── */
@keyframes drift1{0%,100%{transform:translate(0,0)}50%{transform:translate(40px,-30px)}}
@keyframes drift2{0%,100%{transform:translate(0,0)}50%{transform:translate(-35px,25px)}}
@keyframes drift3{0%,100%{transform:translate(0,0)}50%{transform:translate(25px,35px)}}
```

- [ ] **Step 2: Update blob styles with animation and GPU hints**

```css
/* OLD */
.blob{position:absolute;border-radius:50%;filter:blur(100px);pointer-events:none}
.blob:nth-child(1){width:600px;height:350px;top:-5%;left:8%;background:rgba(0,200,150,.13)}
.blob:nth-child(2){width:500px;height:300px;top:28%;right:3%;background:rgba(79,158,255,.11)}
.blob:nth-child(3){width:450px;height:250px;bottom:8%;left:25%;background:rgba(167,139,250,.10)}

/* NEW */
.blob{position:absolute;border-radius:50%;filter:blur(120px);pointer-events:none;will-change:transform}
.blob:nth-child(1){width:600px;height:600px;top:-100px;right:-100px;background:rgba(0,200,150,.07);animation:drift1 25s ease-in-out infinite}
.blob:nth-child(2){width:500px;height:500px;bottom:-150px;left:-100px;background:rgba(232,197,71,.05);animation:drift2 30s ease-in-out infinite}
.blob:nth-child(3){width:400px;height:400px;top:50%;left:40%;background:rgba(0,200,150,.04);animation:drift3 20s ease-in-out infinite}
```

Key changes: bigger blobs, lower opacity, blob 2 is now gold (not blue), drift animations.

- [ ] **Step 3: Gate animations behind reduced-motion**

The existing rule at line 69 (`@media(prefers-reduced-motion:reduce){*,...{animation-duration:.01ms!important}}`) already covers this. Verify it applies to `.blob`.

- [ ] **Step 4: Update light mode blob opacity**

```css
body.light .blob{opacity:.18}
```

(Adjust from current `.25` if needed for the new larger blobs.)

- [ ] **Step 5: Verify in browser**

Blobs should drift slowly, barely noticeable. Green blob top-right, gold blob bottom-left, green blob center.

- [ ] **Step 6: Commit**

```bash
git add dashboard/index.html
git commit -m "style: animated mesh blobs with gold accent, GPU-optimized drift"
```

---

### Task 2.2: Meteor Particles on Hero Strip

**Files:**
- Modify: `dashboard/index.html` (CSS + HTML around stat grid)

- [ ] **Step 1: Add meteor keyframes and styles**

Add to CSS section:

```css
/* ── METEOR PARTICLES ── */
@keyframes meteor{0%{transform:translate(0,0) rotate(-45deg);opacity:0}10%{opacity:1}100%{transform:translate(-300px,300px) rotate(-45deg);opacity:0}}
.meteor-field{position:absolute;inset:0;overflow:hidden;pointer-events:none;z-index:0}
.meteor-field span{position:absolute;width:1px;height:60px;background:linear-gradient(to bottom,rgba(0,200,150,.4),transparent);opacity:0;animation:meteor 4s linear infinite}
.meteor-field span:nth-child(1){top:10%;left:70%;animation-delay:0s}
.meteor-field span:nth-child(2){top:5%;left:40%;animation-delay:1.2s;height:45px}
.meteor-field span:nth-child(3){top:15%;left:85%;animation-delay:2.5s;height:35px}
.meteor-field span:nth-child(4){top:8%;left:55%;animation-delay:3.8s;height:50px}
.meteor-field span:nth-child(5){top:3%;left:25%;animation-delay:0.8s;height:40px}
```

- [ ] **Step 2: Add meteor field to hero stat grid HTML**

Find the `<div class="sgrid"` that contains the 4 hero stat tiles (in `page-today` section). Wrap it with a relative container and add the meteor field:

```html
<div style="position:relative">
  <div class="meteor-field" aria-hidden="true">
    <span></span><span></span><span></span><span></span><span></span>
  </div>
  <div class="sgrid" id="hero-stats">
    <!-- existing stat tiles -->
  </div>
</div>
```

- [ ] **Step 3: Verify meteors are subtle**

They should be barely visible — thin diagonal streaks drifting across the hero section only.

- [ ] **Step 4: Commit**

```bash
git add dashboard/index.html
git commit -m "style: meteor particles on hero stat strip"
```

---

### Task 2.3: Skeleton Loading States

**Files:**
- Modify: `dashboard/index.html` (CSS exists at line ~107, add skeleton HTML + JS show/hide)

- [ ] **Step 1: Add skeleton card templates**

Add HTML right after the hero stat grid (inside `page-today`):

```html
<div id="skeleton-picks" class="skeleton-group" style="display:none">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
    <div class="skeleton" style="height:180px"></div>
    <div class="skeleton" style="height:180px"></div>
    <div class="skeleton" style="height:180px"></div>
    <div class="skeleton" style="height:180px"></div>
  </div>
</div>
```

- [ ] **Step 2: Show skeletons before Promise.all, hide after**

In the JS, just before the `Promise.all` call (around line ~7100), add:

```js
// Show skeleton loading
const _skelPicks=document.getElementById('skeleton-picks');
if(_skelPicks)_skelPicks.style.display='block';
```

Then after Promise.all resolves (inside the `.then` or `await` block), add:

```js
// Hide skeleton loading
if(_skelPicks)_skelPicks.style.display='none';
```

- [ ] **Step 3: Add skeleton styles for hero tiles too**

```css
.skeleton-stat{height:110px;border-radius:var(--r-card)}
```

- [ ] **Step 4: Verify loading state**

Throttle network to Slow 3G in DevTools, reload. Skeleton rectangles should pulse during fetch.

- [ ] **Step 5: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: skeleton loading states during data fetch"
```

---

## Chunk 3: Dynamic Interactive Effects (Spotlight + Tilt + Scroll Reveals)

### Task 3.1: Mouse-Tracking Spotlight on Cards

**Files:**
- Modify: `dashboard/index.html` (CSS + JS)

- [ ] **Step 1: Add spotlight CSS**

```css
/* ── MOUSE SPOTLIGHT ── */
.card{--x:50%;--y:50%}
.card::after{
  content:'';position:absolute;inset:0;border-radius:inherit;
  background:radial-gradient(circle 200px at var(--x) var(--y),rgba(0,200,150,0.08),transparent 60%);
  opacity:0;transition:opacity 300ms ease;pointer-events:none;z-index:1
}
.card:hover::after{opacity:1}
@media(hover:none){.card::after{display:none}}
```

Note: `.card` already has `overflow:hidden` and `position:relative` — the `::after` will be clipped.

But wait — `.card::before` is already used for the gradient top bar. We need to use `::after` for spotlight. Check that `.card::after` isn't already defined. Search for `.card::after` — there's no existing rule, but `.sc::after` is used for stat tiles. Good — `.card::after` is available.

- [ ] **Step 2: Add spotlight JS**

Add in the JS section (after the nav functions, around line ~2920):

```js
// ── MOUSE SPOTLIGHT ──
(function(){
  if(matchMedia('(hover:none)').matches)return;
  document.addEventListener('mousemove',function(e){
    var card=e.target.closest('.card,.sc,.phc');
    if(!card)return;
    var r=card.getBoundingClientRect();
    card.style.setProperty('--x',(e.clientX-r.left)+'px');
    card.style.setProperty('--y',(e.clientY-r.top)+'px');
  },{passive:true});
})();
```

- [ ] **Step 3: Add spotlight to stat tiles too**

```css
.sc{--x:50%;--y:50%}
.sc::before{
  content:'';position:absolute;inset:0;border-radius:inherit;
  background:radial-gradient(circle 180px at var(--x) var(--y),rgba(0,200,150,0.06),transparent 60%);
  opacity:0;transition:opacity 300ms ease;pointer-events:none;z-index:1
}
.sc:hover::before{opacity:1}
@media(hover:none){.sc::before{display:none}}
```

Note: `.sc::after` is already used for the colored top-bar, so we use `::before` for spotlight on stat tiles.

- [ ] **Step 4: Verify spotlight tracks cursor**

Hover over any card — a subtle green glow follows the cursor position.

- [ ] **Step 5: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: mouse-tracking spotlight effect on cards"
```

---

### Task 3.2: 3D Tilt on Hover

**Files:**
- Modify: `dashboard/index.html` (CSS + JS)

- [ ] **Step 1: Add tilt CSS on card parents**

```css
/* ── 3D TILT ── */
.tilt-parent{perspective:800px}
.card,.sc{transition:transform 250ms ease,box-shadow 160ms ease,background 160ms ease,border-color 160ms ease}
@media(hover:none){.card,.sc{transform:none!important}}
```

- [ ] **Step 2: Add tilt JS**

Add after the spotlight IIFE:

```js
// ── 3D TILT ON HOVER ──
(function(){
  if(matchMedia('(hover:none)').matches)return;
  if(matchMedia('(prefers-reduced-motion:reduce)').matches)return;
  var MAX=3; // max degrees
  function handleMove(e){
    var card=e.target.closest('.card,.sc');
    if(!card)return;
    var r=card.getBoundingClientRect();
    var cx=r.left+r.width/2, cy=r.top+r.height/2;
    var rx=((e.clientY-cy)/(r.height/2))*-MAX;
    var ry=((e.clientX-cx)/(r.width/2))*MAX;
    card.style.transform='perspective(800px) rotateX('+rx.toFixed(1)+'deg) rotateY('+ry.toFixed(1)+'deg) translateY(-2px)';
  }
  function handleLeave(e){
    var card=e.target.closest('.card,.sc');
    if(card)card.style.transform='';
  }
  document.addEventListener('mousemove',handleMove,{passive:true});
  document.addEventListener('mouseleave',handleLeave,true);
  // Reset on mouseout from card specifically
  document.addEventListener('mouseout',function(e){
    if(e.target.classList&&(e.target.classList.contains('card')||e.target.classList.contains('sc'))){
      e.target.style.transform='';
    }
  },{passive:true});
})();
```

- [ ] **Step 3: Verify tilt is subtle**

Hover and move cursor across a card — it should tilt 2-3 degrees max, spring back on leave.

- [ ] **Step 4: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: 3D tilt effect on card hover (touch-device safe)"
```

---

### Task 3.3: Scroll-Driven Card Reveals

**Files:**
- Modify: `dashboard/index.html` (CSS)

- [ ] **Step 1: Add scroll-driven animation CSS**

```css
/* ── SCROLL-DRIVEN CARD REVEALS ── */
@keyframes fadeSlideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.card,.sc,.gc{
  animation:fadeSlideUp 0.6s ease both;
  animation-timeline:view();
  animation-range:entry 0% entry 30%;
}
/* Fallback for browsers without scroll-driven animations — just show immediately */
@supports not (animation-timeline:view()){
  .card,.sc,.gc{animation:cardIn .4s cubic-bezier(.16,1,.3,1) both}
}
```

- [ ] **Step 2: Remove old staggered `animation-delay` rules**

Delete lines ~240-244 (the `.card:nth-child(2)` through `.card:nth-child(6)` delay rules), since scroll-driven animations handle timing naturally.

```css
/* DELETE THESE */
.card:nth-child(2),.gc:nth-child(2),.sc:nth-child(2){animation-delay:50ms}
.card:nth-child(3),.gc:nth-child(3),.sc:nth-child(3){animation-delay:.1s}
/* ... etc ... */
```

- [ ] **Step 3: Verify scroll behavior**

Scroll down the page — cards should fade+slide-up as they enter the viewport. On Chrome/Edge this uses native scroll-driven animation. On Firefox it falls back to the existing cardIn animation.

- [ ] **Step 4: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: scroll-driven card reveal animations with fallback"
```

---

## Chunk 4: Navigation Restructure (Hybrid Macro + Sub-Tabs)

### Task 4.1: Macro Tab System (Analytics / Teams & Players / Betting)

**Files:**
- Modify: `dashboard/index.html:1165-1216` (nav HTML), `dashboard/index.html:247-346` (nav CSS), `dashboard/index.html:2806-2910` (nav JS)

This is the most complex task — it changes the entire navigation paradigm.

- [ ] **Step 1: Add sub-tab bar CSS**

Add after the existing `.tab-bar` styles:

```css
/* ── SUB-TAB BAR ── */
.sub-tab-bar{
  display:flex;align-items:center;gap:6px;padding:8px 20px;
  background:rgba(8,9,14,0.6);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border-bottom:1px solid rgba(255,255,255,0.06);
  position:fixed;top:48px;left:0;right:0;z-index:99;
  overflow-x:auto;scrollbar-width:none;
}
.sub-tab-bar::-webkit-scrollbar{display:none}
.sub-tab{
  padding:5px 14px;border-radius:8px;
  border:none;background:transparent;cursor:pointer;
  font-family:var(--f);font-size:12px;font-weight:600;color:var(--t2);
  transition:all 150ms ease;white-space:nowrap
}
.sub-tab:hover{color:var(--t1);background:rgba(255,255,255,0.04)}
.sub-tab.active{background:rgba(0,200,150,0.08);color:var(--green)}
```

- [ ] **Step 2: Replace nav HTML with macro tabs**

Replace the nav-tabs div (lines ~1180-1192) with:

```html
<div class="nav-tabs" role="tablist" aria-label="Dashboard sections" id="main-tablist">
  <button class="nb active" onclick="showMacro('analytics')" id="nb-analytics" role="tab" aria-selected="true">Analytics</button>
  <button class="nb" onclick="showMacro('teams')" id="nb-teams" role="tab" aria-selected="false">Teams & Players</button>
  <button class="nb" onclick="showMacro('betting')" id="nb-betting-macro" role="tab" aria-selected="false">Betting</button>
</div>
```

- [ ] **Step 3: Add sub-tab bar HTML**

Add immediately after the closing `</nav>` tag (before the betting dropdown):

```html
<!-- ═══ SUB-TABS ═══ -->
<div class="sub-tab-bar" id="sub-tab-bar" role="tablist" aria-label="Sub-navigation">
  <!-- Analytics sub-tabs (default) -->
  <button class="sub-tab active" onclick="showSubTab('today')" id="st-today">Today's Picks</button>
  <button class="sub-tab" onclick="showSubTab('standings')" id="st-standings">Standings</button>
  <button class="sub-tab" onclick="showSubTab('rankings')" id="st-rankings">Rankings</button>
  <button class="sub-tab" onclick="showSubTab('history')" id="st-history">History</button>
  <button class="sub-tab" onclick="showSubTab('injuries')" id="st-injuries">Injuries</button>
</div>
```

- [ ] **Step 4: Update `.wrap` padding for double-bar**

```css
/* Two bars: nav (48px) + sub-tab-bar (~40px) + ticker if visible */
.wrap{min-height:100dvh;padding:110px 32px 56px;max-width:1320px;margin:0 auto;position:relative;z-index:10}
```

Also update `.ticker-wrap` top position:

```css
.ticker-wrap{position:fixed;top:88px;...}
```

And if ticker is showing, add extra padding:
```css
.has-ticker .wrap{padding-top:136px!important}
```

- [ ] **Step 5: Write the macro/sub-tab JS**

Add new navigation functions (replace or wrap existing `showFreeTab`):

```js
// ── MACRO + SUB-TAB NAVIGATION ──
var _currentMacro='analytics';
var MACRO_SUBS={
  analytics:['today','standings','rankings','history','injuries'],
  teams:['trends','h2h','players','matchups'],
  betting:['picks','valuebets','props','totals','sharp','linemovement','performance','oddsconverter','tracker']
};
var SUB_TO_PAGE={
  today:'today', standings:'standings', rankings:'rankings', history:'history', injuries:'injuries',
  trends:'trends', h2h:'h2h', players:'players', matchups:'deep',
  picks:'bet-picks', valuebets:'bet-valuebets', props:'bet-props', totals:'bet-totals', sharp:'bet-sharp',
  linemovement:'bet-linemovement', performance:'bet-performance', oddsconverter:'bet-oddsconverter', tracker:'bet-tracker'
};
var SUB_LABELS={
  today:"Today's Picks", standings:'Standings', rankings:'Rankings', history:'Season History', injuries:'Injuries',
  trends:'Team Trends', h2h:'Head-to-Head', players:'Players', matchups:'Matchup Analysis',
  picks:'Picks', valuebets:'Value Bets', props:'Player Props', totals:'Totals', sharp:'Sharp Money',
  linemovement:'Line Movement', performance:'Performance', oddsconverter:'Odds Converter', tracker:'Bet Tracker'
};

function showMacro(macro){
  _currentMacro=macro;
  // Update macro tab active state
  document.querySelectorAll('.nb').forEach(function(b){b.classList.remove('active');b.setAttribute('aria-selected','false')});
  var btn=document.getElementById('nb-'+macro)||document.getElementById('nb-'+macro+'-macro');
  if(btn){btn.classList.add('active');btn.setAttribute('aria-selected','true')}
  // Rebuild sub-tabs
  var bar=document.getElementById('sub-tab-bar');
  if(!bar)return;
  var subs=MACRO_SUBS[macro]||[];
  var html=subs.map(function(s,i){
    return '<button class="sub-tab'+(i===0?' active':'')+'" onclick="showSubTab(\''+s+'\')" id="st-'+s+'">'+SUB_LABELS[s]+'</button>';
  }).join('');
  _setHtml(bar,html);
  // Show first sub-tab
  if(subs.length)showSubTab(subs[0]);
}

function showSubTab(sub){
  // Update sub-tab active state
  document.querySelectorAll('.sub-tab').forEach(function(b){b.classList.remove('active')});
  var btn=document.getElementById('st-'+sub);
  if(btn)btn.classList.add('active');
  // Show corresponding page
  var pageId=SUB_TO_PAGE[sub];
  if(!pageId)return;
  document.querySelectorAll('.page').forEach(function(p){p.classList.remove('active')});
  var page=document.getElementById('page-'+pageId);
  if(page){page.classList.add('active');page.classList.add('tab-fade-in');setTimeout(function(){page.classList.remove('tab-fade-in')},250)}
}
```

- [ ] **Step 6: Remove old betting dropdown system**

Delete the betting dropdown HTML (`#bt-dropdown`, lines ~1206-1216) and the `toggleBettingDropdown()`, `showBettingTab()`, `closeBettingDropdown()` JS functions. Betting is now a macro tab with sub-tabs.

Keep a backward-compat shim:

```js
// Legacy shims
function showBettingTab(tab){showMacro('betting');setTimeout(function(){showSubTab(tab)},50)}
function showFreeTab(tab){
  var macroMap={today:'analytics',players:'teams',standings:'analytics',injuries:'analytics',history:'analytics'};
  var m=macroMap[tab]||'analytics';
  if(_currentMacro!==m)showMacro(m);
  showSubTab(tab);
}

// Initialize on page load
showMacro('analytics');
```

- [ ] **Step 7: Add rankings page section if missing**

Check if `page-rankings` exists. If not, add it to the HTML:

```html
<section id="page-rankings" class="page" role="tabpanel">
  <h2 style="font-size:20px;font-weight:800;margin-bottom:16px">Power Rankings</h2>
  <div id="power-rankings-container"></div>
</section>
```

(The existing power rankings may be inside a tab-content within standings — extract it to its own page.)

- [ ] **Step 8: Verify full navigation flow**

Test: Click Analytics → sub-tabs show (Today, Standings, Rankings, History, Injuries). Click Teams & Players → sub-tabs show (Team Trends, H2H, Players, Matchup Analysis). Click Betting → sub-tabs show (Picks, Value Bets, Props, Sharp Money, Line Movement, Performance, Bet Tracker). Each sub-tab shows the correct page content.

- [ ] **Step 9: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: hybrid macro/sub-tab navigation system"
```

---

## Chunk 5: Hero Section (Slot Machine Ticker + Shimmer Buttons + Notification Feed)

### Task 5.1: Slot Machine Number Ticker

**Files:**
- Modify: `dashboard/index.html` (CSS + JS)

- [ ] **Step 1: Add ticker CSS**

```css
/* ── SLOT MACHINE TICKER ── */
.ticker-num{display:inline-flex;overflow:hidden;line-height:1}
.ticker-col{display:inline-block;height:1em;overflow:hidden}
.ticker-strip{display:flex;flex-direction:column;transition:transform 800ms cubic-bezier(.16,1,.3,1)}
.ticker-strip span{display:block;height:1em;text-align:center}
.ticker-sep,.ticker-unit{display:inline-block}
```

- [ ] **Step 2: Add ticker JS utility**

```js
// ── SLOT MACHINE NUMBER TICKER ──
function tickerHtml(numStr){
  var parts=numStr.split('');
  var html='<span class="ticker-num">';
  var colIdx=0;
  parts.forEach(function(ch){
    if(/\d/.test(ch)){
      var digits='';
      for(var i=0;i<=9;i++)digits+='<span>'+i+'</span>';
      html+='<span class="ticker-col" data-target="'+ch+'" style="--i:'+colIdx+'"><span class="ticker-strip" style="transform:translateY(-'+ch+'em)">'+digits+'</span></span>';
      colIdx++;
    } else {
      html+='<span class="ticker-sep">'+ch+'</span>';
    }
  });
  html+='</span>';
  return html;
}
function animateTickers(){
  document.querySelectorAll('.ticker-col').forEach(function(col,i){
    var strip=col.querySelector('.ticker-strip');
    if(!strip)return;
    var target=col.getAttribute('data-target')||'0';
    strip.style.transform='translateY(0)';
    setTimeout(function(){
      strip.style.transform='translateY(-'+target+'em)';
    },50+(i*80));
  });
}
```

- [ ] **Step 3: Update hero stat rendering to use ticker**

In the Promise.all callback where hero stats are rendered, replace direct number insertion with `tickerHtml()`. Find where `.sv` elements get their values set and wrap them:

```js
// Example: where accuracy gets set
// Wire ticker to real data from performance JSON
var accVal=(perf&&perf.accuracy)?perf.accuracy.toFixed(1):'67.5';
var accEl=document.getElementById('hero-accuracy');
if(accEl)_setHtml(accEl,tickerHtml(accVal)+'<span class="ticker-unit">%</span>');
```

Then call `animateTickers()` after all hero stats are rendered.

- [ ] **Step 4: Verify slot machine effect**

Numbers should "roll" into place from 0, each digit staggered by 80ms.

- [ ] **Step 5: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: slot machine number ticker for hero stats"
```

---

### Task 5.2: Shimmer CTA Buttons

**Files:**
- Modify: `dashboard/index.html` (CSS)

- [ ] **Step 1: Add shimmer keyframes and button class**

```css
/* ── SHIMMER BUTTON ── */
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
.btn-shimmer{
  position:relative;overflow:hidden;
  padding:8px 20px;border-radius:var(--r-pill);
  border:1px solid rgba(232,197,71,0.35);background:transparent;
  color:var(--gold);font-size:12px;font-weight:700;font-family:var(--f);
  cursor:pointer;letter-spacing:.3px;transition:all 200ms ease
}
.btn-shimmer::after{
  content:'';position:absolute;inset:0;
  background:linear-gradient(90deg,transparent 30%,rgba(255,255,255,0.06) 50%,transparent 70%);
  background-size:200% 100%;
  animation:shimmer 3s linear infinite;
  pointer-events:none
}
.btn-shimmer:hover{background:rgba(232,197,71,0.08);border-color:rgba(232,197,71,0.5)}
```

- [ ] **Step 2: Apply shimmer class to key CTA buttons**

Find "See all picks", "View Value Bets" type buttons in the HTML/JS and add `class="btn-shimmer"` to them. These are typically rendered in the JS — search for "See all picks" and "See Value Bets" in the JS code.

- [ ] **Step 3: Commit**

```bash
git add dashboard/index.html
git commit -m "style: shimmer sweep effect on CTA buttons"
```

---

### Task 5.3: Animated Notification Feed

**Files:**
- Modify: `dashboard/index.html` (CSS + HTML + JS)

- [ ] **Step 1: Add notification feed CSS**

```css
/* ── NOTIFICATION FEED ── */
@keyframes slideInRight{from{opacity:0;transform:translateX(30px)}to{opacity:1;transform:translateX(0)}}
.notif-feed{display:flex;flex-direction:column;gap:6px;max-height:200px;overflow:hidden}
.notif-item{
  display:flex;align-items:center;gap:8px;padding:8px 12px;
  background:var(--card);border:1px solid var(--card-border);border-radius:10px;
  font-size:12px;animation:slideInRight .4s ease both
}
.notif-hit{color:var(--green);font-weight:700;font-family:var(--fm)}
.notif-miss{color:var(--red);font-weight:700;font-family:var(--fm)}
.notif-edge{color:var(--gold);font-family:var(--fm);font-size:11px}
```

- [ ] **Step 2: Add notification feed container in Today page**

Add after the hero stat grid section:

```html
<div class="card" style="margin-bottom:20px;padding:14px 18px">
  <div class="label-up" style="margin-bottom:8px">Recent Predictions</div>
  <div class="notif-feed" id="notif-feed"></div>
</div>
```

- [ ] **Step 3: Add notification feed JS (inside Promise.all callback)**

```js
// ── NOTIFICATION FEED ──
(function(){
  var feed=document.getElementById('notif-feed');
  if(!feed||!picks||!picks.length)return;
  // Use yesterday's picks (or today's resolved ones) as notification items
  var items=picks.slice(0,5).map(function(p,i){
    var hit=p.correct!==undefined?p.correct:null;
    var edge=p.edge_pct||p.edge||0;
    var label=hit===true?'<span class="notif-hit">HIT</span>':hit===false?'<span class="notif-miss">MISS</span>':'<span style="color:var(--t2)">PENDING</span>';
    return '<div class="notif-item" style="animation-delay:'+(i*100)+'ms">'
      +label
      +'<span style="color:var(--t1)">'+(p.away_team||'')+' @ '+(p.home_team||'')+'</span>'
      +'<span class="notif-edge" style="margin-left:auto">+'+(typeof edge==='number'?edge.toFixed(1):edge)+'%</span>'
      +'</div>';
  });
  _setHtml(feed,items.join(''));
})();
```

- [ ] **Step 4: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: animated notification feed for recent predictions"
```

---

### Task 5.4: Sparkline Stagger Animation

**Files:**
- Modify: `dashboard/index.html` (CSS)

- [ ] **Step 1: Add sparkline bar stagger CSS**

```css
/* ── SPARKLINE STAGGER ── */
.spark-bar{
  display:inline-block;width:4px;border-radius:2px;
  transform-origin:bottom;transform:scaleY(0);
  animation:sparkGrow 300ms ease forwards;
  animation-delay:calc(var(--i,0) * 50ms);
}
@keyframes sparkGrow{from{transform:scaleY(0)}to{transform:scaleY(1)}}
```

- [ ] **Step 2: Update `_sparklineHtml()` to add stagger delay**

In the existing `_sparklineHtml` helper function, add `style="--i:${index}"` to each bar element so the CSS variable drives the stagger delay.

- [ ] **Step 3: Verify sparklines stagger on load**

Hero stat sparkline bars should grow from bottom, each delayed by 50ms.

- [ ] **Step 4: Commit**

```bash
git add dashboard/index.html
git commit -m "style: sparkline bar stagger animation on load"
```

---

## Chunk 6: Dynamic Text Effects (Gradient Reveal + POTD Sparkles + Chart Dimming)

### Task 6.1: Text Gradient Reveal on Scroll

**Files:**
- Modify: `dashboard/index.html` (CSS)

- [ ] **Step 1: Add gradient text class**

```css
/* ── TEXT GRADIENT REVEAL ── */
.grad-text{
  background:var(--grad-brand);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
}
```

- [ ] **Step 2: Apply to section titles**

Add `class="grad-text"` to key section headings like "Today's Picks", "Value Bets", "Power Rankings". These are rendered in JS — search for the heading text strings and add the class.

- [ ] **Step 3: Commit**

```bash
git add dashboard/index.html
git commit -m "style: gradient text on section headings"
```

---

### Task 6.2: POTD Sparkle Stars

**Files:**
- Modify: `dashboard/index.html` (CSS)

- [ ] **Step 1: Add sparkle keyframes and styles**

```css
/* ── POTD SPARKLES ── */
@keyframes sparkle{0%,100%{opacity:0;transform:scale(0)}50%{opacity:1;transform:scale(1)}}
.card.potd .sparkle{position:absolute;width:6px;height:6px;pointer-events:none;z-index:2}
.card.potd .sparkle::before{
  content:'';position:absolute;inset:0;
  background:var(--gold);clip-path:polygon(50% 0%,61% 35%,98% 35%,68% 57%,79% 91%,50% 70%,21% 91%,32% 57%,2% 35%,39% 35%);
  animation:sparkle 2s ease-in-out infinite;
}
.card.potd .sparkle:nth-child(1){top:8px;right:20px;animation-delay:0s}
.card.potd .sparkle:nth-child(2){top:40%;left:10px;animation-delay:.5s}
.card.potd .sparkle:nth-child(3){bottom:15px;right:40%;animation-delay:1s}
.card.potd .sparkle:nth-child(4){top:20%;right:10%;animation-delay:1.5s}
```

- [ ] **Step 2: Add sparkle elements in POTD card rendering**

In the JS where the POTD card is built, prepend sparkle divs. Find the POTD card building code and add:

```js
var sparkles='<div class="sparkle"></div><div class="sparkle"></div><div class="sparkle"></div><div class="sparkle"></div>';
// Prepend to the POTD card HTML
```

Note: The POTD card should also have `class="card potd"` instead of just `class="card"`.

- [ ] **Step 3: Replace star emoji with SVG in POTD label**

Search for `⭐` in JS/HTML and replace with an inline SVG star:

```js
var starSvg='<svg width="12" height="12" viewBox="0 0 24 24" fill="var(--gold)" xmlns="http://www.w3.org/2000/svg"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>';
```

- [ ] **Step 4: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: POTD sparkle stars and SVG star icon"
```

---

### Task 6.3: Hover Data Dimming on Charts

**Files:**
- Modify: `dashboard/index.html` (JS — Chart.js config)

- [ ] **Step 1: Add hover dimming plugin for Chart.js**

Add as a global Chart.js plugin:

```js
// ── CHART HOVER DIMMING ──
if(typeof Chart!=='undefined'){
  Chart.register({
    id:'hoverDim',
    beforeDraw:function(chart){
      var active=chart.getActiveElements();
      if(!active.length){
        chart.data.datasets.forEach(function(ds){ds.borderColor=ds._origColor||ds.borderColor;ds.backgroundColor=ds._origBg||ds.backgroundColor});
        return;
      }
      var activeIdx=active[0].datasetIndex;
      chart.data.datasets.forEach(function(ds,i){
        if(!ds._origColor){ds._origColor=ds.borderColor;ds._origBg=ds.backgroundColor}
        if(i===activeIdx){
          ds.borderColor=ds._origColor;ds.backgroundColor=ds._origBg;ds.borderWidth=3;
        } else {
          var c=ds._origColor;
          // Handle both rgba(...) and hex color formats
          if(typeof c==='string'&&c.indexOf('rgba')===0){
            ds.borderColor=c.replace(/[\d.]+\)$/,'0.3)');
          } else if(typeof c==='string'){
            ds.borderColor=c;ds.borderOpacity=0.3;
          }
          ds.borderWidth=1;
        }
      });
    }
  });
}
```

- [ ] **Step 2: Verify on performance charts**

Hover over a line in any Chart.js chart — other lines should dim to 30% opacity.

- [ ] **Step 3: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: hover data dimming on Chart.js charts"
```

---

## Chunk 7: Tab-by-Tab Content Polish

### Task 7.1: Apply Card Styles to All Remaining Tabs

**Files:**
- Modify: `dashboard/index.html` (HTML sections for each tab)

- [ ] **Step 1: Audit all page sections**

Go through each `<section id="page-*">` and verify:
- Cards use `.card` class (not inline styles for backgrounds)
- Numbers use `font-family:var(--fm)` (mono)
- Labels use `.label-up` class where appropriate
- Gold accents on money-related elements
- Green accents on performance elements

- [ ] **Step 2: Update Standings table styles**

Ensure standings table header uses `var(--t2)` uppercase labels, numbers use mono font, win% values are color-coded (green/red).

- [ ] **Step 3: Update Injury cards**

Apply `.card` class with impact badges using `.vb-edge.hi` (high = red), `.vb-edge.md` (medium = gold).

- [ ] **Step 4: Update Value Bets cards**

Add `.card.potd` treatment to the highest-edge value bet. Kelly row should use gold accent (`var(--gold)` not `var(--green)` for the monetary element).

- [ ] **Step 5: Verify all tabs visually**

Click through every sub-tab and verify consistent card style, typography, and color usage.

- [ ] **Step 6: Commit**

```bash
git add dashboard/index.html
git commit -m "style: apply Baseline Black card system to all tabs"
```

---

## Chunk 8: Polish & Accessibility

### Task 8.1: Focus Visible Rings

**Files:**
- Modify: `dashboard/index.html` (CSS)

- [ ] **Step 1: Add focus-visible styles**

```css
/* ── FOCUS VISIBLE ── */
:focus-visible{outline:2px solid var(--green);outline-offset:2px;border-radius:4px}
button:focus-visible,a:focus-visible,.nb:focus-visible,.sub-tab:focus-visible,.tab:focus-visible{outline:2px solid var(--green);outline-offset:2px}
```

- [ ] **Step 2: Verify keyboard navigation**

Tab through the page — every interactive element should show a green outline ring.

- [ ] **Step 3: Commit**

```bash
git add dashboard/index.html
git commit -m "a11y: focus-visible rings on all interactive elements"
```

---

### Task 8.2: Reduced Motion Gates

**Files:**
- Modify: `dashboard/index.html` (CSS)

- [ ] **Step 1: Verify all new animations are gated**

The existing rule at line 69 covers `animation-duration` and `transition-duration` globally. But verify that:
- Meteor particles are hidden: add `@media(prefers-reduced-motion:reduce){.meteor-field{display:none}}`
- Shimmer stops: covered by global rule
- Blob drift stops: covered by global rule
- Tilt is disabled: add check in JS `if(matchMedia('(prefers-reduced-motion:reduce)').matches)return;`

- [ ] **Step 2: Add explicit reduced-motion overrides**

```css
@media(prefers-reduced-motion:reduce){
  .meteor-field{display:none}
  .card.potd .sparkle{display:none}
  .ticker-strip{transition:none!important}
  .notif-item{animation:none!important;opacity:1!important;transform:none!important}
}
```

- [ ] **Step 3: Commit**

```bash
git add dashboard/index.html
git commit -m "a11y: reduced-motion gates for all dynamic effects"
```

---

### Task 8.3: Light Mode Update

**Files:**
- Modify: `dashboard/index.html` (CSS light mode block)

- [ ] **Step 1: Update light mode variables for new gold**

```css
body.light{
  ...
  --gold:#C07818; /* darken for light backgrounds */
  --gold-dark:#A86614;
  --gold-d:rgba(192,120,24,0.10);
  ...
}
```

- [ ] **Step 2: Update light mode sub-tab bar**

```css
body.light .sub-tab-bar{background:rgba(232,236,244,0.7);border-bottom-color:rgba(0,0,0,0.08)}
body.light .sub-tab.active{background:rgba(0,145,106,0.08);color:var(--green)}
```

- [ ] **Step 3: Verify light mode contrast**

Toggle to light mode. All text should be readable. Run a contrast check on `--t2` against `--bg`.

- [ ] **Step 4: Commit**

```bash
git add dashboard/index.html
git commit -m "style: light mode update for Baseline Black palette"
```

---

### Task 8.4: About Page Update

**Files:**
- Modify: `dashboard/about.html`

- [ ] **Step 1: Copy updated CSS tokens**

Mirror the `:root` changes (gold token, gradient vars) into about.html's `<style>` block.

- [ ] **Step 2: Update gradient references**

Ensure any green→blue gradients are changed to green→gold to match the brand.

- [ ] **Step 3: Verify visual consistency**

Open about.html — it should feel like the same brand as the dashboard.

- [ ] **Step 4: Commit**

```bash
git add dashboard/about.html
git commit -m "style: about.html aligned with Baseline Black design"
```

---

### Task 8.5: Contrast Audit

**Files:**
- Modify: `dashboard/index.html` (if adjustments needed)

- [ ] **Step 1: Check key color pairs**

| Text | Background | Required Ratio | Check |
|------|-----------|----------------|-------|
| `--t0` (#F1F5FF) | `--bg` (#08090E) | 4.5:1 | ✓ (high contrast) |
| `--t1` (#8892AA) | `--bg` (#08090E) | 4.5:1 | Check |
| `--t2` (#6B7A99) | `--bg` (#08090E) | 4.5:1 | Check — may need lightening |
| `--green` (#00C896) | `--bg` (#08090E) | 4.5:1 | Check |
| `--gold` (#E8C547) | `--bg` (#08090E) | 4.5:1 | Check |

- [ ] **Step 2: Adjust any failing colors**

If `--t2` fails contrast, lighten to `#7B89A8` (current value) or `#8494B2`.

- [ ] **Step 3: Commit any adjustments**

```bash
git add dashboard/index.html
git commit -m "a11y: contrast ratio adjustments for WCAG AA"
```

---

### Task 8.6: Update Design Spec Approval

**Files:**
- Modify: `docs/superpowers/specs/2026-03-13-baseline-black-design.md`

- [ ] **Step 1: Mark implementation plan as complete**

Change `- [ ] Implementation plan pending` to `- [x] Implementation plan complete`.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-03-13-baseline-black-design.md
git commit -m "docs: mark implementation plan as complete in spec"
```

---

## Execution Order Summary

| Phase | Tasks | Estimated Steps |
|-------|-------|----------------|
| 1. Foundation | 1.1-1.4 | ~24 steps |
| 2. Background & Atmosphere | 2.1-2.3 | ~16 steps |
| 3. Dynamic Effects | 3.1-3.3 | ~14 steps |
| 4. Navigation | 4.1 | ~9 steps |
| 5. Hero Section | 5.1-5.3 | ~12 steps |
| 6. Text Effects | 6.1-6.3 | ~9 steps |
| 7. Tab Polish | 7.1 | ~6 steps |
| 8. Polish & A11y | 8.1-8.6 | ~16 steps |
| **Total** | **19 tasks** | **~106 steps** |

Each phase produces a working, visually coherent state. After each chunk commit, the dashboard should be usable (no broken states between phases).

---

## Deferred Items

These items from the spec are intentionally deferred from this plan:

| Item | Spec Reference | Reason |
|------|---------------|--------|
| Stacking cards effect | Phase 6 | Complex 3D z-stacking requires significant layout changes; pursue as follow-up |
| `tab-fade-in` JS usage | Line 354 keyframe | Already defined in CSS — verify during Task 4.1 nav restructure |
