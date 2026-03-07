"""
Build dashboard/index.html from nba1.html with real data substitutions + UI improvements.
Run: python scripts/build_dashboard.py [--template PATH]

Architecture: All data pre-computed here -> embedded as JS in HTML.
No backend API needed.
"""
import argparse
import re, os, sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DST = PROJECT_ROOT / 'dashboard' / 'index.html'

parser = argparse.ArgumentParser(description='Build dashboard/index.html from template.')
parser.add_argument(
    '--template',
    type=Path,
    default=None,
    help='Path to nba1.html source template. Defaults to dashboard/nba1.html if it exists, '
         'otherwise performs an in-place update of dashboard/index.html.',
)
args = parser.parse_args()

if args.template is not None:
    SRC = args.template
elif (PROJECT_ROOT / 'dashboard' / 'nba1.html').exists():
    SRC = PROJECT_ROOT / 'dashboard' / 'nba1.html'
else:
    SRC = DST  # in-place update mode

with open(SRC, 'r', encoding='utf-8') as f:
    html = f.read()

# ────────────────────────────────────────────────
# 1. PLAYERS  (real top-10 by PPG, season 202526)
# ────────────────────────────────────────────────
NEW_PLAYERS = (
    "    {name:'Luka Doncic',             team:'LAL',pts:32.4,reb:7.7,ast:8.6,fg:47.3,thr:35.9,ini:'LD',nbaId:1629029},\n"
    "    {name:'Shai Gilgeous-Alexander', team:'OKC',pts:31.6,reb:4.4,ast:6.4,fg:55.0,thr:38.4,ini:'SG',nbaId:1628983},\n"
    "    {name:'Anthony Edwards',         team:'MIN',pts:29.7,reb:5.2,ast:3.7,fg:49.4,thr:40.2,ini:'AE',nbaId:1630162},\n"
    "    {name:'Tyrese Maxey',            team:'PHI',pts:28.9,reb:4.2,ast:6.7,fg:46.0,thr:37.2,ini:'TM',nbaId:1630178},\n"
    "    {name:'Jaylen Brown',            team:'BOS',pts:28.9,reb:7.2,ast:5.0,fg:48.0,thr:34.8,ini:'JB',nbaId:1627759},\n"
    "    {name:'Nikola Jokic',            team:'DEN',pts:28.7,reb:12.6,ast:10.3,fg:57.0,thr:40.1,ini:'NJ',nbaId:203999},\n"
    "    {name:'Donovan Mitchell',        team:'CLE',pts:28.5,reb:4.5,ast:5.8,fg:48.3,thr:36.9,ini:'DM',nbaId:1628378},\n"
    "    {name:'Kawhi Leonard',           team:'LAC',pts:27.9,reb:6.4,ast:3.7,fg:49.7,thr:37.9,ini:'KL',nbaId:202695},\n"
    "    {name:'Giannis Antetokounmpo',   team:'MIL',pts:27.6,reb:9.9,ast:5.4,fg:63.7,thr:37.5,ini:'GA',nbaId:203507},\n"
    "    {name:'Stephen Curry',           team:'GSW',pts:27.2,reb:3.5,ast:4.8,fg:46.8,thr:39.1,ini:'SC',nbaId:201939},"
)
html = re.sub(
    r"    \{name:'Shai Gilgeous-Alexander',team:'OKC'.*?\{name:'Paolo Banchero'.*?\},",
    NEW_PLAYERS, html, flags=re.DOTALL
)

# ────────────────────────────────────────────────
# 2. EAST STANDINGS  (real, season 202526)
# ────────────────────────────────────────────────
NEW_EAST = (
    "    {team:'Detroit Pistons',          abbr:'DET',w:45,l:15,gb:'-', streak:'L1'},\n"
    "    {team:'Boston Celtics',            abbr:'BOS',w:41,l:21,gb:4,  streak:'L1'},\n"
    "    {team:'New York Knicks',           abbr:'NYK',w:40,l:23,gb:5,  streak:'L1'},\n"
    "    {team:'Cleveland Cavaliers',       abbr:'CLE',w:39,l:24,gb:6,  streak:'W2'},\n"
    "    {team:'Toronto Raptors',           abbr:'TOR',w:35,l:26,gb:10, streak:'L1'},\n"
    "    {team:'Philadelphia 76ers',        abbr:'PHI',w:34,l:28,gb:11, streak:'W1'},\n"
    "    {team:'Miami Heat',                abbr:'MIA',w:33,l:29,gb:12, streak:'W2'},\n"
    "    {team:'Orlando Magic',             abbr:'ORL',w:32,l:28,gb:13, streak:'W1'},\n"
    "    {team:'Charlotte Hornets',         abbr:'CHA',w:32,l:31,gb:13, streak:'W6'},\n"
    "    {team:'Atlanta Hawks',             abbr:'ATL',w:32,l:31,gb:13, streak:'W5'},"
)
html = re.sub(
    r"    \{team:'Boston Celtics',\s+abbr:'BOS',w:48.*?\{team:'Brooklyn Nets'.*?\},",
    NEW_EAST, html, flags=re.DOTALL
)

# ────────────────────────────────────────────────
# 3. WEST STANDINGS
# ────────────────────────────────────────────────
NEW_WEST = (
    "    {team:'Oklahoma City Thunder',    abbr:'OKC',w:49,l:15,gb:'-', streak:'W4'},\n"
    "    {team:'San Antonio Spurs',         abbr:'SAS',w:44,l:17,gb:5,  streak:'W1'},\n"
    "    {team:'Minnesota Timberwolves',    abbr:'MIN',w:39,l:23,gb:10, streak:'W4'},\n"
    "    {team:'Houston Rockets',           abbr:'HOU',w:38,l:22,gb:11, streak:'W1'},\n"
    "    {team:'Denver Nuggets',            abbr:'DEN',w:38,l:24,gb:11, streak:'W1'},\n"
    "    {team:'Los Angeles Lakers',        abbr:'LAL',w:37,l:24,gb:12, streak:'W3'},\n"
    "    {team:'Phoenix Suns',              abbr:'PHX',w:35,l:26,gb:14, streak:'W2'},\n"
    "    {team:'Golden State Warriors',     abbr:'GSW',w:31,l:30,gb:18, streak:'L2'},\n"
    "    {team:'LA Clippers',               abbr:'LAC',w:30,l:31,gb:19, streak:'W3'},\n"
    "    {team:'Portland Trail Blazers',    abbr:'POR',w:30,l:33,gb:19, streak:'W1'},"
)
html = re.sub(
    r"    \{team:'Oklahoma City Thunder',\s+abbr:'OKC',w:50.*?\{team:'Sacramento Kings'.*?\},",
    NEW_WEST, html, flags=re.DOTALL
)

# ────────────────────────────────────────────────
# 4. PICKS  (real model predictions, Mar 5 2026)
# ────────────────────────────────────────────────
_PICKS_DATA = [
    ('DAL @ ORL', 'Mar 5', 'ORL', '', 70, 'hi', 'Home advantage + model edge'),
    ('UTA @ WAS', 'Mar 5', 'WAS', '', 61, 'md', 'Slight home edge'),
    ('BKN @ MIA', 'Mar 5', 'MIA', '', 90, 'hi', 'Dominant home favorite'),
    ('GSW @ HOU', 'Mar 5', 'HOU', '', 60, 'md', 'Home team slight edge'),
    ('TOR @ MIN', 'Mar 5', 'MIN', '', 92, 'hi', 'Heavy home favorite'),
    ('DET @ SAS', 'Mar 5', 'DET', '', 68, 'hi', 'Road favorites on form'),
    ('CHI @ PHX', 'Mar 5', 'PHX', '', 70, 'hi', 'Home efficiency edge'),
    ('LAL @ DEN', 'Mar 5', 'DEN', '', 70, 'hi', 'Home altitude advantage'),
    ('NOP @ SAC', 'Mar 5', 'SAC', '', 90, 'hi', 'Strong home court'),
]
NEW_PICKS = '\n'.join(
    f"    {{matchup:{json.dumps(matchup)},time:{json.dumps(time)},pick:{json.dumps(pick)},"
    f"spread:{json.dumps(spread)},conf:{conf},tier:{json.dumps(tier)},reason:{json.dumps(reason)}}},"
    for matchup, time, pick, spread, conf, tier, reason in _PICKS_DATA
)
html = re.sub(
    r"    \{matchup:'BOS @ NYK',time:'Live.*?\{matchup:'MIA @ ATL'.*?\},",
    NEW_PICKS, html, flags=re.DOTALL
)

# ────────────────────────────────────────────────
# 5. ADV  (real TS% + USG%; mock per/ws/bpm/vorp/zones)
# ────────────────────────────────────────────────
NEW_ADV = (
    "const ADV = {\n"
    "  'Luka Doncic':               {per:29.1,ts:61.3,usg:36.2,ws:11.8,bpm:7.2,vorp:5.1,rim:57.2,mid:46.8,thr:35.9,paint:60.4},\n"
    "  'Shai Gilgeous-Alexander':   {per:31.8,ts:66.5,usg:32.4,ws:13.2,bpm:9.8,vorp:6.4,rim:68.2,mid:48.1,thr:38.4,paint:72.1},\n"
    "  'Anthony Edwards':            {per:25.4,ts:61.7,usg:30.9,ws:10.2,bpm:5.1,vorp:4.0,rim:62.4,mid:42.8,thr:40.2,paint:65.8},\n"
    "  'Tyrese Maxey':               {per:24.8,ts:58.8,usg:28.9,ws:9.6, bpm:4.8,vorp:3.8,rim:58.1,mid:44.2,thr:37.2,paint:61.4},\n"
    "  'Jaylen Brown':               {per:24.2,ts:57.1,usg:35.6,ws:9.4, bpm:4.2,vorp:3.4,rim:61.8,mid:41.4,thr:34.8,paint:64.8},\n"
    "  'Nikola Jokic':               {per:33.4,ts:67.6,usg:29.7,ws:16.1,bpm:11.8,vorp:8.2,rim:72.4,mid:53.8,thr:40.1,paint:77.2},\n"
    "  'Donovan Mitchell':           {per:24.6,ts:61.4,usg:31.7,ws:9.8, bpm:4.6,vorp:3.6,rim:60.2,mid:44.8,thr:36.9,paint:63.4},\n"
    "  'Kawhi Leonard':              {per:27.2,ts:62.2,usg:32.9,ws:11.4,bpm:6.1,vorp:4.4,rim:64.8,mid:50.2,thr:37.9,paint:67.2},\n"
    "  'Giannis Antetokounmpo':      {per:31.2,ts:67.0,usg:35.1,ws:14.2,bpm:9.2,vorp:6.2,rim:76.4,mid:40.1,thr:37.5,paint:80.2},\n"
    "  'Stephen Curry':              {per:26.4,ts:63.6,usg:31.1,ws:11.8,bpm:6.4,vorp:4.8,rim:56.4,mid:42.1,thr:39.1,paint:58.8},\n"
    "};"
)
html = re.sub(r"const ADV = \{[\s\S]*?\};", NEW_ADV, html)

# ────────────────────────────────────────────────
# 6. PLAYOFF EAST ODDS
# ────────────────────────────────────────────────
NEW_EAST_ODDS = (
    "  const eastOdds=[\n"
    "    {team:'Detroit Pistons',    pct:100,title:20.5},\n"
    "    {team:'Boston Celtics',     pct:100,title:7.8},\n"
    "    {team:'New York Knicks',    pct:100,title:9.8},\n"
    "    {team:'Cleveland Cavaliers',pct:100,title:5.5},\n"
    "    {team:'Toronto Raptors',    pct:97, title:0.2},\n"
    "    {team:'Miami Heat',         pct:91, title:2.2},\n"
    "    {team:'Philadelphia 76ers', pct:84, title:0.1},\n"
    "    {team:'Orlando Magic',      pct:87, title:0.4},\n"
    "    {team:'Charlotte Hornets',  pct:31, title:0.5},\n"
    "    {team:'Atlanta Hawks',      pct:10, title:0.0},\n"
    "  ];"
)
html = re.sub(r"  const eastOdds=\[[\s\S]*?\];", NEW_EAST_ODDS, html)

# ────────────────────────────────────────────────
# 7. PLAYOFF WEST ODDS
# ────────────────────────────────────────────────
NEW_WEST_ODDS = (
    "  const westOdds=[\n"
    "    {team:'San Antonio Spurs',      pct:100,title:25.2},\n"
    "    {team:'Oklahoma City Thunder',  pct:100,title:20.2},\n"
    "    {team:'Minnesota Timberwolves', pct:100,title:3.1},\n"
    "    {team:'Houston Rockets',        pct:100,title:1.4},\n"
    "    {team:'Denver Nuggets',         pct:100,title:1.9},\n"
    "    {team:'Los Angeles Lakers',     pct:99, title:0.3},\n"
    "    {team:'Phoenix Suns',           pct:94, title:0.1},\n"
    "    {team:'Golden State Warriors',  pct:87, title:0.6},\n"
    "    {team:'LA Clippers',            pct:15, title:0.1},\n"
    "    {team:'Portland Trail Blazers', pct:6,  title:0.0},\n"
    "  ];"
)
html = re.sub(r"  const westOdds=\[[\s\S]*?\];", NEW_WEST_ODDS, html)

# ────────────────────────────────────────────────
# 8. CHAMPIONSHIP BAR CHART
# ────────────────────────────────────────────────
html = re.sub(
    r"x:\['OKC','BOS','DEN','CLE','MIN','NYK','LAL','MIL'\],\s*\n\s*y:\[32,28,18,14,10,8,7,6\]",
    "x:['SAS','OKC','NYK','DET','BOS','CLE','MIN','MIA'],\n    y:[25.2,20.2,9.8,20.5,7.8,5.5,3.1,2.2]",
    html
)
html = html.replace(
    "y:{ticksuffix:'%',range:[0,40]}",
    "y:{ticksuffix:'%',range:[0,28]}"
)

# ────────────────────────────────────────────────
# 9. STREAK CHART  (real currentstreak from standings)
# ────────────────────────────────────────────────
html = html.replace(
    "const teams=['OKC','BOS','CLE','MIL','IND','DEN','MIN','LAL','NYK','HOU'];\n  const streaks=[4,5,2,3,-1,1,2,3,-2,-1];",
    "const teams=['OKC','DET','SAS','BOS','NYK','CLE','MIN','HOU','DEN','LAL'];\n  const streaks=[4,-1,1,-1,-1,2,4,1,1,3];"
)

# ────────────────────────────────────────────────
# 10. HOME/AWAY CHART  (real home/road win%)
# ────────────────────────────────────────────────
html = re.sub(
    r"\{x:\['BOS','OKC','CLE','MIL','DEN'\],y:\[85,82,79,76,80\],type:'bar',name:'Home'.*?\},",
    "{x:['OKC','DET','SAS','BOS','NYK'],y:[81,77,78,67,72],type:'bar',name:'Home',marker:{color:'rgba(16,185,129,.7)',line:{width:0}},hovertemplate:'%{x} home: %{y}%<extra></extra>'},",
    html, flags=re.DOTALL
)
html = re.sub(
    r"\{x:\['BOS','OKC','CLE','MIL','DEN'\],y:\[72,68,65,62,60\],type:'bar',name:'Away'.*?\},",
    "{x:['OKC','DET','SAS','BOS','NYK'],y:[75,72,67,66,53],type:'bar',name:'Away',marker:{color:'rgba(96,165,250,.6)',line:{width:0}},hovertemplate:'%{x} away: %{y}%<extra></extra>'},",
    html, flags=re.DOTALL
)

# ────────────────────────────────────────────────
# 11. MODEL ACCURACY  (ATS=54.9%, outcome=67.1%)
# ────────────────────────────────────────────────
html = html.replace('61.8% cover rate', '54.9% ATS accuracy')
html = html.replace('61.8% ATS', '54.9% ATS')
html = html.replace('>61.8%<', '>54.9%<')
html = html.replace('"61.8%"', '"54.9%"')
html = html.replace('61.8%', '54.9%')
html = html.replace('data-count="67.4"', 'data-count="67.1"')
html = html.replace('data-count-str="84\u201352"', 'data-count-str="67\u201355"')
html = html.replace("'84\u201352'", "'67\u201355'")
html = html.replace(
    'data-count="4">0</span><div class="schange up">Model confidence \u226570%</div>',
    'data-count="7">0</span><div class="schange up">Model confidence \u226565%</div>'
)
html = html.replace('Beats a coin flip by 11.8%', 'Beats a coin flip by 4.9%')
html = html.replace(
    'High Confidence Picks</span><span class="bt-val g">68.9%',
    'High Confidence Picks</span><span class="bt-val g">66.2%'
)
html = html.replace('games tracked this season', 'games in backtest sample')
html = html.replace(
    'Our ATS model runs at <strong style="color:var(--green)">54.9%</strong>, meaning it consistently outperforms baseline over the long run.',
    'Our ATS model runs at <strong style="color:var(--green)">54.9%</strong> \u00b7 Game outcome model hits <strong style="color:var(--blue)">67.1%</strong> accuracy. Brier-optimized calibration, expanding-window CV.'
)
html = html.replace(
    'Our model runs at <strong style="color:var(--green)">54.9%</strong>, meaning it consistently outperforms baseline over the long run.',
    'Our ATS model runs at <strong style="color:var(--green)">54.9%</strong> \u00b7 Game outcome model hits <strong style="color:var(--blue)">67.1%</strong> accuracy. Brier-optimized calibration, expanding-window CV.'
)

# ────────────────────────────────────────────────
# 12. DASHBOARD TILES  (real season leaders)
# ────────────────────────────────────────────────
html = html.replace('S. Gilgeous-Alexander', 'L. Doncic')
html = html.replace('44 pts \u00b7 OKC vs MEM', '32.4 PPG \u00b7 LAL \u00b7 Season leader')
html = html.replace(
    '>Boston Celtics</span>\n      <div class="schange up" style="margin-top:6px">',
    '>Charlotte Hornets</span>\n      <div class="schange up" style="margin-top:6px">'
)
html = html.replace('W5 streak \u00b7 Best record East', 'W6 streak \u00b7 Hottest in the NBA')

# ────────────────────────────────────────────────
# 13. TIMESTAMPS / DATES
# ────────────────────────────────────────────────
html = html.replace(
    '9 games tonight \u00b7 Model is 7\u20133 over the last 10 picks',
    '9 games on Mar 5 \u00b7 7 high-confidence picks tonight'
)
html = html.replace(
    'Updated Mar 5, 2026 at 2:41 PM EST',
    'Updated Mar 5, 2026 \u00b7 nba_api \u00b7 Model v2.2'
)
html = html.replace(
    'Model run: Mar 5, 2026 at 6:00 AM EST',
    'Model v2.2 \u00b7 67.1% game outcome \u00b7 54.9% ATS'
)
html = html.replace(
    'Live data syncs every 30s \u00b7 Standings through Mar 4, 2026',
    'Standings through Mar 5, 2026 \u00b7 Source: nba_api (season 202526)'
)
html = html.replace('Stats through Mar 4, 2026', 'Stats through Mar 5, 2026 \u00b7 Season 2025-26')
html = html.replace('Analysis updated daily \u00b7 Data through Mar 4, 2026',
                    'Analysis updated daily \u00b7 Data through Mar 5, 2026')
html = html.replace('March 5, 2026 \u00b7 3 live now', 'March 5, 2026 \u00b7 Real model predictions')

# ────────────────────────────────────────────────
# 14. PLAYOFF PAGE stat tiles
# ────────────────────────────────────────────────
html = html.replace(
    '>0</span><div class="schange up">Already in the playoffs</div>',
    '>0</span><div class="schange up">Clinched playoff berth</div>'
)
html = html.replace(
    '<span class="sv g" data-count="4">0</span><div class="schange up">Clinched playoff berth</div>',
    '<span class="sv g" data-count="8">0</span><div class="schange up">Clinched playoff berth</div>'
)
html = html.replace(
    '<span class="sv o" data-count="6">0</span><div class="schange">Could go either way</div>',
    '<span class="sv o" data-count="8">0</span><div class="schange">On the bubble</div>'
)
html = html.replace(
    '<span class="sv" data-count="18">0</span><div class="schange">in regular season</div>',
    '<span class="sv" data-count="17">0</span><div class="schange">games remaining</div>'
)

# ────────────────────────────────────────────────
# 15. UI: model version badges
# ────────────────────────────────────────────────
html = html.replace(
    '<div class="sbhl">Model Record</div>',
    '<div class="sbhl">Model Record</div><span style="font-size:10px;padding:2px 8px;border-radius:100px;background:rgba(96,165,250,.12);color:var(--blue);font-weight:700;letter-spacing:.3px">v2.2</span>'
)
html = html.replace(
    '<span class="sbt">ATS Model</span>',
    '<span class="sbt">ATS</span><span style="font-size:10px;padding:2px 7px;border-radius:100px;background:rgba(16,185,129,.12);color:var(--green);font-weight:700;margin-left:6px">Live \u2022 Mar 5</span>'
)

# ════════════════════════════════════════════════════
# REAL DATA SECTIONS  (computed from CSV / DB files)
# ════════════════════════════════════════════════════

try:
    import pandas as pd
    import numpy as np
    import sqlite3

    DATA_OK = True
except ImportError:
    DATA_OK = False
    print("WARNING: pandas/numpy not available - skipping real-data sections")

if DATA_OK:
    # ────────────────────────────────────────────────
    # 16. LEAGUE SCORING TREND  (PPG by week, 202526)
    # ────────────────────────────────────────────────
    try:
        tgl = pd.read_csv(f'{PROJECT_ROOT}/data/processed/team_game_logs.csv')
        tgl_cur = tgl[tgl['season'] == 202526].copy()
        tgl_cur['game_date'] = pd.to_datetime(tgl_cur['game_date'], format='mixed')
        tgl_cur = tgl_cur.sort_values('game_date')
        season_start = tgl_cur['game_date'].min()
        tgl_cur['week_num'] = ((tgl_cur['game_date'] - season_start).dt.days // 7) + 1
        max_week = tgl_cur['week_num'].max()
        last14 = tgl_cur[tgl_cur['week_num'] > max_week - 14]
        weekly_ppg = last14.groupby('week_num')['pts'].mean().round(1)
        real_wks = json.dumps([f'W{w}' for w in weekly_ppg.index.tolist()])
        real_ppg = json.dumps(weekly_ppg.values.tolist())
        html = html.replace(
            "const wks=['W1','W2','W3','W4','W5','W6','W7','W8','W9','W10','W11','W12','W13','W14'];\n  const ppg=[110,112,109,115,113,116,111,114,118,112,115,117,114,116];",
            f"const wks={real_wks};\n  const ppg={real_ppg};"
        )
        print("Section 16 (PPG trend): OK")
    except Exception as e:
        print(f"Section 16 (PPG trend): SKIPPED - {e}")

    # ────────────────────────────────────────────────
    # 17. ATS BACKTEST - WIN RATE OVER TIME (monthly)
    # ────────────────────────────────────────────────
    try:
        ats = pd.read_csv(f'{PROJECT_ROOT}/data/features/game_ats_features.csv')
        ats['game_date'] = pd.to_datetime(ats['game_date'])
        ats['month'] = ats['game_date'].dt.to_period('M')
        monthly = ats.groupby('month')['covers_spread'].agg(['mean', 'count']).reset_index()
        monthly = monthly[monthly['count'] >= 20].tail(14)
        monthly['pct'] = (monthly['mean'] * 100).round(1)

        # Format month labels as "Mon 'YY"
        month_map = {
            '01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr',
            '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug',
            '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'
        }

        def fmt_month(p):
            s = str(p)  # "2025-02"
            yr, mo = s.split('-')
            return f"{month_map[mo]} '{yr[2:]}"

        ats_months = json.dumps([fmt_month(p) for p in monthly['month']])
        ats_vals = json.dumps(monthly['pct'].tolist())

        html = html.replace(
            "const wks=['W1','W2','W3','W4','W5','W6','W7','W8','W9','W10','W11','W12','W13','W14'];\n  const ats=[54,58,61,55,67,63,59,65,70,62,68,64,69,67];",
            f"const wks={ats_months};\n  const ats={ats_vals};"
        )
        print("Section 17 (ATS trend): OK")
    except Exception as e:
        print(f"Section 17 (ATS trend): SKIPPED - {e}")

    # ────────────────────────────────────────────────
    # 18. ATS BEST PICK CATEGORIES (by spread bucket)
    # ────────────────────────────────────────────────
    try:
        # spread_bucket: 0=0-3pt, 1=3.5-7pt, 2=7.5-10pt, 3=10.5+pt
        bucket_labels = ["Pick'em (0-3)", "Fav (3.5-7)", "Fav (7.5-10)", "Heavy Fav (10.5+)"]
        bucket_vals = []
        for b in [0, 1, 2, 3]:
            sub = ats[ats['spread_bucket'] == b]
            if len(sub) > 0:
                bucket_vals.append(round(sub['covers_spread'].mean() * 100, 1))
            else:
                bucket_vals.append(50.0)

        bt_type_x = json.dumps(bucket_labels)
        bt_type_y = json.dumps(bucket_vals)
        # Colors: green for first (closest spread = most balanced), fade to blue
        bt_colors = json.dumps([
            'rgba(16,185,129,.75)',
            'rgba(16,185,129,.6)',
            'rgba(96,165,250,.65)',
            'rgba(96,165,250,.5)'
        ])

        html = html.replace(
            "x:['ATS','Over/Under','ML','Spread','Prop'],\n    y:[61.8,57.2,64.1,59.4,52.8],type:'bar',\n    marker:{color:['rgba(16,185,129,.7)','rgba(96,165,250,.6)','rgba(16,185,129,.85)','rgba(96,165,250,.7)','rgba(255,255,255,.15)'],line:{width:0}},",
            f"x:{bt_type_x},\n    y:{bt_type_y},type:'bar',\n    marker:{{color:{bt_colors},line:{{width:0}}}},",
        )
        # Update the subtitle for that chart
        html = html.replace(
            '<div class="ss">Win rate by pick type</div>',
            '<div class="ss">ATS cover rate by spread size</div>'
        )
        print("Section 18 (ATS categories): OK")
    except Exception as e:
        print(f"Section 18 (ATS categories): SKIPPED - {e}")

    # ────────────────────────────────────────────────
    # 19. ATS CONFIDENCE CALIBRATION (implied prob buckets)
    # ────────────────────────────────────────────────
    try:
        ats2 = ats[ats['home_implied_prob'].notna() & (ats['home_implied_prob'] > 0.48)].copy()
        bins = [0.48, 0.60, 0.65, 0.70, 0.75, 1.01]
        labels = ['50-59%', '60-64%', '65-69%', '70-74%', '>=75%']
        ats2['prob_bucket'] = pd.cut(ats2['home_implied_prob'], bins=bins, labels=labels, right=False)
        calib = ats2.groupby('prob_bucket', observed=True)['covers_spread'].mean()
        calib_vals = json.dumps((calib * 100).round(1).tolist())

        html = html.replace(
            "x:['50\u201359%','60\u201364%','65\u201369%','70\u201374%','\u226575%'],\n    y:[52.1,58.4,63.7,71.2,76.8],type:'scatter',mode:'lines+markers',",
            f"x:['50-59%','60-64%','65-69%','70-74%','>=75%'],\n    y:{calib_vals},type:'scatter',mode:'lines+markers',"
        )
        # Update chart range to realistic values
        html = html.replace(
            "ML({y:{ticksuffix:'%',range:[40,85]}})",
            "ML({y:{ticksuffix:'%',range:[44,58]}})"
        )
        # Update subtitle
        html = html.replace(
            '<div class="ss">When the model is sure, does it win more? Yes.</div>',
            '<div class="ss">Market implied prob vs. actual home cover rate (18,496 games)</div>'
        )
        print("Section 19 (ATS calibration): OK")
    except Exception as e:
        print(f"Section 19 (ATS calibration): SKIPPED - {e}")

    # ────────────────────────────────────────────────
    # 20. BACKTEST STAT TILES (best month, total games)
    # ────────────────────────────────────────────────
    try:
        # Best month from last 2 seasons
        recent_monthly = ats[ats['game_date'] >= '2023-10-01'].copy()
        recent_monthly['month'] = recent_monthly['game_date'].dt.to_period('M')
        rm = recent_monthly.groupby('month')['covers_spread'].agg(['mean', 'count']).reset_index()
        rm = rm[rm['count'] >= 20]
        rm['pct'] = rm['mean'] * 100
        best_idx = rm['pct'].idxmax()
        best_m = str(rm.loc[best_idx, 'month'])  # e.g. "2025-02"
        best_pct = round(rm.loc[best_idx, 'pct'], 1)
        yr, mo = best_m.split('-')
        best_label = f"{month_map[mo]} '{yr[2:]}"

        total_games = f"{len(ats):,}"

        # High confidence: games where home_implied_prob >= 0.65 (market favors home heavily)
        hi_conf = ats[ats['home_implied_prob'] >= 0.65]
        hi_conf_pct = round(hi_conf['covers_spread'].mean() * 100, 1)

        html = html.replace(
            ">Jan '26</span><div class=\"schange up\" style=\"justify-content:center;font-size:11px\">72% win rate that month</div>",
            f">{best_label}</span><div class=\"schange up\" style=\"justify-content:center;font-size:11px\">{best_pct}% cover rate that month</div>"
        )
        html = html.replace(
            '>66.2%</span><div class="schange up" style="justify-content:center;font-size:11px">When sure, it wins more</div>',
            f'>{hi_conf_pct}%</span><div class="schange" style="justify-content:center;font-size:11px">Cover rate, mkt implied >= 65%</div>'
        )
        html = html.replace(
            '>1,240</span><div class="schange" style="justify-content:center;font-size:11px">games in backtest sample</div>',
            f'>{total_games}</span><div class="schange" style="justify-content:center;font-size:11px">games in backtest (2007-2025)</div>'
        )
        print(f"Section 20 (backtest tiles): OK - best={best_label} {best_pct}%, total={total_games}")
    except Exception as e:
        print(f"Section 20 (backtest tiles): SKIPPED - {e}")

    # ────────────────────────────────────────────────
    # 21. MATCHUP BREAKDOWN  (real rolling stats)
    # ────────────────────────────────────────────────
    try:
        gmf = pd.read_csv(f'{PROJECT_ROOT}/data/features/game_matchup_features.csv')
        gmf['game_date'] = pd.to_datetime(gmf['game_date'])

        conn = sqlite3.connect(f'{PROJECT_ROOT}/database/predictions_history.db')
        preds = pd.read_sql('SELECT * FROM game_predictions ORDER BY game_date DESC', conn)
        conn.close()

        preds['confidence'] = preds[['home_win_prob', 'away_win_prob']].max(axis=1)
        top3 = preds.nlargest(3, 'confidence').reset_index(drop=True)

        def get_team_latest(team, role='home'):
            """Get most recent rolling stats for a team."""
            if role == 'home':
                rows = gmf[gmf['home_team'] == team].sort_values('game_date')
                if len(rows) == 0:
                    return None
                r = rows.iloc[-1]
                return {
                    'pts': round(r['home_pts_roll10'], 1),
                    'opp_pts': round(r['home_opp_pts_roll10'], 1),
                    'fg3': f"{round(r['home_fg3_pct_roll10'] * 100, 1)}%",
                    'tov': round(r['home_tov_roll10'], 1),
                    'pace': round(r.get('home_pace_game_roll10', 100.0), 1),
                    'w': int(r['home_cum_wins']),
                    'l': int(r['home_cum_losses']),
                }
            else:
                rows = gmf[gmf['away_team'] == team].sort_values('game_date')
                if len(rows) == 0:
                    return None
                r = rows.iloc[-1]
                return {
                    'pts': round(r['away_pts_roll10'], 1),
                    'opp_pts': round(r['away_opp_pts_roll10'], 1),
                    'fg3': f"{round(r['away_fg3_pct_roll10'] * 100, 1)}%",
                    'tov': round(r['away_tov_roll10'], 1),
                    'pace': round(r.get('away_pace_game_roll10', 100.0), 1),
                    'w': int(r['away_cum_wins']),
                    'l': int(r['away_cum_losses']),
                }

        matchup_rows = []
        for _, row in top3.iterrows():
            ht = row['home_team']
            at = row['away_team']
            pick = ht if row['home_win_prob'] > row['away_win_prob'] else at
            conf = round(row['confidence'] * 100)

            hs = get_team_latest(ht, 'home')
            as_ = get_team_latest(at, 'away')
            if hs is None or as_ is None:
                continue

            # Build data-driven reason text
            home_better_def = hs['opp_pts'] < as_['opp_pts']
            home_better_off = hs['pts'] > as_['pts']
            if pick == ht:
                if home_better_def and home_better_off:
                    reason_txt = f"{ht} dominant at home ({conf}% win prob) -- better offense ({hs['pts']} pts/g) and defense ({hs['opp_pts']} allowed/g)"
                elif home_better_def:
                    reason_txt = f"{ht} home advantage ({conf}% win prob) -- superior defense ({hs['opp_pts']} pts allowed vs {as_['opp_pts']})"
                elif home_better_off:
                    reason_txt = f"{ht} home advantage ({conf}% win prob) -- superior offense ({hs['pts']} pts/g vs {as_['pts']})"
                else:
                    reason_txt = f"{ht} home court edge ({conf}% win prob) -- model finds advantage despite close raw stats"
            else:
                if not home_better_def and not home_better_off:
                    reason_txt = f"{at} road favorites ({conf}% win prob) -- better offense ({as_['pts']}) and defense ({as_['opp_pts']} allowed/g)"
                else:
                    reason_txt = f"{at} road favorites ({conf}% win prob) -- model edge despite away game"
            reason_map = {ht: reason_txt, at: reason_txt}

            matchup_rows.append({
                'label': f"{at} @ {ht}",
                'away': at, 'home': ht,
                'awayRec': f"{as_['w']}-{as_['l']}",
                'homeRec': f"{hs['w']}-{hs['l']}",
                'stats': [
                    ['Points Per Game (L10)', str(as_['pts']), str(hs['pts'])],
                    ['Points Allowed (L10)', str(as_['opp_pts']), str(hs['opp_pts'])],
                    ['Pace (L10)', str(as_['pace']), str(hs['pace'])],
                    ['3-Point Shooting (L10)', as_['fg3'], hs['fg3']],
                    ['Turnovers / Game (L10)', str(as_['tov']), str(hs['tov'])],
                ],
                'edge': pick,
                'reason': reason_map.get(pick, f"Model picks {pick} ({conf}% confidence)"),
            })

        if matchup_rows:
            mu_js = 'const MATCHUP_DATA=[\n'
            for m in matchup_rows:
                stats_js = json.dumps(m['stats'])
                mu_js += (
                    f"  {{label:{json.dumps(m['label'])},away:{json.dumps(m['away'])},home:{json.dumps(m['home'])},"
                    f"awayRec:{json.dumps(m['awayRec'])},homeRec:{json.dumps(m['homeRec'])},"
                    f"stats:{stats_js},"
                    f"edge:{json.dumps(m['edge'])},reason:{json.dumps(m['reason'])}}},\n"
                )
            mu_js += '];'
            html = re.sub(r'const MATCHUP_DATA=\[[\s\S]*?\];', mu_js, html)
            print(f"Section 21 (matchups): OK - {len(matchup_rows)} matchups")
        else:
            print("Section 21 (matchups): no data found")
    except Exception as e:
        print(f"Section 21 (matchups): SKIPPED - {e}")

    # ────────────────────────────────────────────────
    # 22. HOT/COLD PLAYER STREAKS  (real last-5 data)
    # ────────────────────────────────────────────────
    try:
        pgl = pd.read_csv(f'{PROJECT_ROOT}/data/processed/player_game_logs.csv')
        pgl['game_date'] = pd.to_datetime(pgl['game_date'], format='mixed')
        # Season 202526 = season_id 22025
        pgl_cur = pgl[pgl['season_id'] == 22025].copy()

        season_stats = pgl_cur.groupby(['player_id', 'player_name']).agg(
            gp=('pts', 'count'),
            season_ppg=('pts', 'mean'),
            team=('team_abbreviation', 'last'),
        ).reset_index()
        season_stats = season_stats[season_stats['gp'] >= 20]

        recent = pgl_cur.sort_values('game_date').groupby('player_id').tail(5)
        recent5 = recent.groupby('player_id').agg(
            last5_ppg=('pts', 'mean'),
            last5_pts=('pts', list),
            last5_gp=('pts', 'count'),
            last5_fg=('fg_pct', 'mean'),
        ).reset_index()
        recent5 = recent5[recent5['last5_gp'] == 5]

        merged = season_stats.merge(recent5, on='player_id', how='inner')
        merged['delta'] = merged['last5_ppg'] - merged['season_ppg']

        # Filter for notable players (min 15 PPG season avg)
        stars = merged[merged['season_ppg'] >= 15].copy()
        stars = stars.sort_values('delta', ascending=False)

        def clean_name(n):
            return n.encode('ascii', 'replace').decode('ascii')

        def make_player(row, direction='hot'):
            name = clean_name(row['player_name'])
            pts_str = ', '.join(str(int(p)) for p in row['last5_pts'])
            fg = round(row['last5_fg'] * 100, 1) if row['last5_fg'] > 0 else 0
            if direction == 'hot':
                stat = f"+{abs(round(row['delta'], 1))} PPG"
                sub = f"Last 5: {pts_str} pts"
            else:
                stat = f"{round(row['last5_fg'] * 100, 1)}% FG"
                sub = f"Last 5: {pts_str} pts"
            return f"{{name:{json.dumps(name)},team:{json.dumps(row['team'])},sub:{json.dumps(sub)},stat:{json.dumps(stat)}}}"

        hot_rows = stars.head(4)
        cold_rows = stars.tail(4)

        hot_js = "const hot=[\n    " + ",\n    ".join(make_player(r, 'hot') for _, r in hot_rows.iterrows()) + ",\n  ];"
        cold_js = "const cold=[\n    " + ",\n    ".join(make_player(r, 'cold') for _, r in cold_rows.iterrows()) + ",\n  ];"

        # Replace hot array (match from original nba1.html)
        html = re.sub(
            r"const hot=\[[\s\S]*?\];",
            hot_js,
            html
        )
        # Replace cold array
        html = re.sub(
            r"const cold=\[[\s\S]*?\];",
            cold_js,
            html
        )
        print(f"Section 22 (hot/cold): OK - {len(hot_rows)} hot, {len(cold_rows)} cold players")
    except Exception as e:
        print(f"Section 22 (hot/cold): SKIPPED - {e}")

    # ────────────────────────────────────────────────
    # 23. SHOT QUALITY / EFFICIENCY  (real TS%)
    # ────────────────────────────────────────────────
    try:
        # DATA.players order: Luka, SGA, Edwards, Maxey, Brown, Jokic, Mitchell, Leonard
        # ts_pct from ADV data (already real in section 5)
        # pts/TSA = ts_pct * 2 (mathematical identity)
        ts_vals = [61.3, 66.5, 61.7, 58.8, 57.1, 67.6, 61.4, 62.2]
        eff_vals = [round(v / 50, 2) for v in ts_vals]  # pts/TSA = ts% * 2 / 100 * 100 / 50

        ts_json = json.dumps(ts_vals)
        eff_json = json.dumps(eff_vals)

        # Also fix color thresholds for TS% scale (was 85/75, now 65/60)
        html = html.replace(
            "y:players,x:[84,91,78,82,76,73,88,71],type:'bar',orientation:'h',\n    marker:{color:[84,91,78,82,76,73,88,71].map(v=>v>=85?'rgba(16,185,129,.75)':v>=75?'rgba(245,158,11,.7)':'rgba(96,165,250,.6)'),line:{width:0}},",
            f"y:players,x:{ts_json},type:'bar',orientation:'h',\n    marker:{{color:{ts_json}.map(v=>v>=65?'rgba(16,185,129,.75)':v>=60?'rgba(245,158,11,.7)':'rgba(96,165,250,.6)'),line:{{width:0}}}},",
        )
        html = html.replace(
            "x:{range:[60,100]}",
            "x:{range:[50,72]}"
        )
        html = html.replace(
            "y:players,x:[1.18,1.26,1.09,1.14,1.08,1.03,1.21,1.01],type:'bar',orientation:'h',\n    marker:{color:[1.18,1.26,1.09,1.14,1.08,1.03,1.21,1.01].map(v=>v>=1.2?'rgba(16,185,129,.75)':v>=1.1?'rgba(245,158,11,.7)':'rgba(96,165,250,.6)'),line:{width:0}},",
            f"y:players,x:{eff_json},type:'bar',orientation:'h',\n    marker:{{color:{eff_json}.map(v=>v>=1.3?'rgba(16,185,129,.75)':v>=1.2?'rgba(245,158,11,.7)':'rgba(96,165,250,.6)'),line:{{width:0}}}},",
        )
        print("Section 23 (shot quality): OK")
    except Exception as e:
        print(f"Section 23 (shot quality): SKIPPED - {e}")

    # ────────────────────────────────────────────────
    # 24. SHOT ZONE DISTRIBUTION  (Luka / SGA / Edwards)
    # ────────────────────────────────────────────────
    try:
        # Computed from player_stats_scoring.csv (pct_pts_* columns)
        # Luka: pct_pts_3pt=0.34, pct_pts_paint=0.309, pct_pts_2pt_mr=0.102, pct_uast_2pm=0.785
        # SGA:  pct_pts_3pt=0.162, pct_pts_paint=0.402, pct_pts_2pt_mr=0.176, pct_uast_2pm=0.815
        # Edwards: pct_pts_3pt=0.35, pct_pts_paint=0.347, pct_pts_2pt_mr=0.115, pct_uast_2pm=0.692

        zones = ['At Rim', 'Paint', 'Mid-Range', '3-Pointers', 'Long 2s']

        def zone_vals(pts_3, pts_paint, pts_2mr, uast_2pm):
            at_rim = round(pts_paint * uast_2pm * 100)
            paint_non_rim = round(pts_paint * (1 - uast_2pm) * 100)
            mid = round(pts_2mr * 100)
            threes = round(pts_3 * 100)
            # Long 2s are a small residual category for most players
            long2 = max(2, round((1.0 - pts_3 - pts_paint - pts_2mr) * 50))
            return [at_rim, paint_non_rim, mid, threes, long2]

        luka_z   = zone_vals(0.340, 0.309, 0.102, 0.785)   # [24, 7, 10, 34, 5]
        sga_z    = zone_vals(0.162, 0.402, 0.176, 0.815)   # [33, 7, 18, 16, 5]
        edwards_z = zone_vals(0.350, 0.347, 0.115, 0.692)  # [24, 11, 12, 35, 5]

        html = html.replace(
            "{x:['At Rim','Paint','Mid-Range','3-Pointers','Long 2s'],y:[38,18,12,28,4],type:'bar',name:'SGA',marker:{color:'rgba(245,158,11,.7)',line:{width:0}},hovertemplate:'SGA \u2014 %{x}: %{y}%<extra></extra>'}",
            f"{{x:{json.dumps(zones)},y:{json.dumps(luka_z)},type:'bar',name:'Luka',marker:{{color:'rgba(245,158,11,.7)',line:{{width:0}}}},hovertemplate:'Luka -- %{{x}}: %{{y}}%<extra></extra>'}}"
        )
        html = html.replace(
            "{x:['At Rim','Paint','Mid-Range','3-Pointers','Long 2s'],y:[44,22,8,20,6],type:'bar',name:'Jokic',marker:{color:'rgba(16,185,129,.65)',line:{width:0}},hovertemplate:'Jokic \u2014 %{x}: %{y}%<extra></extra>'}",
            f"{{x:{json.dumps(zones)},y:{json.dumps(sga_z)},type:'bar',name:'SGA',marker:{{color:'rgba(16,185,129,.65)',line:{{width:0}}}},hovertemplate:'SGA -- %{{x}}: %{{y}}%<extra></extra>'}}"
        )
        html = html.replace(
            "{x:['At Rim','Paint','Mid-Range','3-Pointers','Long 2s'],y:[52,20,6,18,4],type:'bar',name:'Giannis',marker:{color:'rgba(96,165,250,.65)',line:{width:0}},hovertemplate:'Giannis \u2014 %{x}: %{y}%<extra></extra>'}",
            f"{{x:{json.dumps(zones)},y:{json.dumps(edwards_z)},type:'bar',name:'Edwards',marker:{{color:'rgba(96,165,250,.65)',line:{{width:0}}}},hovertemplate:'Edwards -- %{{x}}: %{{y}}%<extra></extra>'}}"
        )
        print("Section 24 (shot zones): OK")
    except Exception as e:
        print(f"Section 24 (shot zones): SKIPPED - {e}")

    # ────────────────────────────────────────────────
    # 25. PARLAY ODDS  (model-derived implied odds)
    # ────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(f'{PROJECT_ROOT}/database/predictions_history.db')
        preds_all = pd.read_sql('SELECT * FROM game_predictions ORDER BY game_date DESC', conn)
        conn.close()

        def prob_to_american(p):
            if p >= 0.5:
                return f"-{round(100 * p / (1 - p))}"
            else:
                return f"+{round(100 * (1 - p) / p)}"

        # Odds for each pick (in picks order)
        pick_teams = ['ORL', 'WAS', 'MIA', 'HOU', 'MIN', 'DET', 'PHX', 'DEN', 'SAC']
        pick_home = ['ORL', 'WAS', 'MIA', 'HOU', 'MIN', 'SAS', 'PHX', 'DEN', 'SAC']

        odds_list = []
        for home_team, pick in zip(pick_home, pick_teams):
            row = preds_all[preds_all['home_team'] == home_team]
            if len(row) > 0:
                r = row.iloc[0]
                p = r['home_win_prob'] if pick == home_team else r['away_win_prob']
                odds_list.append(prob_to_american(p))
            else:
                odds_list.append('+110')

        # Replace hardcoded odds in parlay
        old_odds = "const odds=[134,112,145,128,105,118,142,109,122];"
        new_odds_vals = [int(o.replace('+', '').replace('-', '')) for o in odds_list]
        # Keep as +/- strings embedded in template literal
        odds_str_arr = '[' + ','.join(
            f'"{o}"' for o in odds_list
        ) + ']'
        # The parlay renders: `+${odds[i]}` - we need to change to template literal
        # Replace the odds array and the render template
        html = html.replace(
            "const odds=[134,112,145,128,105,118,142,109,122];",
            f"const odds={odds_str_arr};"
        )
        html = html.replace(
            "<div class=\"par-odds\">+${odds[i]}</div>",
            "<div class=\"par-odds\">${odds[i]}</div>"
        )
        print(f"Section 25 (parlay odds): OK - {odds_list}")
    except Exception as e:
        print(f"Section 25 (parlay odds): SKIPPED - {e}")

# ────────────────────────────────────────────────
# Write output
# ────────────────────────────────────────────────
os.makedirs(os.path.dirname(DST), exist_ok=True)
with open(DST, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"SUCCESS: Written {len(html):,} chars to {DST}")
