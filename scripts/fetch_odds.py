"""fetch_odds.py — Daily sportsbook data refresh for NBA Analytics Project.

Pulls NBA game lines (moneylines + spreads) and player props (pts/reb/ast)
from The Odds API, then joins them with model projections to produce the
model_vs_odds.csv comparison file used by the website.

Usage (run from project root):
    python scripts/fetch_odds.py

The script is designed to be run daily, either manually or via a scheduler
(e.g., cron, Windows Task Scheduler, or GitHub Actions). See
docs/odds_integration_notes.md for automation setup details.

Environment variable required (in .env at project root):
    ODDS_API_KEY=<your key from the-odds-api.com>

Output files (written to data/odds/):
    game_lines.csv    — one row per game, moneyline + spread
    player_props.csv  — one row per player per stat type
    model_vs_odds.csv — joined comparison with model projections + flags

Requires:
    python-dotenv, requests, pandas (all already in requirements.txt)
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

# ── Setup ──────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("ODDS_API_KEY")
if not API_KEY:
    log.error("ODDS_API_KEY not found in .env — cannot fetch odds. Exiting.")
    sys.exit(1)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT    = "basketball_nba"
REGION   = "us"
ODDS_FMT = "american"
DATE_FMT = "iso"

ODDS_DIR = PROJECT_ROOT / "data" / "odds"
ODDS_DIR.mkdir(parents=True, exist_ok=True)

# Flag thresholds (from agent_task_plan.md)
PROP_FLAG_GAP   = 1.5    # units (pts / reb / ast)
WINPROB_FLAG_PP = 0.05   # 5 percentage points

# ── Team abbreviation mapping ──────────────────────────────────────────────────
# The Odds API uses full city names; our models use NBA 3-letter abbreviations.
# This mapping covers all 30 current NBA teams.
ODDS_TEAM_TO_ABB = {
    "Atlanta Hawks":          "ATL",
    "Boston Celtics":         "BOS",
    "Brooklyn Nets":          "BKN",
    "Charlotte Hornets":      "CHA",
    "Chicago Bulls":          "CHI",
    "Cleveland Cavaliers":    "CLE",
    "Dallas Mavericks":       "DAL",
    "Denver Nuggets":         "DEN",
    "Detroit Pistons":        "DET",
    "Golden State Warriors":  "GSW",
    "Houston Rockets":        "HOU",
    "Indiana Pacers":         "IND",
    "Los Angeles Clippers":   "LAC",
    "Los Angeles Lakers":     "LAL",
    "Memphis Grizzlies":      "MEM",
    "Miami Heat":             "MIA",
    "Milwaukee Bucks":        "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans":   "NOP",
    "New York Knicks":        "NYK",
    "Oklahoma City Thunder":  "OKC",
    "Orlando Magic":          "ORL",
    "Philadelphia 76ers":     "PHI",
    "Phoenix Suns":           "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings":       "SAC",
    "San Antonio Spurs":      "SAS",
    "Toronto Raptors":        "TOR",
    "Utah Jazz":              "UTA",
    "Washington Wizards":     "WAS",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_odds_api(endpoint: str, params: dict) -> list | dict | None:
    """Call The Odds API and return parsed JSON, or None on error."""
    params["apiKey"] = API_KEY
    url = f"{BASE_URL}/{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=20)
        remaining = r.headers.get("x-requests-remaining", "?")
        used = r.headers.get("x-requests-used", "?")
        log.info(f"API call: {endpoint} → {r.status_code} | used={used} remaining={remaining}")
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 401:
            log.error("API key is invalid or expired.")
        elif r.status_code == 429:
            log.warning("Monthly quota exhausted on free tier. Upgrade plan or wait for reset.")
        else:
            log.error(f"Unexpected status {r.status_code}: {r.text[:300]}")
    except requests.RequestException as e:
        log.error(f"Network error calling {url}: {e}")
    return None


def american_odds_to_implied_prob(ml: int | None) -> float | None:
    """Convert American moneyline odds to implied probability (vig-inclusive)."""
    if ml is None:
        return None
    if ml > 0:
        return round(100 / (ml + 100), 4)
    else:
        return round(abs(ml) / (abs(ml) + 100), 4)


def team_name_to_abb(name: str) -> str:
    """Return 3-letter abbreviation, or the original name if not found."""
    result = ODDS_TEAM_TO_ABB.get(name)
    if result is None:
        log.warning(f"Unknown team name from odds API: '{name}' — no abbreviation mapped")
        return name   # pass through so caller can inspect
    return result


# ── Game lines ─────────────────────────────────────────────────────────────────

def fetch_game_lines() -> pd.DataFrame:
    """Fetch h2h and spreads for all upcoming NBA games."""
    data = get_odds_api(
        f"sports/{SPORT}/odds/",
        {
            "regions": REGION,
            "markets": "h2h,spreads",
            "oddsFormat": ODDS_FMT,
            "dateFormat": DATE_FMT,
        }
    )
    if not data:
        log.warning("No game lines returned from API.")
        return pd.DataFrame(columns=["date","home_team","away_team",
                                     "home_moneyline","away_moneyline","spread"])

    rows = []
    for game in data:
        game_date = game.get("commence_time", "")[:10]   # ISO date only
        home_name = game.get("home_team", "")
        away_name = game.get("away_team", "")
        home_abb  = team_name_to_abb(home_name)
        away_abb  = team_name_to_abb(away_name)

        home_ml = away_ml = spread = None

        for bookmaker in game.get("bookmakers", []):
            if bookmaker["key"] != "draftkings":   # prefer DraftKings; fallback handled below
                continue
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == home_name:
                            home_ml = outcome["price"]
                        elif outcome["name"] == away_name:
                            away_ml = outcome["price"]
                elif market["key"] == "spreads":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == home_name:
                            spread = outcome["point"]   # negative = home favored

        # Fallback: if DraftKings not available, use first bookmaker that has both markets
        if home_ml is None and away_ml is None:
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == home_name and home_ml is None:
                                home_ml = outcome["price"]
                            elif outcome["name"] == away_name and away_ml is None:
                                away_ml = outcome["price"]
                    elif market["key"] == "spreads" and spread is None:
                        for outcome in market["outcomes"]:
                            if outcome["name"] == home_name:
                                spread = outcome["point"]
                if home_ml is not None:
                    break

        rows.append({
            "date":            game_date,
            "home_team":       home_abb,
            "away_team":       away_abb,
            "home_moneyline":  home_ml,
            "away_moneyline":  away_ml,
            "spread":          spread,
        })

    df = pd.DataFrame(rows)
    log.info(f"Fetched {len(df)} game lines")
    return df


# ── Player props ───────────────────────────────────────────────────────────────

# Map The Odds API player prop market keys to our stat names
PROP_MARKET_MAP = {
    "player_points":   "pts",
    "player_rebounds": "reb",
    "player_assists":  "ast",
}

def fetch_player_props(event_ids: list[str], game_dates: dict[str, str]) -> pd.DataFrame:
    """Fetch pts/reb/ast player props for the given event IDs.

    NOTE: Player prop markets count against quota more heavily (1 request per
    event × 3 markets = 3 requests per game). With the free tier's 500
    monthly requests, limit to games within the next 2 days to stay under quota.
    """
    if not event_ids:
        log.warning("No event IDs provided for player props fetch.")
        return pd.DataFrame(columns=["date","player_name","stat","line","over_odds","under_odds"])

    rows = []
    for event_id in event_ids:
        markets_str = ",".join(PROP_MARKET_MAP.keys())
        data = get_odds_api(
            f"sports/{SPORT}/events/{event_id}/odds/",
            {
                "regions": REGION,
                "markets": markets_str,
                "oddsFormat": ODDS_FMT,
                "dateFormat": DATE_FMT,
            }
        )
        if not data:
            continue

        game_date = game_dates.get(event_id, data.get("commence_time", "")[:10])

        for bookmaker in data.get("bookmakers", []):
            if bookmaker["key"] != "draftkings":
                continue
            for market in bookmaker.get("markets", []):
                stat = PROP_MARKET_MAP.get(market["key"])
                if stat is None:
                    continue
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description", outcome.get("name", ""))
                    side   = outcome.get("name", "").lower()   # "Over" or "Under"
                    line   = outcome.get("point")
                    price  = outcome.get("price")

                    # Find or create a row for this (player, stat, line)
                    key = (game_date, player, stat, line)
                    existing = [r for r in rows if
                                r["date"] == game_date and
                                r["player_name"] == player and
                                r["stat"] == stat and
                                r["line"] == line]
                    if existing:
                        if "over" in side:
                            existing[0]["over_odds"] = price
                        else:
                            existing[0]["under_odds"] = price
                    else:
                        rows.append({
                            "date":        game_date,
                            "player_name": player,
                            "stat":        stat,
                            "line":        line,
                            "over_odds":   price if "over" in side else None,
                            "under_odds":  price if "under" in side else None,
                        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["date","player_name","stat","line","over_odds","under_odds"])
    log.info(f"Fetched {len(df)} player prop rows")
    return df


# ── Model projections ──────────────────────────────────────────────────────────

def load_model_game_projections() -> pd.DataFrame:
    """Load model win probabilities from game_matchup_features.csv + trained model.

    Returns a DataFrame with columns: date, home_team, away_team, model_win_prob.
    Falls back to feature-based proxy if model cannot be loaded.
    """
    import pickle
    import numpy as np

    features_path = PROJECT_ROOT / "data" / "features" / "game_matchup_features.csv"
    model_path    = PROJECT_ROOT / "models" / "artifacts" / "game_outcome_model.pkl"
    feats_path    = PROJECT_ROOT / "models" / "artifacts" / "game_outcome_features.pkl"

    df = pd.read_csv(features_path)
    # Use only the current season (most recent data)
    today = datetime.now(timezone.utc).date().isoformat()
    recent = df[df["game_date"] >= "2025-10-01"].copy()

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(feats_path, "rb") as f:
            feature_cols = pickle.load(f)

        X = recent[feature_cols].fillna(0)
        probs = model.predict_proba(X)[:, 1]
        recent = recent.copy()
        recent["model_win_prob"] = probs
        log.info(f"Model win probabilities generated for {len(recent)} games")

    except Exception as e:
        log.warning(f"Could not load trained model ({e}). Using feature-based proxy.")
        def proxy_wp(row):
            diff_wp = row.get("diff_cum_win_pct", 0) or 0
            diff_mg = row.get("diff_plus_minus_roll10", 0) or 0
            score   = diff_wp * 2.5 + diff_mg * 0.03
            return round(1 / (1 + np.exp(-score)), 4)
        recent["model_win_prob"] = recent.apply(proxy_wp, axis=1)

    return recent[["game_date","home_team","away_team","model_win_prob"]].rename(
        columns={"game_date": "date"})


def load_model_player_projections() -> pd.DataFrame:
    """Load model pts/reb/ast projections.

    Returns DataFrame with: player_name, stat, model_projection.
    Falls back to rolling 10-game averages if models cannot be loaded.
    """
    import pickle

    pdf_path = PROJECT_ROOT / "data" / "features" / "player_game_features.csv"
    pdf = pd.read_csv(pdf_path, low_memory=False)

    # Use the most recent game per player to get their rolling averages
    pdf_sorted = pdf.sort_values("game_date")
    latest = pdf_sorted.groupby("player_name").last().reset_index()

    rows = []
    for stat in ["pts", "reb", "ast"]:
        roll_col = f"{stat}_roll10"
        if roll_col not in latest.columns:
            log.warning(f"Column {roll_col} not in player features — skipping {stat} projections")
            continue

        # Try loading the trained model
        model_path = PROJECT_ROOT / "models" / "artifacts" / f"player_{stat}_model.pkl"
        feats_path = PROJECT_ROOT / "models" / "artifacts" / f"player_{stat}_features.pkl"
        model_proj = None

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(feats_path, "rb") as f:
                feature_cols = pickle.load(f)
            X = latest[feature_cols].fillna(0)
            model_proj = model.predict(X)
            log.info(f"Player {stat} model projections generated for {len(latest)} players")
        except Exception as e:
            log.warning(f"Could not load player {stat} model ({e}). Using rolling 10-game avg.")

        for i, row in latest.iterrows():
            proj = float(model_proj[i]) if model_proj is not None else row.get(roll_col)
            if pd.notna(proj):
                rows.append({
                    "player_name":       row["player_name"],
                    "stat":              stat,
                    "model_projection":  round(proj, 2),
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["player_name","stat","model_projection"])


# ── model_vs_odds.csv assembly ─────────────────────────────────────────────────

def build_model_vs_odds(
    game_lines: pd.DataFrame,
    player_props: pd.DataFrame,
    game_projections: pd.DataFrame,
    player_projections: pd.DataFrame,
) -> pd.DataFrame:
    """Join odds data with model projections and flag meaningful gaps."""

    rows = []

    # --- Game win probability rows ---
    gl = game_lines.copy()
    gp = game_projections.copy()
    merged = gl.merge(gp, on=["date","home_team","away_team"], how="left")

    for _, row in merged.iterrows():
        model_wp   = row.get("model_win_prob")
        home_ml    = row.get("home_moneyline")
        implied_wp = american_odds_to_implied_prob(home_ml) if pd.notna(home_ml) else None

        if pd.notna(model_wp) and implied_wp is not None:
            gap     = round(float(model_wp) - float(implied_wp), 4)
            flagged = abs(gap) > WINPROB_FLAG_PP
        else:
            gap     = None
            flagged = None

        rows.append({
            "date":             row["date"],
            "home_team":        row["home_team"],
            "away_team":        row["away_team"],
            "stat":             "win_prob",
            "model_projection": model_wp,
            "sportsbook_line":  implied_wp,
            "gap":              gap,
            "flagged":          flagged,
        })

    # --- Player prop rows ---
    if not player_props.empty and not player_projections.empty:
        pp = player_props.copy()
        proj = player_projections.copy()
        merged_pp = pp.merge(proj, on=["player_name","stat"], how="left")

        for _, row in merged_pp.iterrows():
            model_val = row.get("model_projection")
            sb_line   = row.get("line")

            if pd.notna(model_val) and pd.notna(sb_line):
                gap     = round(float(model_val) - float(sb_line), 2)
                flagged = abs(gap) > PROP_FLAG_GAP
            else:
                gap     = None
                flagged = None

            rows.append({
                "date":             row["date"],
                "home_team":        None,
                "away_team":        None,
                "stat":             row["stat"],
                "model_projection": model_val,
                "sportsbook_line":  sb_line,
                "gap":              gap,
                "flagged":          flagged,
            })

        # Add player_name column for prop rows
        # Note: rows list mixes game rows (no player_name) and prop rows.
        # We'll add it as a proper column below.

    mvs = pd.DataFrame(rows)
    if "player_name" not in mvs.columns:
        mvs["player_name"] = None
    return mvs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("=== fetch_odds.py starting ===")

    # 1. Fetch game lines
    game_lines = fetch_game_lines()
    game_lines.to_csv(ODDS_DIR / "game_lines.csv", index=False)
    log.info(f"Saved game_lines.csv ({len(game_lines)} rows)")

    # 2. Build event_id → date mapping for player props (requires raw event list)
    #    Only fetch props for games today and tomorrow to stay inside free tier quota
    today_str = datetime.now(timezone.utc).date().isoformat()
    event_data = get_odds_api(
        f"sports/{SPORT}/events/",
        {"dateFormat": DATE_FMT}
    )
    event_ids   = []
    game_dates  = {}
    if event_data:
        for event in event_data:
            date = event.get("commence_time", "")[:10]
            if date >= today_str:   # only upcoming games
                event_ids.append(event["id"])
                game_dates[event["id"]] = date

    log.info(f"Events to fetch props for: {len(event_ids)}")

    # 3. Fetch player props
    player_props = fetch_player_props(event_ids, game_dates)
    player_props.to_csv(ODDS_DIR / "player_props.csv", index=False)
    log.info(f"Saved player_props.csv ({len(player_props)} rows)")

    # 4. Load model projections
    log.info("Loading model projections...")
    game_projections   = load_model_game_projections()
    player_projections = load_model_player_projections()

    # 5. Build and save comparison file
    mvs = build_model_vs_odds(game_lines, player_props, game_projections, player_projections)
    mvs.to_csv(ODDS_DIR / "model_vs_odds.csv", index=False)
    log.info(f"Saved model_vs_odds.csv ({len(mvs)} rows, "
             f"{mvs['flagged'].sum() if 'flagged' in mvs else 0} flagged)")

    log.info("=== fetch_odds.py complete ===")


if __name__ == "__main__":
    main()
