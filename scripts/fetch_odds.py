"""fetch_odds.py -- Daily sportsbook data refresh for NBA Analytics Project.

Pulls NBA game lines (moneylines + spreads) from the Pinnacle guest API,
then joins them with model projections to produce the model_vs_odds.csv
comparison file used by the website.

Usage (run from project root):
    python scripts/fetch_odds.py

The script is designed to be run daily, either manually or via a scheduler
(e.g., cron, Windows Task Scheduler, or GitHub Actions). See
docs/odds_integration_notes.md for automation setup details.

No API key or authentication required -- uses the Pinnacle guest API.

Output files (written to data/odds/):
    game_lines.csv    -- one row per game, moneyline + spread
    player_props.csv  -- one row per player per stat type (stub, always empty)
    model_vs_odds.csv -- joined comparison with model projections + flags

Requires:
    python-dotenv, requests, pandas (all already in requirements.txt)
"""

import os
import sys
import logging
import datetime as dt
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

# -- Setup ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Add project root to sys.path so src.* modules are importable when
# deserializing pickled model artifacts (e.g. src.models.calibration._CalibratedWrapper).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.models.odds_utils import american_to_implied_prob as _odds_utils_implied_prob

PINNACLE_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
LEAGUE_ID     = 487   # NBA

ODDS_DIR = PROJECT_ROOT / "data" / "odds"
ODDS_DIR.mkdir(parents=True, exist_ok=True)

# Flag thresholds (from agent_task_plan.md)
PROP_FLAG_GAP   = 1.5    # units (pts / reb / ast)
WINPROB_FLAG_PP = 0.05   # 5 percentage points

# -- Team abbreviation mapping -------------------------------------------------
# Pinnacle uses the same full city+nickname team names as The Odds API.
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

# -- Helpers -------------------------------------------------------------------


def get_pinnacle(endpoint: str) -> list | dict | None:
    """Call the Pinnacle guest API and return parsed JSON, or None on error.

    No authentication is required. The guest API is public and keyless.
    """
    url = f"{PINNACLE_BASE}/{endpoint}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NBAAnalytics/1.0",
        "Accept": "application/json",
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        log.info(f"API call: {endpoint} -> {r.status_code}")
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            log.warning("Rate limited by Pinnacle guest API -- back off and retry.")
        else:
            log.error(f"Unexpected status {r.status_code}: {r.text[:300]}")
    except requests.RequestException as e:
        log.error(f"Network error calling {url}: {e}")
    return None


def american_odds_to_implied_prob(ml: int | None) -> float | None:
    """Convert American moneyline odds to implied probability (vig-inclusive).

    Delegates to odds_utils.american_to_implied_prob for the core math
    (includes zero-odds guard). Rounds to 4 decimal places for CSV output.
    """
    result = _odds_utils_implied_prob(ml)
    return round(result, 4) if result is not None else None


def team_name_to_abb(name: str) -> str:
    """Return 3-letter abbreviation, or the original name if not found."""
    result = ODDS_TEAM_TO_ABB.get(name)
    if result is None:
        log.warning(f"Unknown team name from odds API: {name!r} -- no abbreviation mapped")
        return name   # pass through so caller can inspect
    return result


# -- Game lines ----------------------------------------------------------------


def fetch_game_lines() -> pd.DataFrame:
    """Fetch moneylines, spreads, and totals for all upcoming NBA games from Pinnacle."""
    empty = pd.DataFrame(columns=["date", "home_team", "away_team",
                                   "home_moneyline", "away_moneyline", "spread",
                                   "total"])

    matchups_raw = get_pinnacle(f"leagues/{LEAGUE_ID}/matchups")
    if not matchups_raw:
        log.warning("No matchups returned from Pinnacle API.")
        return empty

    markets_raw = get_pinnacle(f"leagues/{LEAGUE_ID}/markets/straight")
    if not markets_raw:
        log.warning("No markets returned from Pinnacle API.")
        return empty

    # Filter matchups to regular h2h games only:
    # - parentId must be None (child records are alternate/period lines)
    # - must have exactly one "home" and one "away" participant
    #   (excludes futures/playoffs with "neutral" alignment)
    matchups = {}
    for m in matchups_raw:
        if m.get("parentId") is not None:
            continue   # skip alternate spreads, period lines, etc.
        participants = m.get("participants", [])
        alignments = {p["alignment"]: p["name"] for p in participants
                      if p.get("alignment") in ("home", "away")}
        if len(alignments) != 2:
            continue   # skip futures, neutral-site placeholders, etc.
        matchups[m["id"]] = {
            "home_team": alignments["home"],
            "away_team": alignments["away"],
            "date":      m.get("startTime", "")[:10],
        }

    # Index markets by matchupId, keeping only period 0 moneyline, spread, and total.
    # For spread and total the API returns multiple alt lines; keep only the first
    # (primary) line per matchup by checking ``mid not in`` before inserting.
    moneylines: dict[int, dict] = {}   # matchupId -> {home: price, away: price}
    spreads:    dict[int, dict] = {}   # matchupId -> {home_points, away_points}
    totals:     dict[int, float] = {}  # matchupId -> total points (over/under)

    for mkt in markets_raw:
        mid    = mkt.get("matchupId")
        mtype  = mkt.get("type")
        period = mkt.get("period")
        if period != 0 or mid not in matchups:
            continue

        prices = mkt.get("prices", [])
        if mtype == "moneyline":
            entry = {}
            for p in prices:
                if p.get("designation") == "home":
                    entry["home"] = p.get("price")
                elif p.get("designation") == "away":
                    entry["away"] = p.get("price")
            moneylines[mid] = entry

        elif mtype == "spread" and mid not in spreads:
            entry = {}
            for p in prices:
                if p.get("designation") == "home":
                    entry["home_points"] = p.get("points")
                    entry["home_price"]  = p.get("price")
                elif p.get("designation") == "away":
                    entry["away_points"] = p.get("points")
            spreads[mid] = entry

        elif mtype == "total" and mid not in totals:
            for p in prices:
                pts = p.get("points")
                if pts is not None:
                    totals[mid] = float(pts)
                    break

    rows = []
    for mid, info in matchups.items():
        ml = moneylines.get(mid, {})
        sp = spreads.get(mid, {})
        rows.append({
            "date":           info["date"],
            "home_team":      team_name_to_abb(info["home_team"]),
            "away_team":      team_name_to_abb(info["away_team"]),
            "home_moneyline": ml.get("home"),
            "away_moneyline": ml.get("away"),
            "spread":         sp.get("home_points"),   # negative = home favored
            "total":          totals.get(mid),          # over/under points
        })

    df = pd.DataFrame(rows)
    log.info(f"Fetched {len(df)} game lines")
    return df


# -- Player props --------------------------------------------------------------

import re
import time

_STAT_LABEL_MAP = {
    "Points":        "PTS",
    "Rebounds":      "REB",
    "Assists":       "AST",
    "3 Point FG":    "3PM",
    "Pts+Rebs+Asts": "PRA",
    "Double+Double": "DD2",
}

_PROP_DESC_RE = re.compile(r"^(.+)\s+\(([^)]+)\)$")

PROPS_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "player_props_lines.csv"
PROPS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

_PROP_BATCH_LIMIT = 80   # cap to avoid hitting rate limits


def fetch_player_props(event_ids: list[str], game_dates: dict[str, str]) -> pd.DataFrame:
    """Fetch player prop lines from the Pinnacle guest API.

    Calls /leagues/487/matchups to get all prop matchups (type="special",
    special.category="Player Props"), then for each prop matchup fetches
    /matchups/{id}/markets/straight to get the over/under line.

    Returns a DataFrame with columns:
        player_name, stat, line, over_price, under_price, game_date

    Also saves the result to data/processed/player_props_lines.csv.
    """
    empty = pd.DataFrame(columns=[
        "player_name", "stat", "line", "over_price", "under_price", "date",
    ])

    try:
        matchups_raw = get_pinnacle(f"leagues/{LEAGUE_ID}/matchups")
        if not matchups_raw:
            log.warning("No matchups returned from Pinnacle -- cannot fetch player props.")
            return empty

        # Filter to player prop specials only
        prop_matchups = [
            m for m in matchups_raw
            if m.get("type") == "special"
            and (m.get("special") or {}).get("category") == "Player Props"
        ]
        log.info(f"Found {len(prop_matchups)} player prop matchups")

        if not prop_matchups:
            return empty

        today_str = dt.date.today().isoformat()
        rows = []

        for matchup in prop_matchups[:_PROP_BATCH_LIMIT]:
            mid = matchup.get("id")
            # Description is in special.description, not top-level description
            special = matchup.get("special") or {}
            description = special.get("description", "")
            start_time = matchup.get("startTime", "")
            game_date = start_time[:10] if start_time else today_str

            m = _PROP_DESC_RE.match(description)
            if not m:
                log.debug(f"Could not parse prop description: {description!r}")
                continue

            player_name = m.group(1).strip()
            stat_raw = m.group(2).strip()
            stat = _STAT_LABEL_MAP.get(stat_raw, stat_raw)

            # Build participant_id -> name map (Over / Under) from the matchup
            participants = matchup.get("participants", [])
            participant_names: dict[int, str] = {
                p["id"]: p["name"].lower()
                for p in participants
                if p.get("id") and p.get("name")
            }

            time.sleep(0.3)

            markets = get_pinnacle(f"matchups/{mid}/markets/straight")
            if not markets:
                log.debug(f"No markets for prop matchup {mid} ({description!r}) -- skipping")
                continue

            # Handle both list and dict responses
            if isinstance(markets, dict):
                markets = [markets]

            over_price = None
            under_price = None
            line = None

            for mkt in markets:
                prices = mkt.get("prices", [])
                for price in prices:
                    pts = price.get("points")
                    if pts is not None and line is None:
                        line = float(pts)
                    p = price.get("price")
                    # Resolve designation from participant name via participantId
                    pid = price.get("participantId")
                    part_name = participant_names.get(pid, "") if pid else ""
                    if not part_name:
                        # Fall back to designation field if present
                        part_name = (price.get("designation") or "").lower()
                    if "over" in part_name:
                        over_price = p
                    elif "under" in part_name:
                        under_price = p

            if line is None:
                log.debug(f"No line found for {description!r} -- skipping")
                continue

            rows.append({
                "player_name": player_name,
                "stat":        stat,
                "line":        line,
                "over_price":  over_price,
                "under_price": under_price,
                "date":        game_date,
            })

        df = pd.DataFrame(rows) if rows else empty.copy()
        log.info(f"Fetched {len(df)} player prop lines from Pinnacle")

        df.to_csv(PROPS_CSV_PATH, index=False)
        log.info(f"Saved player props lines to {PROPS_CSV_PATH}")

        return df

    except Exception as e:
        log.error(f"fetch_player_props failed: {e}")
        return empty


# -- Model projections ---------------------------------------------------------

def load_model_game_projections() -> pd.DataFrame:
    """Load model win probabilities from game_matchup_features.csv + trained model.

    Returns a DataFrame with columns: date, home_team, away_team, model_win_prob.
    Prefers the calibrated model (game_outcome_model_calibrated.pkl) when available,
    falling back to the uncalibrated model, then to a feature-based proxy.
    """
    import pickle
    import numpy as np

    features_path      = PROJECT_ROOT / "data" / "features" / "game_matchup_features.csv"
    calibrated_path    = PROJECT_ROOT / "models" / "artifacts" / "game_outcome_model_calibrated.pkl"
    model_path         = PROJECT_ROOT / "models" / "artifacts" / "game_outcome_model.pkl"
    feats_path         = PROJECT_ROOT / "models" / "artifacts" / "game_outcome_features.pkl"

    df = pd.read_csv(features_path)
    # Use only the current season (most recent data).
    # NBA regular seasons start in mid-October. If today is before June 1,
    # the current season started last October; otherwise it starts this October.
    today = datetime.now(timezone.utc).date()
    if today < dt.date(today.year, 6, 1):
        season_start = dt.date(today.year - 1, 10, 1)
    else:
        season_start = dt.date(today.year, 10, 1)
    season_start_str = season_start.isoformat()
    recent = df[df["game_date"] >= season_start_str].copy()

    try:
        # Prefer calibrated model; fall back to uncalibrated if not found
        try:
            from src.models.game_outcome_model import _CalibrationUnpickler
            with open(calibrated_path, "rb") as f:
                model = _CalibrationUnpickler(f).load()
            log.info("Loaded calibrated game outcome model")
        except FileNotFoundError:
            log.warning(
                "Calibrated model not found at %s -- using uncalibrated model. "
                "Run src/models/calibrate_model.py to generate the calibrated artifact.",
                calibrated_path,
            )
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        with open(feats_path, "rb") as f:
            feature_cols = pickle.load(f)

        X = recent.reindex(columns=feature_cols)  # Pipeline imputer handles NaN (mean strategy)
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
            log.warning(f"Column {roll_col} not in player features -- skipping {stat} projections")
            continue

        # Try loading the trained model
        model_path = PROJECT_ROOT / "models" / "artifacts" / f"player_{stat}_model.pkl"
        feats_path = PROJECT_ROOT / "models" / "artifacts" / f"player_{stat}_features.pkl"
        # model_pred_series: index-aligned Series (keyed by latest.index) so .loc[] is safe
        model_pred_series: "pd.Series | None" = None

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(feats_path, "rb") as f:
                feature_cols = pickle.load(f)
            X = latest[feature_cols]  # Pipeline imputer handles NaN
            # Wrap numpy array as a Series keyed by latest.index -- prevents positional misalignment
            raw_preds = model.predict(X)
            model_pred_series = pd.Series(raw_preds, index=latest.index)
            log.info(f"Player {stat} model projections generated for {len(latest)} players")
            null_count = int(model_pred_series.isna().sum())
            if null_count > 0:
                log.warning(
                    f"Player {stat}: {null_count} null predictions after model.predict -- "
                    "feature columns may not align with training data"
                )
        except Exception as e:
            log.warning(f"Could not load player {stat} model ({e}). Using rolling 10-game avg.")

        for idx, row in latest.iterrows():
            # Use .loc[idx] on the index-aligned Series -- safe even if index is non-sequential
            if model_pred_series is not None:
                proj = float(model_pred_series.loc[idx])
            else:
                proj = row.get(roll_col)
            if pd.notna(proj):
                rows.append({
                    "player_name":       row["player_name"],
                    "stat":              stat,
                    "model_projection":  round(proj, 2),
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["player_name","stat","model_projection"])


# -- model_vs_odds.csv assembly ------------------------------------------------

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


# -- Main ----------------------------------------------------------------------


def main():
    log.info("=== fetch_odds.py starting ===")

    # 1. Fetch game lines
    game_lines = fetch_game_lines()
    game_lines.to_csv(ODDS_DIR / "game_lines.csv", index=False)
    log.info(f"Saved game_lines.csv ({len(game_lines)} rows)")

    # 1b. Log opening lines for CLV tracking
    try:
        from src.models.clv_tracker import CLVTracker
        tracker = CLVTracker()
        n_logged = 0
        for _, row in game_lines.iterrows():
            logged = tracker.log_opening_line(
                game_date=str(row.get("date", "")),
                home_team=str(row.get("home_team", "")),
                away_team=str(row.get("away_team", "")),
                opening_spread=row.get("spread"),
                opening_home_ml=row.get("home_moneyline"),
                opening_away_ml=row.get("away_moneyline"),
            )
            if logged:
                n_logged += 1
        log.info(f"CLV: logged {n_logged} new opening lines for CLV tracking")
    except Exception as e:
        log.warning(f"CLV tracking skipped (non-fatal): {e}")

    # 2. Player props -- fetch live lines from Pinnacle guest API
    player_props = fetch_player_props([], {})
    player_props.to_csv(ODDS_DIR / "player_props.csv", index=False)
    log.info(f"Saved player_props.csv ({len(player_props)} rows)")

    # 3. Load model projections
    log.info("Loading model projections...")
    game_projections   = load_model_game_projections()
    player_projections = load_model_player_projections()

    # 4. Build and save comparison file
    mvs = build_model_vs_odds(game_lines, player_props, game_projections, player_projections)
    mvs.to_csv(ODDS_DIR / "model_vs_odds.csv", index=False)
    flagged_count = int(mvs["flagged"].dropna().sum()) if "flagged" in mvs.columns else 0
    log.info(f"Saved model_vs_odds.csv ({len(mvs)} rows, {flagged_count} flagged)")

    log.info("=== fetch_odds.py complete ===")


if __name__ == "__main__":
    main()
