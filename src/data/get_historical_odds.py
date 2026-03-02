"""
Download and normalize historical NBA betting odds from the Kaggle dataset.

Data source: https://www.kaggle.com/datasets/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024
Coverage: October 2007 through June 2025
Columns: season, date, regular, playoffs, away, home, score_away, score_home,
         whos_favored, spread, total, moneyline_away, moneyline_home, id_spread, id_total

The Kaggle dataset uses lowercase short team codes (e.g. 'gs', 'sa', 'no', 'utah')
mapped here to project 3-letter NBA abbreviations (e.g. 'GSW', 'SAS', 'NOP', 'UTA').
Season is stored as the end year integer (2008 = 2007-08 season), converted to
project format (200708).

Usage:
    from src.data.get_historical_odds import load_and_normalize_odds
    df = load_and_normalize_odds()

Or run as main:
    python src/data/get_historical_odds.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ODDS_DIR = PROJECT_ROOT / "data" / "raw" / "odds"
DEFAULT_ODDS_PATH = ODDS_DIR / "nba_betting_historical.csv"

# ── Team name mapping (Kaggle format -> 3-letter NBA abbreviation) ─────────────
# Actual Kaggle dataset uses lowercase short codes from SportsBookReviewsOnline.
# Inspected from the real file: nba_2008-2025.csv
# Unique values found: atl bkn bos cha chi cle dal den det gs hou ind lac lal
#   mem mia mil min no ny okc orl phi phx por sa sac tor utah wsh
# ESPN full-name format also included for post-Jan 2023 rows.

KAGGLE_TEAM_TO_ABB = {
    # Lowercase short codes (SportsBookReviewsOnline format)
    "atl":  "ATL",
    "bkn":  "BKN",
    "bos":  "BOS",
    "cha":  "CHA",
    "chi":  "CHI",
    "cle":  "CLE",
    "dal":  "DAL",
    "den":  "DEN",
    "det":  "DET",
    "gs":   "GSW",   # Golden State -- short code without 'w'
    "hou":  "HOU",
    "ind":  "IND",
    "lac":  "LAC",
    "lal":  "LAL",
    "mem":  "MEM",
    "mia":  "MIA",
    "mil":  "MIL",
    "min":  "MIN",
    "no":   "NOP",   # New Orleans
    "ny":   "NYK",   # New York Knicks
    "okc":  "OKC",
    "orl":  "ORL",
    "phi":  "PHI",
    "phx":  "PHX",
    "por":  "POR",
    "sa":   "SAS",   # San Antonio -- short code
    "sac":  "SAC",
    "tor":  "TOR",
    "utah": "UTA",   # Utah Jazz
    "wsh":  "WAS",   # Washington
    # Historical / relocated teams
    "nj":   "NJN",   # New Jersey Nets (pre-BKN)
    "nok":  "NOP",   # New Orleans/Oklahoma City (Hornets era)
    "sea":  "SEA",   # Seattle SuperSonics (pre-OKC)
    "van":  "MEM",   # Vancouver Grizzlies (pre-MEM)
    "cha2": "CHA",   # Charlotte Bobcats alt code
    # ESPN full-name format (Jan 2023+)
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


# ── Helper functions ───────────────────────────────────────────────────────────

def _american_to_raw_prob(ml: float) -> float:
    """Raw (vig-inclusive) implied probability from American moneyline odds."""
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return abs(ml) / (abs(ml) + 100.0)


def _no_vig_probs(home_ml: float, away_ml: float) -> tuple:
    """
    Remove bookmaker vig using multiplicative normalization.
    Returns (home_prob, away_prob) that sum to 1.0.
    """
    raw_home = _american_to_raw_prob(home_ml)
    raw_away = _american_to_raw_prob(away_ml)
    total = raw_home + raw_away  # overround (e.g. ~1.04)
    return raw_home / total, raw_away / total


def compute_home_covers_spread(row: pd.Series) -> float:
    """
    Return 1 if home team covered spread, 0 if not, NaN if push.

    id_spread: 1=favorite covered, 0=underdog covered, 2=push
    whos_favored: 'home' or 'away'
    """
    id_sp = row["id_spread"]
    if id_sp == 2:
        return np.nan  # push -- exclude from training and backtest
    if row["whos_favored"] == "home":
        # home was favored: covered means id_spread==1
        return float(id_sp == 1)
    else:
        # away was favored: underdog (home) covered means id_spread==0
        return float(id_sp == 0)


def _normalize_season(season_val) -> int:
    """
    Convert Kaggle season value to project integer format.

    Kaggle stores season as end-year integer: 2008 means the 2007-08 season.
    Project format: 200708.

    Also handles string formats like '2007-08' or '2007-2008'.
    """
    try:
        val = int(season_val)
    except (ValueError, TypeError):
        return 0

    # If 4-digit year (e.g. 2008), convert: 2008 -> 200708
    if 2000 <= val <= 2099:
        prev_year = val - 1
        return int(f"{prev_year}{val % 100:02d}")
    # If already in project format (e.g. 200708), return as-is
    if 100000 <= val <= 999999:
        return val
    return val


def _map_team(name: str, mapping: dict) -> str:
    """Map a team name to 3-letter abbreviation. Returns original if not found."""
    result = mapping.get(name)
    if result is not None:
        return result
    # Try stripped lowercase version
    stripped = str(name).strip().lower()
    result = mapping.get(stripped)
    if result is not None:
        return result
    # Return original so join failures are visible
    return name


def _compute_implied_probs_vectorized(
    home_ml: pd.Series, away_ml: pd.Series
) -> tuple:
    """
    Vectorized no-vig implied probability computation.
    Returns (home_probs, away_probs) as numpy arrays with NaN for missing rows.
    """
    n = len(home_ml)
    home_probs = np.full(n, np.nan)
    away_probs = np.full(n, np.nan)

    both_valid = home_ml.notna() & away_ml.notna()
    valid_mask = both_valid.to_numpy()

    if valid_mask.sum() == 0:
        return home_probs, away_probs

    h = home_ml.to_numpy()
    a = away_ml.to_numpy()

    # Vectorized raw probability computation
    raw_h = np.where(h > 0, 100.0 / (h + 100.0), np.abs(h) / (np.abs(h) + 100.0))
    raw_a = np.where(a > 0, 100.0 / (a + 100.0), np.abs(a) / (np.abs(a) + 100.0))
    total = raw_h + raw_a

    home_probs[valid_mask] = np.round(raw_h[valid_mask] / total[valid_mask], 4)
    away_probs[valid_mask] = np.round(raw_a[valid_mask] / total[valid_mask], 4)

    return home_probs, away_probs


# ── Main public function ───────────────────────────────────────────────────────

def load_and_normalize_odds(
    odds_path: str = None,
) -> pd.DataFrame:
    """
    Load and normalize the Kaggle NBA betting dataset.

    If the file does not exist at odds_path, prints download instructions
    and returns an empty DataFrame with the expected schema.

    Parameters
    ----------
    odds_path : str or None
        Path to the CSV file. Defaults to data/raw/odds/nba_betting_historical.csv.
        Also checks for data/raw/odds/nba_2008-2025.csv as a fallback.

    Returns
    -------
    pd.DataFrame with columns:
        game_date         : str (YYYY-MM-DD)
        home_team         : str (3-letter abbreviation, e.g. 'GSW')
        away_team         : str (3-letter abbreviation, e.g. 'LAL')
        season            : int (project format, e.g. 200708)
        spread            : float (opening line, home team's perspective)
        home_implied_prob : float or NaN (no-vig implied probability)
        away_implied_prob : float or NaN (no-vig implied probability)
        covers_spread     : float (1.0=home covered, 0.0=away covered, NaN=push)
    """
    ODDS_DIR.mkdir(parents=True, exist_ok=True)

    if odds_path is None:
        odds_path = str(DEFAULT_ODDS_PATH)

    # Try fallback filename if canonical name not found
    if not Path(odds_path).exists():
        fallback = ODDS_DIR / "nba_2008-2025.csv"
        if fallback.exists():
            print(f"Using fallback odds file: {fallback}")
            odds_path = str(fallback)

    if not Path(odds_path).exists():
        print("=" * 70)
        print("HISTORICAL ODDS DATA NOT FOUND")
        print("=" * 70)
        print("")
        print("To use historical NBA betting data, download the Kaggle dataset:")
        print("")
        print("  1. Go to:")
        print("     https://www.kaggle.com/datasets/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024")
        print("")
        print("  2. Click 'Download' (free Kaggle account required)")
        print("")
        print("  3. Extract and place the CSV at one of:")
        print(f"     {DEFAULT_ODDS_PATH}")
        print(f"     {ODDS_DIR / 'nba_2008-2025.csv'}")
        print("")
        print("  The file should have columns: season, date, away, home,")
        print("  whos_favored, spread, moneyline_away, moneyline_home, id_spread")
        print("")
        print("Returning empty DataFrame with expected schema.")
        return pd.DataFrame(columns=[
            "game_date", "home_team", "away_team", "season",
            "spread", "home_implied_prob", "away_implied_prob", "covers_spread"
        ])

    raw = pd.read_csv(odds_path, low_memory=False)
    assert raw.shape[0] > 0, f"Loaded file is empty: {odds_path}"

    print(f"Loaded {raw.shape[0]} rows from {odds_path}")

    # ── Rename date column ─────────────────────────────────────────────────
    if "date" in raw.columns:
        raw = raw.rename(columns={"date": "game_date"})
    elif "game_date" not in raw.columns:
        raise ValueError("No 'date' or 'game_date' column found in odds data")

    # ── Normalize date format to YYYY-MM-DD ────────────────────────────────
    raw["game_date"] = pd.to_datetime(
        raw["game_date"]
    ).dt.strftime("%Y-%m-%d")

    # ── Map team names to 3-letter abbreviations ───────────────────────────
    raw["home_team"] = raw["home"].apply(
        lambda x: _map_team(str(x).strip(), KAGGLE_TEAM_TO_ABB)
    )
    raw["away_team"] = raw["away"].apply(
        lambda x: _map_team(str(x).strip(), KAGGLE_TEAM_TO_ABB)
    )

    # Warn about unmapped teams (single-char or not in ABB dict -> stays as-is)
    unmapped_home = set(raw["home_team"]) - set(KAGGLE_TEAM_TO_ABB.values())
    unmapped_home = {t for t in unmapped_home if len(t) != 3 or not t.isupper()}
    unmapped_away = set(raw["away_team"]) - set(KAGGLE_TEAM_TO_ABB.values())
    unmapped_away = {t for t in unmapped_away if len(t) != 3 or not t.isupper()}
    if unmapped_home or unmapped_away:
        print(f"  WARNING: Unmapped home teams: {sorted(unmapped_home)}")
        print(f"  WARNING: Unmapped away teams: {sorted(unmapped_away)}")

    # ── Normalize season to project integer format ─────────────────────────
    raw["season"] = raw["season"].apply(_normalize_season)

    # ── Compute covers_spread from id_spread + whos_favored ───────────────
    # id_spread: 1=favorite covered, 0=underdog covered, 2=push
    raw["id_spread"] = pd.to_numeric(raw["id_spread"], errors="coerce").fillna(2).astype(int)
    raw["whos_favored"] = raw["whos_favored"].astype(str).str.lower().str.strip()
    raw["covers_spread"] = raw.apply(compute_home_covers_spread, axis=1)

    # ── Compute no-vig implied probabilities (vectorized) ─────────────────
    # Rows with missing moneyline (ESPN-sourced post-Jan 2023): NaN
    home_ml = pd.to_numeric(raw["moneyline_home"], errors="coerce")
    away_ml = pd.to_numeric(raw["moneyline_away"], errors="coerce")

    home_probs, away_probs = _compute_implied_probs_vectorized(home_ml, away_ml)
    raw["home_implied_prob"] = home_probs
    raw["away_implied_prob"] = away_probs

    # ── Normalize spread column ────────────────────────────────────────────
    raw["spread"] = pd.to_numeric(raw["spread"], errors="coerce")

    # ── Select and return final columns ───────────────────────────────────
    result = raw[[
        "game_date", "home_team", "away_team", "season",
        "spread", "home_implied_prob", "away_implied_prob", "covers_spread"
    ]].copy()

    # Drop rows with missing game_date
    result = result.dropna(subset=["game_date"])

    print(f"Normalized {result.shape[0]} rows")
    push_count = result["covers_spread"].isna().sum()
    valid_count = result["covers_spread"].notna().sum()
    print(f"  Covers spread: {valid_count} valid, {push_count} pushes/NaN")
    no_ml = (result["home_implied_prob"].isna()).sum()
    print(f"  Implied prob: {result['home_implied_prob'].notna().sum()} valid, {no_ml} missing")
    print(f"  Season range: {result['season'].min()} - {result['season'].max()}")

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_normalize_odds()

    if df.shape[0] == 0:
        print("No data loaded -- download the Kaggle dataset first.")
        sys.exit(0)

    print("")
    print("=== Verification ===")
    print(f"Total rows: {df.shape[0]}")
    print(f"Columns: {list(df.columns)}")
    print(f"Season range: {df['season'].min()} to {df['season'].max()}")
    print("")
    print("Unique home teams:", sorted(df["home_team"].unique()))
    print("Unique away teams:", sorted(df["away_team"].unique()))
    print("")
    print("Sample rows:")
    print(df.head(3).to_string())
    print("")
    print(f"Push (NaN covers_spread) count: {df['covers_spread'].isna().sum()}")
    print(f"Home covered count: {(df['covers_spread'] == 1.0).sum()}")
    print(f"Away covered count: {(df['covers_spread'] == 0.0).sum()}")
    print("")
    # Verify no-vig probs sum to ~1.0
    valid_probs = df[df["home_implied_prob"].notna()].copy()
    prob_sum = (valid_probs["home_implied_prob"] + valid_probs["away_implied_prob"]).mean()
    print(f"Mean implied prob sum (should be ~1.0): {prob_sum:.4f}")
