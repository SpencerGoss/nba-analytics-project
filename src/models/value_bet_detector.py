"""
Value-Bet Detector
==================
Identifies NBA games where the calibrated win-probability model meaningfully
disagrees with market-implied probabilities from Vegas moneylines. These
disagreements are potential value bets.

Odds Data Sourcing Strategy
---------------------------
Two-tier sourcing:

1. Historical odds (training / backtesting):
   - Source: Kaggle dataset (data/raw/odds/nba_betting_historical.csv)
   - Coverage: October 2007 through June 2025
   - Cost: Free, no authentication required
   - Use case: Offline analysis, feature building, historical ATS performance

2. Current-season upcoming game lines:
   - Source: The Odds API free tier (500 monthly credits)
   - Coverage: Upcoming games only -- NOT historical
   - Cost: Free tier (500 req/month); historical endpoint is paid-only (Pitfall 1)
   - Use case: Daily value-bet scan for upcoming games
   - Quota guard: check_remaining_quota() reads x-requests-remaining before any
     batch call to prevent burning credits

Pitfall 1 (The Odds API historical endpoint is paid-only):
The /historical_odds endpoint costs 10 credits per region per market per call
and is not available on the free tier. Never attempt to use The Odds API for
historical odds -- use the Kaggle dataset instead.

Pitfall 3 (Double-counting vig):
american_odds_to_implied_prob() in fetch_odds.py returns vig-inclusive
probabilities (-110 gives 52.38%, not 50%). Always use no_vig_prob() for
value-bet comparison. Raw model output is vig-free; market comparison must
also be vig-free.

Usage:
    # Historical mode (no API key needed)
    python src/models/value_bet_detector.py

    # Or import for programmatic use:
    from src.models.value_bet_detector import detect_value_bets, run_value_bet_scan
    results = run_value_bet_scan(use_live_odds=False)
"""

import os
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# -- Config ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = str(PROJECT_ROOT / "models" / "artifacts")
ATS_FEATURES_PATH = str(PROJECT_ROOT / "data" / "features" / "game_ats_features.csv")
MATCHUP_FEATURES_PATH = str(PROJECT_ROOT / "data" / "features" / "game_matchup_features.csv")

# Configurable threshold: flag games where |model_prob - market_prob| > threshold
# Default 5pp (same as WINPROB_FLAG_PP in fetch_odds.py)
VALUE_BET_THRESHOLD = float(os.getenv("VALUE_BET_THRESHOLD", "0.05"))

# Minimum API credits before aborting batch call
MIN_QUOTA_DEFAULT = 50


# -- Custom exceptions ----------------------------------------------------------

class QuotaError(Exception):
    """Raised when The Odds API remaining quota is below minimum threshold."""
    pass


# -- Probability helpers --------------------------------------------------------

def no_vig_prob(home_ml, away_ml):
    """Remove bookmaker vig using the multiplicative method.

    Converts raw American moneyline odds to fair (no-vig) implied probabilities.
    The raw implied probability over-counts (sums > 1.0 due to vig); dividing
    each side by the total removes the bookmaker's margin.

    Args:
        home_ml: Home team American moneyline (int or float). e.g., -110 or +150.
        away_ml: Away team American moneyline (int or float). e.g., -110 or +150.

    Returns:
        (home_no_vig, away_no_vig): tuple of float. Both are NaN if either
        input is None or NaN.

    Examples:
        >>> no_vig_prob(-110, -110)
        (0.5, 0.5)  # standard spread bet -- 50/50 after vig removal
        >>> no_vig_prob(-200, +170)
        (~0.634, ~0.366)
    """
    try:
        if home_ml is None or away_ml is None:
            return (float("nan"), float("nan"))
        if pd.isna(home_ml) or pd.isna(away_ml):
            return (float("nan"), float("nan"))

        home_ml = float(home_ml)
        away_ml = float(away_ml)

        # Raw implied probabilities (vig-inclusive)
        if home_ml > 0:
            raw_home = 100.0 / (home_ml + 100.0)
        else:
            raw_home = abs(home_ml) / (abs(home_ml) + 100.0)

        if away_ml > 0:
            raw_away = 100.0 / (away_ml + 100.0)
        else:
            raw_away = abs(away_ml) / (abs(away_ml) + 100.0)

        total = raw_home + raw_away  # overround (e.g., 1.0476 for -110/-110)
        if total <= 0:
            return (float("nan"), float("nan"))

        return (raw_home / total, raw_away / total)

    except (TypeError, ValueError):
        return (float("nan"), float("nan"))


# -- Quota guard ----------------------------------------------------------------

def check_remaining_quota(min_remaining=MIN_QUOTA_DEFAULT):
    """Check The Odds API remaining credits before any batch call.

    Makes a single cheap API call to /sports (1 credit) to read the
    x-requests-remaining response header. Raises QuotaError if remaining
    credits fall below min_remaining.

    Args:
        min_remaining: Minimum acceptable credits before raising QuotaError.
            Default is 50. Pass 0 to disable the guard.

    Returns:
        int: Remaining API credits. Returns -1 with a warning if ODDS_API_KEY
             is not set (non-fatal -- caller can fall back to historical mode).

    Raises:
        QuotaError: If remaining credits < min_remaining.
    """
    import requests
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ODDS_API_KEY", "")

    if not api_key:
        warnings.warn(
            "ODDS_API_KEY is not set. Skipping quota check. "
            "Set ODDS_API_KEY in .env to use The Odds API for live odds.",
            UserWarning,
            stacklevel=2,
        )
        return -1

    base_url = "https://api.the-odds-api.com/v4"
    try:
        r = requests.get(
            f"{base_url}/sports",
            params={"apiKey": api_key, "all": "false"},
            timeout=10,
        )
        remaining_raw = r.headers.get("x-requests-remaining", None)
        used_raw = r.headers.get("x-requests-used", None)

        remaining = int(remaining_raw) if remaining_raw is not None else 0
        used = int(used_raw) if used_raw is not None else 0

        print(f"  Quota status: used={used} remaining={remaining}")

        if remaining < min_remaining:
            raise QuotaError(
                f"Only {remaining} API credits remain (minimum {min_remaining}). "
                "Aborting batch call to prevent quota exhaustion. "
                "Use use_live_odds=False to run in historical mode."
            )

        return remaining

    except QuotaError:
        raise
    except Exception as e:
        warnings.warn(
            f"Could not check API quota: {e}. Proceeding cautiously.",
            UserWarning,
            stacklevel=2,
        )
        return -1


# -- Value-bet detection --------------------------------------------------------

def detect_value_bets(games_df, threshold=VALUE_BET_THRESHOLD):
    """Flag games where calibrated model probability disagrees with market odds.

    Computes the edge (model_win_prob - market_implied_prob) for each game.
    Games where |edge| > threshold are flagged as potential value bets.

    Args:
        games_df: DataFrame with columns:
            - model_win_prob: Calibrated model P(home win) from [0, 1].
            - market_implied_prob: No-vig implied P(home win) from [0, 1].
        threshold: Minimum absolute edge to flag as a value bet. Default 0.05
                   (5 percentage points). Configurable via VALUE_BET_THRESHOLD
                   env var.

    Returns:
        DataFrame with original columns plus:
            - edge: model_win_prob - market_implied_prob (signed)
            - edge_magnitude: abs(edge)
            - is_value_bet: True if edge_magnitude > threshold
            - bet_side: "home" if model favors home (edge > 0), "away" otherwise
    """
    df = games_df.copy()

    df["edge"] = df["model_win_prob"] - df["market_implied_prob"]
    df["edge_magnitude"] = df["edge"].abs()
    df["is_value_bet"] = df["edge_magnitude"] > threshold
    df["bet_side"] = df["edge"].apply(lambda e: "home" if e > 0 else "away")

    n_value_bets = int(df["is_value_bet"].sum())
    n_games = len(df)
    n_valid = df["edge_magnitude"].notna().sum()

    if n_valid > 0:
        avg_edge = df.loc[df["is_value_bet"], "edge_magnitude"].mean()
        avg_edge_str = f"{avg_edge:.4f}" if n_value_bets > 0 else "N/A"
    else:
        avg_edge_str = "N/A"

    print(
        f"  Value bets found: {n_value_bets} of {n_games} games "
        f"(threshold={threshold:.2%}, avg edge={avg_edge_str})"
    )

    return df


# -- Model loading --------------------------------------------------------------

def _load_calibrated_model():
    """Load the calibrated game outcome model from artifacts directory.

    Prefers game_outcome_model_calibrated.pkl. Falls back to uncalibrated
    model with a UserWarning (same behavior as _load_game_outcome_model in
    game_outcome_model.py).

    Returns:
        (model, feature_cols): tuple of (loaded model, list of feature names)
    """
    cal_path = os.path.join(ARTIFACTS_DIR, "game_outcome_model_calibrated.pkl")
    raw_path = os.path.join(ARTIFACTS_DIR, "game_outcome_model.pkl")
    feat_path = os.path.join(ARTIFACTS_DIR, "game_outcome_features.pkl")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Feature list not found at {feat_path}. "
            "Run: python src/models/game_outcome_model.py"
        )
    with open(feat_path, "rb") as f:
        feature_cols = pickle.load(f)

    if os.path.exists(cal_path):
        with open(cal_path, "rb") as f:
            model = pickle.load(f)
        return model, feature_cols

    warnings.warn(
        f"Calibrated model not found at {cal_path}. "
        "Using uncalibrated model -- value-bet edge measurement may be inaccurate. "
        "Run: python src/models/calibration.py to regenerate.",
        UserWarning,
        stacklevel=2,
    )
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No model artifact found in {ARTIFACTS_DIR}. "
            "Run: python src/models/game_outcome_model.py"
        )
    with open(raw_path, "rb") as f:
        model = pickle.load(f)
    return model, feature_cols


# -- Main scan function ---------------------------------------------------------

def run_value_bet_scan(use_live_odds=True, threshold=VALUE_BET_THRESHOLD):
    """High-level daily value-bet scan function.

    Loads the calibrated model, fetches game lines (live or historical),
    computes model win probabilities and no-vig market implied probabilities,
    then flags games where they disagree by more than threshold.

    Args:
        use_live_odds: If True, fetch current-season upcoming game lines from
            The Odds API (requires ODDS_API_KEY). Falls back to historical mode
            if ODDS_API_KEY is not set.
            If False, run in historical mode using game_ats_features.csv
            (no API key needed -- safe for offline use and testing).
        threshold: Value-bet detection threshold (default VALUE_BET_THRESHOLD).

    Returns:
        list[dict]: JSON-serializable list of game dicts, each with:
            - home_team, away_team, game_date (if available)
            - model_win_prob, market_implied_prob
            - edge, edge_magnitude, is_value_bet, bet_side
    """
    print("=" * 60)
    print("VALUE-BET SCAN")
    print("=" * 60)

    # -- Load calibrated model --------------------------------------------------
    print("\nLoading calibrated model...")
    model, feature_cols = _load_calibrated_model()
    print(f"  Model loaded ({len(feature_cols)} features)")

    # -- Choose odds source ------------------------------------------------------
    if use_live_odds:
        # Check API key before attempting live fetch
        api_key = os.getenv("ODDS_API_KEY", "")
        if not api_key:
            warnings.warn(
                "ODDS_API_KEY not set. Falling back to historical mode. "
                "Set ODDS_API_KEY in .env to use live odds.",
                UserWarning,
                stacklevel=2,
            )
            use_live_odds = False
        else:
            # Quota check before batch call
            try:
                remaining = check_remaining_quota()
                print(f"  API quota OK: {remaining} credits remaining")
            except QuotaError as e:
                warnings.warn(
                    f"Quota check failed: {e}. Falling back to historical mode.",
                    UserWarning,
                    stacklevel=2,
                )
                use_live_odds = False

    if use_live_odds:
        print("\nFetching live game lines from The Odds API...")
        # Import here to avoid hard dependency when running in historical mode
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            from fetch_odds import fetch_game_lines
        except ImportError:
            sys.path.insert(0, str(PROJECT_ROOT))
            from scripts.fetch_odds import fetch_game_lines

        lines_df = fetch_game_lines()
        if lines_df.empty:
            warnings.warn("No live game lines returned. Falling back to historical.", UserWarning, stacklevel=2)
            use_live_odds = False

    if not use_live_odds:
        # Historical mode: load from game_ats_features.csv
        print(f"\nLoading historical odds from {ATS_FEATURES_PATH}...")
        if not os.path.exists(ATS_FEATURES_PATH):
            raise FileNotFoundError(
                f"ATS features file not found at {ATS_FEATURES_PATH}. "
                "Run: python src/features/ats_features.py"
            )
        ats_df = pd.read_csv(ATS_FEATURES_PATH, low_memory=False)
        # Use recent seasons for a meaningful sample
        if "season" in ats_df.columns:
            recent_seasons = sorted(ats_df["season"].dropna().unique())[-4:]
            ats_df = ats_df[ats_df["season"].isin(recent_seasons)].copy()
            print(f"  Loaded {len(ats_df):,} games (seasons: {recent_seasons})")
        else:
            print(f"  Loaded {len(ats_df):,} games")

        # Determine market implied probability source:
        # game_ats_features.csv has home_implied_prob (already no-vig, from ats_features.py)
        # If raw moneylines are available, compute no-vig from them instead
        if "home_implied_prob" in ats_df.columns:
            # No-vig probs already computed during ATS feature build
            ml_mask = ats_df["home_implied_prob"].notna()
            ats_df = ats_df[ml_mask].copy()
            print(f"  Games with implied prob data: {len(ats_df):,}")
            market_implied_source = "home_implied_prob"
        elif "home_moneyline" in ats_df.columns and "away_moneyline" in ats_df.columns:
            # Fallback: compute no-vig from raw moneylines
            ml_mask = ats_df["home_moneyline"].notna() & ats_df["away_moneyline"].notna()
            ats_df = ats_df[ml_mask].copy()
            print(f"  Games with moneyline data: {len(ats_df):,}")
            market_implied_source = None  # will compute below
        else:
            print("  WARNING: No moneyline or implied probability columns found.")
            return []

        if ats_df.empty:
            print("  WARNING: No games with odds data found.")
            return []

        # -- Compute model win probabilities -------------------------------------
        print("\nComputing model win probabilities...")
        missing_features = [c for c in feature_cols if c not in ats_df.columns]
        if missing_features:
            print(f"  NOTE: {len(missing_features)} features missing from data (will be imputed as 0)")

        X = ats_df.reindex(columns=feature_cols).fillna(0)
        probs = model.predict_proba(X)[:, 1]
        ats_df = ats_df.copy()
        ats_df["model_win_prob"] = probs

        # -- Set market implied probabilities ------------------------------------
        if market_implied_source == "home_implied_prob":
            # Already no-vig from ats_features.py -- use directly
            ats_df["market_implied_prob"] = ats_df["home_implied_prob"]
            print("Using pre-computed no-vig implied probabilities from game_ats_features.csv")
        else:
            # Compute no-vig from raw moneylines
            print("Computing no-vig market implied probabilities from raw moneylines...")
            vig_results = [
                no_vig_prob(row["home_moneyline"], row["away_moneyline"])
                for _, row in ats_df.iterrows()
            ]
            ats_df["market_implied_prob"] = [r[0] for r in vig_results]

        games_df = ats_df[
            ["model_win_prob", "market_implied_prob"]
            + [c for c in ["home_team", "away_team", "game_date", "season"] if c in ats_df.columns]
        ].copy()

    else:
        # Live mode: lines_df from fetch_game_lines()
        print(f"\nProcessing {len(lines_df)} upcoming games...")

        # Load matchup features for model predictions
        if not os.path.exists(MATCHUP_FEATURES_PATH):
            warnings.warn(
                f"Matchup features not found at {MATCHUP_FEATURES_PATH}. "
                "Cannot compute model win probabilities for live games.",
                UserWarning,
                stacklevel=2,
            )
            return []

        matchup_df = pd.read_csv(MATCHUP_FEATURES_PATH, low_memory=False)

        # For each upcoming game, find the most recent matchup features
        rows = []
        for _, game in lines_df.iterrows():
            home_abb = game["home_team"]
            away_abb = game["away_team"]

            # Find recent home/away context
            home_rows = matchup_df[matchup_df["home_team"] == home_abb].sort_values("game_date")
            away_rows = matchup_df[matchup_df["away_team"] == away_abb].sort_values("game_date")

            if home_rows.empty or away_rows.empty:
                continue

            feature_row = home_rows.iloc[-1].copy()
            away_source = away_rows.iloc[-1]

            for c in matchup_df.columns:
                if c.startswith("away_"):
                    feature_row[c] = away_source.get(c, feature_row.get(c, np.nan))

            for c in feature_cols:
                if c.startswith("diff_"):
                    base = c.replace("diff_", "")
                    h_col, a_col = f"home_{base}", f"away_{base}"
                    if h_col in feature_row.index and a_col in feature_row.index:
                        feature_row[c] = feature_row[h_col] - feature_row[a_col]

            X_row = pd.DataFrame([feature_row]).reindex(columns=feature_cols).fillna(0)
            model_prob = float(model.predict_proba(X_row)[0][1])

            home_no_vig, away_no_vig = no_vig_prob(
                game.get("home_moneyline"), game.get("away_moneyline")
            )

            rows.append({
                "home_team": home_abb,
                "away_team": away_abb,
                "game_date": game.get("date", ""),
                "model_win_prob": model_prob,
                "market_implied_prob": home_no_vig,
            })

        if not rows:
            print("  WARNING: No processable games with model features found.")
            return []

        games_df = pd.DataFrame(rows)

    # -- Detect value bets -------------------------------------------------------
    print(f"\nDetecting value bets (threshold={threshold:.2%})...")
    result_df = detect_value_bets(games_df, threshold=threshold)

    # -- Print summary -----------------------------------------------------------
    value_bets = result_df[result_df["is_value_bet"]].copy()
    print(f"\nValue-bet scan complete.")
    if not value_bets.empty:
        print(f"\nTop value bets (by edge magnitude):")
        display_cols = ["home_team", "away_team", "model_win_prob",
                       "market_implied_prob", "edge", "bet_side"]
        display_cols = [c for c in display_cols if c in value_bets.columns]
        top_bets = value_bets.sort_values("edge_magnitude", ascending=False).head(10)
        for _, row in top_bets.iterrows():
            home = row.get("home_team", "?")
            away = row.get("away_team", "?")
            model_p = row.get("model_win_prob", float("nan"))
            market_p = row.get("market_implied_prob", float("nan"))
            edge = row.get("edge", float("nan"))
            side = row.get("bet_side", "?")
            print(
                f"  {home} vs {away}: model={model_p:.1%} market={market_p:.1%} "
                f"edge={edge:+.1%} -> bet {side}"
            )
    else:
        print("  No value bets found above threshold.")

    # -- Return JSON-serializable list -------------------------------------------
    output_cols = [
        "home_team", "away_team", "game_date", "season",
        "model_win_prob", "market_implied_prob",
        "edge", "edge_magnitude", "is_value_bet", "bet_side",
    ]
    out_cols = [c for c in output_cols if c in result_df.columns]
    result_records = result_df[out_cols].copy()

    # Convert NaN to None for JSON serializability (NFR-3)
    result_records = result_records.where(result_records.notna(), None)

    return result_records.to_dict(orient="records")


# -- Entry point ----------------------------------------------------------------

if __name__ == "__main__":
    # Run in historical mode by default -- no API key required
    results = run_value_bet_scan(use_live_odds=False)

    n_value_bets = sum(1 for r in results if r.get("is_value_bet"))
    print(f"\nTotal games scanned: {len(results)}")
    print(f"Value bets flagged: {n_value_bets}")
    print(f"Non-value bets: {len(results) - n_value_bets}")
