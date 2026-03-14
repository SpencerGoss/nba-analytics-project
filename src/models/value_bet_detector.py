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
   - Source: Pinnacle guest API (free, no key required)
   - Coverage: Upcoming games only -- NOT historical
   - Cost: Free, no authentication required
   - Use case: Daily value-bet scan for upcoming games

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

from src.models.odds_utils import no_vig_odds_ratio as _no_vig_core

# -- Config ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = str(PROJECT_ROOT / "models" / "artifacts")
ATS_FEATURES_PATH = str(PROJECT_ROOT / "data" / "features" / "game_ats_features.csv")
MATCHUP_FEATURES_PATH = str(PROJECT_ROOT / "data" / "features" / "game_matchup_features.csv")

# Configurable threshold: flag games where |model_prob - market_prob| > threshold
# Default 5pp (same as WINPROB_FLAG_PP in fetch_odds.py)
VALUE_BET_THRESHOLD = float(os.getenv("VALUE_BET_THRESHOLD", "0.03"))

# -- Probability helpers --------------------------------------------------------

def no_vig_prob(home_ml, away_ml):
    """Remove bookmaker vig using the multiplicative method.

    Delegates to odds_utils.no_vig_odds_ratio for the core math, converting
    None returns to NaN for pandas compatibility.

    Args:
        home_ml: Home team American moneyline (int or float). e.g., -110 or +150.
        away_ml: Away team American moneyline (int or float). e.g., -110 or +150.

    Returns:
        (home_no_vig, away_no_vig): tuple of float. Both are NaN if either
        input is None or NaN.
    """
    try:
        if home_ml is None or away_ml is None:
            return (float("nan"), float("nan"))
        if pd.isna(home_ml) or pd.isna(away_ml):
            return (float("nan"), float("nan"))
        result = _no_vig_core(float(home_ml), float(away_ml))
        if result[0] is None:
            return (float("nan"), float("nan"))
        return result
    except (TypeError, ValueError):
        return (float("nan"), float("nan"))


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

    # EV and confidence tier enrichment
    try:
        from src.models.odds_utils import expected_value
        df["ev"] = df.apply(
            lambda r: expected_value(r["model_win_prob"], r["market_implied_prob"])
            if r["market_implied_prob"] > 0 else None, axis=1)
    except ImportError:
        df["ev"] = None

    try:
        from src.models.betting_router import confidence_tier
        df["confidence_tier"] = df["edge_magnitude"].apply(
            lambda e: confidence_tier(e, True))
    except ImportError:
        df["confidence_tier"] = "Unknown"

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
        from src.models.game_outcome_model import _CalibrationUnpickler
        with open(cal_path, "rb") as f:
            model = _CalibrationUnpickler(f).load()
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
            Pinnacle guest API (free, no key required).
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
        print("\nFetching live game lines from Pinnacle guest API...")
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

        X = ats_df.reindex(columns=feature_cols)
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

            # Refresh Elo ratings from latest data
            try:
                from src.features.elo import get_current_elos
                current_elos = get_current_elos()
                if home_abb in current_elos and away_abb in current_elos:
                    feature_row["diff_elo"] = current_elos[home_abb] - current_elos[away_abb]
            except Exception:
                pass  # Elo refresh is best-effort; stale values still usable

            X_row = pd.DataFrame([feature_row]).reindex(columns=feature_cols)
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
        "ev", "confidence_tier",
    ]
    out_cols = [c for c in output_cols if c in result_df.columns]
    result_records = result_df[out_cols].copy()

    # Convert NaN to None for JSON serializability (NFR-3)
    result_records = result_records.where(result_records.notna(), None)

    return result_records.to_dict(orient="records")



def _compute_kelly_fraction(bet: dict, kelly_scale: float = 0.5) -> float:
    """Compute fractional Kelly position size (0.5x Kelly criterion).

    Kelly formula: f = (p * b - (1 - p)) / b
    where p = model win probability for the bet side,
          b = decimal odds - 1 derived from no-vig market implied probability.

    Args:
        bet: Bet dict with model_win_prob, market_implied_prob, bet_side.
        kelly_scale: Fractional Kelly multiplier. Default 0.5 (half-Kelly).

    Returns:
        float: Fractional Kelly fraction in [0, 1]. Returns 0.0 if inputs are
        missing, market probability is degenerate, or Kelly is negative.
    """
    model_p = bet.get("model_win_prob")
    market_p = bet.get("market_implied_prob")
    bet_side = bet.get("bet_side", "home")

    if model_p is None or market_p is None:
        return 0.0

    model_p = float(model_p)
    market_p = float(market_p)

    # For away bets, flip both probabilities
    if bet_side == "away":
        p = 1.0 - model_p
        q = 1.0 - market_p   # no-vig market prob of away win
    else:
        p = model_p
        q = market_p          # no-vig market prob of home win

    # b = decimal odds - 1 = (1 - q) / q
    if q <= 0.0 or q >= 1.0:
        return 0.0
    b = (1.0 - q) / q

    kelly = (p * b - (1.0 - p)) / b
    fraction = max(0.0, kelly_scale * kelly)
    # Hard cap at 5% to limit maximum bet size
    fraction = min(fraction, 0.05)
    return round(fraction, 4)


# -- Strong value-bet filter ----------------------------------------------------

STRONG_BET_THRESHOLD = float(os.getenv("STRONG_BET_THRESHOLD", "0.08"))

# Composite score weights (tunable via env vars, must sum to 1.0):
#   composite_score = COMPOSITE_EDGE_WEIGHT * edge_magnitude
#                   + COMPOSITE_ATS_WEIGHT  * (ats_prob - 0.5)
COMPOSITE_EDGE_WEIGHT = float(os.getenv("COMPOSITE_EDGE_WEIGHT", "0.6"))
COMPOSITE_ATS_WEIGHT  = float(os.getenv("COMPOSITE_ATS_WEIGHT",  "0.0"))

# Minimum composite_score to qualify as a strong bet.
COMPOSITE_THRESHOLD = float(os.getenv("COMPOSITE_THRESHOLD", "0.04"))


def _load_ats_model():
    import pickle as _pk
    model_path = os.path.join(ARTIFACTS_DIR, "ats_model.pkl")
    feat_path  = os.path.join(ARTIFACTS_DIR, "ats_model_features.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ATS model artifact not found at {model_path!r}. "
            "Run: python src/models/ats_model.py"
        )
    with open(model_path, "rb") as fh:
        model = _pk.load(fh)
    feat_cols = []
    if os.path.exists(feat_path):
        with open(feat_path, "rb") as fh:
            feat_cols = _pk.load(fh)
    return model, feat_cols


def _score_bets_with_ats(candidate_bets, ats_features_path=ATS_FEATURES_PATH):
    """Enrich candidate bets with ATS model probabilities and composite scores.

    When the ATS model is unavailable (FileNotFoundError), all bets are returned
    with ats_prob=None and ats_model_used=False so callers can distinguish
    "no model" from "prediction succeeded".

    When a matching row is not found in the ATS features file, that bet receives
    ats_prob=None (not 0.5) to signal missing data -- callers should not filter
    on ats_prob alone in that case.

    Args:
        candidate_bets: list of bet dicts (output of run_value_bet_scan).
        ats_features_path: Path to game_ats_features.csv for feature lookup.

    Returns:
        list of dicts, each extended with:
            ats_prob        (float | None) -- ATS model P(home covers); None if unavailable
            composite_score (float)        -- weighted combination of edge and ats_prob
            ats_model_used  (bool)         -- True only when ATS model produced a prediction
    """
    # --- Try to load the ATS model ---------------------------------------------
    ats_model = None
    ats_feat_cols = []
    try:
        ats_model, ats_feat_cols = _load_ats_model()
    except FileNotFoundError as exc:
        warnings.warn(
            f"ATS model unavailable -- falling back to edge-only scoring. ({exc})",
            UserWarning, stacklevel=3,
        )

    # If no model, return bets with ats_prob=None and edge-only composite score
    if ats_model is None:
        return [
            {
                **b,
                "ats_prob": None,
                "composite_score": round(COMPOSITE_EDGE_WEIGHT * (b.get("edge_magnitude") or 0.0), 4),
                "ats_model_used": False,
            }
            for b in candidate_bets
        ]

    # --- Load ATS features file for per-game row lookup -----------------------
    ats_df = None
    if ats_feat_cols and os.path.exists(ats_features_path):
        try:
            ats_df = pd.read_csv(ats_features_path, low_memory=False)
            if "game_date" in ats_df.columns:
                ats_df["game_date"] = (
                    pd.to_datetime(ats_df["game_date"], errors="coerce")
                    .dt.strftime("%Y-%m-%d")
                )
        except Exception as exc:
            warnings.warn(
                f"Could not load ATS features file for join: {exc}. "
                "ATS scoring will use ats_prob=None for all games.",
                UserWarning, stacklevel=3,
            )
            ats_df = None

    # --- Score each bet --------------------------------------------------------
    enriched = []
    for bet in candidate_bets:
        edge_mag = bet.get("edge_magnitude") or 0.0
        ats_prob = None   # None = no prediction available (missing data or no model)
        ats_used = False

        if ats_df is not None:
            home      = bet.get("home_team")
            away      = bet.get("away_team")
            game_date = str(bet.get("game_date") or "")

            if home and away and game_date and "game_date" in ats_df.columns:
                matched = ats_df[
                    (ats_df["home_team"] == home)
                    & (ats_df["away_team"] == away)
                    & (ats_df["game_date"] == game_date)
                ]
            elif home and away:
                mask = (ats_df["home_team"] == home) & (ats_df["away_team"] == away)
                matched = (
                    ats_df[mask].sort_values("game_date").tail(1)
                    if "game_date" in ats_df.columns else ats_df[mask].tail(1)
                )
            else:
                matched = pd.DataFrame()

            if not matched.empty:
                try:
                    X_row    = matched.reindex(columns=ats_feat_cols).head(1)
                    ats_prob = float(ats_model.predict_proba(X_row)[0][1])
                    ats_used = True
                except Exception as exc:
                    warnings.warn(
                        f"ATS model prediction failed for {home} vs {away}: {exc}. "
                        "Using ats_prob=None for this game.",
                        UserWarning, stacklevel=3,
                    )
                    ats_prob = None

        # Composite score: use 0.5 as neutral stand-in when ats_prob is None
        # (edge contribution still counts; ats contribution is zeroed out)
        ats_contrib = (ats_prob - 0.5) if ats_prob is not None else 0.0
        composite = COMPOSITE_EDGE_WEIGHT * edge_mag + COMPOSITE_ATS_WEIGHT * ats_contrib
        enriched.append({
            **bet,
            "ats_prob":        round(ats_prob, 4) if ats_prob is not None else None,
            "composite_score": round(composite, 4),
            "ats_model_used":  ats_used,
        })

    return enriched


# Minimum ATS probability for a bet to pass the ATS filter.
# Bets where ats_prob is not None AND ats_prob < ATS_PROB_THRESHOLD are excluded.
# Bets where ats_prob is None (missing feature data) are kept regardless.
ATS_PROB_THRESHOLD = float(os.getenv("ATS_PROB_THRESHOLD", "0.53"))


def get_strong_value_bets(
    strong_threshold=STRONG_BET_THRESHOLD,
    use_live_odds=False,
    composite_threshold=COMPOSITE_THRESHOLD,
    ats_features_path=ATS_FEATURES_PATH,
):
    """Return bets that pass both the game-outcome edge filter AND the ATS filter.

    Logic flow
    ----------
    1. Run the full value-bet scan (game-outcome model vs. market).
    2. Keep only candidates whose edge_magnitude > strong_threshold.
    3. Enrich each candidate with ATS model probabilities via _score_bets_with_ats().
    4. For each scored bet, determine ats_filtered:
       - True  when ats_prob is None (missing data -- do not penalise)
       - True  when ats_prob >= ATS_PROB_THRESHOLD
       - False when ats_prob < ATS_PROB_THRESHOLD (ATS model disagrees -- exclude)
    5. If the ATS model was available for at least one bet, apply the composite-
       score filter (COMPOSITE_THRESHOLD) AND the ats_filtered flag together.
       If no ATS model is present at all, fall back to edge-only filtering.
    6. Return sorted list of dicts; each dict includes:
       - All original run_value_bet_scan() columns
       - ats_prob        (float | None)
       - ats_filtered    (bool)
       - composite_score (float)
       - ats_model_used  (bool)
       - kelly_fraction (float) -- 0.5x Kelly position size in [0, 1]

    The function signature is unchanged so existing callers are unaffected.
    """
    all_bets = run_value_bet_scan(use_live_odds=use_live_odds, threshold=VALUE_BET_THRESHOLD)
    edge_candidates = [b for b in all_bets if (b.get("edge_magnitude") or 0) > strong_threshold]

    scored = _score_bets_with_ats(edge_candidates, ats_features_path=ats_features_path)

    # --- Annotate ats_filtered for every scored bet ----------------------------
    # ats_filtered=True  means "passes ATS gate (or ATS data unavailable)"
    # ats_filtered=False means "ATS model available AND ats_prob < ATS_PROB_THRESHOLD"
    any_ats_used = any(b.get("ats_model_used") for b in scored)

    for bet in scored:
        ats_prob = bet.get("ats_prob")
        if ats_prob is None:
            # No prediction for this row (missing features or no model) -- keep it
            bet["ats_filtered"] = True
        else:
            bet["ats_filtered"] = ats_prob >= ATS_PROB_THRESHOLD

    # --- Apply composite + ats_filtered filter ---------------------------------
    if any_ats_used:
        # ATS model produced at least one prediction: apply both gates
        strong = [
            b for b in scored
            if (b.get("composite_score") or 0) > composite_threshold
            and b.get("ats_filtered", True)
        ]
        strong_sorted = sorted(strong, key=lambda b: b.get("composite_score") or 0, reverse=True)
        n_ats_filtered_out = sum(1 for b in scored if not b.get("ats_filtered", True))
        print(
            f"Strong value bets (composite > {composite_threshold:.2f}, "
            f"ats_prob >= {ATS_PROB_THRESHOLD:.2f}, "
            f"edge_weight={COMPOSITE_EDGE_WEIGHT}, ats_weight={COMPOSITE_ATS_WEIGHT}): "
            f"{len(strong_sorted)} of {len(all_bets)} games "
            f"({n_ats_filtered_out} removed by ATS filter)"
        )
    else:
        # ATS model unavailable: edge-only fallback; ats_filtered is True for all
        strong = [b for b in scored if (b.get("edge_magnitude") or 0) > strong_threshold]
        strong_sorted = sorted(strong, key=lambda b: b.get("edge_magnitude") or 0, reverse=True)
        print(
            f"Strong value bets (edge-only fallback, edge > {strong_threshold:.0%}): "
            f"{len(strong_sorted)} of {len(all_bets)} games"
        )

    # Add fractional Kelly sizing to each strong bet
    for bet in strong_sorted:
        bet["kelly_fraction"] = _compute_kelly_fraction(bet)

    return strong_sorted

# -- Entry point ----------------------------------------------------------------

if __name__ == "__main__":
    # Run in historical mode by default -- no API key required
    results = run_value_bet_scan(use_live_odds=False)

    n_value_bets = sum(1 for r in results if r.get("is_value_bet"))
    print(f"\nTotal games scanned: {len(results)}")
    print(f"Value bets flagged: {n_value_bets}")
    print(f"Non-value bets: {len(results) - n_value_bets}")
