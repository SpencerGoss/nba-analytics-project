"""
Build the ATS (Against-the-Spread) feature table for the ATS classifier.

IMPORTANT: This file intentionally contains spread and implied probability columns.
The win-probability model (game_outcome_model.py) MUST NOT use this file.
The win-probability model reads ONLY data/features/game_matchup_features.csv.

Output: data/features/game_ats_features.csv

Joins game_matchup_features.csv with historical betting odds on game_date +
home_team + away_team. Adds spread, home_implied_prob, away_implied_prob, and
covers_spread as the ATS classification target.

Excludes:
- Pre-2007 games (no odds data -- inner join handles this naturally)
- COVID bubble / shortened seasons 201920 and 202021

Usage:
    from src.features.ats_features import build_ats_features
    df = build_ats_features()

Or run as main:
    python src/features/ats_features.py
"""

import pandas as pd
from pathlib import Path

from src.data.get_historical_odds import load_and_normalize_odds

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Seasons excluded from training data (COVID bubble / shortened seasons)
# Mirrors EXCLUDED_SEASONS in src/models/game_outcome_model.py
EXCLUDED_SEASONS = [201920, 202021]


# ── Build function ─────────────────────────────────────────────────────────────

def build_ats_features(
    matchup_path: str = "data/features/game_matchup_features.csv",
    odds_path: str = "data/raw/odds/nba_betting_historical.csv",
    output_path: str = "data/features/game_ats_features.csv",
) -> pd.DataFrame:
    """
    Build ATS feature table by joining matchup features with betting odds.

    IMPORTANT: This file intentionally contains spread and implied probability.
    The win-probability model (game_outcome_model.py) MUST NOT use this file.

    Parameters
    ----------
    matchup_path : str
        Path to game_matchup_features.csv (base feature table, no spread data).
    odds_path : str
        Path to nba_betting_historical.csv (Kaggle betting dataset).
    output_path : str
        Path for output CSV file.

    Returns
    -------
    pd.DataFrame with all matchup features plus:
        spread            : float (opening line, home team's perspective)
        home_implied_prob : float (no-vig implied probability for home team)
        away_implied_prob : float (no-vig implied probability for away team)
        covers_spread     : float (1=home covered, 0=away covered, NaN=push)
    """
    # ── Data separation guard: verify matchup file has NO spread columns ──────
    # This is the code-level FR-5.1 guard.
    _check_matchup_path = (
        matchup_path
        if Path(matchup_path).is_absolute()
        else str(PROJECT_ROOT / matchup_path)
    )
    _check_odds_path = (
        odds_path
        if Path(odds_path).is_absolute()
        else str(PROJECT_ROOT / odds_path)
    )
    _check_output_path = (
        output_path
        if Path(output_path).is_absolute()
        else str(PROJECT_ROOT / output_path)
    )

    print("Loading matchup features...")
    matchup_header = pd.read_csv(_check_matchup_path, nrows=1)
    forbidden_cols = {"spread", "covers_spread", "home_implied_prob", "away_implied_prob"}
    found_forbidden = forbidden_cols.intersection(set(matchup_header.columns))
    assert len(found_forbidden) == 0, (
        f"DATA SEPARATION VIOLATION: game_matchup_features.csv contains "
        f"forbidden columns: {found_forbidden}. "
        f"The win-probability model must never see spread data."
    )
    print("  Data separation guard PASSED: no spread/odds columns in matchup features")

    # ── Load matchup features ─────────────────────────────────────────────────
    matchup = pd.read_csv(_check_matchup_path)
    print(f"  Matchup features: {matchup.shape[0]} rows, {matchup.shape[1]} columns")

    # ── Load and normalize odds ───────────────────────────────────────────────
    print("Loading historical odds...")
    odds = load_and_normalize_odds(odds_path=_check_odds_path)
    if odds.shape[0] == 0:
        raise RuntimeError(
            "No odds data loaded. Download the Kaggle NBA betting dataset first. "
            "See src/data/get_historical_odds.py for instructions."
        )
    print(f"  Odds data: {odds.shape[0]} rows")

    # ── Keep only the columns needed from odds for the join ───────────────────
    odds_join = odds[[
        "game_date", "home_team", "away_team",
        "spread", "home_implied_prob", "away_implied_prob", "covers_spread"
    ]].copy()

    # ── Inner join on game_date + home_team + away_team ───────────────────────
    # Inner join intentionally drops pre-2007 games (no odds data available).
    # Team abbreviations must match -- get_historical_odds.py normalizes to
    # the same 3-letter format as game_matchup_features.csv.
    print("Joining matchup features with odds...")
    merged = matchup.merge(
        odds_join,
        on=["game_date", "home_team", "away_team"],
        how="inner",
    )
    assert merged.shape[0] > 0, (
        "Zero rows after merge -- check team name mapping. "
        "home_team/away_team values in matchup and odds files must match."
    )
    print(f"  Joined result: {merged.shape[0]} rows (pre-2007 rows dropped -- expected)")

    # ── Exclude anomalous seasons ─────────────────────────────────────────────
    pre_exclusion = merged.shape[0]
    merged = merged[~merged["season"].isin(EXCLUDED_SEASONS)].copy()
    excluded_count = pre_exclusion - merged.shape[0]
    if excluded_count > 0:
        print(f"  Excluded {excluded_count} rows from seasons {EXCLUDED_SEASONS}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_rows = merged.shape[0]
    valid_covers = merged["covers_spread"].notna().sum()
    push_count = merged["covers_spread"].isna().sum()
    valid_probs = merged["home_implied_prob"].notna().sum()
    season_min = merged["season"].min()
    season_max = merged["season"].max()

    print(f"")
    print(f"ATS Feature Table Summary:")
    print(f"  Total rows:        {total_rows}")
    print(f"  Valid covers:      {valid_covers} (non-push)")
    print(f"  Pushes (NaN):      {push_count}")
    print(f"  Valid impl. probs: {valid_probs}")
    print(f"  Season range:      {season_min} - {season_max}")
    print(f"  Columns:           {merged.shape[1]}")

    # ── Save output ───────────────────────────────────────────────────────────
    output_dir = Path(_check_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(_check_output_path, index=False)
    print(f"  Saved to: {_check_output_path}")

    return merged


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_ats_features()
    print("")
    print("=== Final Verification ===")
    print(f"game_ats_features.csv: {df.shape[0]} rows, {df.shape[1]} columns")
    ats_cols = ["spread", "home_implied_prob", "away_implied_prob", "covers_spread"]
    for col in ats_cols:
        assert col in df.columns, f"Missing required column: {col}"
    print(f"Required ATS columns present: {ats_cols}")

    # Verify no-vig probs sum to ~1.0
    valid_p = df[df["home_implied_prob"].notna()].copy()
    if len(valid_p) > 0:
        prob_sum = (valid_p["home_implied_prob"] + valid_p["away_implied_prob"]).mean()
        print(f"Mean implied prob sum (no-vig check, should be ~1.0): {prob_sum:.4f}")

    # Spot-check covers_spread encoding
    home_covers = (df["covers_spread"] == 1.0).sum()
    away_covers = (df["covers_spread"] == 0.0).sum()
    pushes = df["covers_spread"].isna().sum()
    print(f"Covers spread: home={home_covers}, away={away_covers}, push/NaN={pushes}")
    print("")
    print("DONE")
