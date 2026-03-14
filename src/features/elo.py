"""
Elo Rating System for NBA Teams
================================
Computes pre-game Elo ratings for every team-game row in the dataset.

Uses the FiveThirtyEight-style approach:
  - K-factor of 20
  - Home court advantage of 100 Elo points
  - Margin-of-victory multiplier (log-based)
  - Season carryover: regress 1/3 toward 1500

Each row's elo_pre is the rating BEFORE that game (no leakage).

Also computes a "fast Elo" variant (K=40) that reacts more quickly to
recent results.  elo_momentum = elo_pre_fast - elo_pre captures teams
that are surging or slumping relative to their long-run rating.

Usage:
    from src.features.elo import build_elo_ratings, get_current_elos
    df = build_elo_ratings()
    current = get_current_elos()
"""

import math
import os

import pandas as pd
import logging

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_ELO = 1500
K_FACTOR = 20
K_FACTOR_FAST = 40  # high-reactivity variant for momentum detection
HOME_ADVANTAGE = 100
CARRYOVER_FRACTION = 2 / 3  # keep 2/3 of delta from 1500; regress 1/3 back

INPUT_PATH = "data/processed/team_game_logs.csv"
OUTPUT_PATH = "data/features/elo_ratings.csv"


def _expected_win_prob(elo_a: float, elo_b: float) -> float:
    """Standard Elo expected score for player A vs player B."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def _mov_multiplier(margin: float, elo_diff: float) -> float:
    """
    FiveThirtyEight-style margin-of-victory multiplier.

    Rewards blowouts but with diminishing returns (log-based), and adjusts
    so that heavy favourites don't get over-rewarded for expected blowouts.
    """
    abs_margin = abs(margin)
    return math.log(abs_margin + 1.0) * (2.2 / ((elo_diff * 0.001) + 2.2))


def _regress_to_mean(elo: float) -> float:
    """Regress an Elo rating 1/3 toward BASE_ELO for a new season."""
    return BASE_ELO + CARRYOVER_FRACTION * (elo - BASE_ELO)


def _compute_elo_shift(
    h_elo_pre: float,
    a_elo_pre: float,
    home_won: bool,
    margin: float,
    k_factor: float,
) -> float:
    """Compute the Elo shift for one game given a K-factor."""
    h_expected = _expected_win_prob(h_elo_pre + HOME_ADVANTAGE, a_elo_pre)
    h_actual = 1.0 if home_won else 0.0
    elo_diff_for_mov = abs((h_elo_pre + HOME_ADVANTAGE) - a_elo_pre)
    mov_mult = _mov_multiplier(margin, elo_diff_for_mov) if margin != 0 else 1.0
    return k_factor * mov_mult * (h_actual - h_expected)


def _init_or_regress(
    elos: dict[int, float],
    last_season: dict[int, int],
    tid: int,
    season: int,
) -> None:
    """Initialize a team's Elo at BASE_ELO or regress for a new season."""
    if tid not in elos:
        elos[tid] = BASE_ELO
        last_season[tid] = season
    elif last_season[tid] != season:
        elos[tid] = _regress_to_mean(elos[tid])
        last_season[tid] = season


def _process_game_group(
    game_rows: pd.DataFrame,
    df: pd.DataFrame,
    elos: dict[int, float],
    last_season: dict[int, int],
    elos_fast: dict[int, float],
    last_season_fast: dict[int, int],
    out: dict[str, list],
) -> None:
    """Process one game_id group, appending pre-game values and updating Elos."""
    if len(game_rows) != 2:
        for _ in range(len(game_rows)):
            for key in out:
                out[key].append(BASE_ELO)
        return

    rows = game_rows.to_dict("records")
    home_row = rows[0] if rows[0]["is_home"] == 1 else rows[1]
    away_row = rows[1] if rows[0]["is_home"] == 1 else rows[0]

    h_id, a_id = home_row["team_id"], away_row["team_id"]
    season = home_row["season"]

    # Initialize / regress both Elo tracks
    for tid in (h_id, a_id):
        _init_or_regress(elos, last_season, tid, season)
        _init_or_regress(elos_fast, last_season_fast, tid, season)

    h_pre, a_pre = elos[h_id], elos[a_id]
    h_pre_f, a_pre_f = elos_fast[h_id], elos_fast[a_id]

    # Store pre-game ratings in row order
    for idx_val in game_rows.index:
        row_tid = df.at[idx_val, "team_id"]
        is_home = row_tid == h_id
        out["elo_pre"].append(h_pre if is_home else a_pre)
        out["elo_opp_pre"].append(a_pre if is_home else h_pre)
        out["elo_pre_fast"].append(h_pre_f if is_home else a_pre_f)
        out["elo_opp_pre_fast"].append(a_pre_f if is_home else h_pre_f)

    # Compute shifts and update both tracks
    home_won = home_row["wl"] == "W"
    h_pts = home_row.get("pts", 0) or 0
    a_pts = away_row.get("pts", 0) or 0
    margin = h_pts - a_pts

    shift = _compute_elo_shift(h_pre, a_pre, home_won, margin, K_FACTOR)
    elos[h_id], elos[a_id] = h_pre + shift, a_pre - shift

    shift_fast = _compute_elo_shift(h_pre_f, a_pre_f, home_won, margin, K_FACTOR_FAST)
    elos_fast[h_id], elos_fast[a_id] = h_pre_f + shift_fast, a_pre_f - shift_fast


def build_elo_ratings(
    input_path: str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Compute pre-game Elo ratings for every team-game in team_game_logs.csv.

    Returns a DataFrame with columns:
        game_id, team_id, team_abbreviation, game_date, season,
        elo_pre, elo_opp_pre, elo_diff, elo_expected_win_prob,
        elo_pre_fast, elo_opp_pre_fast, elo_momentum

    elo_pre / elo_pre_fast are ratings BEFORE the game result is applied.
    elo_momentum = elo_pre_fast - elo_pre (captures surging/slumping teams).
    """
    df = pd.read_csv(input_path)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df["is_home"] = df["matchup"].str.contains("vs.").astype(int)
    df = df.sort_values(
        ["game_date", "game_id", "is_home"], ascending=[True, True, False],
    ).reset_index(drop=True)

    elos: dict[int, float] = {}
    last_season: dict[int, int] = {}
    elos_fast: dict[int, float] = {}
    last_season_fast: dict[int, int] = {}

    out: dict[str, list] = {
        "elo_pre": [], "elo_opp_pre": [],
        "elo_pre_fast": [], "elo_opp_pre_fast": [],
    }

    for _game_id, game_rows in df.groupby("game_id", sort=False):
        _process_game_group(
            game_rows, df, elos, last_season, elos_fast, last_season_fast, out,
        )

    for col, values in out.items():
        df[col] = values

    df["elo_diff"] = df["elo_pre"] - df["elo_opp_pre"]
    df["elo_momentum"] = df["elo_pre_fast"] - df["elo_pre"]
    df["elo_expected_win_prob"] = df.apply(
        lambda r: _expected_win_prob(
            r["elo_pre"] + (HOME_ADVANTAGE if r["is_home"] == 1 else 0),
            r["elo_opp_pre"] + (HOME_ADVANTAGE if r["is_home"] == 0 else 0),
        ),
        axis=1,
    )

    result = df[[
        "game_id", "team_id", "team_abbreviation", "game_date", "season",
        "elo_pre", "elo_opp_pre", "elo_diff", "elo_expected_win_prob",
        "elo_pre_fast", "elo_opp_pre_fast", "elo_momentum",
    ]].copy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    log.info(f"Saved {len(result):,} Elo rows -> {output_path}")
    return result


def get_current_elos(
    input_path: str = OUTPUT_PATH,
    extended: bool = False,
) -> dict[str, float] | dict[str, dict[str, float]]:
    """
    Return current Elo ratings for every team based on their most recent game.

    Args:
        input_path: Path to the Elo ratings CSV.
        extended: If False (default), returns {team: elo_pre}.
                  If True, returns {team: {elo: float, elo_fast: float, momentum: float}}.

    If the Elo ratings file doesn't exist, builds it first.
    """
    if not os.path.exists(input_path):
        build_elo_ratings()

    df = pd.read_csv(input_path)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")

    # Get the most recent row per team
    latest = (
        df.sort_values("game_date")
        .groupby("team_abbreviation")
        .tail(1)
        .set_index("team_abbreviation")
    )

    if not extended:
        return latest["elo_pre"].to_dict()

    result: dict[str, dict[str, float]] = {}
    for team, row in latest.iterrows():
        result[str(team)] = {
            "elo": float(row["elo_pre"]),
            "elo_fast": float(row.get("elo_pre_fast", row["elo_pre"])),
            "momentum": float(row.get("elo_momentum", 0.0)),
        }
    return result


if __name__ == "__main__":
    ratings = build_elo_ratings()
    log.info(f"\nElo ratings: {len(ratings)} rows, cols: {list(ratings.columns)}")
    log.info(f"\nTop 10 current Elo ratings:")
    current = get_current_elos()
    for team, elo in sorted(current.items(), key=lambda x: -x[1])[:10]:
        log.info(f"  {team}: {elo:.1f}")
