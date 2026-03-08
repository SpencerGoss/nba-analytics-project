"""
build_explainers.py
-------------------
Generate plain-English explainer bullets for each game in todays_picks.json.
Reads:
  - dashboard/data/todays_picks.json   -- games + model probs + edge
  - dashboard/data/value_bets.json     -- supplemental edge/market_prob data
  - data/features/game_matchup_features.csv -- per-game feature rows
  - dashboard/data/game_context.json   -- optional B2B / rest context

Writes:
  - dashboard/data/explainers.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PICKS_PATH = PROJECT_ROOT / "dashboard" / "data" / "todays_picks.json"
VALUE_BETS_PATH = PROJECT_ROOT / "dashboard" / "data" / "value_bets.json"
MATCHUP_CSV = PROJECT_ROOT / "data" / "features" / "game_matchup_features.csv"
GAME_CONTEXT_PATH = PROJECT_ROOT / "dashboard" / "data" / "game_context.json"
OUTPUT_PATH = PROJECT_ROOT / "dashboard" / "data" / "explainers.json"

# ---------------------------------------------------------------------------
# Bullet-generation thresholds (named constants -- no magic numbers)
# ---------------------------------------------------------------------------
HIGH_WIN_PROB_THRESHOLD = 0.65
STRONG_RECENT_WINS = 7       # home last-10 >= this -> "hot team"
WEAK_RECENT_WINS = 3         # away last-10 <= this -> "struggling"
REST_ADVANTAGE_DAYS = 1      # home_rest - away_rest > this
HIGH_EDGE_THRESHOLD = 0.10
HIGH_NET_RTG_DIFF = 5.0      # points per 100 possessions
HIGH_PYTHAG_DIFF = 0.10      # pythagorean win% gap


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_json(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_game_context(path: Path) -> list[dict[str, Any]]:
    """Return game context list; empty list if file absent."""
    if not path.exists():
        return []
    try:
        return load_json(path)
    except (json.JSONDecodeError, OSError):
        return []


def load_matchup_features(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    return df


def build_matchup_index(df: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    """
    Build {(home_team, away_team) -> most-recent feature row} lookup.
    Uses the last historical row for the matchup pair as the best available
    feature snapshot when the exact game date is not present.
    """
    latest = (
        df.sort_values("game_date")
        .groupby(["home_team", "away_team"], sort=False)
        .last()
        .reset_index()
    )
    index: dict[tuple[str, str], pd.Series] = {}
    for _, row in latest.iterrows():
        key = (str(row["home_team"]), str(row["away_team"]))
        index[key] = row
    return index


def build_value_bet_index(
    vb_list: list[dict[str, Any]]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index value bets by (home_team, away_team, game_date) for fast lookup."""
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for vb in vb_list:
        key = (vb["home_team"], vb["away_team"], vb["game_date"])
        index[key] = vb
    return index


def build_context_index(
    ctx_list: list[dict[str, Any]]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ctx in ctx_list:
        key = (ctx.get("home_team", ""), ctx.get("away_team", ""), ctx.get("game_date", ""))
        index[key] = ctx
    return index


# ---------------------------------------------------------------------------
# Bullet generators
# ---------------------------------------------------------------------------

def _pct(value: float) -> str:
    """Format a probability (0-1) as '63.2%'."""
    return f"{value * 100:.1f}%"


def _spread_implied_prob(spread: float | None) -> float | None:
    """
    Convert a point spread to an approximate implied win probability using
    a simple logistic approximation (each point ~ 3% edge).
    Returns None if spread is None.
    """
    if spread is None:
        return None
    # Positive spread = home is favored by that many points
    # P(home_win) ~ sigmoid(spread / 10)
    import math
    return 1.0 / (1.0 + math.exp(-spread / 10.0))


def generate_bullets(
    pick: dict[str, Any],
    features: pd.Series | None,
    value_bet: dict[str, Any] | None,
    context: dict[str, Any] | None,
) -> list[str]:
    """
    Return 2-4 plain-English bullets explaining the bet recommendation.
    Checks rules in priority order; stops after 4 bullets.
    """
    bullets: list[str] = []

    home = pick["home_team"]
    away = pick["away_team"]
    home_name = pick.get("home_team_name", home)
    away_name = pick.get("away_team_name", away)
    home_prob: float = pick.get("home_win_prob", 0.5)
    away_prob: float = pick.get("away_win_prob", 1.0 - home_prob)
    spread: float | None = pick.get("spread")
    edge_pct: float | None = pick.get("edge_pct") or (
        value_bet["edge_pct"] if value_bet else None
    )
    market_prob: float | None = (
        value_bet["market_prob"] if value_bet else _spread_implied_prob(spread)
    )

    # ------------------------------------------------------------------
    # Rule 1: Away B2B
    # ------------------------------------------------------------------
    away_b2b = False
    if context and context.get("away_is_back_to_back"):
        away_b2b = True
    elif features is not None and "away_is_back_to_back" in features.index:
        away_b2b = bool(features["away_is_back_to_back"])

    if away_b2b:
        bullets.append(
            f"{away_name} playing 2nd game in 2 nights -- fatigue factor"
        )

    if len(bullets) >= 4:
        return bullets

    # ------------------------------------------------------------------
    # Rule 2: Strong home win probability
    # ------------------------------------------------------------------
    if home_prob > HIGH_WIN_PROB_THRESHOLD:
        implied_str = _pct(market_prob) if market_prob is not None else "N/A"
        bullets.append(
            f"Model shows strong home advantage -- {_pct(home_prob)} win probability"
            f" vs {implied_str} implied by the line"
        )

    if len(bullets) >= 4:
        return bullets

    # ------------------------------------------------------------------
    # Rule 3: Home team hot streak (last 10)
    # ------------------------------------------------------------------
    if features is not None and "home_win_pct_roll10" in features.index:
        home_last10_pct = features["home_win_pct_roll10"]
        if pd.notna(home_last10_pct):
            home_last10_wins = round(home_last10_pct * 10)
            home_last10_losses = 10 - home_last10_wins
            if home_last10_wins >= STRONG_RECENT_WINS:
                bullets.append(
                    f"{home_name} is {home_last10_wins}-{home_last10_losses}"
                    f" in their last 10 games"
                )

    if len(bullets) >= 4:
        return bullets

    # ------------------------------------------------------------------
    # Rule 4: Away team struggling (last 10)
    # ------------------------------------------------------------------
    if features is not None and "away_win_pct_roll10" in features.index:
        away_last10_pct = features["away_win_pct_roll10"]
        if pd.notna(away_last10_pct):
            away_last10_wins = round(away_last10_pct * 10)
            if away_last10_wins <= WEAK_RECENT_WINS:
                bullets.append(
                    f"{away_name} has struggled recently -- only {away_last10_wins}"
                    f" wins in their last 10 games"
                )

    if len(bullets) >= 4:
        return bullets

    # ------------------------------------------------------------------
    # Rule 5: Rest advantage
    # ------------------------------------------------------------------
    home_rest: float | None = None
    away_rest: float | None = None

    if context:
        home_rest = context.get("home_days_rest")
        away_rest = context.get("away_days_rest")

    if home_rest is None and features is not None:
        for col in ("home_days_rest", "home_rest_days"):
            if col in features.index and pd.notna(features[col]):
                home_rest = float(features[col])
                break

    if away_rest is None and features is not None:
        for col in ("away_days_rest", "away_rest_days"):
            if col in features.index and pd.notna(features[col]):
                away_rest = float(features[col])
                break

    if home_rest is not None and away_rest is not None:
        if home_rest > away_rest + REST_ADVANTAGE_DAYS:
            bullets.append(
                f"{home_name} has {int(home_rest)} days rest vs {away_name}'s"
                f" {int(away_rest)} -- significant rest advantage"
            )

    if len(bullets) >= 4:
        return bullets

    # ------------------------------------------------------------------
    # Rule 6: High model edge
    # ------------------------------------------------------------------
    if edge_pct is not None and edge_pct > HIGH_EDGE_THRESHOLD:
        bullets.append(
            f"Model edge of {edge_pct * 100:.1f}% -- significantly above our"
            f" threshold for value bets"
        )

    if len(bullets) >= 4:
        return bullets

    # ------------------------------------------------------------------
    # Rule 7: Net rating differential
    # ------------------------------------------------------------------
    if features is not None and "diff_net_rtg_game_roll10" in features.index:
        net_rtg_diff = features["diff_net_rtg_game_roll10"]
        if pd.notna(net_rtg_diff) and net_rtg_diff > HIGH_NET_RTG_DIFF:
            bullets.append(
                f"{home_name} outperforms {away_name} by"
                f" {net_rtg_diff:.1f} points per 100 possessions in net rating"
            )

    if len(bullets) >= 4:
        return bullets

    # ------------------------------------------------------------------
    # Rule 8: Pythagorean win% gap
    # ------------------------------------------------------------------
    if features is not None and "diff_pythagorean_win_pct_roll10" in features.index:
        pythag_diff = features["diff_pythagorean_win_pct_roll10"]
        if pd.notna(pythag_diff) and pythag_diff > HIGH_PYTHAG_DIFF:
            home_pythag = features.get("home_pythagorean_win_pct_roll10", float("nan"))
            away_pythag = features.get("away_pythagorean_win_pct_roll10", float("nan"))
            if pd.notna(home_pythag) and pd.notna(away_pythag):
                bullets.append(
                    f"Pythagorean win% gap favors {home_name}"
                    f" ({_pct(home_pythag)} vs {_pct(away_pythag)})"
                )
            else:
                bullets.append(
                    f"Pythagorean win% gap favors {home_name} by"
                    f" {pythag_diff * 100:.1f}%"
                )

    if len(bullets) >= 4:
        return bullets

    # ------------------------------------------------------------------
    # Rule 9: Default fallback — always ensure at least 2 bullets
    # ------------------------------------------------------------------
    # Primary fallback: model probability vs market line
    implied_str = _pct(market_prob) if market_prob is not None else "N/A"
    primary_fallback = (
        f"Model projects {home_name} as {_pct(home_prob)} favorites"
        f" -- market implies {implied_str}"
    )

    # Secondary fallback: recent form summary for both teams
    home_w10 = None
    away_w10 = None
    if features is not None:
        if "home_win_pct_roll10" in features.index and pd.notna(features["home_win_pct_roll10"]):
            home_w10 = round(features["home_win_pct_roll10"] * 10)
        if "away_win_pct_roll10" in features.index and pd.notna(features["away_win_pct_roll10"]):
            away_w10 = round(features["away_win_pct_roll10"] * 10)

    if home_w10 is not None and away_w10 is not None:
        secondary_fallback = (
            f"Recent form: {home_name} {home_w10}-{10 - home_w10} L10"
            f" vs {away_name} {away_w10}-{10 - away_w10} L10"
        )
    else:
        secondary_fallback = (
            f"Matchup analysis points to {home_name if home_prob >= 0.5 else away_name}"
            f" as the stronger side"
        )

    if not bullets:
        bullets.append(primary_fallback)
        bullets.append(secondary_fallback)
    elif len(bullets) < 2:
        # We have 1 bullet -- add secondary fallback if it differs from what is there
        if secondary_fallback not in bullets and primary_fallback not in bullets:
            bullets.append(secondary_fallback)
        elif primary_fallback not in bullets:
            bullets.append(primary_fallback)
        else:
            bullets.append(secondary_fallback)

    return bullets


# ---------------------------------------------------------------------------
# One-liner generator
# ---------------------------------------------------------------------------

def generate_one_liner(
    pick: dict[str, Any],
    recommended_side: str,
    value_bet: dict[str, Any] | None,
) -> str:
    home_name = pick.get("home_team_name", pick["home_team"])
    away_name = pick.get("away_team_name", pick["away_team"])
    home_prob: float = pick.get("home_win_prob", 0.5)
    away_prob: float = pick.get("away_win_prob", 1.0 - home_prob)
    market_prob: float | None = value_bet["market_prob"] if value_bet else None
    spread: float | None = pick.get("spread")
    projected_margin: float | None = pick.get("projected_margin")

    if recommended_side == pick["home_team"]:
        rec_name = home_name
        rec_model_prob = home_prob
    else:
        rec_name = away_name
        rec_model_prob = away_prob

    model_pct = _pct(rec_model_prob)

    # Build spread/margin suffix for extra context
    margin_suffix = ""
    if projected_margin is not None:
        direction = "by" if projected_margin >= 0 else "by"
        margin_suffix = f" (model projects home win by {abs(projected_margin):.1f})"
    elif spread is not None:
        margin_suffix = f" (line: {spread:+.1f})"

    if market_prob is not None:
        # market_prob from value_bets is always for the recommended side
        market_pct = _pct(market_prob)
        return (
            f"{rec_name} is undervalued -- books give them {market_pct}"
            f" but our model says {model_pct}{margin_suffix}"
        )

    if spread is not None:
        # Positive spread = home is favored; invert for away
        implied_home = _spread_implied_prob(spread)
        if implied_home is not None:
            implied = (
                implied_home
                if recommended_side == pick["home_team"]
                else 1.0 - implied_home
            )
            implied_str = _pct(implied)
        else:
            implied_str = "N/A"
        return (
            f"{rec_name} is undervalued -- books imply {implied_str}"
            f" but our model says {model_pct}{margin_suffix}"
        )

    base = f"Model projects {rec_name} to win with {model_pct} probability"
    return base + margin_suffix if margin_suffix else base


def generate_confidence_explanation(
    pick: dict[str, Any],
    features: "pd.Series | None",
    recommended_side: str,
) -> str:
    """
    Generate a short plain-English explanation of why the model has this
    confidence level (e.g. net rating gap, pythagorean win%, recent form).
    """
    home = pick["home_team"]
    home_name = pick.get("home_team_name", home)
    away_name = pick.get("away_team_name", pick["away_team"])
    home_prob: float = pick.get("home_win_prob", 0.5)
    away_prob: float = pick.get("away_win_prob", 1.0 - home_prob)
    confidence_tier: str = pick.get("confidence_tier", "")

    if recommended_side == home:
        rec_name = home_name
        rec_prob = home_prob
    else:
        rec_name = away_name
        rec_prob = away_prob

    tier_label = confidence_tier.upper() if confidence_tier else "MEDIUM"
    model_pct = f"{rec_prob * 100:.0f}%"

    # Start with the base confidence statement
    base = f"Model favors {rec_name} with {model_pct} confidence"

    if features is None:
        return f"{base} ({tier_label.lower()} confidence, no feature detail available)"

    # Attempt to add a primary driver
    driver_parts: list[str] = []

    # Net rating differential
    net_col = "diff_net_rtg_game_roll10"
    if net_col in features.index and pd.notna(features[net_col]):
        nr = float(features[net_col])
        if abs(nr) >= 2.0:
            sign_txt = "advantage" if nr > 0 else "disadvantage"
            driver_parts.append(f"net rating {sign_txt} of {nr:+.1f} pts/100")

    # Pythagorean win% gap
    pyth_col = "diff_pythagorean_win_pct_roll10"
    if pyth_col in features.index and pd.notna(features[pyth_col]):
        pv = float(features[pyth_col])
        if abs(pv) >= 0.05:
            driver_parts.append(f"pythagorean win% gap {pv:+.1%}")

    # Recent form (L10)
    h_l10_col = "home_win_pct_roll10"
    a_l10_col = "away_win_pct_roll10"
    if h_l10_col in features.index and a_l10_col in features.index:
        h_w10 = features[h_l10_col]
        a_w10 = features[a_l10_col]
        if pd.notna(h_w10) and pd.notna(a_w10):
            hw = round(float(h_w10) * 10)
            aw = round(float(a_w10) * 10)
            driver_parts.append(f"recent form {hw}-{10-hw} vs {aw}-{10-aw} (L10)")

    if driver_parts:
        driver_str = "; ".join(driver_parts[:2])
        return f"{base} based on {driver_str}"

    return f"{base} ({tier_label.lower()} confidence)"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def build_explainers(
    picks: list[dict[str, Any]],
    value_bets: list[dict[str, Any]],
    matchup_df: pd.DataFrame,
    game_context: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matchup_index = build_matchup_index(matchup_df)
    vb_index = build_value_bet_index(value_bets)
    ctx_index = build_context_index(game_context)

    results: list[dict[str, Any]] = []

    for pick in picks:
        home = pick["home_team"]
        away = pick["away_team"]
        game_date = pick.get("game_date", "")

        # Locate feature row: exact game date first, then most-recent fallback
        features: pd.Series | None = None
        exact_key = (home, away)
        if exact_key in matchup_index:
            features = matchup_index[exact_key]

        # Supplemental data
        vb_key = (home, away, game_date)
        value_bet = vb_index.get(vb_key)

        ctx_key = (home, away, game_date)
        context = ctx_index.get(ctx_key)

        # Determine recommended side
        if value_bet:
            recommended_side = value_bet.get("recommended_side", pick.get("predicted_winner", home))
        else:
            recommended_side = pick.get("ats_pick") or pick.get("predicted_winner") or home

        bullets = generate_bullets(pick, features, value_bet, context)
        one_liner = generate_one_liner(pick, recommended_side, value_bet)
        confidence_explanation = generate_confidence_explanation(
            pick, features, recommended_side
        )

        results.append(
            {
                "home_team": home,
                "away_team": away,
                "game_date": game_date,
                "bullets": bullets,
                "recommended_side": recommended_side,
                "one_liner": one_liner,
                "confidence_explanation": confidence_explanation,
            }
        )

    return results


def main() -> None:
    print("Loading picks...")
    picks = load_json(PICKS_PATH)
    print(f"  {len(picks)} games loaded from todays_picks.json")

    print("Loading value bets...")
    value_bets = load_json(VALUE_BETS_PATH)
    print(f"  {len(value_bets)} value bet records loaded")

    print("Loading matchup features...")
    matchup_df = load_matchup_features(MATCHUP_CSV)
    print(f"  {len(matchup_df):,} matchup rows, {len(matchup_df.columns)} columns")

    print("Loading game context (optional)...")
    game_context = load_game_context(GAME_CONTEXT_PATH)
    print(f"  {len(game_context)} context records loaded")

    print("Generating explainers...")
    explainers = build_explainers(picks, value_bets, matchup_df, game_context)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(explainers, fh, indent=2)

    print(f"Wrote {len(explainers)} explainers -> {OUTPUT_PATH}")

    # Print a sample to stdout for verification
    for ex in explainers[:2]:
        print(f"\n  {ex['home_team']} vs {ex['away_team']} ({ex['game_date']})")
        for b in ex["bullets"]:
            print(f"    - {b}")
        print(f"    -> {ex['one_liner']}")
        print(f"    confidence: {ex['confidence_explanation']}")


if __name__ == "__main__":
    main()
