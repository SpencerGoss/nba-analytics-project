"""Walk-forward betting backtest with realistic vig."""
import numpy as np
from typing import Optional


# Standard American odds vig
DEFAULT_VIG = 0.0455  # -110 lines: pay $110 to win $100


def run_backtest(
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    actuals: np.ndarray,
    edge_threshold: float = 0.03,
    vig: float = DEFAULT_VIG,
    unit_size: float = 1.0,
    kelly: bool = False,
    kelly_fraction: float = 0.25,
) -> dict:
    """
    Run a flat-bet or Kelly-sized backtest.

    Parameters
    ----------
    model_probs : P(home_win) from model
    market_probs : P(home_win) implied by market odds (after devigging)
    actuals : 1 = home win, 0 = away win
    edge_threshold : minimum |model - market| to trigger a bet
    vig : bookmaker margin (0.0455 = -110 lines)
    unit_size : base bet size (flat bet mode)
    kelly : if True, size bets by fractional Kelly criterion
    kelly_fraction : Kelly multiplier (0.25 = quarter Kelly)
    """
    edges = model_probs - market_probs

    # Determine which games to bet on
    bet_mask = np.abs(edges) >= edge_threshold
    n_bets = int(np.sum(bet_mask))

    if n_bets == 0:
        return {
            "roi": 0.0,
            "win_rate": 0.0,
            "n_bets": 0,
            "profit": 0.0,
            "max_drawdown": 0.0,
            "avg_edge": 0.0,
            "total_wagered": 0.0,
        }

    # For each bet: positive edge -> bet home, negative -> bet away
    bet_edges = edges[bet_mask]
    bet_actuals = actuals[bet_mask]
    bet_model_probs = model_probs[bet_mask]

    # Bet direction: True = home, False = away
    bet_home = bet_edges > 0

    # Did bet win?
    bet_won = (bet_home & (bet_actuals == 1)) | (~bet_home & (bet_actuals == 0))

    # Bet sizing
    if kelly:
        # Kelly: f = (bp - q) / b where b = payout odds, p = model prob, q = 1-p
        payout_odds = (1 - vig) / 1  # simplified: bet 1 to win (1 - vig)
        bet_prob = np.where(bet_home, bet_model_probs, 1 - bet_model_probs)
        raw_kelly = (payout_odds * bet_prob - (1 - bet_prob)) / payout_odds
        raw_kelly = np.maximum(raw_kelly, 0)  # no negative bets
        bet_sizes = raw_kelly * kelly_fraction * unit_size
    else:
        bet_sizes = np.full(n_bets, unit_size)

    total_wagered = float(np.sum(bet_sizes))

    # P&L per bet: win -> gain (1 - vig) * stake, lose -> lose stake
    pnl = np.where(bet_won, bet_sizes * (1 - vig), -bet_sizes)

    profit = float(np.sum(pnl))
    roi = profit / total_wagered if total_wagered > 0 else 0.0
    win_rate = float(np.mean(bet_won))
    avg_edge = float(np.mean(np.abs(bet_edges)))

    # Max drawdown from cumulative P&L
    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_max
    max_drawdown = float(np.min(drawdowns))

    return {
        "roi": roi,
        "win_rate": win_rate,
        "n_bets": n_bets,
        "profit": profit,
        "max_drawdown": max_drawdown,
        "avg_edge": avg_edge,
        "total_wagered": total_wagered,
    }


def walk_forward_backtest(
    df,
    model_prob_col: str = "model_prob",
    market_prob_col: str = "market_prob",
    actual_col: str = "actual",
    date_col: str = "game_date",
    train_seasons: int = 3,
    test_seasons: int = 1,
    edge_threshold: float = 0.03,
    vig: float = DEFAULT_VIG,
) -> dict:
    """
    Walk-forward backtest: train on N seasons, test on next season, slide forward.

    Returns aggregate metrics across all test windows.
    """
    import pandas as pd

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format="mixed")
    df = df.sort_values(date_col).reset_index(drop=True)

    # Extract season (Oct-Jun -> season starts in Oct)
    df["_season"] = df[date_col].apply(
        lambda d: d.year if d.month >= 10 else d.year - 1
    )
    seasons = sorted(df["_season"].unique())

    all_results = []

    for i in range(train_seasons, len(seasons)):
        test_season = seasons[i]
        test_mask = df["_season"] == test_season
        test_df = df[test_mask]

        if len(test_df) == 0:
            continue

        result = run_backtest(
            model_probs=test_df[model_prob_col].values,
            market_probs=test_df[market_prob_col].values,
            actuals=test_df[actual_col].values,
            edge_threshold=edge_threshold,
            vig=vig,
        )
        result["test_season"] = test_season
        all_results.append(result)

    if not all_results:
        return {"roi": 0.0, "win_rate": 0.0, "n_bets": 0, "seasons": []}

    total_wagered = sum(r["total_wagered"] for r in all_results)
    total_profit = sum(r["profit"] for r in all_results)
    total_bets = sum(r["n_bets"] for r in all_results)
    total_wins = sum(r["win_rate"] * r["n_bets"] for r in all_results)

    return {
        "roi": total_profit / total_wagered if total_wagered > 0 else 0.0,
        "win_rate": total_wins / total_bets if total_bets > 0 else 0.0,
        "n_bets": total_bets,
        "profit": total_profit,
        "total_wagered": total_wagered,
        "seasons": all_results,
    }
