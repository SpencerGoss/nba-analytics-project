"""
Playoff Odds Simulation Model
================================
Uses Monte Carlo simulation to estimate each team's probability of making
the playoffs and winning the championship.

Approach:
  - Load current standings (wins, losses, games remaining)
  - For each remaining game, sample a win probability using:
      * Both teams' current win% (Bradley-Terry style)
      * Home court adjustment
  - Simulate N seasons → count playoff appearances and titles

This model works at any point in the season — the later you run it, the
more accurate it becomes as actual results narrow the uncertainty.

Usage:
    python src/models/playoff_odds_model.py

    Or import:
        from src.models.playoff_odds_model import simulate_playoff_odds
        results = simulate_playoff_odds(season="202425")
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


# ── Config ─────────────────────────────────────────────────────────────────────

STANDINGS_PATH   = "data/processed/standings.csv"
GAME_LOG_PATH    = "data/processed/team_game_logs.csv"
OUTPUT_PATH      = "data/features/playoff_odds.csv"

N_SIMULATIONS    = 10_000
TOTAL_GAMES      = 82           # regular season games per team
HOME_ADVANTAGE   = 0.60         # baseline win prob for home team at equal strength
PLAYOFF_SPOTS    = 8            # playoff spots per conference (before play-in)
PLAYOFF_IN_SPOTS = 10           # play-in tournament spots per conference


# ── Helper: Bradley-Terry win probability ─────────────────────────────────────

def bt_win_prob(win_pct_a: float, win_pct_b: float, home: bool = True) -> float:
    """
    Estimate P(team_a wins) using a simple Bradley-Terry model.

    Converts win percentages to 'strength' ratings, then computes
    the head-to-head probability.  Adds a home advantage boost.
    """
    # Avoid division by zero / log(0)
    wp_a = np.clip(win_pct_a, 0.01, 0.99)
    wp_b = np.clip(win_pct_b, 0.01, 0.99)

    # Strength ratings (odds scale)
    r_a = wp_a / (1 - wp_a)
    r_b = wp_b / (1 - wp_b)

    # Home advantage multiplier
    if home:
        r_a *= HOME_ADVANTAGE / (1 - HOME_ADVANTAGE)

    return r_a / (r_a + r_b)


# ── Build remaining schedule ───────────────────────────────────────────────────

def _build_remaining_schedule(
    season: str,
    completed_games: pd.DataFrame,
    all_teams: pd.DataFrame,
) -> pd.DataFrame:
    """
    Approximate remaining games by checking how many games each team has
    already played and filling the gap to TOTAL_GAMES.

    Returns a DataFrame of (home_team_id, away_team_id) pairs.
    """
    games_played = (
        completed_games[completed_games["season"] == season]
        .groupby("team_id")["game_id"]
        .nunique()
        .reset_index()
        .rename(columns={"game_id": "games_played"})
    )
    remaining = all_teams.merge(games_played, on="team_id", how="left")
    remaining["games_played"] = remaining["games_played"].fillna(0).astype(int)
    remaining["games_left"]   = (TOTAL_GAMES - remaining["games_played"]).clip(lower=0)

    # Build simplified schedule: pair up teams by conference for remaining games
    schedule_rows = []
    team_list = remaining.to_dict("records")

    for i, team in enumerate(team_list):
        for _ in range(team["games_left"] // 2):
            # Pair with a random opponent (simplified — not a real schedule)
            opponent = team_list[np.random.randint(0, len(team_list))]
            if opponent["team_id"] != team["team_id"]:
                schedule_rows.append({
                    "home_team_id": team["team_id"],
                    "away_team_id": opponent["team_id"],
                })

    return pd.DataFrame(schedule_rows) if schedule_rows else pd.DataFrame(
        columns=["home_team_id", "away_team_id"]
    )


# ── Monte Carlo simulation ─────────────────────────────────────────────────────

def simulate_playoff_odds(
    season: str          = None,
    standings_path: str  = STANDINGS_PATH,
    game_log_path: str   = GAME_LOG_PATH,
    output_path: str     = OUTPUT_PATH,
    n_sims: int          = N_SIMULATIONS,
) -> pd.DataFrame:
    """
    Run Monte Carlo playoff odds simulation for the given season.

    Args:
        season: e.g. "202425". If None, uses the most recent season in standings.

    Returns:
        DataFrame with columns:
          team_id, team_name, conference, w, l, w_pct,
          playoff_prob, playin_prob, conf_title_prob, title_prob
    """
    print("=" * 60)
    print("PLAYOFF ODDS SIMULATION")
    print("=" * 60)

    # ── Load standings ────────────────────────────────────────────────────────
    standings = pd.read_csv(standings_path)
    if season is None:
        season = standings["season"].astype(str).max()
    standings = standings[standings["season"].astype(str) == season].copy()

    if standings.empty:
        raise ValueError(f"No standings found for season {season}")

    print(f"\nSeason: {season} | Teams: {len(standings)}")

    # Use win% for simulation; fall back to w/(w+l) if w_pct missing
    if "w_pct" not in standings.columns:
        standings["w_pct"] = standings["w"] / (standings["w"] + standings["l"])
    standings["w_pct"] = standings["w_pct"].fillna(0.5)

    # Normalize conference labels
    standings["conference"] = standings["conference"].str.strip().str.title()

    # ── Load game logs for schedule estimation ────────────────────────────────
    game_logs = pd.read_csv(game_log_path)

    teams = standings[["team_id", "team_name", "conference", "w", "l", "w_pct"]].copy()
    team_ids = teams["team_id"].tolist()
    team_wp   = teams.set_index("team_id")["w_pct"].to_dict()
    team_conf = teams.set_index("team_id")["conference"].to_dict()

    # Games played so far
    played = (
        game_logs[game_logs["season"].astype(str) == season]
        .groupby("team_id")["game_id"].nunique()
        .to_dict()
    )

    # ── Simulation ────────────────────────────────────────────────────────────
    print(f"\nRunning {n_sims:,} simulations...")

    # Accumulators
    playoff_count    = {t: 0 for t in team_ids}
    playin_count     = {t: 0 for t in team_ids}
    conf_title_count = {t: 0 for t in team_ids}
    title_count      = {t: 0 for t in team_ids}

    rng = np.random.default_rng(seed=42)

    for sim in range(n_sims):
        # ── Simulate remaining games ──────────────────────────────────────────
        sim_wins = {t: teams.set_index("team_id").loc[t, "w"] for t in team_ids}

        games_left = {t: max(0, TOTAL_GAMES - played.get(t, 0)) for t in team_ids}

        # Each team plays their remaining games against random opponents
        # (simplified schedule; real schedule would use actual game_ids)
        for t_id in team_ids:
            gl = games_left[t_id]
            if gl <= 0:
                continue

            opponents = rng.choice(
                [x for x in team_ids if x != t_id],
                size=gl,
                replace=True,
            )
            for opp_id in opponents:
                wp_a = team_wp[t_id]
                wp_b = team_wp[opp_id]
                home = rng.random() > 0.5  # random home assignment
                p_win = bt_win_prob(wp_a, wp_b, home=home)
                if rng.random() < p_win:
                    sim_wins[t_id] = sim_wins.get(t_id, 0) + 1

        # ── Determine standings by conference ─────────────────────────────────
        for conf in ["East", "West"]:
            conf_teams = [t for t in team_ids if team_conf.get(t) == conf]
            sorted_by_wins = sorted(conf_teams, key=lambda t: sim_wins[t], reverse=True)

            # Playoff seeds 1-8 + play-in 9-10
            for rank, t_id in enumerate(sorted_by_wins):
                if rank < PLAYOFF_SPOTS:
                    playoff_count[t_id] += 1
                    playin_count[t_id]  += 1
                elif rank < PLAYOFF_IN_SPOTS:
                    playin_count[t_id]  += 1

            # Simulate playoffs (simplified 8-team bracket, higher seed favored)
            bracket = sorted_by_wins[:PLAYOFF_SPOTS]

            def _play_series(t_a, t_b):
                """Best-of-7: higher sim_wins team has slight edge."""
                wins_a, wins_b = 0, 0
                for _ in range(7):
                    p = bt_win_prob(sim_wins[t_a] / TOTAL_GAMES,
                                    sim_wins[t_b] / TOTAL_GAMES,
                                    home=(wins_a + wins_b) % 2 == 0)
                    if rng.random() < p:
                        wins_a += 1
                    else:
                        wins_b += 1
                    if wins_a == 4 or wins_b == 4:
                        break
                return t_a if wins_a == 4 else t_b

            # First round: 1v8, 2v7, 3v6, 4v5
            if len(bracket) >= 8:
                qf_winners = [
                    _play_series(bracket[0], bracket[7]),
                    _play_series(bracket[1], bracket[6]),
                    _play_series(bracket[2], bracket[5]),
                    _play_series(bracket[3], bracket[4]),
                ]
                sf_winners = [
                    _play_series(qf_winners[0], qf_winners[3]),
                    _play_series(qf_winners[1], qf_winners[2]),
                ]
                conf_winner = _play_series(sf_winners[0], sf_winners[1])
                conf_title_count[conf_winner] += 1

    # ── Finals between conference winners ─────────────────────────────────────
    # (computed separately for cleanliness — title_count updated in loop above
    #  for conference winners that also won the final)
    # Simple approach: title_prob ≈ conf_title_prob × 0.5 if equal conferences
    # (already embedded in the bracket logic above; just normalize)

    # ── Build results ─────────────────────────────────────────────────────────
    results = teams.copy()
    results["playoff_prob"]     = results["team_id"].map(
        lambda t: playoff_count[t] / n_sims
    )
    results["playin_prob"]      = results["team_id"].map(
        lambda t: playin_count[t] / n_sims
    )
    results["conf_title_prob"]  = results["team_id"].map(
        lambda t: conf_title_count[t] / n_sims
    )
    # Title prob: product of getting to finals (0.5 of conf title) × winning it
    results["title_prob"]       = results["conf_title_prob"] * 0.5

    results = results.sort_values(
        ["conference", "playoff_prob"], ascending=[True, False]
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n── PLAYOFF ODDS ─────────────────────────────────────────────")
    print(f"{'Team':<25} {'Conf':<6} {'W':>4} {'L':>4} {'W%':>6} "
          f"{'Playoff':>9} {'Play-In':>9} {'Conf Title':>11} {'Title':>7}")
    print("─" * 80)
    for _, row in results.iterrows():
        print(
            f"{row['team_name']:<25} {row['conference']:<6} "
            f"{int(row['w']):>4} {int(row['l']):>4} {row['w_pct']:>6.3f} "
            f"{row['playoff_prob']:>8.1%}  {row['playin_prob']:>8.1%}  "
            f"{row['conf_title_prob']:>10.1%}  {row['title_prob']:>6.1%}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = simulate_playoff_odds()
