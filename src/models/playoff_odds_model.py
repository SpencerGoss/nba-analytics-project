"""
Playoff Odds Simulation Model  (v2 — improved)
================================================
Monte Carlo simulation to estimate each team's probability of making the
playoffs and winning the championship.

Key improvements over v1:
  • ML-based win probabilities  — loads the trained game outcome model
    and each team's most recent rolling stats to produce a calibrated
    win probability for every simulated game.  Falls back gracefully
    to Bradley-Terry if the model is not yet trained.
  • Regression-to-mean        — remaining games apply a small blend
    toward .500 (15 %) to reflect that current form is imperfect.
  • Finals simulation          — properly simulates the NBA Finals as a
    best-of-7 series between the two conference champions instead of the
    previous approximation of halving conf_title_prob.
  • Better uncertainty bounds  — outputs 5th/95th percentile win totals
    from the simulation so you can see the uncertainty range.

Approach:
  1. Load standings (actual W/L) and team rolling features.
  2. For each remaining game, sample a win probability using:
       a. ML model prediction (preferred): builds a synthetic matchup row
          from each team's current rolling stats and runs the classifier.
       b. Bradley-Terry fallback: uses win % with home advantage.
  3. Simulate N seasons → track playoff appearances, conference titles, Finals.
  4. Report probabilities, print table, save CSV.

Usage:
    python src/models/playoff_odds_model.py

    Or import:
        from src.models.playoff_odds_model import simulate_playoff_odds
        results = simulate_playoff_odds(season="202425")
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")


# ── Config ─────────────────────────────────────────────────────────────────────

STANDINGS_PATH       = "data/processed/standings.csv"
GAME_LOG_PATH        = "data/processed/team_game_logs.csv"
TEAM_FEATURES_PATH   = "data/features/team_game_features.csv"
MATCHUP_FEATURES_PATH= "data/features/game_matchup_features.csv"
ARTIFACTS_DIR        = "models/artifacts"
OUTPUT_PATH          = "data/features/playoff_odds.csv"

N_SIMULATIONS        = 2_000
TOTAL_GAMES          = 82
HOME_ADVANTAGE       = 0.60      # win prob for home team at equal strength
PLAYOFF_SPOTS        = 8         # direct playoff seeds per conference
PLAYOFF_IN_SPOTS     = 10        # play-in eligible per conference

# Regression-to-mean blend: remaining games, blend this fraction toward .500
REGRESSION_ALPHA     = 0.15


# ── ML model loader ────────────────────────────────────────────────────────────

def _load_game_model(artifacts_dir: str = ARTIFACTS_DIR):
    """
    Try to load the trained game outcome model + feature list.
    Returns (model, feat_cols) or (None, None) if not available.
    """
    model_path = os.path.join(artifacts_dir, "game_outcome_model.pkl")
    feat_path  = os.path.join(artifacts_dir, "game_outcome_features.pkl")
    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        return None, None
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(feat_path, "rb") as f:
            feat_cols = pickle.load(f)
        return model, feat_cols
    except Exception as e:
        print(f"  Warning: could not load game outcome model: {e}")
        return None, None


def _load_team_current_features(
    team_features_path: str = TEAM_FEATURES_PATH,
) -> pd.DataFrame:
    """
    Load team_game_features.csv and return each team's MOST RECENT row.
    This represents the team's current rolling state (last 10 games, etc.).
    """
    if not os.path.exists(team_features_path):
        return pd.DataFrame()
    df = pd.read_csv(team_features_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    # Most recent game per team
    latest = (
        df.sort_values("game_date")
        .groupby("team_id", as_index=False)
        .last()
    )
    return latest


def _build_matchup_row(
    home_feats: pd.Series,
    away_feats: pd.Series,
    feat_cols: list,
) -> pd.DataFrame:
    """
    Construct a single matchup feature row for the game outcome model from
    the two teams' current rolling-stat vectors.

    The matchup model uses columns prefixed home_*, away_*, and diff_*.
    We assemble these from the raw team feature rows.
    """
    row = {}
    for fc in feat_cols:
        if fc.startswith("home_"):
            base = fc[5:]
            row[fc] = home_feats.get(base, np.nan)
        elif fc.startswith("away_"):
            base = fc[5:]
            row[fc] = away_feats.get(base, np.nan)
        elif fc.startswith("diff_"):
            base = fc[5:]
            row[fc] = home_feats.get(base, np.nan) - away_feats.get(base, np.nan)
        else:
            # Column with no prefix — look in both, prefer home
            row[fc] = home_feats.get(fc, away_feats.get(fc, np.nan))

    return pd.DataFrame([row])[feat_cols].fillna(0)


# ── Win probability ────────────────────────────────────────────────────────────

def bt_win_prob(win_pct_a: float, win_pct_b: float, home: bool = True) -> float:
    """
    Bradley-Terry win probability with home advantage.
    Used as fallback when ML model is unavailable.
    """
    wp_a = np.clip(win_pct_a, 0.01, 0.99)
    wp_b = np.clip(win_pct_b, 0.01, 0.99)
    r_a  = wp_a / (1 - wp_a)
    r_b  = wp_b / (1 - wp_b)
    if home:
        r_a *= HOME_ADVANTAGE / (1 - HOME_ADVANTAGE)
    return r_a / (r_a + r_b)


def get_win_prob(
    home_id: int,
    away_id: int,
    team_wp: dict,
    team_features: pd.DataFrame,
    model,
    feat_cols: list,
    home: bool = True,
    regression_alpha: float = REGRESSION_ALPHA,
) -> float:
    """
    Return P(home team wins) for a single game.

    Priority order:
      1. ML model (if loaded and feature data available)
      2. Bradley-Terry fallback

    `regression_alpha` blends the base win probability toward 0.5 to
    account for uncertainty in future games.
    """
    base_prob = None

    # ── Try ML model ──────────────────────────────────────────────────────────
    if model is not None and not team_features.empty:
        home_row = team_features[team_features["team_id"] == home_id]
        away_row = team_features[team_features["team_id"] == away_id]
        if not home_row.empty and not away_row.empty:
            try:
                matchup_row = _build_matchup_row(
                    home_row.iloc[0],
                    away_row.iloc[0],
                    feat_cols,
                )
                base_prob = float(model.predict_proba(matchup_row)[0][1])
            except Exception:
                pass  # fall through to Bradley-Terry

    # ── Bradley-Terry fallback ────────────────────────────────────────────────
    if base_prob is None:
        wp_home = team_wp.get(home_id, 0.5)
        wp_away = team_wp.get(away_id, 0.5)
        base_prob = bt_win_prob(wp_home, wp_away, home=home)

    # ── Regression-to-mean ────────────────────────────────────────────────────
    return (1 - regression_alpha) * base_prob + regression_alpha * 0.5


# ── Playoff series simulation ──────────────────────────────────────────────────

def _play_series(
    t_a: int,
    t_b: int,
    sim_wp: dict,
    team_features: pd.DataFrame,
    model,
    feat_cols: list,
    rng: np.random.Generator,
    wp_cache: dict = None,
) -> int:
    """Simulate a best-of-7 series. Returns the winner's team_id."""
    wins_a, wins_b = 0, 0
    home_team = t_a     # higher seed hosts games 1,2,5,7
    away_team = t_b

    for game_num in range(1, 8):
        if wins_a == 4 or wins_b == 4:
            break
        if game_num in (1, 2, 5, 7):
            h, a = home_team, away_team
        else:
            h, a = away_team, home_team

        # Use cache if available, else compute on the fly
        if wp_cache is not None and (h, a) in wp_cache:
            p = wp_cache[(h, a)]
        else:
            p = get_win_prob(
                h, a, sim_wp, team_features, model, feat_cols,
                home=True, regression_alpha=0.0,
            )
        if rng.random() < p:
            if h == t_a: wins_a += 1
            else:        wins_b += 1
        else:
            if h == t_a: wins_b += 1
            else:        wins_a += 1

    return t_a if wins_a == 4 else t_b


# ── Monte Carlo simulation ─────────────────────────────────────────────────────

def simulate_playoff_odds(
    season: str          = None,
    standings_path: str  = STANDINGS_PATH,
    game_log_path: str   = GAME_LOG_PATH,
    output_path: str     = OUTPUT_PATH,
    artifacts_dir: str   = ARTIFACTS_DIR,
    n_sims: int          = N_SIMULATIONS,
) -> pd.DataFrame:
    """
    Run Monte Carlo playoff odds simulation for the given season.

    Args:
        season: e.g. "202425". If None, uses the most recent season.

    Returns:
        DataFrame with columns:
          team_id, team_name, conference, w, l, w_pct,
          playoff_prob, playin_prob, conf_title_prob, title_prob,
          sim_wins_p5, sim_wins_p50, sim_wins_p95
    """
    print("=" * 60)
    print("PLAYOFF ODDS SIMULATION  (v2 — improved)")
    print("=" * 60)

    # ── Load standings ────────────────────────────────────────────────────────
    standings = pd.read_csv(standings_path)
    if season is None:
        season = standings["season"].astype(str).max()
    standings = standings[standings["season"].astype(str) == season].copy()

    if standings.empty:
        raise ValueError(f"No standings found for season {season}")

    print(f"\nSeason: {season} | Teams: {len(standings)}")

    if "w_pct" not in standings.columns:
        standings["w_pct"] = standings["w"] / (standings["w"] + standings["l"])
    standings["w_pct"]      = standings["w_pct"].fillna(0.5)
    standings["conference"] = standings["conference"].str.strip().str.title()

    # ── Load game model and team features ────────────────────────────────────
    model, feat_cols = _load_game_model(artifacts_dir)
    team_features    = _load_team_current_features()

    if model is not None:
        print(f"  Using ML model for win probabilities ({len(feat_cols)} features)")
    else:
        print("  ML model not found — using Bradley-Terry win probabilities")
        print("  (run game_outcome_model.py first for ML-based odds)")

    # ── Load game logs for games-played count ─────────────────────────────────
    game_logs = pd.read_csv(game_log_path)
    played = (
        game_logs[game_logs["season"].astype(str) == season]
        .groupby("team_id")["game_id"].nunique()
        .to_dict()
    )

    teams     = standings[["team_id", "team_name", "conference", "w", "l", "w_pct"]].copy()
    team_ids  = teams["team_id"].tolist()
    team_wp   = teams.set_index("team_id")["w_pct"].to_dict()
    team_conf = teams.set_index("team_id")["conference"].to_dict()
    team_wins = teams.set_index("team_id")["w"].to_dict()

    # ── Pre-compute matchup win-probability cache ──────────────────────────────
    # Call the ML model (or BT) once per ordered pair (home, away) so the
    # inner simulation loop only does a dict lookup, not a predict_proba call.
    print("\nPre-computing matchup win probabilities...")
    wp_cache = {}   # (home_id, away_id) -> float P(home wins)
    for h_id in team_ids:
        for a_id in team_ids:
            if h_id == a_id:
                continue
            wp_cache[(h_id, a_id)] = get_win_prob(
                h_id, a_id, team_wp, team_features, model,
                feat_cols if feat_cols else [], home=True,
            )
    print(f"  Cached {len(wp_cache)} matchup probabilities.")

    # ── Simulation ────────────────────────────────────────────────────────────
    print(f"\nRunning {n_sims:,} simulations...")

    playoff_count    = {t: 0 for t in team_ids}
    playin_count     = {t: 0 for t in team_ids}
    conf_title_count = {t: 0 for t in team_ids}
    title_count      = {t: 0 for t in team_ids}
    sim_win_totals   = {t: [] for t in team_ids}

    rng = np.random.default_rng(seed=42)

    for sim in range(n_sims):
        sim_wins = dict(team_wins)       # copy of actual wins so far
        sim_wp_current = dict(team_wp)   # current win% for this simulation

        # ── Simulate remaining regular-season games ───────────────────────────
        games_left = {t: max(0, TOTAL_GAMES - played.get(t, 0)) for t in team_ids}

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
                home = rng.random() > 0.5
                home_id = t_id if home else opp_id
                away_id = opp_id if home else t_id

                # Use pre-computed cache for speed
                p_home_win = wp_cache.get((home_id, away_id), 0.5)
                t_wins = (home_id == t_id) == (rng.random() < p_home_win)
                if t_wins:
                    sim_wins[t_id] = sim_wins.get(t_id, 0) + 1

        # Update win% for playoff sim
        for t in team_ids:
            total = TOTAL_GAMES
            sim_wp_current[t] = sim_wins.get(t, 0) / total

        # Store win totals for percentile calculation
        for t in team_ids:
            sim_win_totals[t].append(sim_wins.get(t, 0))

        # ── Conference standings + playoff seeding ─────────────────────────────
        conf_winners = {}
        for conf in ["East", "West"]:
            conf_teams = [t for t in team_ids if team_conf.get(t) == conf]
            sorted_conf = sorted(conf_teams, key=lambda t: sim_wins.get(t, 0), reverse=True)

            for rank, t_id in enumerate(sorted_conf):
                if rank < PLAYOFF_SPOTS:
                    playoff_count[t_id] += 1
                    playin_count[t_id]  += 1
                elif rank < PLAYOFF_IN_SPOTS:
                    playin_count[t_id]  += 1

            # ── Simulate playoff bracket (top 8 seeds) ─────────────────────────
            bracket = sorted_conf[:PLAYOFF_SPOTS]
            if len(bracket) >= 8:
                qf = [
                    _play_series(bracket[0], bracket[7], sim_wp_current, team_features, None, [], rng, wp_cache),
                    _play_series(bracket[1], bracket[6], sim_wp_current, team_features, None, [], rng, wp_cache),
                    _play_series(bracket[2], bracket[5], sim_wp_current, team_features, None, [], rng, wp_cache),
                    _play_series(bracket[3], bracket[4], sim_wp_current, team_features, None, [], rng, wp_cache),
                ]
                sf = [
                    _play_series(qf[0], qf[3], sim_wp_current, team_features, None, [], rng, wp_cache),
                    _play_series(qf[1], qf[2], sim_wp_current, team_features, None, [], rng, wp_cache),
                ]
                conf_champ = _play_series(sf[0], sf[1], sim_wp_current, team_features, None, [], rng, wp_cache)
                conf_title_count[conf_champ] += 1
                conf_winners[conf] = conf_champ

        # ── NBA Finals ────────────────────────────────────────────────────────
        if "East" in conf_winners and "West" in conf_winners:
            east_champ = conf_winners["East"]
            west_champ = conf_winners["West"]
            home_team = east_champ if sim_wins.get(east_champ, 0) >= sim_wins.get(west_champ, 0) else west_champ
            away_team = west_champ if home_team == east_champ else east_champ
            champion  = _play_series(home_team, away_team, sim_wp_current, team_features, None, [], rng, wp_cache)
            title_count[champion] += 1

        if (sim + 1) % 2000 == 0:
            print(f"  Completed {sim + 1:,} / {n_sims:,} simulations...")

    # ── Build results ─────────────────────────────────────────────────────────
    results = teams.copy()
    results["playoff_prob"]    = results["team_id"].map(lambda t: playoff_count[t]    / n_sims)
    results["playin_prob"]     = results["team_id"].map(lambda t: playin_count[t]     / n_sims)
    results["conf_title_prob"] = results["team_id"].map(lambda t: conf_title_count[t] / n_sims)
    results["title_prob"]      = results["team_id"].map(lambda t: title_count[t]      / n_sims)

    # Win-total percentiles from simulation
    results["sim_wins_p5"]  = results["team_id"].map(lambda t: int(np.percentile(sim_win_totals[t],  5)))
    results["sim_wins_p50"] = results["team_id"].map(lambda t: int(np.percentile(sim_win_totals[t], 50)))
    results["sim_wins_p95"] = results["team_id"].map(lambda t: int(np.percentile(sim_win_totals[t], 95)))

    results = results.sort_values(["conference", "playoff_prob"], ascending=[True, False])

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n── PLAYOFF ODDS ──────────────────────────────────────────────────")
    print(f"{'Team':<25} {'Conf':<5} {'W':>3} {'L':>3} {'W%':>6} "
          f"{'Playoff':>9} {'Play-In':>8} {'ConfTitle':>10} {'Title':>7} "
          f"{'WinRange':>12}")
    print("─" * 95)
    for _, row in results.iterrows():
        print(
            f"{row['team_name']:<25} {row['conference']:<5} "
            f"{int(row['w']):>3} {int(row['l']):>3} {row['w_pct']:>6.3f} "
            f"{row['playoff_prob']:>8.1%}  {row['playin_prob']:>7.1%}  "
            f"{row['conf_title_prob']:>9.1%}  {row['title_prob']:>6.1%}  "
            f"{int(row['sim_wins_p5'])}-{int(row['sim_wins_p50'])}-{int(row['sim_wins_p95'])}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = simulate_playoff_odds()
