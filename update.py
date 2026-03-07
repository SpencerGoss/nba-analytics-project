"""
NBA Analytics Update Script

Refreshes the current NBA season from NBA API endpoints and rebuilds processed CSVs.

Run daily (or on demand):
    python update.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.data.get_game_log import get_team_game_logs
from src.data.get_player_game_logs import get_player_game_logs
from src.data.get_player_master import get_player_master
from src.data.get_player_stats import get_player_stats_all_seasons
from src.data.get_player_stats_advanced import get_player_stats_advanced
from src.data.get_player_stats_clutch import get_player_stats_clutch
from src.data.get_player_stats_scoring import get_player_stats_scoring
from src.data.get_standings import get_standings
from src.data.get_team_stats import get_team_stats_all_seasons
from src.data.get_team_stats_advanced import get_team_stats_advanced
from src.data.get_teams import get_teams
from src.processing.preprocessing import run_preprocessing
from src.features.team_game_features import build_team_game_features, build_matchup_dataset
from src.data.get_odds import refresh_odds_data
from src.data.get_injury_data import get_injury_report
from src.data.get_balldontlie import get_balldontlie_stats


ERROR_LOG_PATH = Path("logs/pipeline_errors.log")


def generate_today_predictions(game_date: str) -> int:
    """Fetch today's NBA schedule and write a prediction per game to predictions_history.db.

    Returns the number of predictions written (0 on failure).
    """
    import time
    from nba_api.stats.endpoints import scoreboardv2
    from nba_api.stats.static import teams as nba_teams_static
    from src.models.game_outcome_model import predict_game

    team_id_to_abbr = {t["id"]: t["abbreviation"] for t in nba_teams_static.get_teams()}

    try:
        time.sleep(1)
        board = scoreboardv2.ScoreboardV2(game_date=game_date, timeout=30)
        games_df = board.get_data_frames()[0]
    except Exception as e:
        print(f"  Could not fetch today's schedule: {e}")
        return 0

    if games_df.empty:
        print("  No games scheduled today.")
        return 0

    written = 0
    for _, row in games_df.iterrows():
        home_abbr = team_id_to_abbr.get(row["HOME_TEAM_ID"])
        away_abbr = team_id_to_abbr.get(row["VISITOR_TEAM_ID"])
        if not home_abbr or not away_abbr:
            print(f"  Skipping unknown team IDs: {row['HOME_TEAM_ID']} / {row['VISITOR_TEAM_ID']}")
            continue
        try:
            result = predict_game(home_abbr, away_abbr, game_date=game_date)
            if "error" not in result:
                written += 1
                print(f"  {away_abbr} @ {home_abbr}: home_win_prob={result['home_win_prob']:.3f}")
            else:
                print(f"  {away_abbr} @ {home_abbr}: {result['error']}")
        except Exception as e:
            print(f"  {away_abbr} @ {home_abbr}: prediction failed ({e})")

    return written


def log_pipeline_error(script_name: str, error: Exception) -> None:
    """Append pipeline errors to logs/pipeline_errors.log."""
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {script_name} failed: {error}\n")


def get_current_season_year(reference: datetime | None = None) -> int:
    """Return NBA season start year (season flips in October)."""
    now = reference or datetime.now()
    return now.year if now.month >= 10 else now.year - 1


def run_fetchers(
    title: str,
    fetchers: list[tuple[str, Callable[..., None]]],
    start_year: int,
    end_year: int,
) -> None:
    print(f"\n=== {title} ===")
    for label, fetcher in fetchers:
        print(f"--- {label} ---")
        fetcher(start_year=start_year, end_year=end_year)


def fetch_current_season_data(current_year: int, now: datetime) -> None:
    regular_fetchers = [
        ("Player stats (basic)", get_player_stats_all_seasons),
        ("Player stats (advanced)", get_player_stats_advanced),
        ("Player stats (scoring)", get_player_stats_scoring),
        ("Player stats (clutch)", get_player_stats_clutch),
        ("Player game logs", get_player_game_logs),
        ("Team stats (basic)", get_team_stats_all_seasons),
        ("Team stats (advanced)", get_team_stats_advanced),
        ("Team game logs", get_team_game_logs),
        ("Standings", get_standings),
    ]
    run_fetchers("Step 1A: Current-season regular data", regular_fetchers, current_year, current_year)

    if current_year >= 2015:
        from src.data.get_hustle_stats import get_hustle_stats

        run_fetchers(
            "Step 1B: Hustle stats",
            [("Hustle stats", get_hustle_stats)],
            current_year,
            current_year,
        )
    else:
        print("\n=== Step 1B: Hustle stats ===")
        print("Skipping hustle stats (NBA tracking starts in 2015-16).")

    if now.month >= 4:
        from src.data.get_player_game_logs_playoffs import get_player_game_logs_playoffs
        from src.data.get_player_stats_playoffs import get_player_stats_playoffs
        from src.data.get_team_game_logs_playoffs import get_team_game_logs_playoffs
        from src.data.get_team_stats_playoffs import get_team_stats_playoffs

        playoff_fetchers = [
            ("Player playoffs stats", get_player_stats_playoffs),
            ("Team playoffs stats", get_team_stats_playoffs),
            ("Player playoffs game logs", get_player_game_logs_playoffs),
            ("Team playoffs game logs", get_team_game_logs_playoffs),
        ]
        run_fetchers("Step 1C: Playoff data", playoff_fetchers, current_year, current_year)
    else:
        print("\n=== Step 1C: Playoff data ===")
        print(f"Skipping playoff pulls (current month={now.month}; playoffs start in April).")

    print("\n=== Step 1D: Reference tables ===")
    print("--- Player master ---")
    get_player_master()
    print("--- Teams ---")
    get_teams()


def _check_env_vars() -> None:
    """Warn if optional API keys are missing so the user knows which features are disabled."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; fall back to checking os.environ directly

    missing = []
    if not os.getenv("BALLDONTLIE_API_KEY"):
        missing.append("BALLDONTLIE_API_KEY")

    for key in missing:
        print(
            f"WARNING: {key} is not set in environment or .env file. "
            f"Features that depend on this key will be skipped."
        )


def main() -> None:
    try:
        _check_env_vars()

        start_time = datetime.now()
        now = datetime.now()
        current_year = get_current_season_year(now)
        current_season = f"{current_year}-{str(current_year + 1)[-2:]}"

        print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting NBA data update...")
        print(f"Current NBA season: {current_season}")

        fetch_current_season_data(current_year, now)

        print("\n=== Step 1E: BallDontLie supplementary game stats ===")
        try:
            bdl_df = get_balldontlie_stats(season=current_year)
            if bdl_df.empty:
                print("BallDontLie returned no data (API key missing or no games yet).")
            else:
                print(f"BallDontLie stats fetched: {len(bdl_df)} game records.")
        except Exception as bdl_err:
            log_pipeline_error("update.py:balldontlie", bdl_err)
            print(f"BallDontLie fetch failed (non-fatal): {bdl_err}")

        print("\n=== Step 2: Updating processed CSVs (incremental) ===")
        run_preprocessing(full_rebuild=False)

        print("\n=== Step 3: Rebuilding derived feature CSVs ===")
        try:
            build_team_game_features()
            build_matchup_dataset()
            try:
                from src.features.player_features import build_player_game_features
                build_player_game_features()
            except Exception as player_feat_err:
                print(f"Player feature rebuild skipped (non-fatal): {player_feat_err}")
        except Exception as feat_err:
            log_pipeline_error("update.py:feature_rebuild", feat_err)
            print(f"Feature rebuild failed (non-fatal): {feat_err}")

        print("\n=== Step 4: Refreshing sportsbook odds ===")
        if not refresh_odds_data():
            print("Odds refresh skipped/failed; continuing update pipeline.")

        print("\n=== Step 5: Fetching today's injury report ===")
        try:
            injury_df = get_injury_report(season_year=current_year)
            if injury_df.empty:
                print("Injury report returned empty (may be off-season or no games today).")
            else:
                print(f"Injury report saved: {len(injury_df)} entries.")
        except Exception as injury_err:
            print(f"Injury report fetch failed (non-fatal): {injury_err}")

        print("\n=== Step 6: Generating today's game predictions ===")
        today_str = now.strftime("%Y-%m-%d")
        try:
            n_written = generate_today_predictions(today_str)
            print(f"  Wrote {n_written} prediction(s) to predictions_history.db")
        except Exception as pred_err:
            log_pipeline_error("update.py:generate_predictions", pred_err)
            print(f"  Prediction generation failed (non-fatal): {pred_err}")

        print("\n=== Step 6b: Running ensemble on today's predictions ===")
        try:
            import pandas as pd
            from src.models.ensemble import NBAEnsemble

            matchup_path = "data/features/game_matchup_features.csv"
            if Path(matchup_path).exists():
                matchup_df = pd.read_csv(matchup_path)
                matchup_df["game_date"] = pd.to_datetime(
                    matchup_df["game_date"], format="mixed"
                )
                today_rows = matchup_df[
                    matchup_df["game_date"].dt.strftime("%Y-%m-%d") == today_str
                ].copy()

                if today_rows.empty:
                    print(f"  No matchup rows for {today_str}; ensemble skipped.")
                else:
                    ens = NBAEnsemble.load()
                    scores = ens.predict(today_rows)

                    import sqlite3
                    db_path = "database/predictions_history.db"
                    with sqlite3.connect(db_path) as conn:
                        conn.execute("""
                            CREATE TABLE IF NOT EXISTS ensemble_predictions (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                game_date TEXT,
                                home_team TEXT,
                                away_team TEXT,
                                win_prob REAL,
                                ats_prob REAL,
                                margin_pred REAL,
                                margin_signal REAL,
                                ensemble_score REAL,
                                ensemble_edge REAL,
                                confidence TEXT,
                                created_at TEXT
                            )
                        """)
                        for idx, row in scores.iterrows():
                            src = today_rows.loc[idx]
                            conn.execute("""
                                INSERT INTO ensemble_predictions
                                    (game_date, home_team, away_team,
                                     win_prob, ats_prob, margin_pred, margin_signal,
                                     ensemble_score, ensemble_edge, confidence,
                                     created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                today_str,
                                str(src.get("home_team", "")),
                                str(src.get("away_team", "")),
                                float(row["win_prob"]) if pd.notna(row["win_prob"]) else None,
                                float(row["ats_prob"]) if pd.notna(row["ats_prob"]) else None,
                                float(row["margin_pred"]) if pd.notna(row["margin_pred"]) else None,
                                float(row["margin_signal"]) if pd.notna(row["margin_signal"]) else None,
                                float(row["ensemble_score"]) if pd.notna(row["ensemble_score"]) else None,
                                float(row["ensemble_edge"]) if pd.notna(row["ensemble_edge"]) else None,
                                str(row["confidence"]),
                                datetime.now().isoformat(),
                            ))
                        conn.commit()

                    n_ens = len(scores)
                    print(f"  Ensemble scores written: {n_ens} row(s) to ensemble_predictions table")
            else:
                print(f"  Matchup CSV not found at {matchup_path}; ensemble skipped.")
        except Exception as ens_err:
            log_pipeline_error("update.py:ensemble", ens_err)
            print(f"  Ensemble step failed (non-fatal): {ens_err}")

        elapsed = datetime.now() - start_time
        total_seconds = int(elapsed.total_seconds())
        print(
            f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Update complete in {total_seconds // 60}m {total_seconds % 60}s"
        )
    except Exception as error:
        log_pipeline_error("update.py", error)
        print(f"Update failed: {error}. See logs/pipeline_errors.log for details.")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
