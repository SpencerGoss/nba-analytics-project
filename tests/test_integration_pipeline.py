"""Integration test: 3-season synthetic pipeline through feature engineering.

Verifies that build_team_game_features can process a small synthetic dataset
(50 games across 3 seasons) without crashing, and that the output has the
expected shape and column names.

Phase 7.7 test addition.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Fixtures ─────────────────────────────────────────────────────────────────

TEAMS = [
    {"team_id": 1, "team_abbreviation": "BOS", "team_name": "Boston Celtics"},
    {"team_id": 2, "team_abbreviation": "LAL", "team_name": "Los Angeles Lakers"},
    {"team_id": 3, "team_abbreviation": "GSW", "team_name": "Golden State Warriors"},
    {"team_id": 4, "team_abbreviation": "MIA", "team_name": "Miami Heat"},
]

SEASONS = [202223, 202324, 202425]


def _generate_synthetic_games(n_games=50, seed=42):
    """Generate a minimal synthetic team_game_logs DataFrame.

    Each game produces two rows (one per team). Total rows = n_games * 2.
    Games are spread across 3 seasons with 4 teams.
    """
    np.random.seed(seed)
    rows = []
    game_counter = 0

    for season_idx, season in enumerate(SEASONS):
        # Distribute games roughly evenly across seasons
        games_this_season = n_games // len(SEASONS)
        if season_idx < n_games % len(SEASONS):
            games_this_season += 1

        base_date = pd.Timestamp(f"{2022 + season_idx}-10-20")

        for g in range(games_this_season):
            game_counter += 1
            game_id = f"002{season_idx}{game_counter:04d}"
            game_date = base_date + pd.Timedelta(days=g * 3)

            # Pick two different teams
            team_indices = np.random.choice(len(TEAMS), size=2, replace=False)
            home_info = TEAMS[team_indices[0]]
            away_info = TEAMS[team_indices[1]]

            # Generate realistic box score stats
            home_pts = int(np.random.normal(110, 12))
            away_pts = int(np.random.normal(108, 12))
            home_pts = max(home_pts, 70)
            away_pts = max(away_pts, 70)

            for is_home, info, pts, opp_pts_val in [
                (True, home_info, home_pts, away_pts),
                (False, away_info, away_pts, home_pts),
            ]:
                opp = away_info if is_home else home_info
                matchup = (
                    f"{info['team_abbreviation']} vs. {opp['team_abbreviation']}"
                    if is_home
                    else f"{info['team_abbreviation']} @ {home_info['team_abbreviation']}"
                )
                wl = "W" if pts > opp_pts_val else ("L" if pts < opp_pts_val else "W")
                plus_minus = pts - opp_pts_val

                fga = int(np.random.normal(88, 5))
                fgm = int(np.clip(fga * np.random.uniform(0.42, 0.52), 30, fga))
                fg3a = int(np.random.normal(35, 5))
                fg3m = int(np.clip(fg3a * np.random.uniform(0.30, 0.42), 5, fg3a))
                fta = int(np.random.normal(22, 5))
                ftm = int(np.clip(fta * np.random.uniform(0.72, 0.85), 5, fta))
                oreb = int(np.random.normal(10, 3))
                dreb = int(np.random.normal(34, 4))

                rows.append({
                    "season_id": f"2{season}",
                    "team_id": info["team_id"],
                    "team_abbreviation": info["team_abbreviation"],
                    "team_name": info["team_name"],
                    "game_id": game_id,
                    "game_date": str(game_date.date()),
                    "matchup": matchup,
                    "wl": wl,
                    "min": 240,
                    "fgm": fgm,
                    "fga": fga,
                    "fg_pct": round(fgm / max(fga, 1), 3),
                    "fg3m": fg3m,
                    "fg3a": fg3a,
                    "fg3_pct": round(fg3m / max(fg3a, 1), 3),
                    "ftm": ftm,
                    "fta": fta,
                    "ft_pct": round(ftm / max(fta, 1), 3),
                    "oreb": oreb,
                    "dreb": dreb,
                    "reb": oreb + dreb,
                    "ast": int(np.random.normal(25, 4)),
                    "stl": int(np.random.normal(7, 2)),
                    "blk": int(np.random.normal(5, 2)),
                    "tov": int(np.random.normal(14, 3)),
                    "pf": int(np.random.normal(20, 3)),
                    "pts": pts,
                    "plus_minus": plus_minus,
                    "video_available": 1,
                    "season": season,
                })

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_csv(tmp_path):
    """Write a synthetic team_game_logs.csv and return its path."""
    df = _generate_synthetic_games(n_games=50, seed=42)
    csv_path = tmp_path / "team_game_logs.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def standings_csv(tmp_path):
    """Write a minimal standings.csv and return its path."""
    rows = []
    for season in SEASONS:
        for t in TEAMS:
            rows.append({
                "team_id": t["team_id"],
                "team_abbreviation": t["team_abbreviation"],
                "conference": "Eastern" if t["team_abbreviation"] in ("BOS", "MIA") else "Western",
                "season": season,
            })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "standings.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def output_csv(tmp_path):
    """Return path for feature output CSV."""
    return str(tmp_path / "team_game_features.csv")


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSyntheticDataset:
    """Verify the synthetic data generator itself."""

    def test_generates_correct_row_count(self):
        df = _generate_synthetic_games(n_games=50, seed=42)
        # Each game produces 2 rows (home + away)
        assert len(df) == 100

    def test_has_required_columns(self):
        df = _generate_synthetic_games(n_games=10, seed=42)
        required = {
            "season_id", "team_id", "team_abbreviation", "team_name",
            "game_id", "game_date", "matchup", "wl", "min",
            "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
            "ftm", "fta", "ft_pct", "oreb", "dreb", "reb",
            "ast", "stl", "blk", "tov", "pf", "pts",
            "plus_minus", "video_available", "season",
        }
        assert required.issubset(set(df.columns))

    def test_spans_three_seasons(self):
        df = _generate_synthetic_games(n_games=50, seed=42)
        assert set(df["season"].unique()) == set(SEASONS)

    def test_deterministic_with_seed(self):
        df1 = _generate_synthetic_games(n_games=20, seed=42)
        df2 = _generate_synthetic_games(n_games=20, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_each_game_has_two_rows(self):
        df = _generate_synthetic_games(n_games=30, seed=42)
        game_counts = df.groupby("game_id").size()
        assert (game_counts == 2).all(), "Every game_id should appear exactly twice"


class TestPipelineIntegration:
    """Run build_team_game_features on synthetic data and verify output."""

    def test_pipeline_does_not_crash(self, synthetic_csv, standings_csv, output_csv):
        """The full feature engineering pipeline should complete without error."""
        from src.features.team_game_features import build_team_game_features

        # Mock external dependencies that require real data files
        with patch("src.features.team_game_features.build_lineup_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.build_elo_ratings", side_effect=Exception("no elo data")), \
             patch("src.features.team_game_features.build_hustle_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.label_eras", side_effect=lambda df, **kw: df), \
             patch("src.features.injury_proxy.build_injury_proxy_features", side_effect=Exception("no injury data")):
            result = build_team_game_features(
                data_path=synthetic_csv,
                standings_path=standings_csv,
                output_path=output_csv,
            )

        assert result is not None
        assert len(result) > 0

    def test_output_shape_matches_input(self, synthetic_csv, standings_csv, output_csv):
        """Output should have same number of rows as input (one per team-game)."""
        from src.features.team_game_features import build_team_game_features

        input_df = pd.read_csv(synthetic_csv)

        with patch("src.features.team_game_features.build_lineup_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.build_elo_ratings", side_effect=Exception("no elo data")), \
             patch("src.features.team_game_features.build_hustle_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.label_eras", side_effect=lambda df, **kw: df), \
             patch("src.features.injury_proxy.build_injury_proxy_features", side_effect=Exception("no injury data")):
            result = build_team_game_features(
                data_path=synthetic_csv,
                standings_path=standings_csv,
                output_path=output_csv,
            )

        assert len(result) == len(input_df), (
            f"Expected {len(input_df)} rows but got {len(result)}"
        )

    def test_output_has_core_columns(self, synthetic_csv, standings_csv, output_csv):
        """Output must contain identity columns and key feature columns."""
        from src.features.team_game_features import build_team_game_features

        with patch("src.features.team_game_features.build_lineup_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.build_elo_ratings", side_effect=Exception("no elo data")), \
             patch("src.features.team_game_features.build_hustle_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.label_eras", side_effect=lambda df, **kw: df), \
             patch("src.features.injury_proxy.build_injury_proxy_features", side_effect=Exception("no injury data")):
            result = build_team_game_features(
                data_path=synthetic_csv,
                standings_path=standings_csv,
                output_path=output_csv,
            )

        # Identity columns
        for col in ["season", "team_id", "team_abbreviation", "game_id", "game_date"]:
            assert col in result.columns, f"Missing identity column: {col}"

        # Context columns
        for col in ["is_home", "opponent_abbr", "days_rest", "is_back_to_back", "win"]:
            assert col in result.columns, f"Missing context column: {col}"

        # Rolling features (at least some should exist)
        roll_cols = [c for c in result.columns if "_roll" in c]
        assert len(roll_cols) > 10, (
            f"Expected many rolling feature columns, got {len(roll_cols)}"
        )

    def test_rolling_features_have_correct_windows(self, synthetic_csv, standings_csv, output_csv):
        """Rolling features must include windows 3, 5, 10, 20."""
        from src.features.team_game_features import build_team_game_features

        with patch("src.features.team_game_features.build_lineup_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.build_elo_ratings", side_effect=Exception("no elo data")), \
             patch("src.features.team_game_features.build_hustle_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.label_eras", side_effect=lambda df, **kw: df), \
             patch("src.features.injury_proxy.build_injury_proxy_features", side_effect=Exception("no injury data")):
            result = build_team_game_features(
                data_path=synthetic_csv,
                standings_path=standings_csv,
                output_path=output_csv,
            )

        for window in [3, 5, 10, 20]:
            window_cols = [c for c in result.columns if f"_roll{window}" in c]
            assert len(window_cols) > 0, f"No rolling features found for window {window}"

    def test_no_future_leakage_in_rolling(self, synthetic_csv, standings_csv, output_csv):
        """First game of each team-season should have NaN rolling features (shift-1)."""
        from src.features.team_game_features import build_team_game_features

        with patch("src.features.team_game_features.build_lineup_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.build_elo_ratings", side_effect=Exception("no elo data")), \
             patch("src.features.team_game_features.build_hustle_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.label_eras", side_effect=lambda df, **kw: df), \
             patch("src.features.injury_proxy.build_injury_proxy_features", side_effect=Exception("no injury data")):
            result = build_team_game_features(
                data_path=synthetic_csv,
                standings_path=standings_csv,
                output_path=output_csv,
            )

        # For each team, the first game should have NaN for shifted rolling features
        for team_id in result["team_id"].unique():
            team_data = result[result["team_id"] == team_id].sort_values("game_date")
            first_row = team_data.iloc[0]
            # pts_roll5 should be NaN for first game (shift-1 means no prior data)
            if "pts_roll5" in result.columns:
                assert pd.isna(first_row["pts_roll5"]), (
                    f"pts_roll5 should be NaN for first game of team {team_id}"
                )

    def test_output_saved_to_csv(self, synthetic_csv, standings_csv, output_csv):
        """The function should save output to the specified CSV path."""
        from src.features.team_game_features import build_team_game_features

        with patch("src.features.team_game_features.build_lineup_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.build_elo_ratings", side_effect=Exception("no elo data")), \
             patch("src.features.team_game_features.build_hustle_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.label_eras", side_effect=lambda df, **kw: df), \
             patch("src.features.injury_proxy.build_injury_proxy_features", side_effect=Exception("no injury data")):
            build_team_game_features(
                data_path=synthetic_csv,
                standings_path=standings_csv,
                output_path=output_csv,
            )

        assert os.path.exists(output_csv), "Output CSV should be written to disk"
        saved = pd.read_csv(output_csv)
        assert len(saved) > 0

    def test_season_game_num_starts_at_zero(self, synthetic_csv, standings_csv, output_csv):
        """season_game_num should start at 0 for each team-season."""
        from src.features.team_game_features import build_team_game_features

        with patch("src.features.team_game_features.build_lineup_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.build_elo_ratings", side_effect=Exception("no elo data")), \
             patch("src.features.team_game_features.build_hustle_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.label_eras", side_effect=lambda df, **kw: df), \
             patch("src.features.injury_proxy.build_injury_proxy_features", side_effect=Exception("no injury data")):
            result = build_team_game_features(
                data_path=synthetic_csv,
                standings_path=standings_csv,
                output_path=output_csv,
            )

        for (team_id, season), group in result.groupby(["team_id", "season"]):
            min_gn = group["season_game_num"].min()
            assert min_gn == 0, (
                f"season_game_num for team {team_id}, season {season} starts at {min_gn}, expected 0"
            )

    def test_multiple_column_count(self, synthetic_csv, standings_csv, output_csv):
        """Output should have substantially more columns than input (features added)."""
        from src.features.team_game_features import build_team_game_features

        input_cols = len(pd.read_csv(synthetic_csv, nrows=1).columns)

        with patch("src.features.team_game_features.build_lineup_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.build_elo_ratings", side_effect=Exception("no elo data")), \
             patch("src.features.team_game_features.build_hustle_features", return_value=pd.DataFrame()), \
             patch("src.features.team_game_features.label_eras", side_effect=lambda df, **kw: df), \
             patch("src.features.injury_proxy.build_injury_proxy_features", side_effect=Exception("no injury data")):
            result = build_team_game_features(
                data_path=synthetic_csv,
                standings_path=standings_csv,
                output_path=output_csv,
            )

        assert len(result.columns) > input_cols * 2, (
            f"Expected output columns ({len(result.columns)}) to be much more than "
            f"input columns ({input_cols})"
        )
