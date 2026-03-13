"""
Tests for src/processing/preprocessing.py

Focuses on:
  - Column cleaning (CamelCase -> snake_case, spaces -> underscores)
  - Type coercion with NaN in integer columns
  - Duplicate row removal preserving unique rows
  - Multi-file concatenation via load_season_folder
"""

import time
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.processing.preprocessing import (
    clean_columns,
    load_season_folder,
    get_stale_seasons,
    load_season_files,
    merge_incremental,
    _season_label,
)


# ── clean_columns ─────────────────────────────────────────────────────────────


class TestCleanColumns:

    def test_uppercase_to_lowercase(self):
        df = pd.DataFrame({"PLAYER_ID": [1], "TEAM_NAME": ["LAL"]})
        result = clean_columns(df)
        assert list(result.columns) == ["player_id", "team_name"]

    def test_spaces_replaced_with_underscores(self):
        df = pd.DataFrame({"Player Name": [1], "Team City": ["LA"]})
        result = clean_columns(df)
        assert list(result.columns) == ["player_name", "team_city"]

    def test_hyphens_replaced_with_underscores(self):
        df = pd.DataFrame({"plus-minus": [5], "fg3-pct": [0.4]})
        result = clean_columns(df)
        assert list(result.columns) == ["plus_minus", "fg3_pct"]

    def test_slashes_replaced_with_underscores(self):
        df = pd.DataFrame({"W/L": ["W"], "FG/FGA": [0.45]})
        result = clean_columns(df)
        assert list(result.columns) == ["w_l", "fg_fga"]

    def test_mixed_case_and_symbols(self):
        df = pd.DataFrame({
            "PLAYER ID": [1],
            "Team-Name": ["LAL"],
            "Win/Loss": ["W"],
            " Leading Spaces ": [10],
        })
        result = clean_columns(df)
        expected = ["player_id", "team_name", "win_loss", "leading_spaces"]
        assert list(result.columns) == expected

    def test_already_clean_columns_unchanged(self):
        df = pd.DataFrame({"player_id": [1], "team_name": ["LAL"]})
        result = clean_columns(df)
        assert list(result.columns) == ["player_id", "team_name"]

    def test_clean_columns_preserves_data(self):
        """Cleaning columns should not modify the data itself."""
        df = pd.DataFrame({"PLAYER ID": [101, 102], "POINTS": [25, 30]})
        result = clean_columns(df)
        assert result["player_id"].tolist() == [101, 102]
        assert result["points"].tolist() == [25, 30]

    def test_clean_columns_returns_new_dataframe(self):
        """
        clean_columns modifies in-place (df.columns = ...),
        but we verify the returned df has correct columns.
        """
        df = pd.DataFrame({"ABC": [1]})
        result = clean_columns(df)
        assert "abc" in result.columns


# ── Type coercion with NaN ────────────────────────────────────────────────────


class TestTypeCoercion:

    def test_int_column_with_nan_coerces_to_float(self):
        """
        Pandas cannot store NaN in int64 columns. When NaN exists in a
        column that should be int, astype(float) is the correct approach.
        """
        df = pd.DataFrame({"gp": [82, np.nan, 75]})
        df["gp"] = df["gp"].astype(float)
        assert df["gp"].dtype == np.float64
        assert np.isnan(df["gp"].iloc[1])
        assert df["gp"].iloc[0] == 82.0

    def test_int_column_without_nan_coerces_to_int(self):
        df = pd.DataFrame({"gp": [82, 75, 60]})
        df["gp"] = df["gp"].astype(int)
        assert df["gp"].dtype == np.int64 or df["gp"].dtype == np.int32

    def test_string_to_datetime_coercion(self):
        df = pd.DataFrame({"game_date": ["2024-10-22", "2024-10-24"]})
        df["game_date"] = pd.to_datetime(df["game_date"])
        assert pd.api.types.is_datetime64_any_dtype(df["game_date"])

    def test_numeric_coerce_handles_non_numeric_strings(self):
        """pd.to_numeric with errors='coerce' turns bad values to NaN."""
        df = pd.DataFrame({"min": ["30", "25:30", "N/A", "28"]})
        df["min"] = pd.to_numeric(df["min"], errors="coerce")
        assert df["min"].iloc[0] == 30.0
        assert np.isnan(df["min"].iloc[1])  # "25:30" is not numeric
        assert np.isnan(df["min"].iloc[2])  # "N/A" is not numeric
        assert df["min"].iloc[3] == 28.0


# ── Duplicate row removal ────────────────────────────────────────────────────


class TestDuplicateRemoval:

    def test_drop_duplicates_removes_exact_copies(self):
        df = pd.DataFrame({
            "player_id": [101, 101, 102],
            "season": [202425, 202425, 202425],
            "pts": [25, 25, 30],
        })
        result = df.drop_duplicates()
        assert len(result) == 2

    def test_drop_duplicates_preserves_unique_rows(self):
        df = pd.DataFrame({
            "player_id": [101, 102, 103],
            "season": [202425, 202425, 202425],
            "pts": [25, 30, 20],
        })
        result = df.drop_duplicates()
        assert len(result) == 3

    def test_drop_duplicates_subset_key(self):
        """drop_duplicates(subset=...) removes rows with matching key columns."""
        df = pd.DataFrame({
            "player_id": [101, 101, 102],
            "player_name": ["Alice", "Alice Updated", "Bob"],
        })
        result = df.drop_duplicates(subset=["player_id"])
        assert len(result) == 2
        # First occurrence is kept by default
        assert result.iloc[0]["player_name"] == "Alice"

    def test_all_duplicates_reduces_to_one(self):
        df = pd.DataFrame({
            "player_id": [101, 101, 101],
            "pts": [25, 25, 25],
        })
        result = df.drop_duplicates()
        assert len(result) == 1


# ── load_season_folder (multi-file concatenation) ────────────────────────────


class TestLoadSeasonFolder:

    def test_load_single_file(self, tmp_path):
        """Loading a single season file returns correct data with season column."""
        season_dir = tmp_path / "player_stats"
        season_dir.mkdir()

        df = pd.DataFrame({
            "PLAYER_ID": [101, 102],
            "PTS": [25, 30],
        })
        df.to_csv(season_dir / "player_stats_202425.csv", index=False)

        result = load_season_folder(
            str(season_dir / "player_stats_*.csv"),
            "player_stats_",
        )
        assert len(result) == 2
        assert "season" in result.columns
        assert result["season"].iloc[0] == "202425"

    def test_load_multiple_files_concatenated(self, tmp_path):
        """Loading multiple season files concatenates them with correct season tags."""
        season_dir = tmp_path / "player_stats"
        season_dir.mkdir()

        df1 = pd.DataFrame({"PLAYER_ID": [101], "PTS": [25]})
        df2 = pd.DataFrame({"PLAYER_ID": [102], "PTS": [30]})
        df1.to_csv(season_dir / "player_stats_202324.csv", index=False)
        df2.to_csv(season_dir / "player_stats_202425.csv", index=False)

        result = load_season_folder(
            str(season_dir / "player_stats_*.csv"),
            "player_stats_",
        )
        assert len(result) == 2
        seasons = set(result["season"].tolist())
        assert seasons == {"202324", "202425"}

    def test_load_preserves_all_columns(self, tmp_path):
        """All original columns plus the added 'season' column should be present."""
        season_dir = tmp_path / "stats"
        season_dir.mkdir()

        df = pd.DataFrame({
            "PLAYER_ID": [101],
            "PTS": [25],
            "AST": [5],
            "REB": [8],
        })
        df.to_csv(season_dir / "stats_202425.csv", index=False)

        result = load_season_folder(str(season_dir / "stats_*.csv"), "stats_")
        expected_cols = {"PLAYER_ID", "PTS", "AST", "REB", "season"}
        assert set(result.columns) == expected_cols

    def test_load_no_files_raises_error(self, tmp_path):
        """Loading from an empty directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_season_folder(
                str(tmp_path / "nonexistent_*.csv"),
                "nonexistent_",
            )

    def test_load_season_extracted_from_filename(self, tmp_path):
        """Season code is extracted by stripping prefix and .csv suffix."""
        season_dir = tmp_path / "team_stats"
        season_dir.mkdir()

        df = pd.DataFrame({"TEAM_ID": [1]})
        df.to_csv(season_dir / "team_stats_199697.csv", index=False)

        result = load_season_folder(
            str(season_dir / "team_stats_*.csv"),
            "team_stats_",
        )
        assert result["season"].iloc[0] == "199697"

    def test_concatenation_resets_index(self, tmp_path):
        """Concatenated result should have a clean 0-based index."""
        season_dir = tmp_path / "data"
        season_dir.mkdir()

        df1 = pd.DataFrame({"X": [1, 2]})
        df2 = pd.DataFrame({"X": [3, 4]})
        df1.to_csv(season_dir / "data_2023.csv", index=False)
        df2.to_csv(season_dir / "data_2024.csv", index=False)

        result = load_season_folder(str(season_dir / "data_*.csv"), "data_")
        assert list(result.index) == [0, 1, 2, 3]


# ── get_stale_seasons ─────────────────────────────────────────────────────────


class TestGetStaleSeasons:

    def test_returns_all_files_when_no_processed_output(self, tmp_path):
        """First run: processed file absent → all raw files considered stale."""
        raw1 = tmp_path / "player_stats_202324.csv"
        raw2 = tmp_path / "player_stats_202425.csv"
        raw1.write_text("col\n1")
        raw2.write_text("col\n2")

        result = get_stale_seasons(
            raw_dir=str(tmp_path / "player_stats_*.csv"),
            processed_path=str(tmp_path / "player_stats.csv"),
        )
        assert sorted(result) == sorted([str(raw1), str(raw2)])

    def test_returns_empty_when_all_raw_older_than_processed(self, tmp_path):
        """Processed file newer than raw → nothing stale."""
        raw = tmp_path / "player_stats_202425.csv"
        raw.write_text("col\n1")

        time.sleep(0.05)
        processed = tmp_path / "player_stats.csv"
        processed.write_text("col\n1")

        result = get_stale_seasons(
            raw_dir=str(tmp_path / "player_stats_*.csv"),
            processed_path=str(processed),
        )
        assert result == []

    def test_returns_only_newer_raw_files(self, tmp_path):
        """Only raw files newer than processed are stale."""
        old_raw = tmp_path / "player_stats_202324.csv"
        old_raw.write_text("col\n1")

        time.sleep(0.05)
        processed = tmp_path / "player_stats.csv"
        processed.write_text("col\n1")

        time.sleep(0.05)
        new_raw = tmp_path / "player_stats_202425.csv"
        new_raw.write_text("col\n2")

        result = get_stale_seasons(
            raw_dir=str(tmp_path / "player_stats_*.csv"),
            processed_path=str(processed),
        )
        assert result == [str(new_raw)]

    def test_returns_empty_list_when_no_raw_files(self, tmp_path):
        """No raw files → empty list, no error."""
        result = get_stale_seasons(
            raw_dir=str(tmp_path / "player_stats_*.csv"),
            processed_path=str(tmp_path / "player_stats.csv"),
        )
        assert result == []


# ── load_season_files ─────────────────────────────────────────────────────────


class TestLoadSeasonFiles:

    def test_adds_season_column_from_filename(self, tmp_path):
        csv = tmp_path / "player_stats_202425.csv"
        csv.write_text("player_id,pts\n1,20\n")

        result = load_season_files([str(csv)], prefix="player_stats_")

        assert "season" in result.columns
        assert (result["season"] == "202425").all()

    def test_concatenates_multiple_files(self, tmp_path):
        csv1 = tmp_path / "player_stats_202324.csv"
        csv2 = tmp_path / "player_stats_202425.csv"
        csv1.write_text("player_id,pts\n1,18\n")
        csv2.write_text("player_id,pts\n2,22\n")

        result = load_season_files([str(csv1), str(csv2)], prefix="player_stats_")

        assert len(result) == 2
        assert set(result["season"].unique()) == {"202324", "202425"}

    def test_raises_on_empty_file_list(self):
        with pytest.raises(FileNotFoundError):
            load_season_files([], prefix="player_stats_")


# ── merge_incremental ─────────────────────────────────────────────────────────


class TestMergeIncremental:

    def test_returns_new_df_when_processed_does_not_exist(self, tmp_path):
        """First run: no processed CSV → return new_df unchanged."""
        new_df = pd.DataFrame({"season": ["202425"], "player_id": [1], "pts": [20]})
        processed_path = str(tmp_path / "player_stats.csv")
        stale = [str(tmp_path / "player_stats_202425.csv")]

        result = merge_incremental(new_df, processed_path, stale, prefix="player_stats_")

        assert len(result) == 1
        assert result["season"].iloc[0] == "202425"

    def test_appends_new_seasons_to_existing(self, tmp_path):
        existing = pd.DataFrame({
            "season": [202324, 202324],
            "player_id": [1, 2],
            "pts": [20, 15],
        })
        processed_path = str(tmp_path / "player_stats.csv")
        existing.to_csv(processed_path, index=False)

        new_df = pd.DataFrame({"season": [202425], "player_id": [3], "pts": [25]})
        stale = [str(tmp_path / "player_stats_202425.csv")]

        result = merge_incremental(new_df, processed_path, stale, prefix="player_stats_")

        assert len(result) == 3
        assert set(result["season"].astype(int).unique()) == {202324, 202425}

    def test_replaces_existing_rows_for_updated_season(self, tmp_path):
        """Stale season rows replaced, not duplicated."""
        existing = pd.DataFrame({
            "season": ["202425", "202425"],
            "player_id": [1, 2],
            "pts": [10, 12],
        })
        processed_path = str(tmp_path / "player_stats.csv")
        existing.to_csv(processed_path, index=False)

        new_df = pd.DataFrame({
            "season": ["202425", "202425"],
            "player_id": [1, 2],
            "pts": [20, 25],
        })
        stale = [str(tmp_path / "player_stats_202425.csv")]

        result = merge_incremental(
            new_df, processed_path, stale,
            prefix="player_stats_",
            dedup_subset=["season", "player_id"],
        )

        assert len(result) == 2
        assert set(result["pts"].tolist()) == {20, 25}


# ── _season_label ─────────────────────────────────────────────────────────────


class TestSeasonLabel:

    def test_six_digit_code_formatted_with_dash(self):
        result = _season_label("data/raw/player_stats/player_stats_202425.csv", "player_stats_")
        assert result == "2024-25"

    def test_early_season_code(self):
        result = _season_label("player_stats_200001.csv", "player_stats_")
        assert result == "2000-01"

    def test_non_six_digit_code_returned_unchanged(self):
        result = _season_label("player_stats_abc.csv", "player_stats_")
        assert result == "abc"
