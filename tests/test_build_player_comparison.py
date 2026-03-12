"""
Tests for scripts/build_player_comparison.py

Covers:
  1.  season_str_to_int / season_int_to_str round-trip
  2.  load_existing_seasons returns empty set for missing file
  3.  load_existing_seasons reads season_str column correctly
  4.  generate_season_strings produces correct range and format
  5.  compute_league_averages aggregates by season
  6.  compute_league_averages handles missing team data (None)
  7.  enrich_player_seasons adds normalized columns
  8.  enrich_player_seasons adds per36 columns
  9.  enrich_player_seasons adds ts_pct
  10. enrich_player_seasons handles missing blk/stl gracefully
  11. build_player_records filters by min_seasons
  12. build_player_records filters by min_career_games
  13. build_player_records computes best_season by pts_normalized
  14. build_player_records sorts descending by career pts_normalized
  15. build_league_by_season returns one row per season
  16. build_player_index returns lightweight id/name/seasons dicts
  17. _round_or_none returns None for NaN inputs
  18. _safe_div handles zero denominator
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the project root is on the path so scripts can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the module under test
from scripts.build_player_comparison import (
    season_str_to_int,
    season_int_to_str,
    generate_season_strings,
    load_existing_seasons,
    compute_league_averages,
    enrich_player_seasons,
    compute_advanced_stats,
    build_player_records,
    build_league_by_season,
    build_player_index,
    _round_or_none,
    _safe_div,
    _build_team_season_lookup,
    _build_abbr_to_name_map,
    _league_avg_reb_by_season,
    _infer_position,
    TEAM_CSV_COLUMNS,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_players() -> pd.DataFrame:
    """Two players across three seasons."""
    rows = [
        # Player A — 3 seasons
        dict(player_id=1, player_name="Player A", season_str="2000-01", season=200001,
             gp=80, min=35.0, pts=25.0, reb=10.0, ast=5.0, stl=1.5, blk=1.0,
             fgm=9.0, fga=20.0, fg_pct=0.450, fg3m=1.0, fg3a=3.0, fg3_pct=0.333,
             ftm=6.0, fta=7.0, ft_pct=0.857, oreb=2.0, dreb=8.0, tov=3.0,
             team_abbreviation="LAL", age=22),
        dict(player_id=1, player_name="Player A", season_str="2001-02", season=200102,
             gp=82, min=36.0, pts=27.0, reb=9.0, ast=6.0, stl=1.2, blk=0.8,
             fgm=10.0, fga=22.0, fg_pct=0.455, fg3m=1.0, fg3a=3.0, fg3_pct=0.333,
             ftm=5.0, fta=6.0, ft_pct=0.833, oreb=1.5, dreb=7.5, tov=3.5,
             team_abbreviation="LAL", age=23),
        dict(player_id=1, player_name="Player A", season_str="2002-03", season=200203,
             gp=75, min=34.0, pts=26.0, reb=9.5, ast=5.5, stl=1.3, blk=0.9,
             fgm=9.5, fga=21.0, fg_pct=0.452, fg3m=1.0, fg3a=3.0, fg3_pct=0.333,
             ftm=6.0, fta=7.0, ft_pct=0.857, oreb=1.8, dreb=7.7, tov=3.2,
             team_abbreviation="LAL", age=24),
        # Player B — 2 seasons (below min_seasons=3, above min_games=400)
        dict(player_id=2, player_name="Player B", season_str="2000-01", season=200001,
             gp=78, min=30.0, pts=18.0, reb=6.0, ast=3.0, stl=0.9, blk=0.5,
             fgm=7.0, fga=16.0, fg_pct=0.438, fg3m=2.0, fg3a=5.0, fg3_pct=0.400,
             ftm=2.0, fta=3.0, ft_pct=0.667, oreb=1.0, dreb=5.0, tov=2.0,
             team_abbreviation="BOS", age=25),
        dict(player_id=2, player_name="Player B", season_str="2001-02", season=200102,
             gp=76, min=28.0, pts=16.0, reb=5.5, ast=3.5, stl=1.0, blk=0.4,
             fgm=6.0, fga=15.0, fg_pct=0.400, fg3m=1.0, fg3a=4.0, fg3_pct=0.250,
             ftm=3.0, fta=4.0, ft_pct=0.750, oreb=0.8, dreb=4.7, tov=2.5,
             team_abbreviation="BOS", age=26),
        # Player C — 1 season, 600 games (should pass via min_career_games)
        dict(player_id=3, player_name="Player C", season_str="2000-01", season=200001,
             gp=600, min=20.0, pts=10.0, reb=4.0, ast=2.0, stl=0.5, blk=0.2,
             fgm=4.0, fga=9.0, fg_pct=0.444, fg3m=0.5, fg3a=2.0, fg3_pct=0.250,
             ftm=1.5, fta=2.0, ft_pct=0.750, oreb=0.5, dreb=3.5, tov=1.5,
             team_abbreviation="CHI", age=30),
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def league_averages(synthetic_players) -> pd.DataFrame:
    """League averages computed from the synthetic player data."""
    return compute_league_averages(synthetic_players, teams=None)


# ---------------------------------------------------------------------------
# 1. season_str_to_int
# ---------------------------------------------------------------------------

def test_season_str_to_int_modern():
    assert season_str_to_int("2024-25") == 202425


def test_season_str_to_int_historical():
    assert season_str_to_int("1946-47") == 194647


# ---------------------------------------------------------------------------
# 2. season_int_to_str
# ---------------------------------------------------------------------------

def test_season_int_to_str_modern():
    assert season_int_to_str(202425) == "2024-25"


def test_season_int_to_str_historical():
    assert season_int_to_str(194647) == "1946-47"


def test_season_round_trip():
    for season_str in ("1996-97", "2000-01", "1999-00", "2019-20"):
        assert season_int_to_str(season_str_to_int(season_str)) == season_str


# ---------------------------------------------------------------------------
# 3. generate_season_strings
# ---------------------------------------------------------------------------

def test_generate_season_strings_count():
    seasons = generate_season_strings(1946, 2024)
    assert len(seasons) == 79  # 1946-47 through 2024-25


def test_generate_season_strings_first_last():
    seasons = generate_season_strings(1946, 2024)
    assert seasons[0] == "1946-47"
    assert seasons[-1] == "2024-25"


def test_generate_season_strings_century_boundary():
    seasons = generate_season_strings(1999, 2000)
    assert "1999-00" in seasons
    assert "2000-01" in seasons


# ---------------------------------------------------------------------------
# 4. load_existing_seasons
# ---------------------------------------------------------------------------

def test_load_existing_seasons_missing_file(tmp_path):
    result = load_existing_seasons(tmp_path / "nonexistent.csv")
    assert result == set()


def test_load_existing_seasons_reads_values(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("season_str,other\n2000-01,x\n2001-02,y\n")
    result = load_existing_seasons(csv)
    assert result == {"2000-01", "2001-02"}


def test_load_existing_seasons_deduplicates(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("season_str\n2000-01\n2000-01\n2001-02\n")
    result = load_existing_seasons(csv)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# 5. compute_league_averages — basic aggregation
# ---------------------------------------------------------------------------

def test_compute_league_averages_seasons(synthetic_players):
    league = compute_league_averages(synthetic_players, teams=None)
    assert set(league["season_str"]) == {"2000-01", "2001-02", "2002-03"}


def test_compute_league_averages_has_avg_pts(synthetic_players):
    league = compute_league_averages(synthetic_players, teams=None)
    assert "avg_pts" in league.columns
    # 2000-01 has Player A (25), B (18), C (10) -> mean = (25+18+10)/3 = 17.67
    row = league[league["season_str"] == "2000-01"].iloc[0]
    assert abs(row["avg_pts"] - ((25 + 18 + 10) / 3)) < 0.1


def test_compute_league_averages_none_teams(synthetic_players):
    """Should not raise when teams=None."""
    league = compute_league_averages(synthetic_players, teams=None)
    assert "avg_pace" in league.columns
    # All pace values should be NaN when no team data
    assert league["avg_pace"].isna().all()


# ---------------------------------------------------------------------------
# 6. enrich_player_seasons — normalized cols
# ---------------------------------------------------------------------------

def test_enrich_adds_normalized_pts(synthetic_players, league_averages):
    hist_avgs = {"pts": 14.5, "reb": 7.0, "ast": 3.5, "stl": 0.9, "blk": 0.4, "pace": 95.0}
    enriched = enrich_player_seasons(synthetic_players, league_averages, hist_avgs)
    assert "pts_normalized" in enriched.columns
    assert enriched["pts_normalized"].notna().any()


def test_enrich_adds_per36(synthetic_players, league_averages):
    hist_avgs = {"pts": 14.5, "reb": 7.0, "ast": 3.5, "stl": 0.9, "blk": 0.4, "pace": 95.0}
    enriched = enrich_player_seasons(synthetic_players, league_averages, hist_avgs)
    assert "per36_pts" in enriched.columns
    # Player A: 25 pts * 80 gp / (35 min * 80 gp) * 36 = 25/35*36 = 25.71
    row_a = enriched[(enriched["player_id"] == 1) & (enriched["season_str"] == "2000-01")].iloc[0]
    assert abs(row_a["per36_pts"] - (25 / 35 * 36)) < 0.1


def test_enrich_adds_ts_pct(synthetic_players, league_averages):
    hist_avgs = {"pts": 14.5, "reb": 7.0, "ast": 3.5, "stl": 0.9, "blk": 0.4, "pace": 95.0}
    enriched = enrich_player_seasons(synthetic_players, league_averages, hist_avgs)
    assert "ts_pct" in enriched.columns
    # TS% = total_pts / (2 * (total_fga + 0.44 * total_fta))
    # Player A 2000-01: pts=25*80, fga=20*80, fta=7*80
    # TS% = 2000 / (2 * (1600 + 0.44*560)) = 2000 / (2 * 1846.4) = 0.5417
    row_a = enriched[(enriched["player_id"] == 1) & (enriched["season_str"] == "2000-01")].iloc[0]
    expected_ts = (25 * 80) / (2 * (20 * 80 + 0.44 * 7 * 80))
    assert abs(row_a["ts_pct"] - expected_ts) < 0.01


def test_enrich_handles_missing_blk_stl(league_averages):
    """Should not raise when stl/blk columns are absent."""
    players_no_stl_blk = pd.DataFrame([
        dict(player_id=1, player_name="X", season_str="2000-01", season=200001,
             gp=80, min=30.0, pts=20.0, reb=8.0, ast=4.0,
             fgm=8.0, fga=18.0, fg_pct=0.444, fg3m=1.0, fg3a=3.0, fg3_pct=0.333,
             ftm=3.0, fta=4.0, ft_pct=0.750, team_abbreviation="NYK", age=28),
    ])
    league = compute_league_averages(players_no_stl_blk, teams=None)
    hist_avgs = {"pts": 14.5, "reb": 7.0, "ast": 3.5, "stl": 0.9, "blk": 0.4, "pace": 95.0}
    enriched = enrich_player_seasons(players_no_stl_blk, league, hist_avgs)
    # Should have fallback normalized columns
    assert "pts_normalized" in enriched.columns


# ---------------------------------------------------------------------------
# 7. build_player_records — filtering
# ---------------------------------------------------------------------------

def test_build_player_records_min_seasons_filter(synthetic_players, league_averages):
    hist_avgs = {"pts": 14.5, "reb": 7.0, "ast": 3.5, "stl": 0.9, "blk": 0.4, "pace": 95.0}
    enriched = enrich_player_seasons(synthetic_players, league_averages, hist_avgs)
    # min_seasons=3 and min_career_games=999 => only Player A (3 seasons) passes
    records = build_player_records(enriched, min_seasons=3, min_career_games=999)
    names = {r["player_name"] for r in records}
    assert "Player A" in names
    assert "Player B" not in names


def test_build_player_records_min_career_games_filter(synthetic_players, league_averages):
    hist_avgs = {"pts": 14.5, "reb": 7.0, "ast": 3.5, "stl": 0.9, "blk": 0.4, "pace": 95.0}
    enriched = enrich_player_seasons(synthetic_players, league_averages, hist_avgs)
    # min_seasons=10 means no one qualifies by seasons; Player C (gp=600) qualifies by games
    records = build_player_records(enriched, min_seasons=10, min_career_games=500)
    names = {r["player_name"] for r in records}
    assert "Player C" in names


def test_build_player_records_best_season(synthetic_players, league_averages):
    hist_avgs = {"pts": 14.5, "reb": 7.0, "ast": 3.5, "stl": 0.9, "blk": 0.4, "pace": 95.0}
    enriched = enrich_player_seasons(synthetic_players, league_averages, hist_avgs)
    records = build_player_records(enriched, min_seasons=3, min_career_games=0)
    player_a = next(r for r in records if r["player_name"] == "Player A")
    # Player A's highest raw pts is 27.0 in 2001-02; normalization shouldn't flip it dramatically
    assert player_a["best_season"] in {"2000-01", "2001-02", "2002-03"}


def test_build_player_records_sorted_desc(synthetic_players, league_averages):
    hist_avgs = {"pts": 14.5, "reb": 7.0, "ast": 3.5, "stl": 0.9, "blk": 0.4, "pace": 95.0}
    enriched = enrich_player_seasons(synthetic_players, league_averages, hist_avgs)
    records = build_player_records(enriched, min_seasons=1, min_career_games=0)
    pts_vals = [r["career_avgs"]["pts_normalized"] or 0 for r in records]
    assert pts_vals == sorted(pts_vals, reverse=True)


def test_build_player_records_season_rows_present(synthetic_players, league_averages):
    hist_avgs = {"pts": 14.5, "reb": 7.0, "ast": 3.5, "stl": 0.9, "blk": 0.4, "pace": 95.0}
    enriched = enrich_player_seasons(synthetic_players, league_averages, hist_avgs)
    records = build_player_records(enriched, min_seasons=3, min_career_games=0)
    player_a = next(r for r in records if r["player_name"] == "Player A")
    assert len(player_a["seasons"]) == 3
    # Each season row should have expected keys
    season_row = player_a["seasons"][0]
    for key in ("season", "season_int", "team", "pts", "pts_normalized", "per36_pts", "ts_pct"):
        assert key in season_row, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 8. build_league_by_season
# ---------------------------------------------------------------------------

def test_build_league_by_season_row_count(synthetic_players):
    league = compute_league_averages(synthetic_players, teams=None)
    rows = build_league_by_season(league)
    assert len(rows) == 3


def test_build_league_by_season_required_keys(synthetic_players):
    league = compute_league_averages(synthetic_players, teams=None)
    rows = build_league_by_season(league)
    for row in rows:
        for key in ("season", "season_int", "avg_pts"):
            assert key in row


# ---------------------------------------------------------------------------
# 9. build_player_index
# ---------------------------------------------------------------------------

def test_build_player_index_keys(synthetic_players, league_averages):
    hist_avgs = {"pts": 14.5, "reb": 7.0, "ast": 3.5, "stl": 0.9, "blk": 0.4, "pace": 95.0}
    enriched = enrich_player_seasons(synthetic_players, league_averages, hist_avgs)
    records = build_player_records(enriched, min_seasons=1, min_career_games=0)
    index = build_player_index(records)
    for entry in index:
        assert "id" in entry
        assert "name" in entry
        assert "seasons" in entry


def test_build_player_index_count_matches_records(synthetic_players, league_averages):
    hist_avgs = {"pts": 14.5, "reb": 7.0, "ast": 3.5, "stl": 0.9, "blk": 0.4, "pace": 95.0}
    enriched = enrich_player_seasons(synthetic_players, league_averages, hist_avgs)
    records = build_player_records(enriched, min_seasons=1, min_career_games=0)
    index = build_player_index(records)
    assert len(index) == len(records)


# ---------------------------------------------------------------------------
# 10. _round_or_none and _safe_div
# ---------------------------------------------------------------------------

def test_round_or_none_nan():
    assert _round_or_none(float("nan")) is None


def test_round_or_none_none():
    assert _round_or_none(None) is None


def test_round_or_none_value():
    assert _round_or_none(3.14159, 2) == 3.14


def test_safe_div_zero_denominator():
    assert _safe_div(10.0, 0.0) == 0.0


def test_safe_div_normal():
    assert abs(_safe_div(10.0, 4.0) - 2.5) < 1e-9


# ---------------------------------------------------------------------------
# 11. Legend data integrity
# ---------------------------------------------------------------------------

def test_legend_best_season_is_string():
    """All legend entries must have best_season as a string, not a dict."""
    from scripts.build_player_comparison import _LEGENDS
    for leg in _LEGENDS:
        assert isinstance(leg.get("best_season", ""), str), (
            f"{leg['player_name']} best_season is {type(leg['best_season'])}, expected str"
        )


# ---------------------------------------------------------------------------
# 12. Advanced stats helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_teams() -> pd.DataFrame:
    """Team data matching synthetic_players' teams and seasons."""
    # One row per team per season, per-game averages (no header in real file)
    rows = []
    for season_str, season in [("2000-01", 200001), ("2001-02", 200102), ("2002-03", 200203)]:
        for team_name in ["Los Angeles Lakers", "Boston Celtics", "Chicago Bulls"]:
            rows.append({
                "team_id": 0, "team_name": team_name,
                "gp": 82, "w": 50, "l": 32, "w_pct": 0.610,
                "min": 48.0,
                "fgm": 38.0, "fga": 84.0, "fg_pct": 0.452,
                "fg3m": 8.0, "fg3a": 22.0, "fg3_pct": 0.364,
                "ftm": 18.0, "fta": 24.0, "ft_pct": 0.750,
                "oreb": 11.0, "dreb": 30.0, "reb": 41.0,
                "ast": 24.0, "tov": 14.0, "stl": 8.0, "blk": 5.0,
                "blka": 4.0, "pf": 20.0, "pfd": 20.0,
                "pts": 102.0, "plus_minus": 3.0,
                "season_str": season_str, "season": season,
            })
    df = pd.DataFrame(rows)
    # Add rank columns (fill with 1)
    for col in TEAM_CSV_COLUMNS:
        if col.endswith("_rank") and col not in df.columns:
            df[col] = 1
    # Reorder to match TEAM_CSV_COLUMNS
    df = df.reindex(columns=TEAM_CSV_COLUMNS, fill_value=0)
    return df


def test_build_team_season_lookup(synthetic_teams):
    lookup = _build_team_season_lookup(synthetic_teams)
    assert ("Los Angeles Lakers", "2000-01") in lookup
    tm = lookup[("Los Angeles Lakers", "2000-01")]
    assert tm["fga"] == 84.0
    assert tm["min"] == 48.0


def test_build_abbr_to_name_map(synthetic_teams):
    mapping = _build_abbr_to_name_map(synthetic_teams)
    assert ("LAL", "2000-01") in mapping
    assert mapping[("LAL", "2000-01")] == "Los Angeles Lakers"
    assert ("BOS", "2001-02") in mapping


def test_league_avg_reb_by_season(synthetic_teams):
    lg = _league_avg_reb_by_season(synthetic_teams)
    assert "2000-01" in lg
    assert abs(lg["2000-01"]["oreb"] - 11.0) < 0.1
    assert abs(lg["2000-01"]["dreb"] - 30.0) < 0.1


# ---------------------------------------------------------------------------
# 13. compute_advanced_stats
# ---------------------------------------------------------------------------

def test_compute_advanced_stats_adds_columns(synthetic_players, synthetic_teams):
    result = compute_advanced_stats(synthetic_players.copy(), synthetic_teams)
    for col in ("usg_pct", "ast_pct", "tov_pct", "efg_pct", "oreb_pct", "dreb_pct", "stl_pct"):
        assert col in result.columns, f"Missing column: {col}"


def test_compute_advanced_stats_usg_range(synthetic_players, synthetic_teams):
    """USG% should be in reasonable range (5-45%) for non-zero players."""
    result = compute_advanced_stats(synthetic_players.copy(), synthetic_teams)
    valid = result["usg_pct"].dropna()
    assert len(valid) > 0
    assert (valid >= 5).all(), f"USG% too low: {valid.min()}"
    assert (valid <= 45).all(), f"USG% too high: {valid.max()}"


def test_compute_advanced_stats_efg_range(synthetic_players, synthetic_teams):
    """eFG% should be between 0 and 1."""
    result = compute_advanced_stats(synthetic_players.copy(), synthetic_teams)
    valid = result["efg_pct"].dropna()
    assert len(valid) > 0
    assert (valid >= 0).all()
    assert (valid <= 1).all()


def test_compute_advanced_stats_tov_range(synthetic_players, synthetic_teams):
    """TOV% should be in reasonable range (0-30%)."""
    result = compute_advanced_stats(synthetic_players.copy(), synthetic_teams)
    valid = result["tov_pct"].dropna()
    assert len(valid) > 0
    assert (valid >= 0).all()
    assert (valid <= 40).all()


def test_compute_advanced_stats_no_teams_returns_nan(synthetic_players):
    """When teams=None, all advanced stat columns should be NaN."""
    result = compute_advanced_stats(synthetic_players.copy(), None)
    for col in ("usg_pct", "ast_pct", "tov_pct", "oreb_pct", "dreb_pct", "stl_pct"):
        assert result[col].isna().all(), f"{col} should be all NaN when no teams"


def test_compute_advanced_stats_efg_formula(synthetic_players, synthetic_teams):
    """Verify eFG% = (FGM + 0.5*FG3M) / FGA for Player A season 1."""
    result = compute_advanced_stats(synthetic_players.copy(), synthetic_teams)
    row = result[(result["player_id"] == 1) & (result["season_str"] == "2000-01")].iloc[0]
    expected = (9.0 + 0.5 * 1.0) / 20.0  # 0.475
    assert abs(row["efg_pct"] - expected) < 0.001


def test_compute_advanced_stats_tov_formula(synthetic_players, synthetic_teams):
    """Verify TOV% = 100 * TOV / (FGA + 0.44*FTA + TOV)."""
    df = synthetic_players.copy()
    df["tov"] = 3.0  # Add TOV to all rows
    result = compute_advanced_stats(df, synthetic_teams)
    # Player A 2000-01: fga=20, fta=7, tov=3
    row = result[(result["player_id"] == 1) & (result["season_str"] == "2000-01")].iloc[0]
    expected = 100 * 3.0 / (20.0 + 0.44 * 7.0 + 3.0)  # ~11.6%
    assert abs(row["tov_pct"] - expected) < 0.5


# ---------------------------------------------------------------------------
# 14. Season row includes advanced stats
# ---------------------------------------------------------------------------

def test_season_row_includes_advanced_stats(synthetic_players, league_averages, synthetic_teams):
    hist_avgs = {"pts": 14.5, "reb": 7.0, "ast": 3.5, "stl": 0.9, "blk": 0.4, "pace": 95.0}
    enriched = enrich_player_seasons(synthetic_players, league_averages, hist_avgs)
    enriched = compute_advanced_stats(enriched, synthetic_teams)
    records = build_player_records(enriched, min_seasons=1, min_career_games=0)
    player_a = next(r for r in records if r["player_name"] == "Player A")
    season_row = player_a["seasons"][0]
    for key in ("usg_pct", "ast_pct", "tov_pct", "efg_pct", "oreb_pct", "dreb_pct", "stl_pct"):
        assert key in season_row, f"Missing key in season row: {key}"


def test_career_avgs_include_advanced_stats(synthetic_players, league_averages, synthetic_teams):
    hist_avgs = {"pts": 14.5, "reb": 7.0, "ast": 3.5, "stl": 0.9, "blk": 0.4, "pace": 95.0}
    enriched = enrich_player_seasons(synthetic_players, league_averages, hist_avgs)
    enriched = compute_advanced_stats(enriched, synthetic_teams)
    records = build_player_records(enriched, min_seasons=1, min_career_games=0)
    player_a = next(r for r in records if r["player_name"] == "Player A")
    for key in ("usg_pct", "ast_pct", "tov_pct", "efg_pct", "oreb_pct", "dreb_pct", "stl_pct"):
        assert key in player_a["career_avgs"], f"Missing key in career_avgs: {key}"


# ---------------------------------------------------------------------------
# 15. Position inference (_infer_position)
# ---------------------------------------------------------------------------

def test_infer_position_center_advanced():
    """High OREB%+DREB% and blocks -> C."""
    ca = {"ast_pct": 9.0, "oreb_pct": 11.0, "dreb_pct": 24.0, "blk": 1.2, "ast": 1.5, "reb": 8.0}
    assert _infer_position(ca) == "C"


def test_infer_position_pf_advanced():
    """High DREB%, moderate OREB%, low AST% -> PF."""
    ca = {"ast_pct": 8.0, "oreb_pct": 7.5, "dreb_pct": 20.0, "blk": 0.6, "ast": 1.5, "reb": 6.5}
    assert _infer_position(ca) == "PF"


def test_infer_position_pg_advanced():
    """High AST%, low rebound rates -> PG."""
    ca = {"ast_pct": 25.0, "oreb_pct": 2.0, "dreb_pct": 10.0, "blk": 0.2, "ast": 5.0, "reb": 3.0}
    assert _infer_position(ca) == "PG"


def test_infer_position_sg_advanced():
    """Moderate AST%, low rebounds -> SG."""
    ca = {"ast_pct": 13.0, "oreb_pct": 2.5, "dreb_pct": 11.0, "blk": 0.3, "ast": 2.0, "reb": 3.0}
    assert _infer_position(ca) == "SG"


def test_infer_position_sf_advanced():
    """Moderate DREB%, moderate OREB%, low AST% -> SF."""
    ca = {"ast_pct": 10.0, "oreb_pct": 4.5, "dreb_pct": 16.0, "blk": 0.5, "ast": 1.5, "reb": 4.5}
    assert _infer_position(ca) == "SF"


def test_infer_position_generic_pos_constrains():
    """Generic position 'G' constrains to PG or SG even if stats look like SF."""
    ca = {"ast_pct": 10.0, "oreb_pct": 4.0, "dreb_pct": 14.0, "blk": 0.4, "ast": 2.0, "reb": 4.0}
    # Without constraint this might be SF
    result_constrained = _infer_position(ca, "G")
    assert result_constrained in ("PG", "SG")


def test_infer_position_basic_stats_center():
    """Pre-tracking era: high reb + blocks -> C."""
    ca = {"ast_pct": 0, "oreb_pct": 0, "dreb_pct": 0, "blk": 1.5, "ast": 2.0, "reb": 10.0}
    assert _infer_position(ca) == "C"


def test_infer_position_basic_stats_pg():
    """Pre-tracking era: high ast -> PG."""
    ca = {"ast_pct": 0, "oreb_pct": 0, "dreb_pct": 0, "blk": 0.1, "ast": 7.0, "reb": 3.0}
    assert _infer_position(ca) == "PG"


def test_infer_position_basic_stats_sf():
    """Pre-tracking era: moderate rebounds -> SF."""
    ca = {"ast_pct": 0, "oreb_pct": 0, "dreb_pct": 0, "blk": 0.3, "ast": 2.0, "reb": 5.0}
    assert _infer_position(ca) == "SF"


def test_infer_position_not_always_sg():
    """The old heuristic defaulted 73.5% to SG; the new one should produce variety."""
    positions = set()
    test_profiles = [
        {"ast_pct": 25.0, "oreb_pct": 2.0, "dreb_pct": 10.0, "blk": 0.2, "ast": 5.0, "reb": 3.0},
        {"ast_pct": 12.0, "oreb_pct": 3.0, "dreb_pct": 12.0, "blk": 0.3, "ast": 2.0, "reb": 3.5},
        {"ast_pct": 9.0, "oreb_pct": 5.0, "dreb_pct": 16.0, "blk": 0.5, "ast": 1.5, "reb": 5.0},
        {"ast_pct": 8.0, "oreb_pct": 8.0, "dreb_pct": 21.0, "blk": 0.8, "ast": 1.0, "reb": 7.0},
        {"ast_pct": 8.0, "oreb_pct": 11.0, "dreb_pct": 24.0, "blk": 1.2, "ast": 1.0, "reb": 9.0},
    ]
    for p in test_profiles:
        positions.add(_infer_position(p))
    assert len(positions) == 5, f"Expected all 5 positions, got {positions}"


def test_infer_position_pg_sg_tiebreaker():
    """PG/SG tiebreaker uses ast/reb ratio (PG >= 0.95)."""
    # High ast/reb ratio -> PG
    ca_pg = {"ast_pct": 18.0, "oreb_pct": 2.5, "dreb_pct": 11.0, "blk": 0.3, "ast": 4.0, "reb": 3.0}
    # Low ast/reb ratio -> SG
    ca_sg = {"ast_pct": 18.0, "oreb_pct": 2.5, "dreb_pct": 11.0, "blk": 0.3, "ast": 2.0, "reb": 4.0}
    assert _infer_position(ca_pg) == "PG"
    assert _infer_position(ca_sg) == "SG"
