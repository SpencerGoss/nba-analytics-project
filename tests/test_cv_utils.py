"""Tests for shared CV utilities."""
import pandas as pd
import pytest

from src.models.cv_utils import expanding_season_splits


def _make_df(seasons):
    """Create minimal DataFrame with season column."""
    rows = []
    for s in seasons:
        for _ in range(10):
            rows.append({"season": s, "x": 1.0, "y": 0})
    return pd.DataFrame(rows)


def test_expanding_splits_basic():
    df = _make_df([201314, 201415, 201516, 201617, 201718, 201819, 201920])
    splits = expanding_season_splits(df, min_train_seasons=4)
    # With 7 seasons and min_train=4, first split trains on 3 seasons (idx 0..2), validates on idx 3
    assert len(splits) >= 3
    for tr, va, label in splits:
        assert not tr.empty
        assert not va.empty
        assert label != "date_fallback"


def test_expanding_splits_returns_tuples():
    df = _make_df([201920, 202021, 202122, 202223, 202324, 202425])
    splits = expanding_season_splits(df, min_train_seasons=3)
    for item in splits:
        assert len(item) == 3
        assert isinstance(item[0], pd.DataFrame)
        assert isinstance(item[1], pd.DataFrame)


def test_fallback_when_few_seasons():
    df = _make_df([202425])
    splits = expanding_season_splits(df, min_train_seasons=6)
    assert len(splits) == 1
    _, _, label = splits[0]
    assert label == "date_fallback"


def test_no_overlap_between_train_and_valid():
    df = _make_df([201920, 202021, 202122, 202223, 202324])
    splits = expanding_season_splits(df, min_train_seasons=3)
    for tr, va, label in splits:
        train_seasons = set(tr["season"].unique())
        valid_seasons = set(va["season"].unique())
        assert train_seasons.isdisjoint(valid_seasons)


def test_empty_dataframe():
    df = pd.DataFrame(columns=["season", "x", "y"])
    splits = expanding_season_splits(df, min_train_seasons=3)
    assert splits == []


def test_min_train_respected():
    df = _make_df([201920, 202021, 202122, 202223, 202324])
    splits = expanding_season_splits(df, min_train_seasons=4)
    # With min_train=4, first valid split is at index 3 (training on 3 seasons)
    for tr, va, label in splits:
        if label != "date_fallback":
            train_seasons = sorted(tr["season"].unique())
            assert len(train_seasons) >= 3
