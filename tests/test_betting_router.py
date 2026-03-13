"""Tests for BettingRouter: confidence tiers, model agreement, market outputs."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.betting_router import BettingRouter, confidence_tier, model_agreement


class TestConfidenceTier:
    def test_best_bet(self):
        assert confidence_tier(edge=0.10, models_agree=True) == "Best Bet"

    def test_solid_pick(self):
        assert confidence_tier(edge=0.05, models_agree=True) == "Solid Pick"

    def test_lean(self):
        assert confidence_tier(edge=0.025, models_agree=True) == "Lean"

    def test_skip_low_edge(self):
        assert confidence_tier(edge=0.01, models_agree=True) == "Skip"

    def test_skip_disagreement(self):
        assert confidence_tier(edge=0.10, models_agree=False) == "Skip"

    def test_boundary_best_bet(self):
        assert confidence_tier(edge=0.08, models_agree=True) == "Best Bet"

    def test_boundary_solid_pick(self):
        assert confidence_tier(edge=0.04, models_agree=True) == "Solid Pick"

    def test_boundary_lean(self):
        assert confidence_tier(edge=0.02, models_agree=True) == "Lean"

    def test_just_below_lean(self):
        assert confidence_tier(edge=0.019, models_agree=True) == "Skip"


class TestModelAgreement:
    def test_both_favor_home(self):
        assert model_agreement(win_prob=0.65, pred_margin=3.5) is True

    def test_both_favor_away(self):
        assert model_agreement(win_prob=0.35, pred_margin=-5.0) is True

    def test_disagree(self):
        assert model_agreement(win_prob=0.60, pred_margin=-2.0) is False

    def test_neutral_margin(self):
        assert model_agreement(win_prob=0.55, pred_margin=0.5) is True

    def test_neutral_margin_away(self):
        assert model_agreement(win_prob=0.40, pred_margin=-1.0) is True

    def test_disagree_away_prob_home_margin(self):
        assert model_agreement(win_prob=0.40, pred_margin=5.0) is False


def _make_router(residual_std: float = 10.5) -> BettingRouter:
    """Create a BettingRouter without loading artifacts from disk."""
    router = BettingRouter.__new__(BettingRouter)
    router.residual_std = residual_std
    return router


class TestBettingRouterSpread:
    def test_favorite_covers(self):
        router = _make_router()
        result = router.spread(pred_margin=7.0, spread_line=-5.5, win_prob=0.65)
        assert result["cover_prob"] > 0.8
        assert result["confidence_tier"] in ("Best Bet", "Solid Pick", "Lean", "Skip")

    def test_with_market_prob(self):
        router = _make_router()
        result = router.spread(
            pred_margin=7.0, spread_line=-5.5, win_prob=0.65, market_spread_prob=0.5
        )
        assert result["edge"] > 0
        assert result["ev"] is not None

    def test_underdog_spread(self):
        router = _make_router()
        # pred_margin=-3, spread_line=+5.5 -> CDF((-3-5.5)/10.5) ~ 0.21
        # Home team predicted to lose by 3 but getting 5.5 pts is still
        # evaluated as (pred_margin - spread_line), so cover_prob < 0.5
        result = router.spread(pred_margin=-3.0, spread_line=5.5, win_prob=0.40)
        assert 0.0 < result["cover_prob"] < 0.5

    def test_no_market_prob_zero_edge(self):
        router = _make_router()
        result = router.spread(pred_margin=3.0, spread_line=-2.0, win_prob=0.55)
        assert result["edge"] == 0.0
        assert result["ev"] is None


class TestBettingRouterMoneyline:
    def test_with_odds(self):
        router = _make_router()
        result = router.moneyline(
            win_prob=0.65, pred_margin=5.0, home_ml=-200, away_ml=170
        )
        assert "prob" in result
        assert "confidence_tier" in result
        assert result["models_agree"] is True

    def test_without_odds(self):
        router = _make_router()
        result = router.moneyline(win_prob=0.55, pred_margin=2.0)
        assert result["edge"] == 0.0
        assert result["ev"] is None

    def test_edge_positive_when_model_exceeds_market(self):
        router = _make_router()
        # -150/+130: home implied ~0.5357 after devig
        result = router.moneyline(
            win_prob=0.70, pred_margin=8.0, home_ml=-150, away_ml=130
        )
        assert result["edge"] > 0.10

    def test_edge_negative_when_model_below_market(self):
        router = _make_router()
        result = router.moneyline(
            win_prob=0.45, pred_margin=-3.0, home_ml=-200, away_ml=170
        )
        assert result["edge"] < 0

    def test_props_returns_dict(self):
        """props() returns a dict with expected keys when models exist."""
        from src.models.conformal import calibrate_conformal, save_conformal_quantiles
        from src.models.player_minutes_model import train_minutes_model
        from src.models.player_stat_models import train_quantile_models, train_stat_models

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "player_id": np.repeat(range(1, 6), 20),
            "game_date": pd.date_range("2025-01-01", periods=20).tolist() * 5,
            "minutes": np.random.normal(28, 5, n).clip(5, 48),
            "pts_per36": np.random.normal(18, 5, n).clip(0),
            "reb_per36": np.random.normal(7, 3, n).clip(0),
            "ast_per36": np.random.normal(4, 2, n).clip(0),
            "fg3m_per36": np.random.normal(2, 1, n).clip(0),
            "minutes_ewma": np.random.normal(28, 3, n),
            "usage_rate_ewma": np.random.normal(20, 5, n),
            "pts_ewma": np.random.normal(15, 5, n),
            "reb_ewma": np.random.normal(5, 2, n),
            "ast_ewma": np.random.normal(3, 2, n),
            "fg3m_ewma": np.random.normal(1.5, 1, n),
            "is_home": np.random.randint(0, 2, n),
            "is_b2b": np.random.randint(0, 2, n),
            "season_game_num": np.tile(range(1, 21), 5),
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            train_minutes_model(df, artifacts_dir=tmpdir)
            train_stat_models(df, artifacts_dir=tmpdir)
            train_quantile_models(df, artifacts_dir=tmpdir)

            # Calibrate conformal
            residuals = np.random.normal(0, 3, 50)
            quantiles = {}
            for stat in ["pts", "reb", "ast", "fg3m"]:
                quantiles[stat] = float(calibrate_conformal(residuals))
            save_conformal_quantiles(quantiles, tmpdir)

            router = BettingRouter(artifacts_dir=tmpdir)
            features = df.iloc[0:1]
            result = router.props(
                features=features,
                stat="pts",
                line=22.5,
                spread=5.0,
            )

            assert isinstance(result, dict)
            assert "median" in result
            assert "p25" in result
            assert "p75" in result
            assert "pred_minutes" in result
            assert "over_prob" in result
            assert "confidence_tier" in result
            assert "interval" in result
            assert "lower" in result["interval"]
            assert "upper" in result["interval"]
            assert result["stat"] == "pts"
            assert result["line"] == 22.5

    def test_props_invalid_stat(self):
        """props() raises ValueError for unknown stat."""
        router = _make_router()
        router.artifacts_dir = Path("models/artifacts")
        with pytest.raises(ValueError):
            router.props(features=pd.DataFrame(), stat="INVALID", line=10.0)


class TestOverProbSymmetry:
    """Verify over_prob formula is symmetric around p50."""

    def test_over_prob_at_p50_equals_half(self):
        """When line == p50, over_prob must be exactly 0.5."""
        # Simulate the formula directly
        p25, p50, p75 = 15.0, 22.0, 30.0
        line = p50
        spread_width = max(p75 - p25, 0.1)
        over_prob = 0.5 + (p50 - line) / spread_width
        over_prob = max(0.05, min(0.95, over_prob))
        assert over_prob == 0.5

    def test_over_prob_at_p25_near_one(self):
        """When line == p25, over_prob should be close to 1.0 (most outcomes above)."""
        p25, p50, p75 = 15.0, 22.0, 30.0
        line = p25
        spread_width = max(p75 - p25, 0.1)
        over_prob = 0.5 + (p50 - line) / spread_width
        over_prob = max(0.05, min(0.95, over_prob))
        assert over_prob > 0.9

    def test_over_prob_at_p75_near_zero(self):
        """When line == p75, over_prob should be close to 0.0 (most outcomes below)."""
        p25, p50, p75 = 15.0, 22.0, 30.0
        line = p75
        spread_width = max(p75 - p25, 0.1)
        over_prob = 0.5 + (p50 - line) / spread_width
        over_prob = max(0.05, min(0.95, over_prob))
        assert over_prob < 0.1

    def test_over_prob_skewed_at_p50(self):
        """Even with skewed quantiles, line==p50 must give exactly 0.5."""
        # Right-skewed: p50 closer to p25 than p75
        p25, p50, p75 = 10.0, 13.0, 25.0
        line = p50
        spread_width = max(p75 - p25, 0.1)
        over_prob = 0.5 + (p50 - line) / spread_width
        over_prob = max(0.05, min(0.95, over_prob))
        assert over_prob == 0.5
