"""Tests for BettingRouter: confidence tiers, model agreement, market outputs."""
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

    def test_props_raises(self):
        router = _make_router()
        with pytest.raises(NotImplementedError):
            router.props(player_id=1, stat="PTS", line=25.5)
