"""BettingRouter: market-specific prediction routing with confidence tiers.

Deliberate deviation from spec: takes pre-computed win_prob/pred_margin instead
of team abbreviations. This keeps BettingRouter decoupled from NBAEnsemble --
callers (build_picks.py, build_value_bets.py) already have ensemble outputs.
The spec's team-based interface would couple BettingRouter to model loading.

Wraps NBAEnsemble outputs and provides separate outputs per betting market:
- Moneyline: calibrated P(home_win)
- Spread: P(cover) via normal CDF on (pred_margin - spread) / residual_std
- Props: stub for Phase 4

Confidence tiers (strict, plain English, no jargon):
- Best Bet:   edge >= 8%, models agree
- Solid Pick: edge >= 4%, models agree
- Lean:       edge >= 2%
- Skip:       edge < 2% OR models disagree
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import norm

from src.models.odds_utils import expected_value, no_vig_odds_ratio

BEST_BET_EDGE = 0.08
SOLID_PICK_EDGE = 0.04
LEAN_EDGE = 0.02
MARGIN_DEAD_ZONE = 1.5
DEFAULT_RESIDUAL_STD = 10.5


def confidence_tier(edge: float, models_agree: bool) -> str:
    """Assign strict confidence tier.

    Note: This is "bootstrap mode" per spec 2.2 -- uses edge thresholds only.
    After 100+ tracked games validate tier boundaries (Phase 3 backtest),
    tighten to require historical win rate > 65% for Best Bet.
    """
    if not models_agree:
        return "Skip"
    if edge >= BEST_BET_EDGE:
        return "Best Bet"
    if edge >= SOLID_PICK_EDGE:
        return "Solid Pick"
    if edge >= LEAN_EDGE:
        return "Lean"
    return "Skip"


def model_agreement(win_prob: float, pred_margin: float) -> bool:
    """Check if outcome model and margin model agree on direction."""
    if abs(pred_margin) < MARGIN_DEAD_ZONE:
        return True
    return (win_prob > 0.5) == (pred_margin > 0)


class BettingRouter:
    """Routes predictions to market-specific outputs with confidence tiers."""

    def __init__(self, artifacts_dir: str = "models/artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.residual_std = self._load_residual_std()

    def _load_residual_std(self) -> float:
        """Load residual std from margin model training artifacts."""
        path = self.artifacts_dir / "margin_residual_std.json"
        if path.exists():
            with open(path) as f:
                return float(json.load(f)["residual_std"])
        return DEFAULT_RESIDUAL_STD

    def moneyline(
        self,
        win_prob: float,
        pred_margin: float,
        home_ml: int | float | None = None,
        away_ml: int | float | None = None,
    ) -> dict:
        """Moneyline market output with edge, EV, and confidence tier."""
        agree = model_agreement(win_prob, pred_margin)
        edge = 0.0
        ev = None
        if home_ml is not None and away_ml is not None:
            market_home, _ = no_vig_odds_ratio(home_ml, away_ml)
            if market_home is not None:
                edge = win_prob - market_home
                ev = expected_value(win_prob, market_home)

        tier = confidence_tier(abs(edge), agree)
        return {
            "prob": round(win_prob, 4),
            "pred_margin": round(pred_margin, 2),
            "edge": round(edge, 4),
            "ev": round(ev, 4) if ev is not None else None,
            "confidence_tier": tier,
            "models_agree": agree,
        }

    def spread(
        self,
        pred_margin: float,
        spread_line: float,
        win_prob: float,
        market_spread_prob: float | None = None,
    ) -> dict:
        """Spread market output. Uses margin model standalone via normal CDF."""
        cover_prob = float(
            norm.cdf((pred_margin - spread_line) / self.residual_std)
        )
        agree = model_agreement(win_prob, pred_margin)
        edge = 0.0
        ev = None
        if market_spread_prob is not None and market_spread_prob > 0:
            edge = cover_prob - market_spread_prob
            ev = expected_value(cover_prob, market_spread_prob)

        tier = confidence_tier(abs(edge), agree)
        return {
            "cover_prob": round(cover_prob, 4),
            "pred_margin": round(pred_margin, 2),
            "spread_line": spread_line,
            "edge": round(edge, 4),
            "ev": round(ev, 4) if ev is not None else None,
            "confidence_tier": tier,
            "models_agree": agree,
        }

    def props(
        self,
        player_id: int,
        stat: str,
        line: float,
        date: str | None = None,
    ) -> dict:
        """Player prop output. Stub until Phase 4."""
        raise NotImplementedError(
            "Player prop predictions not yet implemented. See Phase 4."
        )
