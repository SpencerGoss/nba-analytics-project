"""
build_sharp_money.py  --  produce dashboard/data/sharp_money.json

Aggregates sharp money signals from multiple dashboard data sources
to produce a composite sharp score for each game today.

Sources:
  - dashboard/data/value_bets.json     (edge pct, kelly fraction)
  - dashboard/data/game_context.json   (situational flags)
  - dashboard/data/matchup_analysis.json (offensive/defensive ratings)
  - dashboard/data/todays_picks.json   (model confidence)

Sharp score components (0-100):
  - edge_pct contribution (weight 40%):  normalized edge from value_bets
  - model_confidence (weight 30%):  |home_win_prob - 0.5| normalized to 0-1
  - situational_advantage (weight 30%):  flag-based score from game_context

Output per record:
  home_team, away_team, game_date, sharp_score, sharp_side,
  sharp_rating, edge_pct, model_confidence, situational_advantage,
  key_factors, fade_alert, kelly_fraction
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import logging

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "dashboard" / "data"
OUT_JSON = DATA_DIR / "sharp_money.json"

# Weights for sharp score (must sum to 1.0)
WEIGHT_EDGE = 0.40
WEIGHT_CONFIDENCE = 0.30
WEIGHT_SITUATIONAL = 0.30

# Rating thresholds
STRONG_THRESHOLD = 70
MODERATE_THRESHOLD = 50

# Situational flags that confer advantage to the home team
HOME_ADV_FLAGS = {"HOME_HOT", "AWAY_COLD", "REST_ADV", "HOME_REST_ADV"}
# Flags that confer advantage to the away team
AWAY_ADV_FLAGS = {"AWAY_HOT", "HOME_COLD", "AWAY_REST_ADV"}

# Max edge_pct we normalise against (cap at this to score 100% on edge component)
EDGE_CAP = 0.25


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _index_by_game(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Index records by (home_team, away_team)."""
    return {
        (str(r.get("home_team", "")), str(r.get("away_team", ""))): r
        for r in records
        if isinstance(r, dict)
    }


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def _edge_component(vbet: dict | None) -> tuple[float, float | None, float | None]:
    """
    Returns (score_0_to_1, edge_pct_raw, kelly_fraction).
    score = min(edge_pct / EDGE_CAP, 1.0)
    """
    if vbet is None:
        return 0.0, None, None
    edge = vbet.get("edge_pct")
    kelly = vbet.get("kelly_fraction")
    if edge is None:
        return 0.0, None, kelly
    edge_float = float(edge)
    score = min(edge_float / EDGE_CAP, 1.0)
    return score, edge_float, kelly


def _confidence_component(pick: dict | None) -> tuple[float, float, str]:
    """
    Returns (score_0_to_1, confidence_pct, sharp_side).
    score = |home_win_prob - 0.5| / 0.5
    """
    if pick is None:
        return 0.0, 50.0, ""
    home_prob = float(pick.get("home_win_prob") or 0.5)
    away_prob = float(pick.get("away_win_prob") or 0.5)
    score = abs(home_prob - 0.5) / 0.5
    confidence_pct = round(max(home_prob, away_prob) * 100, 1)
    sharp_side = (
        str(pick.get("home_team", ""))
        if home_prob >= away_prob
        else str(pick.get("away_team", ""))
    )
    return score, confidence_pct, sharp_side


def _situational_component(ctx: dict | None) -> tuple[float, str, list[str]]:
    """
    Returns (score_0_to_1, primary_flag, key_factors_list).
    Each flag in HOME_ADV_FLAGS / AWAY_ADV_FLAGS adds 0.33 (capped at 1.0).
    """
    if ctx is None:
        return 0.0, "", []

    flags: list[str] = ctx.get("situational_flags") or []
    home_adv_count = sum(1 for f in flags if f in HOME_ADV_FLAGS)
    away_adv_count = sum(1 for f in flags if f in AWAY_ADV_FLAGS)

    net_home_adv = home_adv_count - away_adv_count
    score = min(abs(net_home_adv) / 3.0, 1.0)

    primary_flag = ""
    if flags:
        # Prefer the first flag with known advantage direction
        for f in flags:
            if f in HOME_ADV_FLAGS or f in AWAY_ADV_FLAGS:
                primary_flag = f
                break
        if not primary_flag:
            primary_flag = flags[0]

    factors: list[str] = []
    home_team = str(ctx.get("home_team", ""))
    away_team = str(ctx.get("away_team", ""))

    if "HOME_HOT" in flags:
        factors.append(f"Home team ({home_team}) on hot streak")
    if "AWAY_COLD" in flags:
        factors.append(f"Away team ({away_team}) cold streak")
    if "AWAY_HOT" in flags:
        factors.append(f"Away team ({away_team}) on hot streak")
    if "HOME_COLD" in flags:
        factors.append(f"Home team ({home_team}) cold streak")
    if "REST_ADV" in flags or "HOME_REST_ADV" in flags:
        factors.append(f"{home_team} rest advantage")
    if "AWAY_REST_ADV" in flags:
        factors.append(f"{away_team} rest advantage")
    for f in flags:
        if f not in HOME_ADV_FLAGS and f not in AWAY_ADV_FLAGS and f not in ("HOME_HOT", "AWAY_COLD", "AWAY_HOT", "HOME_COLD"):
            factors.append(f.replace("_", " ").title())

    return score, primary_flag, factors


def _sharp_rating(score: int) -> str:
    if score >= STRONG_THRESHOLD:
        return "STRONG"
    if score >= MODERATE_THRESHOLD:
        return "MODERATE"
    return "LEAN"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_sharp_money() -> list[dict]:
    picks = _load_json(DATA_DIR / "todays_picks.json")
    value_bets = _load_json(DATA_DIR / "value_bets.json")
    contexts = _load_json(DATA_DIR / "game_context.json")

    if not picks:
        log.warning("  todays_picks.json empty or missing -- outputting []")
        return []

    picks_idx = _index_by_game(picks)
    vbets_idx = _index_by_game(value_bets)
    ctx_idx = _index_by_game(contexts)

    results: list[dict] = []

    for (home_team, away_team), pick in picks_idx.items():
        game_date = str(pick.get("game_date", ""))
        vbet = vbets_idx.get((home_team, away_team))
        ctx = ctx_idx.get((home_team, away_team))

        # Score components
        edge_score, edge_pct_raw, kelly = _edge_component(vbet)
        conf_score, model_confidence, conf_sharp_side = _confidence_component(pick)
        sit_score, primary_flag, sit_factors = _situational_component(ctx)

        # Composite sharp score 0-100
        raw_score = (
            WEIGHT_EDGE * edge_score
            + WEIGHT_CONFIDENCE * conf_score
            + WEIGHT_SITUATIONAL * sit_score
        )
        sharp_score = int(round(raw_score * 100))

        # Sharp side: prefer value bet recommended side, else confidence side
        if vbet and vbet.get("recommended_side"):
            sharp_side = str(vbet["recommended_side"])
        elif conf_sharp_side:
            sharp_side = conf_sharp_side
        else:
            sharp_side = home_team

        # Key factors
        key_factors: list[str] = []
        if edge_pct_raw is not None and edge_pct_raw > 0:
            key_factors.append(f"High model edge ({edge_pct_raw * 100:.1f}%)")
        if model_confidence > 60:
            key_factors.append(f"Strong model confidence ({model_confidence:.0f}%)")
        key_factors.extend(sit_factors)
        if not key_factors:
            key_factors.append("Marginal model lean")

        # Fade alert: true if sharp_side is the public underdog
        # (heuristic: if home_win_prob < 0.40, public likely on away -- sharp home = fade)
        home_prob = float(pick.get("home_win_prob") or 0.5)
        fade_alert = (
            sharp_side == home_team and home_prob < 0.40
        ) or (
            sharp_side == away_team and home_prob > 0.60
        )

        results.append({
            "home_team": home_team,
            "away_team": away_team,
            "game_date": game_date,
            "sharp_score": sharp_score,
            "sharp_side": sharp_side,
            "sharp_rating": _sharp_rating(sharp_score),
            "edge_pct": round(edge_pct_raw * 100, 2) if edge_pct_raw is not None else None,
            "model_confidence": model_confidence,
            "situational_advantage": primary_flag if primary_flag else None,
            "key_factors": key_factors,
            "fade_alert": fade_alert,
            "kelly_fraction": float(kelly) if kelly is not None else None,
        })

    # Sort by sharp_score descending
    results.sort(key=lambda x: x["sharp_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Building sharp money signals...")
    records = build_sharp_money()
    log.info(f"  {len(records)} game records")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    log.info(f"  Written -> {OUT_JSON}")
    for rec in records[:5]:
        log.info(f"  {rec['away_team']} @ {rec['home_team']}  "
            f"score={rec['sharp_score']}  side={rec['sharp_side']}  "
            f"rating={rec['sharp_rating']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
