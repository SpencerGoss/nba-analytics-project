# v1.0 Integration Gap Closure — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the 3 audit gaps (INT-01, INT-02, FLOW-01) by wiring ATS training + calibration into `train_all_models.py` and adding `ats` / `value-bet` subcommands to `predict_cli.py`.

**Architecture:** Edit two existing files only. `train_all_models.py` gains two additional training steps (calibration + ATS) after the existing three. `predict_cli.py` gains two argparse subcommands that delegate to existing model functions. No new files.

**Tech Stack:** Python, argparse, pandas, existing model modules (ats_model, calibration, value_bet_detector, game_outcome_model)

---

## Task 1: Wire ATS + Calibration into train_all_models.py

**Files:**
- Modify: `src/models/train_all_models.py`

**Step 1: Add imports**

At line 22 (after the existing playoff_odds import), add:

```python
from src.models.calibration import run_calibration_analysis
from src.models.ats_model import train_ats_model
```

**Step 2: Update header text in main()**

Change the print block (lines 52-55) from:

```python
    print("This run trains one workflow per core task:")
    print("  1) Game outcomes")
    print("  2) Player performance")
    print("  3) Playoff odds")
```

To:

```python
    print("This run trains one workflow per core task:")
    print("  1) Game outcomes")
    print("  2) Player performance")
    print("  3) Playoff odds")
    print("  4) Calibration analysis")
    print("  5) ATS spread model")
```

**Step 3: Add calibration step after playoff odds (before the elapsed/summary block)**

After the playoff odds block (after line 72), add:

```python
    print("\n" + "-" * 72)
    print("TASK 4/5 -- CALIBRATION")
    print("-" * 72)
    cal_metrics = run_calibration_analysis()

    print("\n" + "-" * 72)
    print("TASK 5/5 -- ATS SPREAD MODEL")
    print("-" * 72)
    _, ats_metrics = train_ats_model()
```

**Step 4: Update run summary block**

After the existing player performance summary line, before the Elapsed line, add:

```python
    print(
        f"Calibration -> Brier={cal_metrics.get('brier_calibrated', 0):.5f} | "
        f"ECE={cal_metrics.get('ece_calibrated', 0):.5f}"
    )
    print(
        f"ATS model -> model={ats_metrics.get('model_type', 'n/a')} | "
        f"test_acc={ats_metrics.get('test_accuracy', 0):.4f}"
    )
```

Also update the task counter in existing print lines from "TASK 1/3" to "TASK 1/5", "TASK 2/3" to "TASK 2/5", "TASK 3/3" to "TASK 3/5".

**Step 5: Verify by running with --help**

Run: `python src/models/train_all_models.py --help`
Expected: Shows help with --rebuild-features flag. No import errors.

**Step 6: Commit**

```bash
git add src/models/train_all_models.py
git commit -m "feat: wire ATS training + calibration into train_all_models.py

Closes INT-01 and FLOW-01: train_all_models.py now runs all 5 tasks
(game outcome, player performance, playoff odds, calibration, ATS).
With --rebuild-features, this is the single retrain command."
```

---

## Task 2: Add ats subcommand to predict_cli.py

**Files:**
- Modify: `src/models/predict_cli.py`

**Step 1: Add imports**

After the existing imports (line 18), add:

```python
from src.models.ats_model import predict_ats
from src.models.value_bet_detector import run_value_bet_scan
```

Also add at top:
```python
import numpy as np
import pandas as pd
```

**Step 2: Add ats subparser in parse_args()**

After the `player` subparser block (after line 36), add:

```python
    ats = sp.add_parser("ats", help="Predict ATS (against the spread) outcome")
    ats.add_argument("--home", required=True, help="Home team abbreviation (e.g., BOS)")
    ats.add_argument("--away", required=True, help="Away team abbreviation (e.g., LAL)")
    ats.add_argument("--spread", required=True, type=float, help="Point spread (home perspective, e.g., -3.5)")
    ats.add_argument("--home-ml", type=float, default=None, help="Home moneyline (e.g., -150)")
    ats.add_argument("--away-ml", type=float, default=None, help="Away moneyline (e.g., +130)")
```

**Step 3: Add ats handler in main()**

In the main() function, after the existing `else` (player) branch, replace with elif/else chain:

```python
def main() -> None:
    args = parse_args()
    if args.command == "game":
        out = predict_game(args.home, args.away, game_date=args.date)
    elif args.command == "player":
        out = predict_player_next_game(args.name)
    elif args.command == "ats":
        out = _handle_ats(args)
    else:
        out = _handle_value_bet(args)

    print(json.dumps(out, indent=2, default=str))
```

**Step 4: Implement _handle_ats helper**

Add before main():

```python
def _handle_ats(args) -> dict:
    """Build a feature row from latest matchup data + CLI spread/moneylines, then predict ATS."""
    matchup_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "features", "game_matchup_features.csv",
    )
    if not os.path.exists(matchup_path):
        return {"error": f"Matchup features not found at {matchup_path}. Run feature build first."}

    df = pd.read_csv(matchup_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Find most recent context for this matchup (same logic as predict_game)
    exact = df[(df["home_team"] == args.home) & (df["away_team"] == args.away)]
    if not exact.empty:
        row = exact.sort_values("game_date").iloc[-1].copy()
    else:
        home_rows = df[df["home_team"] == args.home].sort_values("game_date")
        away_rows = df[df["away_team"] == args.away].sort_values("game_date")
        if home_rows.empty or away_rows.empty:
            return {"error": f"Not enough history for {args.home} vs {args.away}."}
        row = home_rows.iloc[-1].copy()
        away_source = away_rows.iloc[-1]
        for c in df.columns:
            if c.startswith("away_"):
                row[c] = away_source.get(c, row.get(c, np.nan))
        # Recompute diff columns
        for c in df.columns:
            if c.startswith("diff_"):
                base = c.replace("diff_", "")
                h_col, a_col = f"home_{base}", f"away_{base}"
                if h_col in row.index and a_col in row.index:
                    row[c] = row[h_col] - row[a_col]

    # Inject CLI-provided spread and moneyline data
    row["spread"] = args.spread
    if args.home_ml is not None and args.away_ml is not None:
        from src.models.value_bet_detector import no_vig_prob
        home_nv, away_nv = no_vig_prob(args.home_ml, args.away_ml)
        row["home_implied_prob"] = home_nv
        row["away_implied_prob"] = away_nv

    row_df = row.to_frame().T
    result = predict_ats(row_df)

    return {
        "home_team": args.home,
        "away_team": args.away,
        "spread": args.spread,
        "covers_spread_prob": round(float(result["covers_spread_prob"].iloc[0]), 4),
        "covers_spread_pred": int(result["covers_spread_pred"].iloc[0]),
    }
```

**Step 5: Verify ats --help**

Run: `python src/models/predict_cli.py ats --help`
Expected: Shows --home, --away, --spread (required) and --home-ml, --away-ml (optional).

**Step 6: Commit**

```bash
git add src/models/predict_cli.py
git commit -m "feat: add ats subcommand to predict_cli.py

Closes INT-02 (partial): predict_cli.py now supports:
  python src/models/predict_cli.py ats --home BOS --away LAL --spread -3.5"
```

---

## Task 3: Add value-bet subcommand to predict_cli.py

**Files:**
- Modify: `src/models/predict_cli.py`

**Step 1: Add value-bet subparser in parse_args()**

After the `ats` subparser block, add:

```python
    vb = sp.add_parser("value-bet", help="Scan for value bets (model vs market disagreement)")
    vb.add_argument("--live", action="store_true", help="Use live odds from The Odds API (requires ODDS_API_KEY)")
    vb.add_argument("--threshold", type=float, default=0.05, help="Edge threshold for flagging value bets (default: 0.05)")
```

**Step 2: Implement _handle_value_bet helper**

Add before main():

```python
def _handle_value_bet(args) -> list:
    """Run value-bet scan and return results."""
    return run_value_bet_scan(
        use_live_odds=args.live,
        threshold=args.threshold,
    )
```

**Step 3: Verify value-bet --help**

Run: `python src/models/predict_cli.py value-bet --help`
Expected: Shows --live and --threshold flags.

**Step 4: Commit**

```bash
git add src/models/predict_cli.py
git commit -m "feat: add value-bet subcommand to predict_cli.py

Closes INT-02: predict_cli.py now supports all prediction types:
  game, player, ats, value-bet"
```

---

## Task 4: Update docs and verify

**Files:**
- Modify: `docs/project_overview.md` (section 9, Known Limitations)

**Step 1: Remove closed gaps from Known Limitations**

In `docs/project_overview.md` section 9 "Known Limitations and Tech Debt", remove or update:
- "train_all_models.py does not call train_ats_model() or run_calibration_analysis()" -> mark as resolved
- "predict_cli.py has no ats or value-bet subcommand" -> mark as resolved
- "Full retrain flow requires 4 manual commands" -> update to note single command

**Step 2: Verify full --help**

Run: `python src/models/predict_cli.py --help`
Expected: Shows all 4 subcommands: game, player, ats, value-bet

Run: `python src/models/train_all_models.py --help`
Expected: Shows --rebuild-features flag

**Step 3: Commit**

```bash
git add docs/project_overview.md
git commit -m "docs: update known limitations after v1.0 gap closure"
```
