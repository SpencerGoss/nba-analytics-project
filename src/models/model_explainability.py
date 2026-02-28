"""
Model Explainability
======================
Produces SHAP-based explanations for all three prediction models.
If SHAP is not installed, falls back to sklearn's permutation importance
(which is a solid alternative that doesn't require the extra dependency).

Why SHAP over raw feature importances:
  - Raw GBM importances only tell you which features were used most
    across all splits — they don't tell you the *direction* or *magnitude*
    of impact for a specific prediction.
  - SHAP values tell you exactly how much each feature pushed the prediction
    up or down for any given game or player.
  - The global SHAP summary plot shows both importance AND direction
    (e.g., "high rolling win% strongly increases home win probability").

Outputs (saved to reports/explainability/):
  Game outcome model:
    - shap_summary_game_outcome.png  (or permutation_importance_game.png)
    - shap_game_outcome.csv          — raw SHAP values for test set sample
    - feature_direction_game.csv     — mean SHAP per feature (sign = direction)

  Player model (per target: pts, reb, ast):
    - shap_summary_player_{target}.png
    - feature_direction_player_{target}.csv

Usage:
    python src/models/model_explainability.py

    Or:
        from src.models.model_explainability import (
            explain_game_outcome_model,
            explain_player_model,
            explain_prediction,
        )

    # Explain a specific game prediction:
        from src.models.model_explainability import explain_prediction
        explain_prediction("BOS", "LAL")
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")


# ── Optional SHAP import ───────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.inspection import permutation_importance


# ── Config ─────────────────────────────────────────────────────────────────────

MATCHUP_PATH    = "data/features/game_matchup_features.csv"
PLAYER_PATH     = "data/features/player_game_features.csv"
ARTIFACTS_DIR   = "models/artifacts"
OUTPUT_DIR      = "reports/explainability"

# Number of test-set rows to use for SHAP computation (SHAP is slow on large sets)
SHAP_SAMPLE     = 500
TEST_SEASONS    = ["202324", "202425"]

PLAYER_TARGETS  = ["pts", "reb", "ast"]
TARGET_CLASS    = "home_win"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_artifact(name: str, artifacts_dir: str = ARTIFACTS_DIR):
    path = os.path.join(artifacts_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            f"Run the relevant model script first to generate it."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _friendly_feature_name(col: str) -> str:
    """Convert snake_case feature names to readable labels for plots."""
    col = col.replace("home_", "HM ").replace("away_", "AW ")
    col = col.replace("_roll5", " (L5)").replace("_roll10", " (L10)").replace("_roll20", " (L20)")
    col = col.replace("_pct", "%").replace("_", " ")
    return col.title()


def _diverging_bar_chart(
    series: pd.Series,
    title: str,
    output_path: str,
    n: int = 30,
) -> None:
    """
    Draw a horizontal bar chart showing positive (green) and negative (red) SHAP.
    Used as a fallback when the SHAP library's own beeswarm plot isn't available.
    """
    top = series.abs().nlargest(n).index
    data = series[top].sort_values()
    labels = [_friendly_feature_name(c) for c in data.index]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in data.values]

    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.28)))
    bars = ax.barh(labels, data.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean SHAP value (positive = increases prediction)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    pos_patch = mpatches.Patch(color="#2ecc71", label="Increases prediction")
    neg_patch = mpatches.Patch(color="#e74c3c", label="Decreases prediction")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {output_path}")


def _permutation_bar_chart(
    feature_names: list,
    importances_mean: np.ndarray,
    importances_std:  np.ndarray,
    title: str,
    output_path: str,
    n: int = 25,
) -> None:
    """Horizontal bar chart of permutation importance with error bars."""
    idx = np.argsort(importances_mean)[-n:]
    labels   = [_friendly_feature_name(feature_names[i]) for i in idx]
    values   = importances_mean[idx]
    errors   = importances_std[idx]

    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.3)))
    ax.barh(
        labels, values, xerr=errors,
        color="#3498db", alpha=0.8, edgecolor="white", linewidth=0.5,
        error_kw={"elinewidth": 1, "ecolor": "gray"},
    )
    ax.set_xlabel("Mean decrease in score when feature is permuted", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {output_path}")


# ── Game outcome explainability ────────────────────────────────────────────────

def explain_game_outcome_model(
    matchup_path:  str = MATCHUP_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    output_dir:    str = OUTPUT_DIR,
    shap_sample:   int = SHAP_SAMPLE,
    test_seasons:  list = TEST_SEASONS,
) -> pd.DataFrame:
    """
    Explain the game outcome classifier.

    Returns a DataFrame of mean SHAP values (or permutation importances)
    per feature, sorted by absolute magnitude.
    """
    print("=" * 60)
    print("EXPLAINABILITY — Game Outcome Model")
    print("=" * 60)

    # ── Load model and data ───────────────────────────────────────────────────
    model     = _load_artifact("game_outcome_model.pkl", artifacts_dir)
    feat_cols = _load_artifact("game_outcome_features.pkl", artifacts_dir)

    df = pd.read_csv(matchup_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"]    = df["season"].astype(str)

    test = df[df["season"].isin(test_seasons)].copy()
    print(f"\nTest set size: {len(test):,} games  |  Features: {len(feat_cols)}")

    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="mean")
    X_raw = test[feat_cols]
    X_imp = imp.fit_transform(X_raw)
    X_df  = pd.DataFrame(X_imp, columns=feat_cols)

    # Sample for SHAP (full set is slow)
    sample_idx = np.random.default_rng(42).choice(len(X_df), size=min(shap_sample, len(X_df)), replace=False)
    X_sample   = X_df.iloc[sample_idx]

    os.makedirs(output_dir, exist_ok=True)

    if SHAP_AVAILABLE:
        print("\nComputing SHAP values (this may take a minute)...")
        # v2 model artifact is CalibratedClassifierCV wrapping a Pipeline;
        # SHAP needs the raw tree estimator, not the calibration wrapper.
        from sklearn.calibration import CalibratedClassifierCV as _CalCV
        if isinstance(model, _CalCV):
            clf = model.estimator.named_steps["clf"]   # inner GBM
        else:
            clf = model.named_steps["clf"]             # v1 plain Pipeline
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            sv = shap_values[1]   # use class 1 = home win
        else:
            sv = shap_values

        # ── SHAP beeswarm summary plot ─────────────────────────────────────
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, feature_names=feat_cols, show=False, max_display=25)
        plt.title("SHAP Feature Impact — Game Outcome Model", fontsize=12, fontweight="bold", pad=14)
        plt.tight_layout()
        beeswarm_path = os.path.join(output_dir, "shap_summary_game_outcome.png")
        plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  SHAP beeswarm saved → {beeswarm_path}")

        # ── Mean SHAP per feature ──────────────────────────────────────────
        mean_shap = pd.Series(sv.mean(axis=0), index=feat_cols, name="mean_shap")
        abs_shap  = pd.Series(np.abs(sv).mean(axis=0), index=feat_cols, name="abs_mean_shap")
        direction = pd.DataFrame({"mean_shap": mean_shap, "abs_mean_shap": abs_shap})
        direction = direction.sort_values("abs_mean_shap", ascending=False)

        # ── Diverging bar chart ────────────────────────────────────────────
        _diverging_bar_chart(
            mean_shap,
            "Mean SHAP Value per Feature — Game Outcome Model\n"
            "(positive = increases P(home win))",
            os.path.join(output_dir, "shap_direction_game_outcome.png"),
        )

        # ── Save raw SHAP values ───────────────────────────────────────────
        shap_df = pd.DataFrame(sv, columns=feat_cols)
        shap_df.to_csv(os.path.join(output_dir, "shap_game_outcome.csv"), index=False)

        print("\nTop 15 features by mean |SHAP|:")
        for feat, row in direction.head(15).iterrows():
            direction_str = "↑" if row["mean_shap"] > 0 else "↓"
            print(f"  {direction_str}  {feat:<45} {row['abs_mean_shap']:>8.5f}")

    else:
        # ── Fallback: permutation importance ──────────────────────────────
        print("\nSHAP not installed — using permutation importance as fallback.")
        print("Install SHAP with: pip install shap")
        print("\nComputing permutation importance (this takes ~1 minute)...")

        perm = permutation_importance(
            model, X_df, test[TARGET_CLASS].values,
            n_repeats=10, random_state=42, scoring="roc_auc",
        )

        direction = pd.DataFrame({
            "abs_mean_shap": perm.importances_mean,
            "mean_shap":     perm.importances_mean,
        }, index=feat_cols).sort_values("abs_mean_shap", ascending=False)

        _permutation_bar_chart(
            feat_cols, perm.importances_mean, perm.importances_std,
            "Permutation Importance — Game Outcome Model\n"
            "(mean decrease in ROC-AUC when feature is shuffled)",
            os.path.join(output_dir, "permutation_importance_game_outcome.png"),
        )

        print("\nTop 15 features by permutation importance:")
        for feat, row in direction.head(15).iterrows():
            print(f"  {feat:<45} {row['abs_mean_shap']:>8.5f}")

    # ── Save direction CSV ─────────────────────────────────────────────────────
    direction_path = os.path.join(output_dir, "feature_direction_game.csv")
    direction.reset_index().rename(columns={"index": "feature"}).to_csv(direction_path, index=False)
    print(f"\nFeature direction table saved → {direction_path}")

    return direction


# ── Player model explainability ────────────────────────────────────────────────

def explain_player_model(
    player_path:   str  = PLAYER_PATH,
    artifacts_dir: str  = ARTIFACTS_DIR,
    output_dir:    str  = OUTPUT_DIR,
    shap_sample:   int  = SHAP_SAMPLE,
    test_seasons:  list = TEST_SEASONS,
    targets:       list = PLAYER_TARGETS,
) -> dict:
    """
    Explain the player performance regression models.

    Returns a dict of {target: mean_shap_DataFrame}.
    """
    print("\n" + "=" * 60)
    print("EXPLAINABILITY — Player Performance Models")
    print("=" * 60)

    df = pd.read_csv(player_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"]    = df["season"].astype(str)

    test = df[df["season"].isin(test_seasons)].copy()
    print(f"\nTest set size: {len(test):,} rows")

    from sklearn.impute import SimpleImputer

    all_directions = {}

    for target in targets:
        print(f"\n{'─'*50}")
        print(f"Target: {target.upper()}")

        try:
            model     = _load_artifact(f"player_{target}_model.pkl", artifacts_dir)
            feat_cols = _load_artifact(f"player_{target}_features.pkl", artifacts_dir)
        except FileNotFoundError as e:
            print(f"  Skipping — {e}")
            continue

        t_test = test.dropna(subset=[target]).copy()
        if len(t_test) < 50:
            print(f"  Skipping — insufficient test data ({len(t_test)} rows)")
            continue

        imp   = SimpleImputer(strategy="mean")
        X_raw = t_test[feat_cols]
        X_imp = imp.fit_transform(X_raw)
        X_df  = pd.DataFrame(X_imp, columns=feat_cols)

        sample_idx = np.random.default_rng(42).choice(
            len(X_df), size=min(shap_sample, len(X_df)), replace=False
        )
        X_sample = X_df.iloc[sample_idx]

        if SHAP_AVAILABLE:
            print(f"  Computing SHAP values for {target}...")
            reg       = model.named_steps["model"]
            explainer   = shap.TreeExplainer(reg)
            shap_values = explainer.shap_values(X_sample)

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feat_cols, show=False, max_display=20)
            plt.title(f"SHAP Feature Impact — Player {target.upper()} Model",
                      fontsize=12, fontweight="bold", pad=14)
            plt.tight_layout()
            beeswarm_path = os.path.join(output_dir, f"shap_summary_player_{target}.png")
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  SHAP beeswarm saved → {beeswarm_path}")

            mean_shap = pd.Series(shap_values.mean(axis=0), index=feat_cols, name="mean_shap")
            abs_shap  = pd.Series(np.abs(shap_values).mean(axis=0), index=feat_cols, name="abs_mean_shap")
            direction = pd.DataFrame({"mean_shap": mean_shap, "abs_mean_shap": abs_shap})
            direction = direction.sort_values("abs_mean_shap", ascending=False)

            _diverging_bar_chart(
                mean_shap,
                f"Mean SHAP Value per Feature — Player {target.upper()} Model",
                os.path.join(output_dir, f"shap_direction_player_{target}.png"),
            )

        else:
            print(f"  Using permutation importance for {target}...")
            from sklearn.metrics import make_scorer
            from sklearn.metrics import mean_absolute_error as mae_fn
            neg_mae_scorer = make_scorer(mae_fn, greater_is_better=False)

            perm = permutation_importance(
                model, X_df, t_test[target].values,
                n_repeats=8, random_state=42, scoring=neg_mae_scorer,
            )
            direction = pd.DataFrame({
                "abs_mean_shap": np.abs(perm.importances_mean),
                "mean_shap":     perm.importances_mean,
            }, index=feat_cols).sort_values("abs_mean_shap", ascending=False)

            _permutation_bar_chart(
                feat_cols, np.abs(perm.importances_mean), perm.importances_std,
                f"Permutation Importance — Player {target.upper()} Model",
                os.path.join(output_dir, f"permutation_importance_player_{target}.png"),
            )

        # Save direction CSV
        direction_path = os.path.join(output_dir, f"feature_direction_player_{target}.csv")
        os.makedirs(output_dir, exist_ok=True)
        direction.reset_index().rename(columns={"index": "feature"}).to_csv(direction_path, index=False)
        print(f"  Feature direction saved → {direction_path}")

        print(f"\n  Top 10 features for {target}:")
        for feat, row in direction.head(10).iterrows():
            direction_str = "↑" if row["mean_shap"] > 0 else "↓"
            print(f"    {direction_str}  {feat:<40} {row['abs_mean_shap']:>8.5f}")

        all_directions[target] = direction

    return all_directions


# ── Single-prediction explainability ──────────────────────────────────────────

def explain_prediction(
    home_team_abbr:  str,
    away_team_abbr:  str,
    matchup_path:    str = MATCHUP_PATH,
    artifacts_dir:   str = ARTIFACTS_DIR,
    output_dir:      str = OUTPUT_DIR,
    top_n:           int = 10,
) -> dict:
    """
    Explain WHY the model predicts a specific matchup the way it does.

    Prints the top N features driving the home team's predicted win probability
    up or down, compared to the average prediction.

    Args:
        home_team_abbr: e.g. "BOS"
        away_team_abbr: e.g. "LAL"

    Returns:
        dict with win probabilities and feature explanations.
    """
    if not SHAP_AVAILABLE:
        print("SHAP is required for single-prediction explanations.")
        print("Install with: pip install shap")
        return {}

    model     = _load_artifact("game_outcome_model.pkl", artifacts_dir)
    feat_cols = _load_artifact("game_outcome_features.pkl", artifacts_dir)

    df = pd.read_csv(matchup_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    home_row = (
        df[df["home_team"] == home_team_abbr]
        .sort_values("game_date").iloc[-1]
    )
    away_row = (
        df[df["away_team"] == away_team_abbr]
        .sort_values("game_date").iloc[-1]
    )

    # Build feature vector from the home_row (it already has home_ and away_ cols
    # from the matchup feature table — we need the home team's latest game as home)
    from sklearn.impute import SimpleImputer
    row_df = home_row[feat_cols].to_frame().T
    imp    = SimpleImputer(strategy="mean")

    background = pd.read_csv(matchup_path)[feat_cols].sample(200, random_state=42)
    imp.fit(background)
    X_background = imp.transform(background)
    X_row        = imp.transform(row_df)

    # v2 model artifact is CalibratedClassifierCV — extract inner GBM for SHAP
    from sklearn.calibration import CalibratedClassifierCV as _CalCV
    if isinstance(model, _CalCV):
        clf = model.estimator.named_steps["clf"]
    else:
        clf = model.named_steps["clf"]
    explainer = shap.TreeExplainer(clf, data=X_background)
    sv        = explainer.shap_values(X_row)

    if isinstance(sv, list):
        sv = sv[1]

    win_prob = float(model.predict_proba(row_df.fillna(0))[0][1])

    shap_series = pd.Series(sv[0], index=feat_cols).sort_values(key=abs, ascending=False)

    print(f"\n{'='*55}")
    print(f"PREDICTION EXPLANATION: {home_team_abbr} (home) vs {away_team_abbr}")
    print(f"{'='*55}")
    print(f"Predicted home win probability: {win_prob:.1%}")
    print(f"\nTop {top_n} factors driving this prediction:")
    print(f"{'Feature':<45} {'SHAP':>8}  {'Direction'}")
    print("─" * 65)
    for feat, val in shap_series.head(top_n).items():
        arrow = "↑ increases" if val > 0 else "↓ decreases"
        print(f"  {_friendly_feature_name(feat):<43} {val:>+8.4f}  {arrow} P(home win)")

    # ── Waterfall plot ─────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 6))
    top_features = shap_series.head(top_n)
    labels   = [_friendly_feature_name(f) for f in top_features.index]
    values   = top_features.values
    colors   = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    plt.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("SHAP value (impact on P(home win))", fontsize=10)
    plt.title(
        f"Prediction Explanation: {home_team_abbr} vs {away_team_abbr}\n"
        f"P({home_team_abbr} wins) = {win_prob:.1%}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"explanation_{home_team_abbr}_vs_{away_team_abbr}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nWaterfall plot saved → {out}")

    return {
        "home_team": home_team_abbr,
        "away_team": away_team_abbr,
        "home_win_prob": win_prob,
        "top_factors": shap_series.head(top_n).to_dict(),
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    explain_game_outcome_model()
    explain_player_model()
    print("\nExplainability analysis complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}/")
