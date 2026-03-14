"""
Model Calibration Analysis
============================
Evaluates whether predicted win probabilities are trustworthy - i.e.,
when the model says "70% chance the home team wins," does that actually
happen about 70% of the time?

This is called *calibration*, and it's the most important diagnostic for
any model that outputs probabilities rather than just labels. A model with
60% accuracy but perfect calibration is far more useful than one with 62%
accuracy whose probabilities are all over the place.

Three calibration diagnostics are computed:

1. Reliability diagram (calibration curve)
   - Bins predictions into deciles (0-10%, 10-20%, ..., 90-100%)
   - Compares the mean predicted probability in each bin to the actual
     win rate in that bin
   - A perfectly calibrated model lies on the diagonal

2. Brier score
   - The mean squared error between predicted probabilities and actual
     outcomes (0 = perfect, 0.25 = completely uninformative)
   - Lower is better; the theoretical floor for NBA game prediction is
     around 0.22-0.24 because the sport has genuine randomness

3. Expected Calibration Error (ECE)
   - The weighted average gap between predicted probabilities and actual
     win rates across all bins
   - More interpretable than the raw curve - a single number summarizing
     how miscalibrated the model is

Outputs (saved to reports/calibration/):
    calibration_curve.png          - reliability diagram
    calibration_metrics.csv        - Brier score, ECE, per-bin stats
    calibration_by_era.csv         - calibration metrics per historical era
    calibration_by_season.csv      - per-season Brier score trend

Usage:
    python src/models/calibration.py

    Or:
        from src.models.calibration import run_calibration_analysis
        metrics = run_calibration_analysis()
"""

import logging
import os
import pickle
import warnings
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
warnings.filterwarnings("ignore")

from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# -- Config ---------------------------------------------------------------------

MATCHUP_PATH   = "data/features/game_matchup_features.csv"
ARTIFACTS_DIR  = "models/artifacts"
OUTPUT_DIR     = "reports/calibration"
from src.models.game_outcome_model import TEST_SEASONS
TARGET         = "home_win"
N_BINS         = 10            # number of calibration bins


# -- Calibrated model wrapper --------------------------------------------------

class _CalibratedWrapper:
    """Wraps a base model + IsotonicRegression into a single object with
    predict_proba(). Lightweight replacement for CalibratedClassifierCV
    with cv='prefit' (removed in sklearn 1.6)."""

    def __init__(self, base_model, isotonic: IsotonicRegression):
        self.base_model = base_model
        self.isotonic = isotonic
        self.calibration_method = "isotonic"

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.isotonic.predict(raw)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _PlattWrapper:
    """Wraps a base model + Platt scaling (logistic sigmoid) into a single
    object with predict_proba(). Uses a 1-feature LogisticRegression fitted
    on the base model's raw probabilities -- only 2 parameters (slope +
    intercept), so much less prone to overfitting than isotonic regression
    on small calibration sets."""

    def __init__(self, base_model, platt_lr: _LR):
        self.base_model = base_model
        self.platt_lr = platt_lr
        self.calibration_method = "platt"

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.platt_lr.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _TemperatureWrapper:
    """Wraps a base model + temperature scaling into a single predict_proba()
    interface.  Temperature scaling applies a single scalar T to the logits:
        calibrated_prob = sigmoid(logit(raw_prob) / T)
    T > 1 softens overconfident predictions, T < 1 sharpens, T = 1 is identity.
    """

    def __init__(self, base_model, temperature: float):
        self.base_model = base_model
        self.temperature = temperature
        self.calibration_method = "temperature"

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        raw_clipped = np.clip(raw, 1e-7, 1 - 1e-7)
        logits = np.log(raw_clipped / (1 - raw_clipped))
        scaled = logits / self.temperature
        cal = 1.0 / (1.0 + np.exp(-scaled))
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Find the optimal temperature T that minimizes NLL (log loss).

    Uses bounded scalar optimization on [0.1, 10.0].
    """
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import log_loss

    probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = np.log(probs_clipped / (1 - probs_clipped))

    def nll(T):
        scaled = logits / T
        cal = 1.0 / (1.0 + np.exp(-scaled))
        return log_loss(labels, cal)

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


# -- Helpers --------------------------------------------------------------------

def _load_model(artifacts_dir: str = ARTIFACTS_DIR):
    path = os.path.join(artifacts_dir, "game_outcome_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained model not found at {path}.\n"
            "Run src/models/game_outcome_model.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_features(artifacts_dir: str = ARTIFACTS_DIR) -> list:
    path = os.path.join(artifacts_dir, "game_outcome_features.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = N_BINS,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum over bins of (bin_size / total) * |mean_confidence - mean_accuracy|
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)

    return float(ece)


def _bin_calibration_stats(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """Return per-bin calibration statistics."""
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_low":    round(bins[i], 2),
            "bin_high":   round(bins[i + 1], 2),
            "n_games":    int(mask.sum()),
            "mean_pred":  float(y_prob[mask].mean()),
            "actual_rate": float(y_true[mask].mean()),
            "gap":        float(y_prob[mask].mean() - y_true[mask].mean()),
        })

    return pd.DataFrame(rows)


# -- Calibration curve plot -----------------------------------------------------

def _plot_calibration_curve(
    y_true: np.ndarray,
    y_prob_uncal: np.ndarray,
    y_prob_cal:   np.ndarray,
    brier_uncal:  float,
    brier_cal:    float,
    ece_uncal:    float,
    ece_cal:      float,
    output_dir:   str,
    cal_method:   str = "isotonic",
) -> None:
    """
    Plot the reliability diagram comparing the raw model to the
    calibrated version (isotonic or Platt scaling).
    """
    frac_pos_u, mean_pred_u = calibration_curve(y_true, y_prob_uncal, n_bins=N_BINS)
    frac_pos_c, mean_pred_c = calibration_curve(y_true, y_prob_cal,   n_bins=N_BINS)

    cal_label = "Isotonic" if cal_method == "isotonic" else "Platt"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # -- Reliability diagram ----------------------------------------------------
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration", alpha=0.6)
    ax.plot(mean_pred_u, frac_pos_u, "o-", color="#e74c3c", linewidth=2,
            markersize=7, label=f"Base GBM (Brier={brier_uncal:.4f}, ECE={ece_uncal:.4f})")
    ax.plot(mean_pred_c, frac_pos_c, "s-", color="#2ecc71", linewidth=2,
            markersize=7, label=f"GBM + {cal_label} (Brier={brier_cal:.4f}, ECE={ece_cal:.4f})")

    ax.fill_between(mean_pred_u, frac_pos_u, mean_pred_u,
                    alpha=0.12, color="#e74c3c", label="Miscalibration area")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives (Actual Win Rate)", fontsize=11)
    ax.set_title("Reliability Diagram\n(Game Outcome Model)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.grid(True, alpha=0.3)

    # -- Prediction distribution histogram -------------------------------------
    ax2 = axes[1]
    ax2.hist(y_prob_uncal[y_true == 1], bins=20, alpha=0.6, color="#2ecc71",
             label="Home Win", density=True)
    ax2.hist(y_prob_uncal[y_true == 0], bins=20, alpha=0.6, color="#e74c3c",
             label="Home Loss", density=True)
    ax2.set_xlabel("Predicted P(Home Win)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Predicted Probability Distribution\nby Actual Outcome",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Game Outcome Model -- Calibration Analysis", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "calibration_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Calibration curve saved -> {out}")


# -- Per-season Brier score trend -----------------------------------------------

BRIER_MIN_SEASON = 200001   # only walk-forward from modern era (avoids 70+ fits)


def _compute_season_brier(
    model,
    df:       pd.DataFrame,
    feat_cols: list,
    all_seasons: list,
    min_train: int = 5,
) -> pd.DataFrame:
    """
    Walk forward and compute the Brier score for each test season.
    Lightweight version -- uses only 50 estimators for speed.
    Restricted to BRIER_MIN_SEASON onward to avoid 70+ GBM fits on historical data.
    """
    log.info("\nComputing per-season Brier score (walk-forward)...")
    rows = []

    for i in range(min_train, len(all_seasons)):
        train_seasons = all_seasons[:i]
        test_season   = all_seasons[i]

        if test_season < BRIER_MIN_SEASON:
            continue

        train = df[df["season"].isin(train_seasons)]
        test  = df[df["season"] == test_season]

        if len(test) < 50:
            continue

        from sklearn.ensemble import GradientBoostingClassifier

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", GradientBoostingClassifier(
                n_estimators=50, max_depth=3,
                learning_rate=0.05, random_state=42,
            )),
        ])

        pipe.fit(train[feat_cols], train[TARGET])
        proba = pipe.predict_proba(test[feat_cols])[:, 1]
        brier = brier_score_loss(test[TARGET].values, proba)

        rows.append({
            "season":      test_season,
            "n_games":     len(test),
            "brier_score": round(brier, 5),
        })
        log.info(f"  {test_season}: Brier = {brier:.5f}  ({len(test)} games)")

    return pd.DataFrame(rows)


def _plot_brier_trend(df: pd.DataFrame, output_dir: str) -> None:
    """Line chart of Brier score over time with era bands."""
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(df["season"], df["brier_score"], color="#3498db", linewidth=2, marker="o",
            markersize=4, alpha=0.85, label="Brier score (lower = better)")

    # Rolling 5-season average — work on a copy to avoid mutating the caller's DataFrame
    df = df.copy()
    df["brier_roll5"] = df["brier_score"].rolling(5, min_periods=1).mean()
    ax.plot(df["season"], df["brier_roll5"], color="#e67e22", linewidth=2.5,
            linestyle="--", label="5-season rolling avg")

    # Era shading
    era_colors = ["#eaf4fb", "#fef9e7", "#eafaf1", "#fdf2f8", "#f0f3fd", "#fef5e7"]
    era_breaks = [
        (194647, 195354, "Pre-Shot Clock", era_colors[0]),
        (195455, 197879, "Shot Clock Era", era_colors[1]),
        (197980, 199394, "3-Point Intro",  era_colors[2]),
        (199495, 200304, "Physical / Iso", era_colors[3]),
        (200405, 201415, "Open Court",     era_colors[4]),
        (201516, 202526, "3-Pt Revolution",era_colors[5]),
    ]
    all_seasons = df["season"].tolist()

    for start, end, name, color in era_breaks:
        idx_start = next((i for i, s in enumerate(all_seasons) if s >= start), None)
        idx_end   = next((i for i, s in enumerate(all_seasons) if s > end), len(all_seasons))
        if idx_start is None:
            continue
        ax.axvspan(idx_start - 0.5, idx_end - 0.5, alpha=0.25, color=color, label=name)

    ax.set_xlabel("Season", fontsize=11)
    ax.set_ylabel("Brier Score", fontsize=11)
    ax.set_title("Brier Score by Season -- Game Outcome Model\n"
                 "(lower = better calibration; NBA floor ~ 0.22-0.24)",
                 fontsize=12, fontweight="bold")

    # Thin out x-axis labels so they don't overlap
    tick_step = max(1, len(all_seasons) // 15)
    ax.set_xticks(range(0, len(all_seasons), tick_step))
    ax.set_xticklabels(all_seasons[::tick_step], rotation=45, ha="right", fontsize=8)

    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(output_dir, "brier_score_by_season.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Brier trend chart saved -> {out}")


# -- Main analysis --------------------------------------------------------------

def run_calibration_analysis(
    matchup_path:  str  = MATCHUP_PATH,
    artifacts_dir: str  = ARTIFACTS_DIR,
    output_dir:    str  = OUTPUT_DIR,
    test_seasons:  list = TEST_SEASONS,
    calibration_season: int = 202122,
) -> dict:
    """
    Run the full calibration analysis suite.

    Returns:
        dict with keys: brier_score, ece, bin_stats, season_brier
    """
    print("=" * 60)
    log.info("CALIBRATION ANALYSIS -- Game Outcome Model")
    print("=" * 60)

    # -- Load ------------------------------------------------------------------
    model     = _load_model(artifacts_dir)
    feat_cols = _load_features(artifacts_dir)

    df = pd.read_csv(matchup_path)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df["season"]    = df["season"].astype(int)
    df = df.sort_values("game_date").reset_index(drop=True)
    df = df.dropna(subset=[TARGET])         # drop unplayed / future games

    test = df[df["season"].isin(test_seasons)].copy()
    log.info(f"\nTest set: {len(test):,} games  ({', '.join(str(s) for s in test_seasons)})")

    X_test = test[feat_cols]
    y_test = test[TARGET].values

    # -- Raw model predictions -------------------------------------------------
    # The v2 model artifact is a CalibratedClassifierCV wrapper around the base
    # GBM Pipeline.  We extract the inner pipeline for "pre-calibration" numbers
    # so the chart can show the improvement from calibration.
    # The v1 model is a plain Pipeline -- handled the same way for compatibility.
    if isinstance(model, (CalibratedClassifierCV, _CalibratedWrapper, _PlattWrapper, _TemperatureWrapper)):
        if isinstance(model, (_CalibratedWrapper, _PlattWrapper, _TemperatureWrapper)):
            raw_model = model.base_model
        else:
            raw_model = model.estimator
        existing_method = getattr(model, "calibration_method", "isotonic")
        log.info(f"\n  (v2 model detected - comparing base GBM vs {existing_method}-calibrated output)")
    else:
        raw_model = model             # v1 plain Pipeline

    y_prob = raw_model.predict_proba(X_test)[:, 1]

    brier_uncal = brier_score_loss(y_test, y_prob)
    ece_uncal   = _expected_calibration_error(y_test, y_prob)
    bin_stats   = _bin_calibration_stats(y_test, y_prob)

    log.info(f"\n-- Base Model (pre-calibration) --------------------------")
    log.info(f"  Brier score : {brier_uncal:.5f}  (lower is better; 0.25 = coin flip)")
    log.info(f"  ECE         : {ece_uncal:.5f}  (lower is better; 0 = perfect)")

    # -- Calibrated predictions ------------------------------------------------
    # v2: the loaded artifact is already the calibrated model -- use it directly.
    # v1: fit both isotonic and Platt calibrators on the calibration holdout,
    #     compare Brier scores, and keep the better one.
    if isinstance(model, (CalibratedClassifierCV, _CalibratedWrapper, _PlattWrapper, _TemperatureWrapper)):
        existing_method = getattr(model, "calibration_method", "isotonic")
        log.info(f"\nUsing pre-fitted {existing_method} calibration from trained model artifact...")
        y_prob_cal = model.predict_proba(X_test)[:, 1]
        cal_model  = model   # already saved; no extra write needed
        selected_method = existing_method
    else:
        train_all = df[~df["season"].isin(test_seasons)].copy()
        train_all = train_all.dropna(subset=[TARGET])

        # Use held-out calibration season if available -- avoids in-sample leakage
        calib_rows = None
        if calibration_season and calibration_season in train_all["season"].values:
            calib_rows = train_all[train_all["season"] == calibration_season].copy()

        if calib_rows is not None and len(calib_rows) >= 100:
            log.info(f"\nFitting calibrators on held-out season {calibration_season} "
                  f"({len(calib_rows):,} games) -- out-of-sample, no leakage...")
            calib_probs = model.predict_proba(calib_rows[feat_cols])[:, 1]
            calib_y = calib_rows[TARGET].values
        else:
            log.info("\nFitting calibrators on training data (v1 model, in-sample)...")
            calib_probs = model.predict_proba(train_all[feat_cols])[:, 1]
            calib_y = train_all[TARGET].values

        # --- Isotonic regression (non-parametric) ---
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(calib_probs, calib_y)
        y_prob_iso = iso.predict(y_prob)
        brier_iso = brier_score_loss(y_test, y_prob_iso)

        # --- Platt scaling (logistic sigmoid, 2 parameters) ---
        platt_lr = _LR(solver="lbfgs", max_iter=1000)
        platt_lr.fit(calib_probs.reshape(-1, 1), calib_y)
        y_prob_platt = platt_lr.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        brier_platt = brier_score_loss(y_test, y_prob_platt)

        # --- Temperature scaling (single parameter T) ---
        T_opt = _fit_temperature(calib_probs, calib_y)
        y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        logits_test = np.log(y_prob_clipped / (1 - y_prob_clipped))
        scaled_test = logits_test / T_opt
        y_prob_temp = 1.0 / (1.0 + np.exp(-scaled_test))
        brier_temp = brier_score_loss(y_test, y_prob_temp)

        log.info(f"\n-- Calibration Method Comparison --------------------------")
        log.info(f"  Isotonic    Brier : {brier_iso:.5f}")
        log.info(f"  Platt       Brier : {brier_platt:.5f}")
        log.info(f"  Temperature Brier : {brier_temp:.5f}  (T={T_opt:.3f})")
        log.info(f"  Base model  Brier : {brier_uncal:.5f}")

        # Select the calibration method with the lowest Brier score.
        candidates = [
            ("isotonic", brier_iso, y_prob_iso, _CalibratedWrapper(model, iso)),
            ("platt", brier_platt, y_prob_platt, _PlattWrapper(model, platt_lr)),
            ("temperature", brier_temp, y_prob_temp, _TemperatureWrapper(model, T_opt)),
        ]
        candidates.sort(key=lambda x: x[1])
        selected_method, best_brier, y_prob_cal, cal_model = candidates[0]
        log.info(f"\n  -> Selected: {selected_method} (Brier {best_brier:.5f})")

        if brier_uncal < best_brier:
            log.info(f"  NOTE: Base model Brier ({brier_uncal:.5f}) is already better than "
                  f"all calibrators. Calibration may be overfitting on a small holdout set.")

        # Save the calibrated wrapper alongside the raw model so downstream
        # scripts can load calibrated probabilities without re-fitting.
        os.makedirs(artifacts_dir, exist_ok=True)
        cal_path = os.path.join(artifacts_dir, "game_outcome_model_calibrated.pkl")
        with open(cal_path, "wb") as _f:
            pickle.dump(cal_model, _f)
        log.info(f"\nCalibrated model saved -> {cal_path}")

    brier_cal = brier_score_loss(y_test, y_prob_cal)
    ece_cal   = _expected_calibration_error(y_test, y_prob_cal)

    cal_label = {"platt": "Platt", "isotonic": "Isotonic", "temperature": "Temperature"}.get(selected_method, selected_method.title())
    log.info(f"\n-- {cal_label} Calibrated Model -----------------------------")
    log.info(f"  Brier score : {brier_cal:.5f}")
    log.info(f"  ECE         : {ece_cal:.5f}")
    log.info(f"\n  Brier improvement : {brier_uncal - brier_cal:+.5f}")
    log.info(f"  ECE   improvement : {ece_uncal - ece_cal:+.5f}")

    # -- Per-bin stats ---------------------------------------------------------
    log.info("\n-- Calibration by Bin ------------------------------------")
    log.info(f"{'Bin':<12} {'N Games':>9} {'Pred %':>8} {'Actual %':>9} {'Gap':>8}")
    print("-" * 52)
    for _, row in bin_stats.iterrows():
        gap_str = f"{row['gap']:+.4f}"
        flag    = " [!]" if abs(row["gap"]) > 0.05 else ""
        log.info(f"  {row['bin_low']:.0%}-{row['bin_high']:.0%}   "
              f"{int(row['n_games']):>9,}   {row['mean_pred']:>7.1%}   "
              f"{row['actual_rate']:>7.1%}   {gap_str}{flag}")

    # -- Plot ------------------------------------------------------------------
    _plot_calibration_curve(
        y_test, y_prob, y_prob_cal,
        brier_uncal, brier_cal,
        ece_uncal, ece_cal,
        output_dir,
        cal_method=selected_method,
    )

    # -- Per-season Brier trend -------------------------------------------------
    all_seasons = sorted(df["season"].unique().tolist())
    season_brier = _compute_season_brier(model, df, feat_cols, all_seasons)
    _plot_brier_trend(season_brier, output_dir)

    # -- Era-level breakdown ----------------------------------------------------
    log.info("\n-- Calibration by Era ------------------------------------")
    try:
        from src.features.era_labels import get_era
        test = test.copy()
        test["era_name"] = test["season"].apply(lambda s: get_era(s)["era_name"])
        era_groups = []
        for era_name, grp in test.groupby("era_name"):
            grp_proba = model.predict_proba(grp[feat_cols])[:, 1]
            era_groups.append({
                "era_name":   era_name,
                "n_games":    len(grp),
                "brier":      round(brier_score_loss(grp[TARGET].values, grp_proba), 5),
                "ece":        round(_expected_calibration_error(grp[TARGET].values, grp_proba), 5),
            })
        era_df = pd.DataFrame(era_groups)
        log.info(f"  {'Era':<30} {'N Games':>9} {'Brier':>9} {'ECE':>8}")
        log.info("  " + "-" * 60)
        for _, r in era_df.iterrows():
            log.info(f"  {r['era_name']:<30} {r['n_games']:>9,} {r['brier']:>9.5f} {r['ece']:>8.5f}")
        os.makedirs(output_dir, exist_ok=True)
        era_df.to_csv(os.path.join(output_dir, "calibration_by_era.csv"), index=False)
    except Exception as e:
        log.error(f"  Could not compute era breakdown: {e}")

    # -- Save outputs ----------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    metrics_df = pd.DataFrame([{
        "metric": "brier_score_raw",       "value": round(brier_uncal, 6)},
        {"metric": "brier_score_calibrated","value": round(brier_cal, 6)},
        {"metric": "ece_raw",               "value": round(ece_uncal, 6)},
        {"metric": "ece_calibrated",        "value": round(ece_cal, 6)},
        {"metric": "calibration_method",    "value": selected_method},
    ])
    metrics_path = os.path.join(output_dir, "calibration_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"\nMetrics saved -> {metrics_path}")

    bin_path = os.path.join(output_dir, "calibration_bins.csv")
    bin_stats.to_csv(bin_path, index=False)
    log.info(f"Bin stats saved -> {bin_path}")

    season_path = os.path.join(output_dir, "calibration_by_season.csv")
    season_brier.to_csv(season_path, index=False)
    log.info(f"Season Brier saved -> {season_path}")

    return {
        "brier_score":         brier_uncal,
        "brier_calibrated":    brier_cal,
        "ece":                 ece_uncal,
        "ece_calibrated":      ece_cal,
        "calibration_method":  selected_method,
        "bin_stats":           bin_stats,
        "season_brier":        season_brier,
    }


# -- Entry point ----------------------------------------------------------------

if __name__ == "__main__":
    metrics = run_calibration_analysis()
    log.info(f"\nCalibration analysis complete.")
    log.info(f"  Brier score: {metrics['brier_score']:.5f}")
    log.info(f"  ECE:         {metrics['ece']:.5f}")
    log.info(f"\nOutputs saved to: {OUTPUT_DIR}/")
