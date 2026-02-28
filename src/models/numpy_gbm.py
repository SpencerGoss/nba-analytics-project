"""
numpy_gbm.py  —  Pure-numpy Gradient Boosted Machine
======================================================
A self-contained GBM implementation that depends only on numpy and pandas.
Replaces scikit-learn's GradientBoostingClassifier / GradientBoostingRegressor
with an implementation that:

  • Uses fully vectorised numpy operations for split-finding (cumulative-sum
    trick) so training is orders of magnitude faster than naive Python loops.
  • Supports subsampling (stochastic GBM) and feature subsampling.
  • Handles NaN values (NaN rows always go to the "left" child).
  • Includes a lightweight Platt-scaling probability calibrator.
  • Exposes scikit-learn-compatible interfaces (fit / predict / predict_proba
    / feature_importances_) so existing code needs minimal changes.

Classes
-------
  FastTree              – vectorised regression tree (depth-limited CART)
  FastGBMClassifier     – binary GBM with log-loss
  FastGBMRegressor      – GBM with squared-error loss
  PlattCalibrator       – logistic calibration layer
"""

import numpy as np
import pandas as pd
import warnings

# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _log_odds(p):
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised regression tree
# ──────────────────────────────────────────────────────────────────────────────

class FastTree:
    """
    Regression tree that uses the cumulative-sum trick for O(n) best-split
    finding per feature instead of O(n²).

    Parameters
    ----------
    max_depth : int
    min_samples_leaf : int
    max_features : float or None
        Fraction of features to consider at each split (None = all features).
    random_state : int or None
    """

    def __init__(self, max_depth=3, min_samples_leaf=20,
                 max_features=None, random_state=None):
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features     = max_features
        self.random_state     = random_state
        self._rng             = np.random.RandomState(random_state)

        # Tree stored as flat arrays; node 0 = root
        self._feat      = []   # split feature index  (-1 = leaf)
        self._thresh    = []   # split threshold
        self._left      = []   # index of left child  (-1 = leaf)
        self._right     = []   # index of right child (-1 = leaf)
        self._value     = []   # leaf prediction value
        self._n_feat    = None
        self.feature_importances_ = None

    # ------------------------------------------------------------------
    # Internal: vectorised best-split search
    # ------------------------------------------------------------------

    def _best_split(self, X, y, feat_indices):
        """Return (best_feat, best_thresh, best_gain) using cumsum trick."""
        n = len(y)
        if n < 2 * self.min_samples_leaf:
            return -1, 0.0, 0.0

        base_ss  = np.sum(y ** 2) - (y.sum() ** 2) / n
        best_gain, best_feat, best_thresh = 0.0, -1, 0.0

        for j in feat_indices:
            col      = X[:, j]
            sort_idx = np.argsort(col, kind="mergesort")  # stable sort for NaN
            ys       = y[sort_idx]
            cs       = col[sort_idx]

            # Find valid split range
            lo = self.min_samples_leaf
            hi = n - self.min_samples_leaf
            if lo >= hi:
                continue

            # Cumulative stats (fully vectorised)
            cum_y   = np.cumsum(ys)
            cum_y2  = np.cumsum(ys ** 2)

            total_y  = cum_y[-1]
            total_y2 = cum_y2[-1]

            k_arr  = np.arange(lo, hi)          # candidate split AFTER index k
            nl     = k_arr + 1
            nr     = n - nl

            ly     = cum_y[k_arr]
            ly2    = cum_y2[k_arr]
            ry     = total_y  - ly
            ry2    = total_y2 - ly2

            # Variance reduction
            gains = base_ss - (ly2 - ly ** 2 / nl) - (ry2 - ry ** 2 / nr)

            # Mask out splits on identical consecutive values
            valid = cs[k_arr] != cs[k_arr + 1]
            gains = np.where(valid, gains, -np.inf)

            if gains.max() > best_gain:
                ki         = int(np.argmax(gains))
                best_gain  = float(gains[ki])
                best_feat  = j
                best_thresh = 0.5 * (cs[lo + ki] + cs[lo + ki + 1])

        return best_feat, best_thresh, best_gain

    # ------------------------------------------------------------------
    # Internal: recursive tree building
    # ------------------------------------------------------------------

    def _build(self, X, y, depth, gain_acc):
        node_id = len(self._feat)
        # Pre-allocate node as leaf
        self._feat.append(-1)
        self._thresh.append(0.0)
        self._left.append(-1)
        self._right.append(-1)
        self._value.append(float(np.mean(y)))

        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return node_id

        # Feature subsampling
        p = X.shape[1]
        if self.max_features is not None:
            nf = max(1, int(round(p * self.max_features)))
            fi = self._rng.choice(p, size=nf, replace=False)
        else:
            fi = np.arange(p)

        feat, thresh, gain = self._best_split(X, y, fi)
        if feat == -1 or gain <= 0.0:
            return node_id

        # Record gain for feature importance
        gain_acc[feat] = gain_acc.get(feat, 0.0) + gain

        # Update internal node
        self._feat[node_id]   = feat
        self._thresh[node_id] = thresh

        lm = X[:, feat] <= thresh
        rm = ~lm

        left_id  = self._build(X[lm], y[lm], depth + 1, gain_acc)
        right_id = self._build(X[rm], y[rm], depth + 1, gain_acc)

        self._left[node_id]  = left_id
        self._right[node_id] = right_id

        return node_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Fit tree on (X, y)."""
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(float)
        else:
            X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Replace NaN in X with column medians
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X = X.copy()
        X[nan_mask] = col_medians[np.where(nan_mask)[1]]

        self._n_feat = X.shape[1]
        self._feat = []
        self._thresh = []
        self._left = []
        self._right = []
        self._value = []

        gain_acc = {}
        self._build(X, y, 0, gain_acc)

        # Convert to numpy arrays for fast prediction
        self._feat_arr   = np.array(self._feat,   dtype=np.int32)
        self._thresh_arr = np.array(self._thresh,  dtype=float)
        self._left_arr   = np.array(self._left,    dtype=np.int32)
        self._right_arr  = np.array(self._right,   dtype=np.int32)
        self._value_arr  = np.array(self._value,   dtype=float)

        # Feature importances
        imps = np.zeros(self._n_feat)
        for feat, gain in gain_acc.items():
            imps[feat] = gain
        total = imps.sum()
        self.feature_importances_ = imps / total if total > 0 else imps

        return self

    def predict(self, X):
        """Vectorised prediction."""
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(float)
        else:
            X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # NaN → median (use 0 as fallback since we don't store medians)
        X = np.where(np.isnan(X), 0.0, X)

        node_idx = np.zeros(n, dtype=np.int32)  # all start at root

        for _ in range(self.max_depth + 2):
            lefts  = self._left_arr[node_idx]
            is_leaf = lefts == -1
            if is_leaf.all():
                break

            feats   = self._feat_arr[node_idx]
            threshs = self._thresh_arr[node_idx]

            # Vectorised feature value lookup: X[i, feats[i]]
            feat_vals = X[np.arange(n), np.clip(feats, 0, X.shape[1] - 1)]
            goes_left = feat_vals <= threshs

            rights  = self._right_arr[node_idx]
            new_idx = np.where(is_leaf, node_idx,
                      np.where(goes_left, lefts, rights))

            if np.array_equal(new_idx, node_idx):
                break
            node_idx = new_idx

        return self._value_arr[node_idx]


# ──────────────────────────────────────────────────────────────────────────────
# Gradient Boosting Classifier (binary log-loss)
# ──────────────────────────────────────────────────────────────────────────────

class FastGBMClassifier:
    """
    Binary gradient boosting classifier with log-loss.

    Parameters
    ----------
    n_estimators   : int     – number of trees
    learning_rate  : float   – shrinkage per tree
    max_depth      : int     – maximum tree depth
    min_samples_leaf : int
    subsample      : float   – row subsampling fraction per tree
    max_features   : float   – column subsampling fraction per tree
    random_state   : int or None
    """

    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=3,
                 min_samples_leaf=20, subsample=0.75, max_features=0.5,
                 random_state=42):
        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample        = subsample
        self.max_features     = max_features
        self.random_state     = random_state

        self._rng              = np.random.RandomState(random_state)
        self._trees            = []        # list of FastTree
        self._f0               = 0.0      # initial log-odds
        self._col_medians      = None
        self.feature_importances_ = None

    # ------------------------------------------------------------------

    def _impute(self, X):
        """Replace NaN with stored column medians."""
        if self._col_medians is None:
            return X
        nan_mask = np.isnan(X)
        X = X.copy()
        X[nan_mask] = self._col_medians[np.where(nan_mask)[1]]
        return X

    def fit(self, X, y):
        """Fit GBM classifier."""
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X = X.values.astype(float)
        else:
            self._feature_names = None
            X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n, p = X.shape

        # Store medians for NaN imputation
        self._col_medians = np.nanmedian(X, axis=0)
        X = self._impute(X)

        # Initial prediction (log-odds of base rate)
        p_mean = np.clip(y.mean(), 1e-7, 1.0 - 1e-7)
        self._f0 = float(np.log(p_mean / (1.0 - p_mean)))
        F = np.full(n, self._f0)

        imp_acc = np.zeros(p)

        for i in range(self.n_estimators):
            # Row subsampling
            n_sub = max(1, int(n * self.subsample))
            sub_idx = self._rng.choice(n, size=n_sub, replace=False)
            Xs, Fs, ys = X[sub_idx], F[sub_idx], y[sub_idx]

            # Negative gradient of log-loss = y - sigmoid(F)
            residuals = ys - _sigmoid(Fs)

            tree = FastTree(
                max_depth        = self.max_depth,
                min_samples_leaf = self.min_samples_leaf,
                max_features     = self.max_features,
                random_state     = self._rng.randint(0, 2**31),
            )
            tree.fit(Xs, residuals)

            # Update F on the FULL dataset
            F += self.learning_rate * tree.predict(X)

            # Accumulate importances
            imp_acc += tree.feature_importances_

            self._trees.append(tree)

        # Normalise importances
        total = imp_acc.sum()
        self.feature_importances_ = imp_acc / total if total > 0 else imp_acc

        return self

    def _raw_score(self, X):
        """Return raw F scores (log-odds)."""
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(float)
        else:
            X = np.asarray(X, dtype=float)
        X = self._impute(X)
        F = np.full(X.shape[0], self._f0)
        for tree in self._trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        p = _sigmoid(self._raw_score(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient Boosting Regressor (squared-error / MSE loss)
# ──────────────────────────────────────────────────────────────────────────────

class FastGBMRegressor:
    """
    Gradient boosting regressor with squared-error loss.

    Parameters same as FastGBMClassifier (minus classification-specific ones).
    """

    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=3,
                 min_samples_leaf=15, subsample=0.75, max_features=0.5,
                 random_state=42):
        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample        = subsample
        self.max_features     = max_features
        self.random_state     = random_state

        self._rng              = np.random.RandomState(random_state)
        self._trees            = []
        self._f0               = 0.0
        self._col_medians      = None
        self.feature_importances_ = None

    def _impute(self, X):
        if self._col_medians is None:
            return X
        nan_mask = np.isnan(X)
        X = X.copy()
        X[nan_mask] = self._col_medians[np.where(nan_mask)[1]]
        return X

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X = X.values.astype(float)
        else:
            self._feature_names = None
            X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n, p = X.shape

        self._col_medians = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self._f0 = float(y.mean())
        F = np.full(n, self._f0)

        imp_acc = np.zeros(p)

        for i in range(self.n_estimators):
            n_sub = max(1, int(n * self.subsample))
            sub_idx = self._rng.choice(n, size=n_sub, replace=False)
            Xs, Fs, ys = X[sub_idx], F[sub_idx], y[sub_idx]

            residuals = ys - Fs  # MSE negative gradient

            tree = FastTree(
                max_depth        = self.max_depth,
                min_samples_leaf = self.min_samples_leaf,
                max_features     = self.max_features,
                random_state     = self._rng.randint(0, 2**31),
            )
            tree.fit(Xs, residuals)

            F += self.learning_rate * tree.predict(X)
            imp_acc += tree.feature_importances_
            self._trees.append(tree)

        total = imp_acc.sum()
        self.feature_importances_ = imp_acc / total if total > 0 else imp_acc

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(float)
        else:
            X = np.asarray(X, dtype=float)
        X = self._impute(X)
        F = np.full(X.shape[0], self._f0)
        for tree in self._trees:
            F += self.learning_rate * tree.predict(X)
        return F


# ──────────────────────────────────────────────────────────────────────────────
# Platt scaling calibrator
# ──────────────────────────────────────────────────────────────────────────────

class PlattCalibrator:
    """
    Fits a logistic function  p_cal = sigmoid(a * s + b)  on top of a
    classifier's raw scores to produce calibrated probabilities.
    Uses gradient descent with L-BFGS-style line search.
    """

    def __init__(self, max_iter=1000, tol=1e-6):
        self.max_iter = max_iter
        self.tol      = tol
        self.a_       = 1.0
        self.b_       = 0.0

    def fit(self, scores, y):
        """Fit on raw classifier scores and binary targets."""
        scores = np.asarray(scores, dtype=float)
        y      = np.asarray(y,      dtype=float)

        a, b = 1.0, 0.0
        lr   = 0.01

        for _ in range(self.max_iter):
            p    = _sigmoid(a * scores + b)
            grad_a = np.mean((p - y) * scores)
            grad_b = np.mean( p - y)

            a_new = a - lr * grad_a
            b_new = b - lr * grad_b

            loss_old = -np.mean(y * np.log(np.clip(p,          1e-10, 1)) +
                                (1-y) * np.log(np.clip(1 - p,  1e-10, 1)))
            p_new    = _sigmoid(a_new * scores + b_new)
            loss_new = -np.mean(y * np.log(np.clip(p_new,      1e-10, 1)) +
                                (1-y) * np.log(np.clip(1-p_new,1e-10, 1)))

            if loss_new < loss_old:
                a, b = a_new, b_new
            else:
                lr *= 0.5

            if abs(grad_a) < self.tol and abs(grad_b) < self.tol:
                break

        self.a_ = a
        self.b_ = b
        return self

    def predict_proba(self, scores):
        scores = np.asarray(scores, dtype=float)
        p = _sigmoid(self.a_ * scores + self.b_)
        return np.column_stack([1.0 - p, p])


# ──────────────────────────────────────────────────────────────────────────────
# Calibrated GBM classifier  (mirrors CalibratedClassifierCV interface)
# ──────────────────────────────────────────────────────────────────────────────

class CalibratedGBMClassifier:
    """
    Wraps FastGBMClassifier + PlattCalibrator.

    Exposes:
      fit(X_train, y_train, X_cal, y_cal)  – train GBM on train set,
                                             fit calibrator on cal set
      predict_proba(X)                     – calibrated probabilities
      predict(X)
      feature_importances_                 – from inner GBM

    The 'estimator' attribute exposes the raw GBM (parallel to sklearn's
    CalibratedClassifierCV.estimator).
    """

    def __init__(self, **gbm_kwargs):
        self.estimator   = FastGBMClassifier(**gbm_kwargs)
        self._calibrator = PlattCalibrator()

    def fit(self, X_train, y_train, X_cal=None, y_cal=None):
        self.estimator.fit(X_train, y_train)
        cal_X = X_cal if X_cal is not None else X_train
        cal_y = y_cal if y_cal is not None else y_train
        raw_scores = self.estimator._raw_score(cal_X)
        self._calibrator.fit(raw_scores, cal_y)
        self.feature_importances_ = self.estimator.feature_importances_
        return self

    def predict_proba(self, X):
        raw = self.estimator._raw_score(X)
        return self._calibrator.predict_proba(raw)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers (replace sklearn.metrics)
# ──────────────────────────────────────────────────────────────────────────────

def accuracy_score(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def roc_auc_score(y_true, y_score):
    """Trapezoidal AUC via Mann-Whitney U statistic."""
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 0.5
    # Vectorised: count (pos_score > neg_score) + 0.5*(pos_score == neg_score)
    pos_scores = y_score[pos_idx][:, None]
    neg_scores = y_score[neg_idx][None, :]
    u = np.sum(pos_scores > neg_scores) + 0.5 * np.sum(pos_scores == neg_scores)
    return float(u / (len(pos_idx) * len(neg_idx)))


def brier_score_loss(y_true, y_prob):
    """Mean squared error between probabilities and labels."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_true - y_prob) ** 2))


def log_loss(y_true, y_prob, eps=1e-15):
    """Binary cross-entropy."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(y_prob,    eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) +
                          (1.0 - y_true) * np.log(1.0 - y_prob)))


def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Feature selection helpers
# ──────────────────────────────────────────────────────────────────────────────

def remove_collinear_features(X, y, threshold=0.92):
    """
    Drop one feature from each pair whose |Pearson r| > threshold.
    Keeps the feature more correlated with the target y.

    Parameters
    ----------
    X : pd.DataFrame
    y : array-like
    threshold : float

    Returns
    -------
    keep_cols : list[str]
    """
    cols   = list(X.columns)
    y_arr  = np.asarray(y, dtype=float)
    X_arr  = X[cols].values.astype(float)

    # Pairwise correlation (nan-safe via pandas)
    corr_matrix = X[cols].corr().values
    np.fill_diagonal(corr_matrix, 0.0)

    # Correlation of each feature with target
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_corr = np.array([
            abs(float(np.corrcoef(X_arr[:, j], y_arr)[0, 1]))
            if np.std(X_arr[:, j]) > 0 else 0.0
            for j in range(len(cols))
        ])

    drop = set()
    for i in range(len(cols)):
        if i in drop:
            continue
        for j in range(i + 1, len(cols)):
            if j in drop:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                # Drop the one less correlated with y
                if y_corr[i] >= y_corr[j]:
                    drop.add(j)
                else:
                    drop.add(i)
                    break  # i is dropped; move to next i

    keep_cols = [c for k, c in enumerate(cols) if k not in drop]
    return keep_cols


def prune_by_importance(importances, feature_names, cumulative_frac=0.999,
                        min_features=10):
    """
    Keep the smallest set of features that together account for
    `cumulative_frac` of total importance (minimum `min_features`).

    Parameters
    ----------
    importances   : np.ndarray  – feature importance scores
    feature_names : list[str]
    cumulative_frac : float
    min_features  : int

    Returns
    -------
    keep_names : list[str]
    """
    order    = np.argsort(importances)[::-1]
    cumsum   = np.cumsum(importances[order])
    total    = cumsum[-1]
    if total == 0:
        return list(feature_names)

    cutoff_idx = np.searchsorted(cumsum, cumulative_frac * total)
    cutoff_idx = max(cutoff_idx, min_features - 1)
    keep_idx   = order[: cutoff_idx + 1]
    return [feature_names[i] for i in sorted(keep_idx)]
