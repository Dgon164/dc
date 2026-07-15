"""BASELINES ONLY — the segment plan's INTEGRITY law restated:
sklearn runs baselines, labeled BASELINE, never the cognition. All
finding/deciding on Isaac's side is engine. A baseline here answers
with a bare score and no receipt — that asymmetry is part of what the
bench reports, not an oversight."""
import numpy as np


# ── B1: flaggers (marginal — they never learn the law) ──

def zscore_flags(y, k=3.0):
    mu, sd = float(np.mean(y)), float(np.std(y))
    if sd == 0:
        return set()
    return {i for i, v in enumerate(y) if abs(v - mu) > k * sd}


def iqr_flags(y, k=1.5):
    q1, q3 = np.percentile(y, [25, 75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return {i for i, v in enumerate(y) if v < lo or v > hi}


def isoforest_flags(rows, seed=0):
    from sklearn.ensemble import IsolationForest
    lab = IsolationForest(random_state=seed).fit_predict(rows)
    return {i for i, v in enumerate(lab) if v == -1}


# ── B2: imputers (they always answer; refusal is not in their
#      vocabulary — the bench scores that too) ──

def mean_impute(y_obs, n_masked):
    return np.full(n_masked, float(np.mean(y_obs)))


def knn_impute(M, target_col, masked_idx, seed=0):
    """M: full matrix with NaN at masked target cells."""
    from sklearn.impute import KNNImputer
    out = KNNImputer(n_neighbors=5).fit_transform(M)
    return out[masked_idx, target_col]


def iterative_impute(M, target_col, masked_idx, seed=0):
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    out = IterativeImputer(random_state=seed).fit_transform(M)
    return out[masked_idx, target_col]
