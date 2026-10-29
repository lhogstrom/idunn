"""
AUC confidence-interval utilities.

DeLong's analytic 95% CI for ROC-AUC, based on the Mann-Whitney U-statistic
variance. Implemented via the Sun & Xu 2014 fast O(n log n) midrank
algorithm — no extra dependencies beyond numpy + scipy.stats.norm.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of x (1-indexed), with ties resolved by averaging."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n, dtype=float)
    out[j] = t
    return out


def _fast_delong(predictions_sorted: np.ndarray, label_1_count: int):
    """Sun & Xu (2014) fast DeLong; positives must come first. Returns (auc, var(auc))."""
    m = label_1_count
    n = len(predictions_sorted) - m
    positive_examples = predictions_sorted[:m]
    negative_examples = predictions_sorted[m:]

    tx = _compute_midrank(positive_examples)
    ty = _compute_midrank(negative_examples)
    tz = _compute_midrank(predictions_sorted)

    aucs = (tz[:m].sum() / (m * n)) - (m + 1.0) / (2.0 * n)

    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m

    sx = np.var(v01, ddof=1) if m > 1 else 0.0
    sy = np.var(v10, ddof=1) if n > 1 else 0.0
    auc_var = sx / m + sy / n
    return float(aucs), float(auc_var)


def delong_auc_ci(y_true, y_score, alpha: float = 0.05):
    """DeLong analytic CI for ROC-AUC. Returns (auc, ci_low, ci_high); NaN bounds if a class has <2 samples."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    pos_mask = y_true == 1
    n_pos = int(pos_mask.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos < 2 or n_neg < 2:
        return float("nan"), float("nan"), float("nan")

    # positives first, negatives last
    order = np.concatenate([np.where(pos_mask)[0], np.where(~pos_mask)[0]])
    auc, auc_var = _fast_delong(y_score[order], n_pos)

    if auc_var <= 0 or not np.isfinite(auc_var):
        return float(auc), float("nan"), float("nan")

    # logit-symmetric CI (as pROC) keeps bounds within [0, 1]
    eps = 1e-12
    auc_clip = min(max(auc, eps), 1 - eps)
    logit = np.log(auc_clip / (1 - auc_clip))
    logit_se = np.sqrt(auc_var) / (auc_clip * (1 - auc_clip))
    z = stats.norm.ppf(1 - alpha / 2.0)
    lo = 1.0 / (1.0 + np.exp(-(logit - z * logit_se)))
    hi = 1.0 / (1.0 + np.exp(-(logit + z * logit_se)))
    return float(auc), float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))
