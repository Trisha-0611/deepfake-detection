"""Fast DeLong test for two correlated ROC-AUCs (DeLong et al., 1988;
Sun and Xu, 2014). Used for every AUC comparison in Chapter 7."""
import numpy as np
from scipy.stats import norm


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_test(y_true, score_a, score_b):
    """Returns (auc_a, auc_b, two-sided p-value)."""
    y = np.asarray(y_true).astype(bool)
    preds = np.vstack([score_a, score_b])
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(p) for p in pos])
    ty = np.array([_midrank(q) for q in neg])
    tz = np.array([_midrank(np.concatenate([p, q])) for p, q in zip(pos, neg)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2 * norm.sf(abs(z)))
