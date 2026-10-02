"""Statistical helpers: q-values, bootstrap CIs, calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def storey_qvalues(pvals: np.ndarray, lam: float = 0.5) -> np.ndarray:
    """Storey q-values (stable for many tests)."""
    p = np.asarray(pvals, dtype=float)
    pi0 = max(min((p > lam).mean() / (1 - lam), 1.0), 1e-8)
    return np.clip(bh_qvalues(p) * pi0, 0, 1)


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "auc",
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return point estimate + 95% CI via stratified bootstrap."""
    rng = rng or np.random.default_rng(0)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fn = {
        "auc": roc_auc_score,
        "auprc": average_precision_score,
        "brier": brier_score_loss,
        "pearson": lambda a, b: float(np.corrcoef(a, b)[0, 1]),
    }[metric]
    point = float(fn(y_true, y_pred))
    boots = np.empty(n_boot)
    n = len(y_true)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boots[i] = fn(y_true[idx], y_pred[idx])
        except ValueError:
            boots[i] = np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frac, mean = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"bin_mean_pred": mean, "fraction_pos": frac})


def expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 15
) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_pred, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        ece += abs(y_pred[mask].mean() - y_true[mask].mean()) * mask.mean()
    return float(ece)
