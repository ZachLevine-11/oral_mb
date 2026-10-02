"""External cohort replication harness + meta-analysis."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .associations import associations
from .data import Cohort, load_cohort
from .stats_utils import bh_qvalues

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplicationResult:
    feature: str
    outcome: str
    beta_discovery: float
    beta_replication: float
    concordant_direction: bool
    p_replication: float
    significant_replication: bool


def replicate(
    discovery_assoc: pd.DataFrame,
    replication_cohort: Cohort,
    feature_kind: str = "clr",
    use_bmi: bool = False,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Re-test discovery-significant features in replication cohort."""
    sig = discovery_assoc[discovery_assoc["q_within_outcome"] < 0.05]
    rep = associations(replication_cohort, feature_kind=feature_kind, use_bmi=use_bmi)

    merged = sig.merge(
        rep, on=["feature", "outcome"], how="left",
        suffixes=("_disc", "_rep"),
    )
    merged["concordant_direction"] = (
        np.sign(merged["beta_disc"]) == np.sign(merged["beta_rep"])
    ).fillna(False)
    merged["significant_replication"] = merged["pvalue_rep"] < alpha
    return merged


def inverse_variance_meta(
    betas: list[float], ses: list[float]
) -> tuple[float, float, float]:
    """Fixed-effects inverse-variance meta-analysis. Returns beta, se, p."""
    b = np.asarray(betas, float)
    s = np.asarray(ses, float)
    w = 1.0 / (s ** 2)
    beta_meta = float((w * b).sum() / w.sum())
    se_meta = float(np.sqrt(1.0 / w.sum()))
    z = beta_meta / se_meta
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return beta_meta, se_meta, p


def meta_analysis(assocs: list[pd.DataFrame]) -> pd.DataFrame:
    """Pool multiple cohorts' association tables by (feature, outcome)."""
    long = pd.concat(
        [a.assign(cohort=i)[["feature", "outcome", "beta", "se", "pvalue", "cohort"]]
         for i, a in enumerate(assocs)],
        ignore_index=True,
    )
    out: list[dict] = []
    for (f, o), g in long.groupby(["feature", "outcome"]):
        if len(g) < 2:
            continue
        b, s, p = inverse_variance_meta(g["beta"].tolist(), g["se"].tolist())
        i2 = _heterogeneity_i2(g["beta"].values, g["se"].values)
        out.append({"feature": f, "outcome": o, "beta_meta": b, "se_meta": s, "p_meta": p,
                    "i2": i2, "k": len(g)})
    df = pd.DataFrame(out)
    if not df.empty:
        df["q_meta"] = bh_qvalues(df["p_meta"].values)
    return df


def _heterogeneity_i2(betas: np.ndarray, ses: np.ndarray) -> float:
    w = 1.0 / ses ** 2
    bm = (w * betas).sum() / w.sum()
    Q = float((w * (betas - bm) ** 2).sum())
    k = len(betas)
    return max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0


def load_external(parquet_dir: Path, level: str = "species") -> Cohort:
    """External cohort packaged in same layout as HPP."""
    return load_cohort(level=level, data_dir=parquet_dir)
