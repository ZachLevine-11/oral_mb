"""Per-feature association tests w/ FDR, sex-stratified option, AUPRC for binary."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

from .config import COVARS_BASE, COVARS_FULL, FDR_ALPHA
from .data import Cohort, outcome_meta, transform_features
from .stats_utils import bh_qvalues

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssocResult:
    feature: str
    outcome: str
    beta: float
    se: float
    pvalue: float
    n: int


def _fit_one(
    y: pd.Series, x: pd.Series, covars: pd.DataFrame, kind: str
) -> AssocResult | None:
    df = pd.concat([y.rename("y"), x.rename("x"), covars], axis=1).dropna()
    if df["y"].nunique() < 2 or len(df) < 30:
        return None
    X = sm.add_constant(df[["x", *covars.columns]], has_constant="add")
    try:
        if kind == "binary":
            res = sm.Logit(df["y"].astype(int), X).fit(disp=0, method="lbfgs", maxiter=200)
        else:
            res = sm.OLS(df["y"].astype(float), X).fit()
    except Exception as e:  # convergence / singular matrix
        log.debug("fit failed feature=%s outcome=%s: %s", x.name, y.name, e)
        return None
    return AssocResult(
        feature=str(x.name),
        outcome=str(y.name),
        beta=float(res.params["x"]),
        se=float(res.bse["x"]),
        pvalue=float(res.pvalues["x"]),
        n=int(len(df)),
    )


def associations(
    cohort: Cohort,
    feature_kind: str = "clr",
    use_bmi: bool = False,
    sex_stratum: str | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Test every feature x every outcome. Return long table with q-values."""
    feats = transform_features(cohort.microbiome, kind=feature_kind)
    cov_cols = list(COVARS_FULL if use_bmi else COVARS_BASE)
    cov = cohort.covariates[cov_cols].copy()

    if sex_stratum is not None:
        mask = cohort.covariates["gender"] == sex_stratum
        feats = feats.loc[mask]
        cov = cov.drop(columns=["gender"]).loc[mask]
        cohort = Cohort(
            microbiome=feats,
            symptoms=cohort.symptoms.loc[mask],
            lifestyle=cohort.lifestyle.loc[mask],
            covariates=cov,
        )

    meta = outcome_meta()
    outcomes = cohort.symptoms.join(cohort.lifestyle, how="outer")

    jobs = [
        (outcomes[o], feats[f], cov, meta.get(o, "continuous"))
        for o in outcomes.columns
        for f in feats.columns
    ]
    log.info("associations: %d tests on %d jobs", len(jobs), n_jobs)
    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_fit_one)(y, x, c, k) for y, x, c, k in jobs
    )
    df = pd.DataFrame([r.__dict__ for r in rows if r is not None])
    if df.empty:
        return df

    # FDR within outcome (primary; stricter — corrects across features).
    df["q_within_outcome"] = (
        df.groupby("outcome")["pvalue"].transform(lambda p: bh_qvalues(p.values))
    )
    # FDR within feature (Nastya-style — corrects across outcomes for each feature).
    df["q_per_feature"] = (
        df.groupby("feature")["pvalue"].transform(lambda p: bh_qvalues(p.values))
    )
    # Joint FDR across all tests.
    df["q_global"] = bh_qvalues(df["pvalue"].values)
    df["significant"] = df["q_within_outcome"] < FDR_ALPHA
    return df.sort_values(["outcome", "pvalue"]).reset_index(drop=True)


def shared_signature(
    sym_assoc: pd.DataFrame, life_assoc: pd.DataFrame, alpha: float = FDR_ALPHA
) -> pd.DataFrame:
    """Features significant in >= 1 symptom AND >= 1 lifestyle outcome."""
    s_sig = set(sym_assoc.loc[sym_assoc["q_within_outcome"] < alpha, "feature"])
    l_sig = set(life_assoc.loc[life_assoc["q_within_outcome"] < alpha, "feature"])
    shared = sorted(s_sig & l_sig)
    return pd.DataFrame({"feature": shared})
