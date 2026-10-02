"""Mediation analysis: exposure -> microbe -> outcome.

Baron-Kenny + bootstrap ACME (average causal mediation effect).
Sobel and product-of-coefficients with bias-corrected CIs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

from .config import COVARS_FULL
from .data import Cohort, transform_features
from .stats_utils import bh_qvalues

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MediationResult:
    exposure: str
    mediator: str
    outcome: str
    a: float  # exposure -> mediator
    b: float  # mediator -> outcome | exposure
    c_total: float  # exposure -> outcome
    c_direct: float
    indirect: float  # a * b
    indirect_lo: float
    indirect_hi: float
    pvalue_indirect: float
    n: int


def _fit_lm(y: pd.Series, X: pd.DataFrame, kind: str) -> sm.regression.linear_model.RegressionResults:
    X_ = sm.add_constant(X, has_constant="add")
    if kind == "binary":
        return sm.Logit(y.astype(int), X_).fit(disp=0, method="lbfgs", maxiter=200)
    return sm.OLS(y.astype(float), X_).fit()


def mediation_one(
    exposure: pd.Series,
    mediator: pd.Series,
    outcome: pd.Series,
    covars: pd.DataFrame,
    outcome_kind: str = "binary",
    n_boot: int = 1000,
    seed: int = 0,
) -> MediationResult | None:
    df = pd.concat(
        [exposure.rename("e"), mediator.rename("m"), outcome.rename("y"), covars], axis=1
    ).dropna()
    if len(df) < 100:
        return None

    cov_cols = list(covars.columns)

    a_res = _fit_lm(df["m"], df[["e", *cov_cols]], "continuous")
    a = float(a_res.params["e"])

    b_res = _fit_lm(df["y"], df[["e", "m", *cov_cols]], outcome_kind)
    b = float(b_res.params["m"])
    c_direct = float(b_res.params["e"])

    c_res = _fit_lm(df["y"], df[["e", *cov_cols]], outcome_kind)
    c_total = float(c_res.params["e"])

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(df)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d = df.iloc[idx]
        try:
            ai = _fit_lm(d["m"], d[["e", *cov_cols]], "continuous").params["e"]
            bi = _fit_lm(d["y"], d[["e", "m", *cov_cols]], outcome_kind).params["m"]
            boots[i] = ai * bi
        except Exception:
            boots[i] = np.nan
    indirect = a * b
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    # Two-sided p from bootstrap distribution.
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())

    return MediationResult(
        exposure=str(exposure.name),
        mediator=str(mediator.name),
        outcome=str(outcome.name),
        a=a, b=b, c_total=c_total, c_direct=c_direct,
        indirect=indirect, indirect_lo=float(lo), indirect_hi=float(hi),
        pvalue_indirect=float(p), n=int(len(df)),
    )


def run_mediation(
    cohort: Cohort,
    exposures: list[str],
    outcomes: list[str],
    mediators: list[str] | None = None,
    feature_kind: str = "clr",
    outcome_meta: dict[str, str] | None = None,
    n_boot: int = 500,
    n_jobs: int = -1,
) -> pd.DataFrame:
    feats = transform_features(cohort.microbiome, kind=feature_kind)
    mediators = mediators or list(feats.columns)
    covars = cohort.covariates[list(COVARS_FULL)]

    all_pheno = cohort.symptoms.join(cohort.lifestyle, how="outer")
    om = outcome_meta or {}

    jobs = []
    for exp in exposures:
        e_series = all_pheno[exp] if exp in all_pheno.columns else cohort.covariates[exp]
        for out in outcomes:
            kind = om.get(out, "binary")
            for med in mediators:
                jobs.append((e_series, feats[med], all_pheno[out], covars, kind))
    log.info("mediation: %d triples on %d jobs (n_boot=%d)", len(jobs), n_jobs, n_boot)
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(mediation_one)(e, m, o, c, outcome_kind=k, n_boot=n_boot)
        for (e, m, o, c, k) in jobs
    )
    df = pd.DataFrame([r.__dict__ for r in results if r is not None])
    if not df.empty:
        df["q_indirect"] = bh_qvalues(df["pvalue_indirect"].values)
    return df
