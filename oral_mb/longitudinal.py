"""Within-person fixed-effects models for repeat-visit HPP data.

Assumes a `visit_id` column and stable subject id. If only one visit
available, raises a helpful error.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .stats_utils import bh_qvalues

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FELongResult:
    feature: str
    outcome: str
    beta_within: float
    se: float
    pvalue: float
    n_obs: int
    n_subj: int


def to_long(panel: pd.DataFrame, id_col: str, visit_col: str = "visit") -> pd.DataFrame:
    """Pass-through helper; expected columns: id_col, visit_col, features..."""
    if id_col not in panel.columns or visit_col not in panel.columns:
        raise ValueError(f"panel must contain {id_col!r} and {visit_col!r}")
    return panel.sort_values([id_col, visit_col]).reset_index(drop=True)


def within_person_demean(df: pd.DataFrame, id_col: str, cols: list[str]) -> pd.DataFrame:
    g = df.groupby(id_col)[cols].transform("mean")
    out = df.copy()
    out[cols] = df[cols] - g
    return out


def fixed_effects_assoc(
    panel: pd.DataFrame,
    id_col: str,
    outcomes: list[str],
    features: list[str],
    time_col: str = "visit",
    cluster: bool = True,
) -> pd.DataFrame:
    """OLS with subject fixed effects via demeaning. SEs clustered by subject."""
    if panel[id_col].duplicated().sum() == 0:
        raise ValueError(
            "panel has no repeated subjects — longitudinal model not applicable. "
            "Provide a long-format dataframe with multiple visits per subject."
        )

    rows: list[FELongResult] = []
    for outcome in outcomes:
        for feat in features:
            d = panel[[id_col, time_col, outcome, feat]].dropna()
            if d[id_col].nunique() < 30:
                continue
            d = within_person_demean(d, id_col, [outcome, feat])
            try:
                mod = smf.ols(f"{outcome} ~ {feat}", data=d)
                kwargs = {"cov_type": "cluster", "cov_kwds": {"groups": d[id_col]}} if cluster else {}
                res = mod.fit(**kwargs)
            except Exception as e:
                log.debug("fe fit failed: %s", e)
                continue
            rows.append(
                FELongResult(
                    feature=feat, outcome=outcome,
                    beta_within=float(res.params[feat]),
                    se=float(res.bse[feat]),
                    pvalue=float(res.pvalues[feat]),
                    n_obs=int(len(d)),
                    n_subj=int(d[id_col].nunique()),
                )
            )
    out = pd.DataFrame([r.__dict__ for r in rows])
    if not out.empty:
        out["q"] = bh_qvalues(out["pvalue"].values)
    return out
