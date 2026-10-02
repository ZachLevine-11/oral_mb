"""Sensitivity analyses: BMI adj, sex-stratified, hygiene proxy, batch."""
from __future__ import annotations

import logging

import pandas as pd

from .associations import associations
from .data import Cohort
from .prediction import run_all_outcomes

log = logging.getLogger(__name__)


def sensitivity_bundle(
    cohort: Cohort,
    feature_kind: str = "log10",
    predict_transform: str = "log10",
) -> dict[str, pd.DataFrame]:
    """Associations use CLR (compositional). Prediction uses raw log10 (Nastya parity)."""
    out: dict[str, pd.DataFrame] = {}

    out["assoc_base"] = associations(cohort, feature_kind, use_bmi=False)
    out["assoc_bmi"] = associations(cohort, feature_kind, use_bmi=True)
    out["assoc_male"] = associations(cohort, feature_kind, sex_stratum=1.0)
    out["assoc_female"] = associations(cohort, feature_kind, sex_stratum=0.0)

    out["pred_base"] = run_all_outcomes(cohort, model="lgbm", transform=predict_transform, use_bmi=False)
    out["pred_bmi"] = run_all_outcomes(cohort, model="lgbm", transform=predict_transform, use_bmi=True)
    return out


def compare_betas(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    m = a.merge(b, on=["feature", "outcome"], suffixes=(f"_{label_a}", f"_{label_b}"))
    m[f"delta_beta_{label_a}_vs_{label_b}"] = m[f"beta_{label_a}"] - m[f"beta_{label_b}"]
    m[f"sign_flip"] = (
        (m[f"beta_{label_a}"].apply(lambda x: x > 0))
        != (m[f"beta_{label_b}"].apply(lambda x: x > 0))
    )
    return m
