"""Predict medical-condition diagnoses (ICD-11) from oral / gut microbiome.

Question: can the (cheap) oral swab predict disease as well as the (expensive)
gut sample? Sources compared per disease: ``oral``, ``gut``, ``oral+gut``,
plus a covariates-only baseline (age/gender) so any microbiome gain is visible.

Diagnoses come from ``LabData.DataLoaders.MedicalConditionLoader`` (cluster
only). GI/digestive conditions are ICD-11 chapter 13 (codes starting ``DA``..
``DE``); they are flagged, not filtered, so the full diagnosis panel is run.

Leakage safety is inherited from :mod:`prediction` (``_run_one_source``):
test rows are held out before any fit, preprocessing is refit per fold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import COVARS_BASE, COVARS_FULL
from .data import Cohort
from .prediction import (
    CVResult,
    PredictionConfig,
    _gut_features,
    _oral_features,
    _run_one_source,
)

log = logging.getLogger(__name__)

# ICD-11 chapter 13 (diseases of the digestive system).
GI_PREFIXES: tuple[str, ...] = ("DA", "DB", "DC", "DD", "DE")

# Codes that are noise rather than disease signal (see RNA_fm/downstream.py).
EXCLUDE_CODES: frozenset[str] = frozenset({"6A05", "RA01"})

DEFAULT_STUDY_IDS: tuple[int, ...] = tuple(range(100)) + tuple(range(1000, 1011))


def is_gi(code: str) -> bool:
    return any(code.startswith(p) for p in GI_PREFIXES)


@dataclass(frozen=True)
class DiseaseConfig:
    min_prevalence: float = 0.02      # fraction of cohort with the diagnosis
    min_cases: int = 50               # absolute floor for a trainable outcome
    only_baseline: bool = False       # first visit per person only
    gi_only: bool = False
    study_ids: tuple[int, ...] = DEFAULT_STUDY_IDS


def load_disease_matrix(
    index: pd.Index, cfg: DiseaseConfig | None = None
) -> pd.DataFrame:
    """People x diagnosis 0/1 matrix restricted to ``index``.

    Lazy LabData import so non-cluster runs fail loudly but only here.
    """
    from LabData.DataLoaders.MedicalConditionLoader import MedicalConditionLoader  # type: ignore

    cfg = cfg or DiseaseConfig()
    b = MedicalConditionLoader().get_data(study_ids=list(cfg.study_ids)).df.reset_index()
    if "Date" in b.columns:
        b = b.sort_values(by="Date")
    if cfg.only_baseline:
        b = b.loc[~b["RegistrationCode"].duplicated(keep="first"), :]

    people = pd.Index(sorted(set(index)))
    b = b.loc[b["RegistrationCode"].isin(people), :]
    if b.empty:
        raise ValueError("no medical conditions overlap the cohort index")

    counts = b.drop_duplicates(["RegistrationCode", "medical_condition"])[
        "medical_condition"
    ].value_counts()
    n = len(people)
    keep = [
        c for c, k in counts.items()
        if (k / n) >= cfg.min_prevalence
        and k >= cfg.min_cases
        and c not in EXCLUDE_CODES
        and not str(c).startswith("Block")
    ]
    if cfg.gi_only:
        keep = [c for c in keep if is_gi(c)]
    log.info("diseases: %d codes pass prevalence>=%.3f & cases>=%d (n=%d people)",
             len(keep), cfg.min_prevalence, cfg.min_cases, n)

    res = pd.DataFrame(0, index=people, columns=keep, dtype=int)
    res.index.name = "RegistrationCode"
    for code in keep:
        cases = b.loc[b["medical_condition"] == code, "RegistrationCode"].unique()
        res.loc[res.index.isin(cases), code] = 1
    return res


def _covariate_features(cohort: Cohort, use_bmi: bool) -> tuple[pd.DataFrame, list[str]]:
    cols = list(COVARS_FULL if use_bmi else COVARS_BASE)
    return cohort.covariates[cols].copy(), []


def predict_disease(
    cohort: Cohort,
    y: pd.Series,
    cfg: PredictionConfig | None = None,
    sources: tuple[str, ...] = ("covars", "oral", "gut", "oral+gut"),
    gut_pack: tuple[pd.DataFrame, list[str]] | None = None,
) -> dict[str, CVResult]:
    """One binary diagnosis, all requested feature sources."""
    cfg = cfg or PredictionConfig(use_gut=True)
    y = y.dropna().astype(float)
    out: dict[str, CVResult] = {}

    X_oral, oral_cols = _oral_features(cohort, cfg.use_bmi)

    if "covars" in sources:
        X_cov, _ = _covariate_features(cohort, cfg.use_bmi)
        for k, v in _run_one_source(X_cov, y, "binary", cfg, [], "covars").items():
            out[f"covars/{k}"] = v

    if "oral" in sources:
        for k, v in _run_one_source(X_oral, y, "binary", cfg, oral_cols, "oral").items():
            out[f"oral/{k}"] = v

    if gut_pack is not None and ("gut" in sources or "oral+gut" in sources):
        X_gut, gut_cols = gut_pack
        if "gut" in sources:
            gut_cfg = PredictionConfig(**{**cfg.__dict__,
                                          "prevalence_min": cfg.gut_prevalence_min})
            for k, v in _run_one_source(X_gut, y, "binary", gut_cfg, gut_cols,
                                        "gut", absent_log10=0.0).items():
                out[f"gut/{k}"] = v
        if "oral+gut" in sources:
            idx = X_oral.index.intersection(X_gut.index)
            if len(idx) >= 30:
                ren = {c: f"gut__{c}" for c in gut_cols}
                X_joint = X_oral.loc[idx].join(X_gut.loc[idx, gut_cols].rename(columns=ren))
                joint_cfg = PredictionConfig(**{**cfg.__dict__, "prevalence_min": 0.0})
                for k, v in _run_one_source(X_joint, y, "binary", joint_cfg,
                                            list(oral_cols) + list(ren.values()),
                                            "oral+gut").items():
                    out[f"oral+gut/{k}"] = v
    return out


def _one(cohort: Cohort, code: str, y: pd.Series, cfg: PredictionConfig,
         sources: tuple[str, ...], gut_pack) -> list[dict[str, Any]]:
    try:
        res = predict_disease(cohort, y, cfg, sources, gut_pack)
    except Exception as e:  # noqa: BLE001
        log.warning("skip %s: %s", code, e)
        return []
    rows = []
    for tag, r in res.items():
        source, model = tag.split("/", 1)
        rows.append(dict(disease=code, gi=is_gi(code), source=source, model=model,
                         n_cases=int(y.sum()), n=int(y.notna().sum()), **r.metrics))
        log.info("[dis] %-8s %-9s %-8s cv_auc=%.3f±%.3f test_auc=%.3f prev=%.3f",
                 code, source, model, r.metrics.get("auc", np.nan),
                 r.metrics.get("auc_std", np.nan), r.metrics.get("auc_test", np.nan),
                 r.metrics.get("prevalence", np.nan))
    return rows


def run_all_diseases(
    cohort: Cohort,
    cfg: PredictionConfig | None = None,
    dcfg: DiseaseConfig | None = None,
    sources: tuple[str, ...] = ("covars", "oral", "gut", "oral+gut"),
    n_jobs: int = 4,
) -> pd.DataFrame:
    """Full panel: every qualifying diagnosis x every feature source."""
    cfg = cfg or PredictionConfig(models=("lgbm", "ridge"), ensemble=True, use_gut=True)
    dcfg = dcfg or DiseaseConfig()
    dis = load_disease_matrix(cohort.samples, dcfg)

    gut_pack = None
    if cfg.use_gut and ("gut" in sources or "oral+gut" in sources):
        gut_pack = _gut_features(cohort, cfg)
        if gut_pack is None:
            log.warning("gut features unavailable; running oral/covars only")

    nested = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one)(cohort, c, dis[c].astype(float), cfg, sources, gut_pack)
        for c in dis.columns
    )
    rows = [r for chunk in nested for r in chunk]
    return pd.DataFrame(rows)
