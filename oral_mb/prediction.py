"""Predictive modeling — leakage-safe with explicit train/eval/test splits.

Splits
------
- **test**: held-out, fixed-seed fraction. Never touched until the final scoring
  of a fully refit model. Not used for any imputation, scaling, feature
  selection, prevalence filtering, augmentation, or hyperparameter choice.
- **dev** (= remainder): used for K-fold CV. Each fold inside dev splits into
  **train** (fit imputers/scalers/prevalence filters/models) and **eval**
  (= validation; model scored, never fit on).
- Final test scoring: imputer/scaler/prevalence filter refit on ALL of dev
  only, model refit on dev, then applied once to test.

All preprocessing (imputer, optional CLR/std scaler, prevalence filter) is fit
on train (per fold) or on dev (for the final test prediction). Test rows are
held out at index level before any fitting.

Models
------
- ``lgbm``   — LightGBM, Nastya-parity hyperparams.
- ``ridge``  — RidgeClassifier (binary, with calibrated decision_function)
               / Ridge (continuous).
- ``ols``    — LogisticRegression(penalty=None) / LinearRegression.
- ``ensemble`` — mean of the above models' predictions (probabilities for
                 binary, raw for continuous).

Feature sources
---------------
- Oral microbiome (from ``cohort``).
- Optional gut microbiome via ``LabData.DataLoaders.GutMBLoader`` — configurable
  filtering (study_ids, prevalence) applied within the leakage-safe pipeline,
  i.e. prevalence on train only.
- Optional gut+oral ensemble: separate models per source, predictions averaged.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .config import COVARS_BASE, COVARS_FULL, SEED
from .data import Cohort, outcome_meta

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------
@dataclass
class FoldPrediction:
    y_true: np.ndarray
    y_pred: np.ndarray


@dataclass
class CVResult:
    """Back-compat container. ``folds`` are eval-fold predictions on dev."""

    outcome: str
    kind: str  # "binary" | "continuous"
    model: str
    metrics: dict[str, float] = field(default_factory=dict)
    folds: list[FoldPrediction] = field(default_factory=list)
    test: FoldPrediction | None = None  # held-out test predictions


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitConfig:
    test_frac: float = 0.2
    n_folds: int = 5
    seed: int = SEED


@dataclass(frozen=True)
class PredictionConfig:
    """Knobs for the leakage-safe pipeline."""

    models: tuple[str, ...] = ("lgbm", "ridge", "ols")
    ensemble: bool = True
    use_bmi: bool = False
    transform: str = "log10"            # oral microbiome treatment
    prevalence_min: float = 0.0         # 0 disables; applied on TRAIN only
    standardize: bool = False           # fit StandardScaler on TRAIN only
    split: SplitConfig = field(default_factory=SplitConfig)
    num_threads: int = 8

    # Gut microbiome ablation/ensemble
    use_gut: bool = False               # also train gut-only model
    gut_loader_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "df": "mpa_species",          # mpa_species | mpa_genus | mpa_family
            "study_ids": [10],
            "min_col_present_frac": 0.2,
            "groupby_reg": "first",
        }
    )
    gut_prevalence_min: float = 0.0


# -----------------------------------------------------------------------------
# Preprocessing (fit-on-train-only)
# -----------------------------------------------------------------------------
class _Preprocessor:
    """Imputer + optional prevalence filter + optional standardizer.

    Fit on a training matrix, then applied to held-out rows. No state leaks.
    """

    def __init__(self, prevalence_min: float = 0.0, standardize: bool = False,
                 micro_cols: list[str] | None = None, absent_log10: float = -4.0):
        self.prevalence_min = prevalence_min
        self.standardize = standardize
        self.micro_cols = micro_cols or []
        self.absent_log10 = absent_log10
        self.keep_cols_: list[str] | None = None
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None

    def fit(self, X_train: pd.DataFrame) -> "_Preprocessor":
        if self.prevalence_min > 0.0 and self.micro_cols:
            micro = X_train[self.micro_cols]
            present = (micro > self.absent_log10).mean(axis=0)
            keep_micro = present[present >= self.prevalence_min].index.tolist()
            other = [c for c in X_train.columns if c not in self.micro_cols]
            self.keep_cols_ = keep_micro + other
        else:
            self.keep_cols_ = list(X_train.columns)
        Xt = X_train[self.keep_cols_]
        self.imputer_ = SimpleImputer(strategy="mean").fit(Xt)
        if self.standardize:
            self.scaler_ = StandardScaler().fit(self.imputer_.transform(Xt))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.keep_cols_ is None or self.imputer_ is None:
            raise RuntimeError("_Preprocessor not fit")
        arr = self.imputer_.transform(X[self.keep_cols_])
        if self.scaler_ is not None:
            arr = self.scaler_.transform(arr)
        return arr


# -----------------------------------------------------------------------------
# Model factories — each returns (fit_fn, predict_fn) closure
# -----------------------------------------------------------------------------
def _fit_predict_lgbm(kind: str, num_threads: int) -> Callable:
    import lightgbm as lgb

    def go(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        if kind == "binary":
            m = lgb.LGBMClassifier(
                n_estimators=1000, learning_rate=0.002, max_depth=4,
                min_child_samples=20, feature_fraction=0.08,
                random_state=422, num_threads=num_threads, verbose=-1,
            )
            m.fit(X_tr, y_tr)
            return m.predict_proba(X_te)[:, 1]
        m = lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.002, max_depth=4,
            min_child_samples=20, feature_fraction=0.2,
            random_state=422, num_threads=num_threads, verbose=-1,
        )
        m.fit(X_tr, y_tr)
        return m.predict(X_te)

    return go


def _fit_predict_ridge(kind: str) -> Callable:
    from sklearn.linear_model import LogisticRegression, Ridge

    def go(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        if kind == "binary":
            m = LogisticRegression(penalty="l2", C=1.0, max_iter=5000,
                                   solver="lbfgs")
            m.fit(X_tr, y_tr)
            return m.predict_proba(X_te)[:, 1]
        m = Ridge(alpha=1.0)
        m.fit(X_tr, y_tr)
        return m.predict(X_te)

    return go


def _fit_predict_ols(kind: str) -> Callable:
    from sklearn.linear_model import LinearRegression, LogisticRegression

    def go(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        if kind == "binary":
            m = LogisticRegression(penalty=None, max_iter=5000, solver="lbfgs")
            m.fit(X_tr, y_tr)
            return m.predict_proba(X_te)[:, 1]
        m = LinearRegression()
        m.fit(X_tr, y_tr)
        return m.predict(X_te)

    return go


_MODEL_FACTORIES: dict[str, Callable[[str, int], Callable]] = {
    "lgbm":  lambda kind, nt: _fit_predict_lgbm(kind, nt),
    "ridge": lambda kind, nt: _fit_predict_ridge(kind),
    "ols":   lambda kind, nt: _fit_predict_ols(kind),
}


# -----------------------------------------------------------------------------
# Feature builders
# -----------------------------------------------------------------------------
def _oral_features(cohort: Cohort, use_bmi: bool) -> tuple[pd.DataFrame, list[str]]:
    """Oral microbiome (log10 as stored) joined with covariates. No leakage:
    join is purely index-aligned, no per-column statistics computed here."""
    mb = cohort.microbiome
    cov = cohort.covariates[list(COVARS_FULL if use_bmi else COVARS_BASE)]
    X = mb.join(cov)
    return X, list(mb.columns)


def load_gut_microbiome(
    df: str = "mpa_species",
    study_ids: list[int] | None = None,
    min_col_present_frac: float = 0.2,
    groupby_reg: str = "first",
    **loader_kwargs: Any,
) -> pd.DataFrame:
    """Load gut microbiome via LabData GutMBLoader (Nastya recipe).

        gml = GutMBLoader().get_data(df='mpa_species', study_ids=[10],
                                     min_col_present_frac=0.2,
                                     groupby_reg='first')
        gmdf = gml.df.join(gml.df_metadata[['RegistrationCode']])\
                     .set_index('RegistrationCode')

    ``df`` selects the taxonomic level: ``mpa_species`` | ``mpa_genus`` |
    ``mpa_family``. Lazy import so non-cluster runs don't break.
    """
    from LabData.DataLoaders.GutMBLoader import GutMBLoader  # type: ignore

    sids = study_ids if study_ids is not None else [10]
    gml = GutMBLoader().get_data(
        df=df, study_ids=sids,
        min_col_present_frac=min_col_present_frac,
        groupby_reg=groupby_reg,
        **loader_kwargs,
    )
    gmdf = gml.df.join(gml.df_metadata[["RegistrationCode"]]).set_index("RegistrationCode")
    gmdf = gmdf.loc[~gmdf.index.duplicated(keep="first")]
    gmdf.index.name = "RegistrationCode"
    return gmdf


def _gut_features(cohort: Cohort, cfg: PredictionConfig) -> tuple[pd.DataFrame, list[str]] | None:
    # Bulletproof: override Nastya's global min_col_present_frac=0.2 to 0 so
    # the column-presence filter does not peek at held-out test samples. The
    # 0.2 threshold is then re-applied per fold on TRAIN ONLY via the
    # _Preprocessor (gut_prevalence_min).
    gut_kwargs = dict(cfg.gut_loader_kwargs)
    user_frac = gut_kwargs.get("min_col_present_frac", 0.2)
    gut_kwargs["min_col_present_frac"] = 0.0
    try:
        gut = load_gut_microbiome(**gut_kwargs)
    except Exception as e:  # ImportError on non-cluster, or empty data
        log.warning("gut microbiome unavailable: %s", e)
        return None
    # Hoist the user's intended prevalence floor into the leakage-safe path.
    if cfg.gut_prevalence_min == 0.0 and user_frac > 0.0:
        log.info("gut: applying min_col_present_frac=%.2f as TRAIN-ONLY "
                 "prevalence filter (leakage-safe)", user_frac)
        object.__setattr__(cfg, "gut_prevalence_min", user_frac)
    cov = cohort.covariates[list(COVARS_FULL if cfg.use_bmi else COVARS_BASE)]
    idx = gut.index.intersection(cov.index)
    if len(idx) < 20:
        log.warning("gut microbiome overlap with cohort=%d; skipping", len(idx))
        return None
    X = gut.loc[idx].join(cov.loc[idx])
    return X, list(gut.columns)


# -----------------------------------------------------------------------------
# Core leakage-safe routine — one outcome, one feature source
# -----------------------------------------------------------------------------
def _make_splits(idx: pd.Index, y: pd.Series, kind: str, split: SplitConfig
                 ) -> tuple[pd.Index, pd.Index, list[tuple[pd.Index, pd.Index]]]:
    """Returns (dev_idx, test_idx, [(train_idx, eval_idx), ...]).

    test_idx is held out before any fold splitting.
    """
    stratify = y.loc[idx] if kind == "binary" and y.loc[idx].nunique() > 1 else None
    dev_idx, test_idx = train_test_split(
        idx, test_size=split.test_frac, random_state=split.seed,
        stratify=stratify,
    )
    dev_idx = pd.Index(dev_idx); test_idx = pd.Index(test_idx)

    # No sample appears in both dev and test.
    assert dev_idx.intersection(test_idx).empty, \
        "leakage: dev ∩ test is non-empty"
    assert len(dev_idx) + len(test_idx) == len(idx), \
        "split lost or duplicated samples"

    if kind == "binary" and y.loc[dev_idx].nunique() > 1:
        kf = StratifiedKFold(n_splits=split.n_folds, shuffle=True,
                             random_state=split.seed)
        folds = [(dev_idx[tr], dev_idx[ev])
                 for tr, ev in kf.split(np.zeros(len(dev_idx)), y.loc[dev_idx].values)]
    else:
        kf = KFold(n_splits=split.n_folds, shuffle=True, random_state=split.seed)
        folds = [(dev_idx[tr], dev_idx[ev]) for tr, ev in kf.split(dev_idx)]

    # Per-fold leakage asserts: train ∩ eval = ∅; both ⊂ dev; both ∩ test = ∅.
    for tr_idx, ev_idx in folds:
        assert pd.Index(tr_idx).intersection(pd.Index(ev_idx)).empty, \
            "leakage: train ∩ eval is non-empty"
        assert pd.Index(tr_idx).intersection(test_idx).empty, \
            "leakage: train ∩ test is non-empty"
        assert pd.Index(ev_idx).intersection(test_idx).empty, \
            "leakage: eval ∩ test is non-empty"
        assert pd.Index(tr_idx).difference(dev_idx).empty, \
            "leakage: train escaped dev"
        assert pd.Index(ev_idx).difference(dev_idx).empty, \
            "leakage: eval escaped dev"
    return dev_idx, test_idx, folds


def _run_one_source(
    X: pd.DataFrame, y: pd.Series, kind: str, cfg: PredictionConfig,
    micro_cols: list[str], source_tag: str, absent_log10: float = -4.0,
) -> dict[str, CVResult]:
    """Train all configured models on one feature source, with leakage-safe
    splits. Returns dict model_name -> CVResult (incl. test preds)."""
    common = X.index.intersection(y.dropna().index)
    if len(common) < 30 or y.loc[common].nunique() < 2:
        return {}
    X = X.loc[common]; y = y.loc[common]

    # Sanity: indices are unique (else split assertions could lie).
    assert X.index.is_unique, f"{source_tag}: X has duplicate indices"
    assert y.index.is_unique, f"{source_tag}: y has duplicate indices"
    assert X.index.equals(y.index), f"{source_tag}: X and y indices misaligned"

    dev_idx, test_idx, folds = _make_splits(X.index, y, kind, cfg.split)
    test_set = frozenset(test_idx)

    def _fit_check(fit_idx: pd.Index, ctx: str) -> None:
        """Hard guarantee: no test row ever enters a .fit() call."""
        bad = test_set.intersection(fit_idx)
        if bad:
            raise AssertionError(
                f"LEAKAGE [{source_tag}/{ctx}]: {len(bad)} test rows would be fit on"
            )

    model_names = list(cfg.models)
    fold_preds: dict[str, list[FoldPrediction]] = {m: [] for m in model_names}
    test_preds: dict[str, np.ndarray] = {}

    # ---- CV inside dev: fit preprocessor + model on train, score eval ----
    for tr_idx, ev_idx in folds:
        _fit_check(tr_idx, "fold-preprocessor")
        _fit_check(tr_idx, "fold-model")
        # Also assert eval rows do not bleed into train.
        assert pd.Index(tr_idx).intersection(pd.Index(ev_idx)).empty
        pp = _Preprocessor(prevalence_min=cfg.prevalence_min,
                           standardize=cfg.standardize,
                           micro_cols=micro_cols,
                           absent_log10=absent_log10).fit(X.loc[tr_idx])
        X_tr = pp.transform(X.loc[tr_idx])
        X_ev = pp.transform(X.loc[ev_idx])
        y_tr = y.loc[tr_idx].values
        y_ev = y.loc[ev_idx].values
        for name in model_names:
            fit_pred = _MODEL_FACTORIES[name](kind, cfg.num_threads)
            yhat = fit_pred(X_tr, y_tr, X_ev)
            fold_preds[name].append(FoldPrediction(y_true=y_ev, y_pred=np.asarray(yhat)))

    # ---- Final refit on ALL of dev, scored once on held-out test ----
    _fit_check(dev_idx, "final-preprocessor")
    _fit_check(dev_idx, "final-model")
    pp_final = _Preprocessor(prevalence_min=cfg.prevalence_min,
                             standardize=cfg.standardize,
                             micro_cols=micro_cols,
                             absent_log10=absent_log10).fit(X.loc[dev_idx])
    X_dev = pp_final.transform(X.loc[dev_idx])
    X_te = pp_final.transform(X.loc[test_idx])
    y_dev = y.loc[dev_idx].values
    y_te = y.loc[test_idx].values
    for name in model_names:
        fit_pred = _MODEL_FACTORIES[name](kind, cfg.num_threads)
        test_preds[name] = np.asarray(fit_pred(X_dev, y_dev, X_te))

    results: dict[str, CVResult] = {}
    for name in model_names:
        res = _summarize(f"{source_tag}", kind, name, fold_preds[name])
        res.test = FoldPrediction(y_true=y_te, y_pred=test_preds[name])
        res.metrics.update(_test_metrics(y_te, test_preds[name], kind))
        results[name] = res

    if cfg.ensemble and len(model_names) > 1:
        ens_folds = _ensemble_folds([fold_preds[m] for m in model_names])
        ens_test_pred = np.mean(np.stack([test_preds[m] for m in model_names]), axis=0)
        res = _summarize(source_tag, kind, "ensemble", ens_folds)
        res.test = FoldPrediction(y_true=y_te, y_pred=ens_test_pred)
        res.metrics.update(_test_metrics(y_te, ens_test_pred, kind))
        results["ensemble"] = res

    return results


def _ensemble_folds(per_model_folds: list[list[FoldPrediction]]) -> list[FoldPrediction]:
    """Average per-fold predictions across models (folds aligned by order)."""
    out: list[FoldPrediction] = []
    n = len(per_model_folds[0])
    for i in range(n):
        ys = per_model_folds[0][i].y_true
        ps = np.mean(np.stack([per_model_folds[m][i].y_pred
                               for m in range(len(per_model_folds))]), axis=0)
        out.append(FoldPrediction(y_true=ys, y_pred=ps))
    return out


def _test_metrics(y_true: np.ndarray, y_pred: np.ndarray, kind: str) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score
    from scipy.stats import pearsonr
    if kind == "binary":
        if len(np.unique(y_true)) < 2:
            return {}
        return {"auc_test": float(roc_auc_score(y_true, y_pred))}
    return {"pearson_test": float(pearsonr(y_true, y_pred)[0])}


def _summarize(outcome: str, kind: str, model: str, folds: list[FoldPrediction]
               ) -> CVResult:
    from sklearn.metrics import roc_auc_score
    from scipy.stats import pearsonr
    if not folds:
        return CVResult(outcome=outcome, kind=kind, model=model, metrics={}, folds=[])
    if kind == "binary":
        scores = [float(roc_auc_score(f.y_true, f.y_pred)) for f in folds
                  if len(np.unique(f.y_true)) >= 2]
        y_true_all = np.concatenate([f.y_true for f in folds])
        m = dict(auc=float(np.mean(scores)) if scores else float("nan"),
                 auc_std=float(np.std(scores)) if scores else float("nan"),
                 prevalence=float(y_true_all.mean()))
    else:
        scores = [float(pearsonr(f.y_true, f.y_pred)[0]) for f in folds]
        m = dict(pearson=float(np.mean(scores)), pearson_std=float(np.std(scores)))
    return CVResult(outcome=outcome, kind=kind, model=model, metrics=m, folds=folds)


# -----------------------------------------------------------------------------
# Public per-outcome entry
# -----------------------------------------------------------------------------
def predict_outcome(
    cohort: Cohort, outcome: str, cfg: PredictionConfig | None = None,
) -> dict[str, CVResult]:
    """Run all configured models on oral (and optionally gut, and oral+gut
    ensemble) features for one outcome. Leakage-safe by construction.

    Returns keyed by ``"{source}/{model}"`` e.g. ``"oral/lgbm"``,
    ``"oral/ensemble"``, ``"gut/ridge"``, ``"oral+gut/ensemble"``.
    """
    cfg = cfg or PredictionConfig()
    meta = outcome_meta()
    if outcome not in meta:
        raise ValueError(f"unknown outcome {outcome!r}")
    kind = meta[outcome]
    y = cohort.symptoms.join(cohort.lifestyle, how="outer")[outcome]

    out: dict[str, CVResult] = {}

    X_oral, oral_cols = _oral_features(cohort, cfg.use_bmi)
    oral_res = _run_one_source(X_oral, y, kind, cfg, oral_cols, source_tag="oral")
    for k, v in oral_res.items():
        out[f"oral/{k}"] = v
        v.outcome = outcome

    if cfg.use_gut:
        gut_pack = _gut_features(cohort, cfg)
        if gut_pack is not None:
            X_gut, gut_cols = gut_pack
            gut_cfg = PredictionConfig(**{**cfg.__dict__,
                                          "prevalence_min": cfg.gut_prevalence_min})
            # Joint oral+gut source: one model on concatenated features,
            # restricted to the sample intersection. Replaces the old
            # prediction-averaging block below for the use_gut path.
            idx_joint = X_oral.index.intersection(X_gut.index)
            if len(idx_joint) >= 20:
                # Prefix gut columns to avoid name collisions with oral (both
                # use s__/g__/f__ prefixes that overlap across body sites).
                gut_renamed = {c: f"gut__{c}" for c in gut_cols}
                gut_sub = (X_gut.loc[idx_joint, gut_cols]
                                .rename(columns=gut_renamed))
                X_joint = X_oral.loc[idx_joint].join(gut_sub)
                joint_cols = list(oral_cols) + list(gut_renamed.values())
                joint_cfg = PredictionConfig(**{**cfg.__dict__,
                                                "prevalence_min": 0.0})
                joint_res = _run_one_source(
                    X_joint, y, kind, joint_cfg, joint_cols,
                    source_tag="oral+gut", absent_log10=-4.0,
                )
                for k, v in joint_res.items():
                    out[f"oral+gut/{k}"] = v
                    v.outcome = outcome
            # Gut data is not log10; treat values ≤0 as absent for the
            # train-only prevalence filter. This is what makes the filter
            # leakage-safe (Nastya's global min_col_present_frac uses all rows
            # including held-out test).
            gut_res = _run_one_source(X_gut, y, kind, gut_cfg, gut_cols,
                                      source_tag="gut", absent_log10=0.0)
            for k, v in gut_res.items():
                out[f"gut/{k}"] = v
                v.outcome = outcome

    return out


# -----------------------------------------------------------------------------
# Back-compat thin wrappers — keep the old API working
# -----------------------------------------------------------------------------
def cv_predict_one(
    cohort: Cohort, outcome: str, model: str = "lgbm",
    transform: str = "log10", n_folds: int | None = None,
    use_bmi: bool = False, seed: int = SEED, n_boot: int = 0,
    num_threads: int = 100,
) -> CVResult:
    """Back-compat. Runs the leakage-safe pipeline for a single model and
    returns the CV result for the oral feature source."""
    if model == "lgbm_legacy":
        raise ValueError("legacy model removed")
    cfg = PredictionConfig(
        models=(model,), ensemble=False, use_bmi=use_bmi, transform=transform,
        split=SplitConfig(n_folds=n_folds or 5, seed=seed),
        num_threads=num_threads,
    )
    res = predict_outcome(cohort, outcome, cfg)
    return res.get(f"oral/{model}") or CVResult(outcome=outcome,
                                                kind=outcome_meta()[outcome],
                                                model=model)


def _safe_predict_one(cohort: Cohort, outcome: str, cfg: PredictionConfig
                      ) -> list[dict[str, Any]] | None:
    try:
        results = predict_outcome(cohort, outcome, cfg)
        if not results:
            log.info("[pred] %-30s skipped (too few samples / single class)", outcome)
            return None
        rows = []
        for tag, r in results.items():
            source, model = tag.split("/", 1)
            row = dict(outcome=outcome, kind=r.kind, source=source, model=model,
                       **r.metrics)
            rows.append(row)
            if r.kind == "binary":
                log.info("[pred] %-26s %-10s %-8s cv_auc=%.3f±%.3f test_auc=%.3f prev=%.3f",
                         outcome, source, model,
                         r.metrics.get("auc", float("nan")),
                         r.metrics.get("auc_std", float("nan")),
                         r.metrics.get("auc_test", float("nan")),
                         r.metrics.get("prevalence", float("nan")))
            else:
                log.info("[pred] %-26s %-10s %-8s cv_r=%.3f±%.3f  test_r=%.3f",
                         outcome, source, model,
                         r.metrics.get("pearson", float("nan")),
                         r.metrics.get("pearson_std", float("nan")),
                         r.metrics.get("pearson_test", float("nan")))
        return rows
    except Exception as e:  # noqa: BLE001
        log.warning("skip %s: %s", outcome, e)
        return None


def run_all_outcomes(
    cohort: Cohort, model: str = "lgbm", transform: str = "log10",
    n_jobs: int | None = None, cfg: PredictionConfig | None = None,
    **kw: Any,
) -> pd.DataFrame:
    """Parallel over outcomes. Pass ``cfg`` to control models/ensemble/gut.

    Back-compat: if ``cfg`` is None, builds a single-model config from kwargs.
    """
    if cfg is None:
        cfg = PredictionConfig(
            models=(model,) if model != "all" else ("lgbm", "ridge", "ols"),
            ensemble=(model == "all"),
            use_bmi=kw.get("use_bmi", False),
            transform=transform,
            num_threads=kw.get("num_threads", 8),
        )
    meta = outcome_meta()
    total_cores = os.cpu_count() or 1
    if n_jobs is None:
        n_jobs = min(len(meta), max(1, total_cores // 8))
    log.info("prediction: %d outcomes on %d workers (cores=%d)",
             len(meta), n_jobs, total_cores)
    nested = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_safe_predict_one)(cohort, o, cfg) for o in meta
    )
    rows: list[dict[str, Any]] = []
    for r in nested:
        if r:
            rows.extend(r)
    return pd.DataFrame(rows)
