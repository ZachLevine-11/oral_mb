"""Revision-response pipeline: 4 independent, re-runnable stages addressing
reviewer points (a) differential missingness, (c) mediation robustness,
(d) clinical anchor for self-reported bleeding gums, (e) corrected ML
reporting. Each stage reads the Cohort once and writes its own artifact
under --out, so any single stage can be re-run without redoing the others.

Usage:
    python scripts/run_revision.py --level species --out revision_out --stage all
    python scripts/run_revision.py --stage missingness
    python scripts/run_revision.py --stage mediation
    python scripts/run_revision.py --stage clinical
    python scripts/run_revision.py --stage prediction
"""
from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.special import expit

from oral_mb.config import COVARS_FULL, DATA_DIR, SYMPTOMS_BINARY
from oral_mb.data import Cohort, load_cohort, outcome_meta, stack_outcomes, transform_features
from oral_mb.mediation import _fit_lm, run_mediation
from oral_mb.prediction import CVResult, FoldPrediction, PredictionConfig, SplitConfig, predict_outcome
from oral_mb.stats_utils import bootstrap_metric

log = logging.getLogger("run_revision")

EXPOSURES = ["med_score_per_day", "UPF_score", "vegetarian_score_per_day", "smoke_tobacco_now"]
KEY_EXPOSURES_FOR_MISSINGNESS = [*EXPOSURES, "age", "bmi"]


def _n_cores() -> int:
    """Usable worker count. Honours a REVISION_CORES override (default: all
    logical cores) so the heavy stages saturate the box the reviewer asked us
    to parallelise over."""
    env = os.environ.get("REVISION_CORES")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, os.cpu_count() or 1)


MEDIATION_ESTIMAND_NOTE = (
    "cross-sectional decomposition; associational, NOT a causal effect "
    "(single time point, unverified no-unmeasured-confounding / no exposure-"
    "induced mediator-outcome confounding assumptions)"
)


def _missingness_crosstab(
    answered: pd.Series, exposures: pd.DataFrame, out: Path, headline: str
) -> None:
    """Responder vs non-responder summary of each key exposure, with a
    univariate test: Welch t for continuous exposures, chi-square for
    binary/categorical. Answers the reviewer's 'cross-tabulate answered-status
    against the key exposures' directly, alongside the joint logistic model."""
    from scipy.stats import chi2_contingency, ttest_ind

    r = answered.astype(bool)
    rows = []
    for col in exposures.columns:
        s = exposures[col]
        pair = pd.concat([answered.rename("ans"), s.rename("x")], axis=1).dropna()
        if pair.empty:
            continue
        resp = pair.loc[pair["ans"] == 1, "x"]
        nonr = pair.loc[pair["ans"] == 0, "x"]
        is_cat = (not np.issubdtype(pair["x"].dtype, np.number)) or pair["x"].nunique() <= 2
        row = {"exposure": col, "n_responder": int(len(resp)),
               "n_nonresponder": int(len(nonr))}
        if is_cat:
            ct = pd.crosstab(pair["ans"], pair["x"])
            try:
                _, p, _, _ = chi2_contingency(ct)
            except ValueError:
                p = float("nan")
            row.update(test="chi2", p_value=float(p),
                       responder_summary=f"{100*resp.astype(float).mean():.1f}% pos"
                       if resp.nunique() <= 2 else "categorical",
                       nonresponder_summary=f"{100*nonr.astype(float).mean():.1f}% pos"
                       if nonr.nunique() <= 2 else "categorical")
        else:
            try:
                p = float(ttest_ind(resp, nonr, equal_var=False, nan_policy="omit").pvalue)
            except Exception:  # noqa: BLE001
                p = float("nan")
            row.update(test="welch_t", p_value=p,
                       responder_summary=f"{resp.mean():.3f} ({resp.std():.3f})",
                       nonresponder_summary=f"{nonr.mean():.3f} ({nonr.std():.3f})")
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / f"missingness_crosstab_{headline}.csv", index=False)


def _permanova_euclidean(
    X: pd.DataFrame, labels: np.ndarray, n_perm: int = 999, seed: int = 0
) -> dict:
    """PERMANOVA (Anderson 2001) on Aitchison distance = Euclidean distance in
    CLR space. Uses the Euclidean SS identity (SS = sum of squared distances to
    the group centroid) so no O(n^2) distance matrix is materialised, and
    permutations fan out across cores. Two groups (responder vs non-responder).
    """
    Xv = np.ascontiguousarray(X.to_numpy(dtype=float))
    labels = np.asarray(labels)
    n, a = len(labels), 2
    grand = Xv.mean(axis=0)
    ss_total = float(((Xv - grand) ** 2).sum())

    def ss_within(lab: np.ndarray) -> float:
        s = 0.0
        for g in (0, 1):
            m = lab == g
            if not m.any():
                continue
            s += float(((Xv[m] - Xv[m].mean(axis=0)) ** 2).sum())
        return s

    def pseudo_f(ssw: float) -> float:
        ssb = ss_total - ssw
        denom = ssw / (n - a)
        return (ssb / (a - 1)) / denom if denom > 0 else float("nan")

    ssw_obs = ss_within(labels)
    f_obs = pseudo_f(ssw_obs)

    def _batch(batch_seed: int, k: int) -> int:
        rng = np.random.default_rng(batch_seed)
        ge = 0
        for _ in range(k):
            ge += int(pseudo_f(ss_within(rng.permutation(labels))) >= f_obs)
        return ge

    n_jobs = _n_cores()
    per = [n_perm // n_jobs + (1 if i < n_perm % n_jobs else 0) for i in range(n_jobs)]
    per = [k for k in per if k > 0]
    counts = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_batch)(seed + i + 1, k) for i, k in enumerate(per)
    )
    p_value = (sum(counts) + 1) / (n_perm + 1)
    return {
        "pseudo_F": f_obs, "p_value": float(p_value), "n_perm": n_perm,
        "R2": float((ss_total - ssw_obs) / ss_total) if ss_total > 0 else float("nan"),
        "n_responder": int((labels == 1).sum()), "n_nonresponder": int((labels == 0).sum()),
        "distance": "Aitchison (Euclidean in CLR space)",
    }


def _e_value_from_rr(rr: float) -> float:
    """VanderWeele & Ding (2017) E-value for a risk-ratio-scaled estimate."""
    if not np.isfinite(rr) or rr <= 0:
        return float("nan")
    rr = rr if rr >= 1 else 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


# --------------------------------------------------------------------------
# (a) Differential missingness on symptom outcomes (bleeding_gums headline)
# --------------------------------------------------------------------------

def stage_missingness(cohort: Cohort, out: Path, headline: str = "bleeding_gums") -> None:
    """Per-outcome answered/unanswered counts + logistic model of
    answered-status ~ exposures, to check whether missingness on the
    headline outcome is exposure-related (MAR/MNAR signal) rather than MCAR.
    """
    out.mkdir(parents=True, exist_ok=True)
    all_pheno = stack_outcomes(cohort)
    cov = cohort.covariates

    # 1. Answered vs unanswered per symptom.
    counts = []
    for sym in SYMPTOMS_BINARY:
        if sym not in all_pheno.columns:
            continue
        answered = all_pheno[sym].notna()
        counts.append({"outcome": sym, "n_answered": int(answered.sum()),
                        "n_total": int(len(answered)),
                        "frac_answered": float(answered.mean())})
    pd.DataFrame(counts).to_csv(out / "missingness_counts.csv", index=False)

    # 2. Logistic model: is answering the headline outcome exposure-related?
    y = all_pheno[headline].notna().astype(int)
    X_cols = [c for c in KEY_EXPOSURES_FOR_MISSINGNESS if c in all_pheno.columns or c in cov.columns]
    X = pd.DataFrame(index=y.index)
    for c in X_cols:
        X[c] = all_pheno[c] if c in all_pheno.columns else cov[c]
    df = pd.concat([y.rename("answered"), X], axis=1).dropna()
    X_ = sm.add_constant(df[X_cols], has_constant="add")
    model = sm.Logit(df["answered"], X_).fit(disp=0)
    model.summary2().tables[1].to_csv(out / f"missingness_model_{headline}.csv")

    # 2b. Per-exposure cross-tab of answered-status (incl. sex), with a
    # univariate test per exposure, complementing the joint logistic model.
    ct_exposures = X.copy()
    if "gender" in cov.columns and "gender" not in ct_exposures.columns:
        ct_exposures["gender"] = cov["gender"].reindex(y.index)
    _missingness_crosstab(y, ct_exposures, out, headline)

    # 3. Responder vs non-responder microbiome comparison: top-species CLR
    # mean-diff descriptor PLUS a PERMANOVA on Aitchison distance for a proper
    # p-value on "responders differ compositionally from non-responders."
    feats = transform_features(cohort.microbiome, kind="clr")
    labels = y.reindex(feats.index).fillna(0).astype(int)
    responder = labels.to_numpy().astype(bool)
    diff = feats[responder].mean() - feats[~responder].mean()
    diff.sort_values(key=np.abs, ascending=False).head(30).to_frame("clr_mean_diff").to_csv(
        out / f"missingness_microbiome_diff_{headline}.csv"
    )
    permanova = _permanova_euclidean(feats, labels.to_numpy(), n_perm=999)
    pd.Series(permanova).to_csv(out / f"missingness_permanova_{headline}.csv")
    log.info("missingness PERMANOVA %s: pseudo-F=%.3f p=%.4f R2=%.4f",
             headline, permanova["pseudo_F"], permanova["p_value"], permanova["R2"])

    # 4. IPW-adjusted re-fit of the headline association, compared to
    # unweighted, to see if top species betas move once responder-selection
    # is corrected for.
    ipw = 1.0 / model.predict(X_).clip(0.05, 0.95)
    ipw_map = pd.Series(ipw, index=df.index)
    _refit_headline_with_weights(cohort, headline, feats, ipw_map, out)

    log.info("missingness stage done -> %s", out)


def _refit_headline_with_weights(
    cohort: Cohort, headline: str, feats: pd.DataFrame, weights: pd.Series, out: Path
) -> None:
    """Weighted vs unweighted logistic species associations for the headline
    outcome; the comparison shows whether top betas are stable once
    inverse-probability-of-response weighting is applied. IPW SEs use an HC0
    sandwich (Huber-White) estimator -- naive weighted SEs understate variance
    because the weights are estimated, not true frequencies. Runs the full
    feature set (fanned out across cores), not a top-200 slice.
    """
    all_pheno = stack_outcomes(cohort)
    y = all_pheno[headline]
    cov = cohort.covariates[list(COVARS_FULL)]

    def _one(sp: str) -> dict | None:
        df = pd.concat([y.rename("y"), feats[sp].rename("m"), cov], axis=1).dropna()
        if df["y"].nunique() < 2 or len(df) < 100:
            return None
        w = weights.reindex(df.index).fillna(1.0)
        X = sm.add_constant(df[["m", *cov.columns]], has_constant="add")
        yb = df["y"].astype(int)
        try:
            uw = sm.Logit(yb, X).fit(disp=0)
            # HC0 sandwich SE for the IPW-weighted fit.
            wt = sm.GLM(yb, X, family=sm.families.Binomial(), freq_weights=w).fit(cov_type="HC0")
        except Exception:  # noqa: BLE001
            return None
        return {"species": sp,
                "beta_unweighted": float(uw.params["m"]), "se_unweighted": float(uw.bse["m"]),
                "beta_ipw": float(wt.params["m"]), "se_ipw_robust": float(wt.bse["m"]),
                "n": int(len(df))}

    rows = Parallel(n_jobs=_n_cores(), backend="loky", verbose=0)(
        delayed(_one)(sp) for sp in feats.columns
    )
    pd.DataFrame([r for r in rows if r is not None]).to_csv(
        out / f"missingness_ipw_beta_shift_{headline}.csv", index=False
    )


# --------------------------------------------------------------------------
# (c) Mediation: counterfactual NDE/NIE + sensitivity, in place of raw
# Baron-Kenny proportion-mediated.
# --------------------------------------------------------------------------

MAX_GFORMULA_PATHS = 25  # cap: full-refit bootstrap g-formula is expensive; see log for what's dropped


def stage_mediation(cohort: Cohort, out: Path, n_boot: int = 5000,
                     n_boot_gformula: int = 200, n_mc: int = 200) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # Baseline Baron-Kenny run at the higher bootstrap count the reviewer
    # asked for, reusing the existing implementation.
    bk = run_mediation(
        cohort, exposures=EXPOSURES, outcomes=list(SYMPTOMS_BINARY),
        n_boot=n_boot,
    )
    if not bk.empty:
        bk = bk.assign(estimand_note=MEDIATION_ESTIMAND_NOTE)
    bk.to_csv(out / "mediation_baron_kenny.csv", index=False)

    surviving = bk[bk["q_indirect"] < 0.05] if not bk.empty else bk
    surviving.to_csv(out / "mediation_surviving_paths.csv", index=False)

    # Counterfactual NDE/NIE via Monte Carlo g-formula (Imai/Keele/Tingley
    # style): fit mediator|exposure,covars and outcome|exposure,mediator,covars,
    # then simulate outcome under (exposure=e1, mediator=M(e0)) vs
    # (exposure=e1, mediator=M(e1)) to decompose the total effect on the
    # outcome's natural (risk-difference / mean-difference) scale. This
    # replaces the product-of-coefficients estimand, which is undefined/
    # sign-flippable once the outcome model is nonlinear (logistic) while
    # the mediator model is linear.
    ranked = surviving.reindex(surviving["indirect"].abs().sort_values(ascending=False).index)
    capped, dropped = ranked.iloc[:MAX_GFORMULA_PATHS], ranked.iloc[MAX_GFORMULA_PATHS:]
    if len(dropped):
        log.warning("g-formula: capped at %d/%d surviving paths by |indirect|; "
                    "%d paths skipped (see mediation_surviving_paths.csv for the full set)",
                    MAX_GFORMULA_PATHS, len(ranked), len(dropped))

    # One g-formula triple per worker: each does its own MC + bootstrap refit,
    # so the 25 capped paths run across all cores instead of serially.
    nde_nie = Parallel(n_jobs=_n_cores(), backend="loky", verbose=0)(
        delayed(_gformula_nde_nie)(
            cohort, row["exposure"], row["mediator"], row["outcome"],
            n_mc=n_mc, n_boot=n_boot_gformula,
        )
        for _, row in capped.iterrows()
    )
    nde_nie_rows = [r for r in nde_nie if r is not None]
    pd.DataFrame(nde_nie_rows).to_csv(out / "mediation_counterfactual.csv", index=False)

    # E-value sensitivity analysis for unmeasured confounding of the
    # mediator->outcome arm, for surviving paths.
    om = outcome_meta()
    evalues = [_e_value(row["b"], om.get(row["outcome"])) for _, row in surviving.iterrows()]
    (surviving.assign(e_value_my_arm=evalues, estimand_note=MEDIATION_ESTIMAND_NOTE)
     .to_csv(out / "mediation_e_values.csv", index=False))

    log.info("mediation stage done -> %s (labelled cross-sectional decomposition, not causal)", out)


def _linpred(res: sm.regression.linear_model.RegressionResults, cov_df: pd.DataFrame,
             e: np.ndarray | float, m: np.ndarray | float | None = None) -> np.ndarray:
    """Linear predictor from a fitted OLS/Logit result, computed by name
    against ``res.params`` rather than via ``.predict(DataFrame)`` -- avoids
    any risk of column-order mismatch between the fitted design matrix and
    a hand-built counterfactual exog frame."""
    p = res.params
    eta = p["const"] + p["e"] * np.asarray(e, dtype=float)
    if m is not None:
        eta = eta + p["m"] * np.asarray(m, dtype=float)
    for c in cov_df.columns:
        eta = eta + p[c] * cov_df[c].to_numpy(dtype=float)
    return eta


def _gformula_nde_nie(
    cohort: Cohort, exposure: str, mediator: str, outcome: str,
    n_mc: int = 200, n_boot: int = 200, seed: int = 0,
) -> dict | None:
    """Monte Carlo g-formula natural direct/indirect effect for one
    exposure->mediator->outcome triple, on the outcome's natural scale
    (risk difference for binary outcomes, mean difference for continuous).

    Contrast: binary exposures use e0=0/e1=1; continuous exposures use a
    1-SD contrast (e0=mean, e1=mean+1sd), the standard convention when the
    "exposure" is a continuous diet score rather than a treatment indicator.
    """
    om = outcome_meta()
    kind = om.get(outcome, "binary")
    feats = transform_features(cohort.microbiome, kind="clr")
    all_pheno = stack_outcomes(cohort)
    cov = cohort.covariates[list(COVARS_FULL)]
    e_series = all_pheno[exposure] if exposure in all_pheno.columns else cohort.covariates[exposure]

    df = pd.concat(
        [e_series.rename("e"), feats[mediator].rename("m"), all_pheno[outcome].rename("y"), cov],
        axis=1,
    ).dropna()
    if len(df) < 100:
        return None
    cov_cols = list(cov.columns)
    is_binary_exposure = set(df["e"].unique()) <= {0.0, 1.0}
    e0, e1 = (0.0, 1.0) if is_binary_exposure else (float(df["e"].mean()), float(df["e"].mean() + df["e"].std()))

    def _fit_and_simulate(
        data: pd.DataFrame, rng: np.random.Generator
    ) -> tuple[float, float, float, float, float]:
        m_res = _fit_lm(data["m"], data[["e", *cov_cols]], "continuous")
        resid_sd = float(np.std(m_res.resid))
        y_res = _fit_lm(data["y"], data[["e", "m", *cov_cols]], kind)
        cov_df = data[cov_cols]
        n = len(data)

        def sim_m(e_val: float) -> np.ndarray:
            mean_m = _linpred(m_res, cov_df, e=np.full(n, e_val))
            return mean_m[None, :] + rng.normal(0.0, resid_sd, size=(n_mc, n))

        def pred_y(e_val: float, m_draws: np.ndarray) -> float:
            eta = _linpred(y_res, cov_df, e=np.full(n, e_val), m=m_draws)
            pred = expit(eta) if kind == "binary" else eta
            return float(pred.mean())

        m0 = sim_m(e0)
        m1 = sim_m(e1)
        y_e0_m0 = pred_y(e0, m0)
        y_e1_m0 = pred_y(e1, m0)
        y_e1_m1 = pred_y(e1, m1)
        # y_e1_m0 / y_e1_m1 are the reference / shifted-mediator outcome means
        # holding exposure at e1; their ratio is the NIE on the risk-ratio scale
        # (binary), which the E-value needs.
        return y_e1_m0 - y_e0_m0, y_e1_m1 - y_e1_m0, y_e1_m1 - y_e0_m0, y_e1_m0, y_e1_m1

    rng = np.random.default_rng(seed)
    try:
        nde, nie, total, y_e1_m0, y_e1_m1 = _fit_and_simulate(df, rng)
    except Exception as exc:  # noqa: BLE001
        log.warning("g-formula fit failed for %s->%s->%s: %s", exposure, mediator, outcome, exc)
        return None

    n = len(df)
    boots_nde, boots_nie = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = df.iloc[idx].reset_index(drop=True)
        try:
            b_nde, b_nie, _, _, _ = _fit_and_simulate(d, rng)
        except Exception:  # noqa: BLE001
            continue
        boots_nde.append(b_nde)
        boots_nie.append(b_nie)
    boots_nde_arr = np.asarray(boots_nde)
    boots_nie_arr = np.asarray(boots_nie)
    nde_lo, nde_hi = (np.nanpercentile(boots_nde_arr, [2.5, 97.5]) if len(boots_nde_arr)
                      else (float("nan"), float("nan")))
    nie_lo, nie_hi = (np.nanpercentile(boots_nie_arr, [2.5, 97.5]) if len(boots_nie_arr)
                      else (float("nan"), float("nan")))
    prop_mediated = float(nie / total) if total != 0 else float("nan")

    # E-value on the indirect effect itself (not the M->Y arm coefficient):
    # for binary outcomes the NIE has a natural risk-ratio y_e1_m1 / y_e1_m0,
    # so the VanderWeele-Ding E-value applies directly to the point estimate
    # and to the CI limit nearest the null. Continuous outcomes have no RR
    # scale -> NaN (the M->Y-arm E-value in mediation_e_values.csv covers those).
    if kind == "binary" and y_e1_m0 > 0:
        rr_nie = y_e1_m1 / y_e1_m0
        e_value_nie = _e_value_from_rr(rr_nie)
        nie_bound = nie_lo if abs(nie_lo) < abs(nie_hi) else nie_hi
        rr_bound = (y_e1_m0 + nie_bound) / y_e1_m0
        e_value_nie_ci = _e_value_from_rr(rr_bound)
    else:
        rr_nie = e_value_nie = e_value_nie_ci = float("nan")

    return dict(
        exposure=exposure, mediator=mediator, outcome=outcome, outcome_kind=kind,
        nde=nde, nde_ci_lo=float(nde_lo), nde_ci_hi=float(nde_hi),
        nie=nie, nie_ci_lo=float(nie_lo), nie_ci_hi=float(nie_hi),
        total_effect=total, prop_mediated=prop_mediated,
        nie_risk_ratio=float(rr_nie), e_value_nie=float(e_value_nie),
        e_value_nie_ci=float(e_value_nie_ci),
        n=n, n_boot_used=len(boots_nde_arr),
        scale="probability (risk difference)" if kind == "binary" else "outcome units (mean difference)",
        estimand_note=MEDIATION_ESTIMAND_NOTE,
    )


def _e_value(b_mediator_outcome: float, outcome_kind: str | None) -> float:
    """E-value (VanderWeele & Ding 2017) for unmeasured confounding of the
    mediator->outcome arm. ``b`` is the covariate-adjusted mediator
    coefficient from mediation.mediation_one's outcome model
    (mediation.py:_fit_lm): for binary outcomes this is already a logistic
    log-odds coefficient, so OR = exp(b) directly and the standard E-value
    formula applies via the RR approximation (valid since the CLR mediator
    effect here is not on a common outcome, so OR approximates RR). For
    continuous outcomes ``b`` is an OLS slope with no natural odds-ratio
    scale -- returned as NaN rather than forced through an ad hoc
    linear-to-OR rule of thumb, which would misstate precision.
    """
    if outcome_kind != "binary" or not np.isfinite(b_mediator_outcome):
        return float("nan")
    return _e_value_from_rr(float(np.exp(b_mediator_outcome)))


# --------------------------------------------------------------------------
# (d) Clinical anchor for self-reported bleeding gums
# --------------------------------------------------------------------------

CLINICAL_FIELD_CANDIDATES = (
    "probing_depth", "bleeding_on_probing", "plaque_index", "clinical_attachment_level",
)


def stage_clinical(cohort: Cohort, out: Path, headline: str = "bleeding_gums",
                    data_dir: Path | None = None) -> None:
    """Look for clinical periodontal fields in the HPP extract; if present,
    validate self-report against them (sensitivity/specificity, kappa) and
    re-run the headline association restricted to that subsample. If absent,
    fail loudly with a flag file rather than silently proceeding, per the
    reviewer's instruction to keep manuscript language honest.
    """
    out.mkdir(parents=True, exist_ok=True)
    data_dir = data_dir or Path(cohort.covariates.attrs.get("data_dir", DATA_DIR))

    clinical = _load_clinical_fields(data_dir)
    if clinical is None or clinical.empty:
        flag = out / "CLINICAL_VALIDATION_MISSING.flag"
        flag.write_text(
            "No clinical periodontal fields found in HPP extract "
            f"(looked for {CLINICAL_FIELD_CANDIDATES}). "
            f"'{headline}' remains an unvalidated self-report outcome. "
            "Manuscript must state this explicitly; do not claim clinical "
            "validation without this file being replaced by real output.\n"
        )
        log.warning("clinical stage: no clinical fields found, wrote %s", flag)
        return

    all_pheno = stack_outcomes(cohort)
    joined = all_pheno[[headline]].join(clinical, how="inner").dropna()
    if joined.empty:
        (out / "CLINICAL_VALIDATION_MISSING.flag").write_text(
            f"Clinical fields present but no overlap with '{headline}' respondents.\n"
        )
        return

    # Agreement stats against whichever binary/ordinal clinical field is
    # available (bleeding_on_probing is the natural anchor for self-reported
    # bleeding gums).
    if "bleeding_on_probing" in joined.columns:
        self_report = joined[headline].astype(int)
        clinical_bop = (joined["bleeding_on_probing"] > 0).astype(int)
        agreement = _binary_agreement(self_report, clinical_bop)
        pd.Series(agreement).to_csv(out / "clinical_agreement_stats.csv")

    # Re-run headline association restricted to the clinically-anchored
    # subsample.
    restricted_cohort = Cohort(
        microbiome=cohort.microbiome.loc[joined.index],
        symptoms=cohort.symptoms.loc[joined.index],
        lifestyle=cohort.lifestyle.loc[joined.index],
        covariates=cohort.covariates.loc[joined.index],
    )
    from oral_mb.associations import associations  # local import: avoid cycle
    # associations() tests every feature x every outcome on the restricted
    # cohort; keep only the headline rows (it has no per-outcome filter arg).
    assoc = associations(restricted_cohort, feature_kind="clr", use_bmi=True, n_jobs=_n_cores())
    if not assoc.empty:
        assoc = assoc[assoc["outcome"] == headline]
    assoc.to_csv(out / f"clinical_restricted_association_{headline}.csv", index=False)

    log.info("clinical stage done -> %s (n=%d clinically-anchored)", out, len(joined))


def _load_clinical_fields(data_dir: Path) -> pd.DataFrame | None:
    """TODO: point at the actual HPP clinical-exam parquet/csv once its path
    is known; return None (not an empty frame) if the source simply doesn't
    exist so stage_clinical can distinguish 'no file' from 'file, no rows'."""
    candidate = data_dir / "clinical_periodontal.parquet"
    if not candidate.exists():
        return None
    return pd.read_parquet(candidate)


def _binary_agreement(a: pd.Series, b: pd.Series) -> dict:
    tp = int(((a == 1) & (b == 1)).sum())
    tn = int(((a == 0) & (b == 0)).sum())
    fp = int(((a == 1) & (b == 0)).sum())
    fn = int(((a == 0) & (b == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    po = (tp + tn) / (tp + tn + fp + fn)
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (tp + tn + fp + fn) ** 2
    kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return {"sensitivity": sens, "specificity": spec, "kappa": kappa, "n": tp + tn + fp + fn}


# --------------------------------------------------------------------------
# (e) Corrected prediction reporting: repeated nested CV, bootstrap CIs,
# AUPRC-first for imbalanced outcomes, leakage audit for high-AUC outcomes.
# --------------------------------------------------------------------------

def _fold_point_metric(fold: FoldPrediction, kind: str) -> dict[str, float]:
    from scipy.stats import pearsonr
    from sklearn.metrics import roc_auc_score
    if kind == "binary":
        if len(np.unique(fold.y_true)) < 2:
            return {"auc": float("nan")}
        return {"auc": float(roc_auc_score(fold.y_true, fold.y_pred))}
    return {"pearson": float(pearsonr(fold.y_true, fold.y_pred)[0])}


def _safe_predict_outcome_raw(cohort: Cohort, outcome: str, cfg: PredictionConfig
                              ) -> dict[str, CVResult] | None:
    """Like prediction._safe_predict_one, but keeps the raw CVResult objects
    (folds + test FoldPrediction) instead of collapsing to summary metrics,
    so this stage can compute bootstrap CIs and real AUPRC downstream."""
    try:
        results = predict_outcome(cohort, outcome, cfg)
        if not results:
            log.info("[pred] %-30s skipped (too few samples / single class)", outcome)
            return None
        return results
    except Exception as e:  # noqa: BLE001
        log.warning("skip %s: %s", outcome, e)
        return None


def stage_prediction(cohort: Cohort, out: Path, n_repeats: int = 5,
                      leakage_auc_threshold: float = 0.90, n_boot: int = 2000) -> None:
    """Repeated holdout (different SplitConfig.seed per repeat) instead of a
    single 80/20 split, with the CV-fold distribution reported alongside a
    bootstrap CI on pooled held-out test predictions -- both computed from
    the raw per-sample predictions in CVResult.folds/.test, not the
    already-summarized rows run_all_outcomes returns.
    """
    out.mkdir(parents=True, exist_ok=True)
    # num_threads=1: parallelism is at the outcome level (n_jobs workers), so
    # each LightGBM must stay single-threaded or workers*threads oversubscribes
    # the box. On a 4-core machine the old total_cores//8 gave n_jobs=1 (no
    # parallelism); drive one worker per core instead.
    cores = _n_cores()
    cfg_base = PredictionConfig(models=("lgbm", "ridge", "ols"), ensemble=True, num_threads=1)
    meta = outcome_meta()
    n_jobs = min(len(meta), cores)

    summary_rows: list[dict] = []
    # (outcome, source, model, kind) -> list of (y_true, y_pred) test arrays, one per repeat
    pooled_test: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    for seed in range(n_repeats):
        cfg = PredictionConfig(**{
            **cfg_base.__dict__,
            "split": SplitConfig(test_frac=cfg_base.split.test_frac,
                                  n_folds=cfg_base.split.n_folds, seed=seed),
        })
        raw_per_outcome = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_safe_predict_outcome_raw)(cohort, o, cfg) for o in meta
        )
        for outcome, results in zip(meta, raw_per_outcome):
            if not results:
                continue
            for tag, r in results.items():
                source, model = tag.split("/", 1)
                for fold in r.folds:
                    summary_rows.append(dict(
                        outcome=outcome, source=source, model=model, kind=r.kind,
                        repeat=seed, split="cv", **_fold_point_metric(fold, r.kind),
                    ))
                if r.test is not None and len(r.test.y_true):
                    summary_rows.append(dict(
                        outcome=outcome, source=source, model=model, kind=r.kind,
                        repeat=seed, split="test", **_fold_point_metric(r.test, r.kind),
                    ))
                    pooled_test[(outcome, source, model, r.kind)].append(
                        (r.test.y_true, r.test.y_pred)
                    )
        log.info("prediction repeat %d/%d done", seed + 1, n_repeats)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "prediction_repeated_cv.csv", index=False)

    # Distribution (not point estimate) per outcome/source/model/split.
    metric_cols = [c for c in ("auc", "pearson") if c in summary_df.columns]
    dist = (
        summary_df.groupby(["outcome", "source", "model", "split"])[metric_cols]
        .agg(["mean", "std", "min", "max"])
    )
    dist.to_csv(out / "prediction_cv_distribution.csv")

    # Bootstrap CI on pooled held-out test predictions (pooled across
    # repeats = a repeated-holdout bootstrap, per reviewer ask: "no more
    # test > CV therefore not overfit" reported off a single split). AUPRC
    # computed for real here (bootstrap_metric already supports "auprc" via
    # sklearn.average_precision_score), with prevalence printed alongside.
    rng = np.random.default_rng(0)
    ci_rows = []
    for (outcome, source, model, kind), pairs in pooled_test.items():
        y_true = np.concatenate([p[0] for p in pairs])
        y_pred = np.concatenate([p[1] for p in pairs])
        row = dict(outcome=outcome, source=source, model=model, kind=kind,
                    n_pooled_test=int(len(y_true)))
        if kind == "binary":
            if len(np.unique(y_true)) < 2:
                continue
            auc_pt, auc_lo, auc_hi = bootstrap_metric(y_true, y_pred, "auc", n_boot=n_boot, rng=rng)
            auprc_pt, auprc_lo, auprc_hi = bootstrap_metric(y_true, y_pred, "auprc", n_boot=n_boot, rng=rng)
            row.update(auc=auc_pt, auc_ci_lo=auc_lo, auc_ci_hi=auc_hi,
                       auprc=auprc_pt, auprc_ci_lo=auprc_lo, auprc_ci_hi=auprc_hi,
                       prevalence=float(y_true.mean()))
        else:
            r_pt, r_lo, r_hi = bootstrap_metric(y_true, y_pred, "pearson", n_boot=n_boot, rng=rng)
            row.update(pearson=r_pt, pearson_ci_lo=r_lo, pearson_ci_hi=r_hi)
        ci_rows.append(row)
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(out / "prediction_bootstrap_ci.csv", index=False)

    # AUPRC-first table for binary outcomes: AUPRC + CI + prevalence next to
    # it, so a high AUC on a rare outcome (e.g. keto, ~91/6857) can't be read
    # without the class-imbalance context.
    if not ci_df.empty and (ci_df["kind"] == "binary").any():
        auprc_tbl = (
            ci_df[ci_df["kind"] == "binary"]
            [["outcome", "source", "model", "auprc", "auprc_ci_lo", "auprc_ci_hi",
              "auc", "auc_ci_lo", "auc_ci_hi", "prevalence", "n_pooled_test"]]
            .sort_values("auprc", ascending=False)
        )
    else:
        auprc_tbl = pd.DataFrame(columns=["outcome", "source", "model", "auprc", "prevalence"])
    auprc_tbl.to_csv(out / "prediction_auprc_prevalence.csv", index=False)

    # Leakage audit flag for any (ensemble-model) outcome whose mean test
    # AUC crosses the threshold -- doesn't run the audit itself (needs
    # top-feature dump + batch/site/collection-date confound checks), just
    # flags what needs it so the rebuttal can't silently skip it.
    if not ci_df.empty:
        high_auc = (
            ci_df[(ci_df["kind"] == "binary") & (ci_df["model"] == "ensemble")]
            .groupby("outcome")["auc"].mean().reset_index()
            .query("auc >= @leakage_auc_threshold")
        )
    else:
        high_auc = pd.DataFrame(columns=["outcome", "auc"])
    high_auc.to_csv(out / "prediction_leakage_audit_needed.csv", index=False)
    if not high_auc.empty:
        log.warning("outcomes needing leakage audit (AUC>=%.2f): %s",
                    leakage_auc_threshold, high_auc["outcome"].tolist())

    log.info("prediction stage done -> %s", out)


# --------------------------------------------------------------------------

STAGES = {
    "missingness": stage_missingness,
    "mediation": stage_mediation,
    "clinical": stage_clinical,
    "prediction": stage_prediction,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="species")
    ap.add_argument("--out", default="revision_out", type=Path)
    ap.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cohort = load_cohort(level=args.level)

    stages = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
    for name, fn in stages.items():
        log.info("=== stage: %s ===", name)
        fn(cohort, args.out / name)


if __name__ == "__main__":
    main()
