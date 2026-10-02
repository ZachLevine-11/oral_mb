#!/usr/bin/env python
"""Which gut species associate with a ketogenic diet?

Single-claim script: gut microbiome (GutMBLoader mpa_species) x the ``keto``
lifestyle flag only, age+gender adjusted logistic regression per species with
BH FDR across species. Deliberately narrow so it runs in seconds, unlike the
full associations sweep (every feature x every outcome).

Motivation: in prediction_with_gut.csv keto is the one outcome where gut beats
buccal, and only for LGBM (gut AUC 0.95 vs ridge 0.65), i.e. the signal looks
like a few specific gut taxa rather than a diffuse linear shift.

    python scripts/gut_keto_assoc.py --out <dir>/gut_keto_assoc.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

from oral_mb.config import COVARS_BASE, COVARS_FULL, FDR_ALPHA
from oral_mb.data import clr_transform, load_cohort, to_relative
from oral_mb.prediction import load_gut_microbiome
from oral_mb.stats_utils import bh_qvalues

log = logging.getLogger("gut_keto_assoc")


def _fit(y: pd.Series, x: pd.Series, cov: pd.DataFrame) -> dict | None:
    d = pd.concat([y.rename("y"), x.rename("x"), cov], axis=1).dropna()
    if d["y"].nunique() < 2 or len(d) < 30 or d["x"].std() == 0:
        return None
    X = sm.add_constant(d[["x", *cov.columns]], has_constant="add")
    try:
        res = sm.Logit(d["y"].astype(int), X).fit(disp=0, method="lbfgs", maxiter=200)
    except Exception as e:  # separation / singular design
        log.debug("fit failed %s: %s", x.name, e)
        return None
    return {
        "feature": str(x.name), "outcome": "keto",
        "beta": float(res.params["x"]), "se": float(res.bse["x"]),
        "pvalue": float(res.pvalues["x"]), "n": int(len(d)),
        "n_cases": int(d["y"].sum()),
        "prevalence_feature": float((x.loc[d.index] > x.min()).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--level", default="mpa_species",
                    choices=["mpa_species", "mpa_genus", "mpa_family"])
    ap.add_argument("--use-bmi", action="store_true")
    ap.add_argument("--min-prev", type=float, default=0.05,
                    help="keep gut features present in >= this fraction of samples")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cohort = load_cohort("species")
    if "keto" not in cohort.lifestyle.columns:
        raise SystemExit("no 'keto' column in lifestyle table")
    y = cohort.lifestyle["keto"].dropna()

    gut = load_gut_microbiome(df=args.level, study_ids=[10],
                              min_col_present_frac=0.0, groupby_reg="first")
    gut = gut.loc[~gut.index.duplicated(keep="first")]

    idx = gut.index.intersection(y.index).intersection(cohort.covariates.index)
    log.info("overlap gut x cohort = %d (keto cases=%d)",
             len(idx), int(y.loc[idx].sum()))
    gut = gut.loc[idx]

    # Prevalence filter on the analysed samples only (no held-out set here).
    present = (gut > gut.min().min()).mean(axis=0)
    keep = present[present >= args.min_prev].index
    log.info("prevalence filter %.2f: %d/%d gut features kept",
             args.min_prev, len(keep), gut.shape[1])
    gut = gut[keep]

    feats = clr_transform(to_relative(gut))
    cov = cohort.covariates.loc[idx, list(COVARS_FULL if args.use_bmi else COVARS_BASE)]
    yk = y.loc[idx]

    rows = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(_fit)(yk, feats[f], cov) for f in feats.columns
    )
    df = pd.DataFrame([r for r in rows if r is not None])
    if df.empty:
        raise SystemExit("no gut species fit successfully")

    df["q"] = bh_qvalues(df["pvalue"].to_numpy())
    df["significant"] = df["q"] < FDR_ALPHA
    df = df.sort_values("pvalue").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("wrote %s (%d species, %d significant at q<%.2f)",
             args.out, len(df), int(df["significant"].sum()), FDR_ALPHA)
    with pd.option_context("display.width", 200):
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
