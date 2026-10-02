#!/usr/bin/env python
"""Predict ICD-11 diagnoses from oral vs gut microbiome.

Claim under test: the cheap buccal swab predicts disease (GI conditions like
IBS in particular) as well as the expensive stool sample. Every qualifying
diagnosis is run against four feature sources — covariates-only (age/gender),
oral, gut, oral+gut — so both the microbiome gain over covariates and the
oral-vs-gut gap are readable off one table.

    python scripts/run_disease_prediction.py --out <dir>/disease_prediction.csv
    python scripts/run_disease_prediction.py --gi-only --level genus
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from oral_mb.config import OUT_DIR
from oral_mb.data import load_cohort
from oral_mb.diseases import DiseaseConfig, run_all_diseases
from oral_mb.prediction import PredictionConfig, SplitConfig

log = logging.getLogger("disease_prediction")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", default="species")
    p.add_argument("--out", type=Path, default=OUT_DIR / "disease_prediction.csv")
    p.add_argument("--gi-only", action="store_true",
                   help="restrict to ICD-11 chapter 13 (digestive) codes")
    p.add_argument("--min-prevalence", type=float, default=0.02)
    p.add_argument("--min-cases", type=int, default=50)
    p.add_argument("--only-baseline", action="store_true",
                   help="first visit per person only")
    p.add_argument("--sources", default="covars,oral,gut,oral+gut")
    p.add_argument("--models", default="lgbm,ridge")
    p.add_argument("--use-bmi", action="store_true")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--num-threads", type=int, default=8)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())

    cohort = load_cohort(level=args.level)
    cfg = PredictionConfig(
        models=models,
        ensemble=len(models) > 1,
        use_bmi=args.use_bmi,
        use_gut=("gut" in sources or "oral+gut" in sources),
        split=SplitConfig(n_folds=args.n_folds),
        num_threads=args.num_threads,
    )
    dcfg = DiseaseConfig(
        min_prevalence=args.min_prevalence,
        min_cases=args.min_cases,
        only_baseline=args.only_baseline,
        gi_only=args.gi_only,
    )

    df = run_all_diseases(cohort, cfg=cfg, dcfg=dcfg, sources=sources,
                          n_jobs=args.n_jobs)
    if df.empty:
        log.error("no results — check diagnosis prevalence thresholds")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["disease", "source", "model"]).to_csv(args.out, index=False)
    log.info("wrote %s (%d rows)", args.out, len(df))

    best = (df[df.model == ("ensemble" if len(models) > 1 else models[0])]
            .pivot_table(index=["disease", "gi"], columns="source", values="auc"))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        log.info("CV AUC by source:\n%s", best.round(3))


if __name__ == "__main__":
    main()
