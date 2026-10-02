"""Replicate discovery signature in an external cohort + meta-analyze."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from oral_mb.associations import associations
from oral_mb.data import load_cohort, prevalence_filter
from oral_mb.replication import meta_analysis, replicate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def _filtered(c):
    return type(c)(
        microbiome=prevalence_filter(c.microbiome),
        symptoms=c.symptoms, lifestyle=c.lifestyle, covariates=c.covariates,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery-dir", type=Path, required=True)
    ap.add_argument("--replication-dir", type=Path, required=True)
    ap.add_argument("--level", default="species")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    disc = _filtered(load_cohort(level=args.level, data_dir=args.discovery_dir))
    rep = _filtered(load_cohort(level=args.level, data_dir=args.replication_dir))

    disc_a = associations(disc, feature_kind="clr")
    rep_a = associations(rep, feature_kind="clr")

    disc_a.to_csv(args.out / "assoc_discovery.csv", index=False)
    rep_a.to_csv(args.out / "assoc_replication.csv", index=False)

    rep_table = replicate(disc_a, rep)
    rep_table.to_csv(args.out / "replication_table.csv", index=False)

    meta = meta_analysis([disc_a, rep_a])
    meta.to_csv(args.out / "meta_analysis.csv", index=False)


if __name__ == "__main__":
    main()
