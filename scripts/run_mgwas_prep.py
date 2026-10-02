"""Prepare PLINK2 phenotype + covariate files for mGWAS, emit runner scripts."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from oral_mb.config import DATA_DIR, OUT_DIR
from oral_mb.data import load_cohort, prevalence_filter
from oral_mb.mgwas import (
    emit_runner_scripts, submit_plink_array,
    write_covariate_file, write_plink_phenotypes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="species")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "mgwas")
    ap.add_argument("--bfile", type=Path, required=False)
    args = ap.parse_args()

    cohort = load_cohort(level=args.level, data_dir=args.data_dir)
    cohort = type(cohort)(
        microbiome=prevalence_filter(cohort.microbiome),
        symptoms=cohort.symptoms, lifestyle=cohort.lifestyle, covariates=cohort.covariates,
    )

    pheno_path = write_plink_phenotypes(cohort, args.out)
    cov_path = write_covariate_file(cohort, args.out)
    runner, _ = emit_runner_scripts(args.out, pheno_path, cov_path)
    if args.bfile:
        submit_plink_array(args.bfile, pheno_path, args.out, runner)


if __name__ == "__main__":
    main()
