"""Microbiome GWAS preparation + heritability estimation.

Writes PLINK-compatible phenotype files for each microbiome feature. Real GWAS
runs externally (PLINK2 / REGENIE / GCTA). Provides h^2 SNP via simple GREML
wrapper template + per-feature inverse-normal transform for non-normal taxa.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data import Cohort, transform_features

log = logging.getLogger(__name__)


def inverse_normal_transform(s: pd.Series) -> pd.Series:
    """Rank-based inverse-normal transform (Blom)."""
    x = s.dropna()
    r = stats.rankdata(x.values, method="average")
    z = stats.norm.ppf((r - 0.5) / len(r))
    return pd.Series(z, index=x.index, name=s.name).reindex(s.index)


def write_plink_phenotypes(
    cohort: Cohort,
    out_dir: Path,
    feature_kind: str = "clr",
    id_map: dict[str, tuple[str, str]] | None = None,
) -> Path:
    """Emit PLINK pheno file: FID IID feat1 feat2 ...

    `id_map` maps RegistrationCode -> (FID, IID). If absent, uses RegCode for both.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = transform_features(cohort.microbiome, kind=feature_kind)
    feats_int = feats.apply(inverse_normal_transform)

    rows = []
    for reg, row in feats_int.iterrows():
        fid, iid = id_map.get(reg, (reg, reg)) if id_map else (reg, reg)
        rows.append([fid, iid, *row.values.tolist()])
    cols = ["FID", "IID", *feats_int.columns.tolist()]
    out = pd.DataFrame(rows, columns=cols)
    p = out_dir / "microbiome_phenos_INT.tsv"
    out.to_csv(p, sep="\t", index=False, na_rep="NA")
    log.info("wrote %d phenotypes for %d samples -> %s", feats_int.shape[1], len(out), p)
    return p


def write_covariate_file(cohort: Cohort, out_dir: Path, n_pcs: int = 10) -> Path:
    """Covariate file template: FID IID age sex PC1..PCk. PCs left as NA placeholders."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cov = cohort.covariates[["age", "gender"]].copy()
    cov.columns = ["age", "sex"]
    for k in range(1, n_pcs + 1):
        cov[f"PC{k}"] = np.nan
    cov = cov.reset_index().rename(columns={"RegistrationCode": "IID"})
    cov.insert(0, "FID", cov["IID"])
    p = out_dir / "covariates.tsv"
    cov.to_csv(p, sep="\t", index=False, na_rep="NA")
    return p


PLINK2_SCRIPT_TEMPLATE = r"""#!/usr/bin/env bash
# Template: run PLINK2 GWAS for one microbiome feature.
# Usage: bash plink2_mgwas.sh BFILE PHENO_NAME OUT_PREFIX
set -euo pipefail
BFILE=$1
PHENO=$2
OUT=$3
plink2 \
  --bfile "$BFILE" \
  --pheno {pheno_file} \
  --pheno-name "$PHENO" \
  --covar {covar_file} \
  --covar-name age,sex,PC1-PC10 \
  --glm hide-covar \
  --out "$OUT"
"""


GCTA_GREML_TEMPLATE = r"""#!/usr/bin/env bash
# Template: GCTA GREML for SNP-heritability per feature.
# Usage: bash gcta_h2.sh GRM PHENO_FILE PHENO_COL OUT
set -euo pipefail
GRM=$1; PHENO=$2; COL=$3; OUT=$4
gcta64 --grm $GRM --pheno $PHENO --mpheno $COL \
  --qcovar {qcovar_file} --reml --out $OUT
"""


def emit_runner_scripts(
    out_dir: Path, pheno_file: Path, covar_file: Path
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "plink2_mgwas.sh"
    p1.write_text(PLINK2_SCRIPT_TEMPLATE.format(pheno_file=pheno_file, covar_file=covar_file))
    p2 = out_dir / "gcta_h2.sh"
    p2.write_text(GCTA_GREML_TEMPLATE.format(qcovar_file=covar_file))
    for p in (p1, p2):
        p.chmod(0o755)
    return p1, p2


def parse_plink_glm(path: Path, p_threshold: float = 5e-8) -> pd.DataFrame:
    """Parse PLINK2 .glm.linear output and return genome-wide significant hits."""
    df = pd.read_csv(path, sep="\t", comment=None)
    df.columns = [c.lstrip("#") for c in df.columns]
    if "P" not in df.columns:
        raise ValueError(f"no P column in {path}")
    return df[df["P"] < p_threshold].sort_values("P").reset_index(drop=True)


def colocalize_with_periodontitis_gwas(
    mqtl_hits: pd.DataFrame, periodontitis_sumstats: pd.DataFrame, window_bp: int = 500_000
) -> pd.DataFrame:
    """Naive coloc: SNPs within `window_bp` of a periodontitis hit at p<5e-8."""
    perio_sig = periodontitis_sumstats[periodontitis_sumstats["P"] < 5e-8][["CHR", "BP"]].copy()
    rows = []
    for _, hit in mqtl_hits.iterrows():
        near = perio_sig[
            (perio_sig["CHR"] == hit["CHROM"])
            & ((perio_sig["BP"] - hit["POS"]).abs() < window_bp)
        ]
        if len(near):
            rows.append({**hit.to_dict(), "n_perio_near": len(near)})
    return pd.DataFrame(rows)


def submit_plink_array(
    bfile: Path, pheno_file: Path, out_dir: Path, runner: Path, queue: str = "long"
) -> None:
    """Stub: emit qsub commands for cluster submission per feature."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pheno_cols = pd.read_csv(pheno_file, sep="\t", nrows=0).columns.tolist()[2:]
    submit_log = out_dir / "submit_log.sh"
    with submit_log.open("w") as f:
        for p in pheno_cols:
            f.write(
                f"qsub -q {queue} -N mgwas_{p} -- {runner} {bfile} {p} {out_dir}/{p}\n"
            )
    log.info("submission script: %s", submit_log)
