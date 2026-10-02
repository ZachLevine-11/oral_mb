"""Strain-level + functional virulence analyses.

Wraps StrainPhlAn4 + HUMAnN3 outputs; tests virulence-gene carriage vs symptom.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import COVARS_FULL
from .stats_utils import bh_qvalues

log = logging.getLogger(__name__)

# Canonical virulence genes by species.
VIRULENCE_GENES: dict[str, tuple[str, ...]] = {
    "Porphyromonas_gingivalis": ("kgp", "rgpA", "rgpB", "fimA", "hagA"),
    "Fusobacterium_nucleatum": ("fadA", "fap2"),
    "Tannerella_forsythia": ("bspA", "karilysin"),
    "Aggregatibacter_actinomycetemcomitans": ("ltxA", "cdtABC"),
}


@dataclass(frozen=True)
class VirulenceAssoc:
    species: str
    gene: str
    outcome: str
    beta: float
    pvalue: float
    n_carriers: int
    n: int


def run_strainphlan(
    samples_dir: Path, db: Path, out_dir: Path, n_threads: int = 8, clade: str | None = None
) -> Path:
    """Invoke StrainPhlAn4 for one clade. Returns marker alignment path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "strainphlan",
        "--samples", *map(str, samples_dir.glob("*.pkl")),
        "--database", str(db),
        "--output_dir", str(out_dir),
        "--nproc", str(n_threads),
    ]
    if clade:
        cmd += ["--clade", clade]
    log.info("strainphlan: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_dir


def run_humann(
    fastq: Path, db_chocophlan: Path, db_uniref: Path, out_dir: Path, n_threads: int = 8
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "humann", "--input", str(fastq), "--output", str(out_dir),
        "--threads", str(n_threads),
        "--nucleotide-database", str(db_chocophlan),
        "--protein-database", str(db_uniref),
    ]
    log.info("humann: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_dir


def load_panphlan_presence(path: Path) -> pd.DataFrame:
    """Load PanPhlAn gene presence/absence: rows=samples, cols=genes."""
    return pd.read_csv(path, sep="\t", index_col=0).T.astype(int)


def virulence_associations(
    gene_pa: pd.DataFrame,
    species: str,
    phenotypes: pd.DataFrame,
    covariates: pd.DataFrame,
    outcome_kinds: dict[str, str],
) -> pd.DataFrame:
    candidates = VIRULENCE_GENES.get(species, ())
    cols = [g for g in candidates if g in gene_pa.columns]
    if not cols:
        log.warning("no virulence genes found for %s in PanPhlAn output", species)
        return pd.DataFrame()

    rows: list[VirulenceAssoc] = []
    for outcome, kind in outcome_kinds.items():
        if outcome not in phenotypes.columns:
            continue
        for g in cols:
            d = pd.concat(
                [phenotypes[outcome].rename("y"), gene_pa[g].rename("x"), covariates], axis=1
            ).dropna()
            if d["x"].nunique() < 2 or len(d) < 30:
                continue
            X = sm.add_constant(d[["x", *covariates.columns]], has_constant="add")
            try:
                res = (
                    sm.Logit(d["y"].astype(int), X).fit(disp=0, method="lbfgs", maxiter=200)
                    if kind == "binary"
                    else sm.OLS(d["y"].astype(float), X).fit()
                )
            except Exception as e:
                log.debug("vir fit failed: %s", e)
                continue
            rows.append(
                VirulenceAssoc(
                    species=species, gene=g, outcome=outcome,
                    beta=float(res.params["x"]), pvalue=float(res.pvalues["x"]),
                    n_carriers=int(d["x"].sum()), n=int(len(d)),
                )
            )
    df = pd.DataFrame([r.__dict__ for r in rows])
    if not df.empty:
        df["q"] = bh_qvalues(df["pvalue"].values)
    return df


def pathway_associations(
    pathway_abund: pd.DataFrame,
    phenotypes: pd.DataFrame,
    covariates: pd.DataFrame,
    outcome_kinds: dict[str, str],
) -> pd.DataFrame:
    """Generic HUMAnN3 pathway abundance vs outcomes."""
    rows = []
    for outcome, kind in outcome_kinds.items():
        if outcome not in phenotypes.columns:
            continue
        for pw in pathway_abund.columns:
            d = pd.concat(
                [phenotypes[outcome].rename("y"), pathway_abund[pw].rename("x"), covariates],
                axis=1,
            ).dropna()
            if len(d) < 50:
                continue
            X = sm.add_constant(d[["x", *covariates.columns]], has_constant="add")
            try:
                res = (
                    sm.Logit(d["y"].astype(int), X).fit(disp=0, method="lbfgs", maxiter=200)
                    if kind == "binary"
                    else sm.OLS(d["y"].astype(float), X).fit()
                )
            except Exception:
                continue
            rows.append(
                dict(pathway=pw, outcome=outcome, beta=float(res.params["x"]),
                     pvalue=float(res.pvalues["x"]), n=int(len(d)))
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q"] = bh_qvalues(df["pvalue"].values)
    return df
