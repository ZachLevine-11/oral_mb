"""Data loading + transforms (CLR, prevalence filter, joins)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ABSENT_LOG10,
    COVARS_BASE,
    COVARS_FULL,
    DATA_DIR,
    LIFESTYLE_BINARY,
    LIFESTYLE_CONT,
    MICROBIOME_LEVELS,
    PREVALENCE_MIN,
    SYMPTOMS_BINARY,
)

log = logging.getLogger(__name__)

INDEX_COL_CANDIDATES = ("RegistrationCode", "Unnamed: 0")


@dataclass(frozen=True)
class Cohort:
    """Aligned cohort: microbiome features + phenotypes."""

    microbiome: pd.DataFrame  # samples x features
    symptoms: pd.DataFrame
    lifestyle: pd.DataFrame
    covariates: pd.DataFrame

    @property
    def samples(self) -> pd.Index:
        return self.microbiome.index

    def aligned(self) -> "Cohort":
        idx = (
            self.microbiome.index.intersection(self.symptoms.index)
            .intersection(self.lifestyle.index)
            .intersection(self.covariates.index)
        )
        log.info("aligned cohort n=%d", len(idx))
        return Cohort(
            microbiome=self.microbiome.loc[idx].copy(),
            symptoms=self.symptoms.loc[idx].copy(),
            lifestyle=self.lifestyle.loc[idx].copy(),
            covariates=self.covariates.loc[idx].copy(),
        )


def _read_indexed(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in INDEX_COL_CANDIDATES:
        if c in df.columns:
            df = df.set_index(c)
            break
    df.index.name = "RegistrationCode"
    return df


def load_microbiome(level: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    if level not in MICROBIOME_LEVELS:
        raise ValueError(f"unknown level {level!r}; choose {MICROBIOME_LEVELS}")
    return _read_indexed(data_dir / f"{level}.parquet")


def load_phenotypes(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sym = _read_indexed(data_dir / "symptoms.parquet")
    life = _read_indexed(data_dir / "lifestyle.parquet")
    cov = _read_indexed(data_dir / "covariates.parquet")
    return sym, life, cov


def load_cohort(level: str = "species", data_dir: Path = DATA_DIR) -> Cohort:
    micro = load_microbiome(level, data_dir)
    sym, life, cov = load_phenotypes(data_dir)
    missing_life = [c for c in (*LIFESTYLE_BINARY, *LIFESTYLE_CONT) if c not in life.columns]
    if missing_life:
        log.warning("lifestyle parquet missing expected columns: %s", missing_life)
    missing_sym = [c for c in SYMPTOMS_BINARY if c not in sym.columns]
    if missing_sym:
        log.warning("symptoms parquet missing expected columns: %s", missing_sym)
    # Binarize smoke_tobacco_now to match Nastya original code: threshold > 0.7.
    if "smoke_tobacco_now" in life.columns:
        life = life.copy()
        life["smoke_tobacco_now"] = (life["smoke_tobacco_now"] > 0.7).astype(float)
    cohort = Cohort(microbiome=micro, symptoms=sym, lifestyle=life, covariates=cov).aligned()
    for c in (*LIFESTYLE_BINARY, *LIFESTYLE_CONT):
        if c in cohort.lifestyle.columns:
            nn = int(cohort.lifestyle[c].notna().sum())
            log.info("lifestyle %-26s n_nonnull=%d", c, nn)
        else:
            log.warning("lifestyle %-26s ABSENT", c)
    return cohort


def prevalence_filter(
    micro: pd.DataFrame, min_prev: float = PREVALENCE_MIN, absent: float = ABSENT_LOG10
) -> pd.DataFrame:
    """Keep features present in >= min_prev fraction of samples."""
    present = (micro > absent).mean(axis=0)
    keep = present[present >= min_prev].index
    log.info("prevalence filter %.2f: %d/%d kept", min_prev, len(keep), micro.shape[1])
    return micro.loc[:, keep]


def to_relative(micro_log10: pd.DataFrame, absent: float = ABSENT_LOG10) -> pd.DataFrame:
    """Invert provided log10 transform to relative abundances; absent -> 0."""
    arr = np.where(micro_log10.values <= absent, 0.0, np.power(10.0, micro_log10.values))
    return pd.DataFrame(arr, index=micro_log10.index, columns=micro_log10.columns)


def clr_transform(rel: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """Centered log-ratio transform on relative-abundance matrix."""
    x = rel.values + pseudocount
    log_x = np.log(x)
    gm = log_x.mean(axis=1, keepdims=True)
    return pd.DataFrame(log_x - gm, index=rel.index, columns=rel.columns)


def transform_features(micro_log10: pd.DataFrame, kind: str = "clr") -> pd.DataFrame:
    if kind == "raw":
        return micro_log10.copy()
    if kind == "log10":
        # Already log10 from MetaPhlAn pipeline.
        return micro_log10.copy()
    if kind == "clr":
        return clr_transform(to_relative(micro_log10))
    raise ValueError(f"unknown transform {kind!r}")


def build_design_matrix(
    cohort: Cohort, use_bmi: bool = False, features: pd.DataFrame | None = None
) -> pd.DataFrame:
    cols = list(COVARS_FULL if use_bmi else COVARS_BASE)
    out = cohort.covariates[cols].copy()
    if features is not None:
        out = out.join(features, how="inner")
    return out


def stack_outcomes(cohort: Cohort) -> pd.DataFrame:
    """All symptom + lifestyle outcomes in one frame."""
    return cohort.symptoms.join(cohort.lifestyle, how="outer")


# Table 1 display config: (group, pretty label, kind). kind "cont" -> mean (s.d.);
# kind "bin" -> count (%). Column lookup order: covariates, lifestyle, symptoms.
TABLE1_SPEC: tuple[tuple[str, str, str], ...] = (
    ("Demographics", "age", "cont"),
    ("Demographics", "bmi", "cont"),
    ("Oral health symptoms", "bleeding_gums", "bin"),
    ("Oral health symptoms", "bad_breath_lose", "bin"),
    ("Oral health symptoms", "bad_breath_strict", "bin"),
    ("Oral health symptoms", "abscess_or_gingivitis", "bin"),
    ("Oral health symptoms", "mouth_sores", "bin"),
    ("Oral health symptoms", "gum_pain", "bin"),
    ("Oral health symptoms", "tooth_ache", "bin"),
    ("Oral health symptoms", "dentures", "bin"),
    ("Lifestyle | smoking", "smoke_tobacco_now", "bin"),
    ("Lifestyle | diet", "is_vegetarian", "bin"),
    ("Lifestyle | diet", "keto", "bin"),
    ("Lifestyle | diet", "is_fasting", "bin"),
    ("Lifestyle | diet", "med_score_per_day", "cont"),
    ("Lifestyle | diet", "UPF_score", "cont"),
    ("Lifestyle | diet", "vegetarian_score_per_day", "cont"),
)

TABLE1_LABELS: dict[str, str] = {
    "age": "Age (years)",
    "bmi": "BMI (kg/m2)",
    "bleeding_gums": "Bleeding gums",
    "bad_breath_lose": "Bad breath (loose)",
    "bad_breath_strict": "Bad breath (strict)",
    "abscess_or_gingivitis": "Abscess or gingivitis",
    "mouth_sores": "Mouth sores",
    "gum_pain": "Gum pain",
    "tooth_ache": "Tooth ache",
    "dentures": "Dentures",
    "smoke_tobacco_now": "Current smoking",
    "is_vegetarian": "Vegetarian status",
    "keto": "Ketogenic diet",
    "is_fasting": "Intermittent fasting",
    "med_score_per_day": "Mediterranean score (per day)",
    "UPF_score": "UPF score",
    "vegetarian_score_per_day": "Vegetarian score (per day)",
}


def cohort_table1(cohort: Cohort, sex_col: str = "gender") -> pd.DataFrame:
    """Table 1: cohort characteristics, grouped by variable domain and
    labelled with human-readable names (not raw lower_case column names),
    split by sex. Mean (s.d.) for continuous rows, count (%) for binary."""
    extra_covars = [c for c in ("age", "bmi") if c in cohort.covariates.columns]
    wide = (
        cohort.covariates[[sex_col]]
        .join(cohort.symptoms, how="left")
        .join(cohort.lifestyle, how="left")
        .join(cohort.covariates[extra_covars], how="left")
    )
    sexes = sorted(wide[sex_col].dropna().unique().tolist())
    n_row = {"group": "", "variable": "N"}
    for sx in [*sexes, "overall"]:
        n_row[str(sx)] = str(len(wide) if sx == "overall" else int((wide[sex_col] == sx).sum()))
    rows = [n_row]
    for group, col, kind in TABLE1_SPEC:
        if col not in wide.columns:
            continue
        row = {"group": group, "variable": TABLE1_LABELS.get(col, col)}
        for sx in [*sexes, "overall"]:
            sub = wide[col] if sx == "overall" else wide.loc[wide[sex_col] == sx, col]
            sub = sub.dropna()
            if kind == "cont":
                row[str(sx)] = f"{sub.mean():.1f} ({sub.std():.1f})"
            else:
                n = int(sub.sum())
                pct = 100 * sub.mean() if len(sub) else float("nan")
                row[str(sx)] = f"{n} ({pct:.1f}%)"
        rows.append(row)
    out = pd.DataFrame(rows).set_index(["group", "variable"])
    return out


def outcome_meta() -> dict[str, str]:
    """Return outcome -> 'binary'|'continuous'."""
    return {
        **{c: "binary" for c in SYMPTOMS_BINARY},
        **{c: "binary" for c in LIFESTYLE_BINARY},
        **{c: "continuous" for c in LIFESTYLE_CONT},
    }
