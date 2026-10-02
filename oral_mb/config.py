"""Paths, constants, column groups."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/oral_microbiome/paper_dfs"
)
OUT_DIR = Path("/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/oral_mb")

MICROBIOME_LEVELS: tuple[str, ...] = ("species", "genus", "family", "phylum", "pathways")

SYMPTOMS_BINARY: tuple[str, ...] = (
    "abscess_or_gingivitis",
    "bleeding_gums",
    "bad_breath_lose",
    "bad_breath_strict",
    "mouth_sores",
    "dentures",
    "gum_pain",
    "tooth_ache",
)

LIFESTYLE_BINARY: tuple[str, ...] = ("keto", "is_vegetarian", "is_fasting", "smoke_tobacco_now")
LIFESTYLE_CONT: tuple[str, ...] = ("med_score_per_day", "UPF_score", "vegetarian_score_per_day")

COVARS_BASE: tuple[str, ...] = ("age", "gender")
COVARS_FULL: tuple[str, ...] = ("age", "gender", "bmi")

# MetaPhlAn4 default pseudocount sentinel value (data already log10-transformed).
ABSENT_LOG10 = -4.0

# Statistical thresholds.
FDR_ALPHA = 0.05
PREVALENCE_MIN = 0.05  # 5%
SEED = 42


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration for a single analysis."""

    level: str = "species"
    n_folds: int = 5
    n_boot: int = 500
    use_bmi: bool = False
    sex_stratified: bool = False
    transform: str = "clr"  # "clr" | "log10" | "raw"
    prevalence_min: float = PREVALENCE_MIN
    seed: int = SEED
    out_dir: Path = field(default_factory=lambda: OUT_DIR)
