"""End-to-end orchestration: discovery + sensitivity + prediction + mediation + plots.

Usage:
    python -m scripts.run_pipeline --level species --out /path/to/out
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from oral_mb.associations import associations, shared_signature
from oral_mb.config import (
    DATA_DIR, LIFESTYLE_BINARY, LIFESTYLE_CONT, MICROBIOME_LEVELS, OUT_DIR,
    SYMPTOMS_BINARY,
)
from oral_mb.data import cohort_table1, load_cohort, outcome_meta
from oral_mb.mediation import run_mediation
from oral_mb.plots import (
    calibration_plot, clustermap_shared, heatmap_shared, mediation_forest_plot,
    prediction_bars, prediction_points_by_level_combined, upf_smoking_beta_scatter,
)
from oral_mb.prediction import (
    PredictionConfig, SplitConfig, cv_predict_one, run_all_outcomes,
)
from oral_mb.sensitivity import sensitivity_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_pipeline")


def split_symptom_lifestyle(assoc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if assoc.empty or "outcome" not in assoc.columns:
        return assoc.copy(), assoc.copy()
    sym_set = set(SYMPTOMS_BINARY)
    return (
        assoc[assoc["outcome"].isin(sym_set)].copy(),
        assoc[~assoc["outcome"].isin(sym_set)].copy(),
    )


def run_level(level: str, out_root: Path, data_dir: Path, mediate: bool,
              use_gut: bool = False, ensemble_models: bool = False) -> None:
    out = out_root / level
    out.mkdir(parents=True, exist_ok=True)
    log.info("=== level=%s ===", level)

    cohort = load_cohort(level=level, data_dir=data_dir)
    # Nastya parity: no prevalence filter.

    log.info("associations (raw log10, Nastya parity)")
    assoc = associations(cohort, feature_kind="log10", use_bmi=False)
    assoc.to_csv(out / "associations_all.csv", index=False)
    sym_a, life_a = split_symptom_lifestyle(assoc)
    sym_a.to_csv(out / "associations_symptoms.csv", index=False)
    life_a.to_csv(out / "associations_lifestyle.csv", index=False)

    if assoc.empty:
        log.warning("no associations returned at level=%s; skipping shared+heatmap", level)
        shared = pd.DataFrame(columns=["feature"])
    else:
        shared = shared_signature(sym_a, life_a)
    shared.to_csv(out / "shared_signature.csv", index=False)
    if len(shared):
        heatmap_shared(sym_a, life_a, shared["feature"].tolist(), out / "heatmap_shared.png")
        clustermap_shared(sym_a, life_a, shared["feature"].tolist(), out / "clustermap_shared.png")
    if not life_a.empty:
        upf_smoking_beta_scatter(
            life_a, shared["feature"].tolist() if len(shared) else None,
            out / "upf_vs_smoking_beta.png",
        )

    log.info("table 1: cohort characteristics")
    cohort_table1(cohort).to_csv(out / "table1_cohort.csv")

    log.info("sensitivity")
    sb = sensitivity_bundle(cohort)
    for name, df in sb.items():
        df.to_csv(out / f"sensitivity_{name}.csv", index=False)

    log.info("prediction (lgbm, leakage-safe with held-out test)")
    pred_lgbm = run_all_outcomes(cohort, model="lgbm")
    pred_lgbm.to_csv(out / "prediction_lgbm.csv", index=False)

    if ensemble_models or use_gut:
        log.info("prediction (ridge+ols+lgbm ensemble; gut=%s)", use_gut)
        gut_df_map = {"species": "mpa_species", "genus": "mpa_genus",
                      "family": "mpa_family"}
        gut_kwargs = {
            "df": gut_df_map.get(level, "mpa_species"),
            "study_ids": [10],
            "min_col_present_frac": 0.2,
            "groupby_reg": "first",
        }
        cfg = PredictionConfig(
            models=("lgbm", "ridge", "ols"),
            ensemble=True,
            use_gut=use_gut,
            gut_loader_kwargs=gut_kwargs,
            split=SplitConfig(test_frac=0.2, n_folds=5, seed=42),
        )
        pred_full = run_all_outcomes(cohort, cfg=cfg)
        tag = "with_gut" if use_gut else "ensemble"
        pred_full.to_csv(out / f"prediction_{tag}.csv", index=False)
    else:
        pred_full = None

    if "auc" in pred_lgbm.columns:
        prediction_bars(pred_lgbm.dropna(subset=["auc"]), "auc", out / "auc_bars.png")
    if "pearson" in pred_lgbm.columns:
        prediction_bars(pred_lgbm.dropna(subset=["pearson"]), "pearson", out / "pearson_bars.png")

    if pred_full is not None:
        # Fig. 3a+3c: buccal vs. gut (source), ensemble model only. The
        # separate by-model breakdown was dropped (four near-identical
        # model colours per outcome added no information beyond the
        # source comparison the text actually makes); binary and
        # continuous outcomes are combined into one two-panel figure.
        ens = pred_full[pred_full["model"] == "ensemble"]
        # oral+gut is computed (for the record) but not plotted: it never beats
        # oral alone, so showing it would suggest combining sources helps.
        ens_plot = ens[ens["source"] != "oral+gut"] if "source" in ens.columns else ens
        auc_df = ens_plot[ens_plot["auc"].notna()] if "auc" in ens_plot.columns else pd.DataFrame()
        pearson_df = ens_plot[ens_plot["pearson"].notna()] if "pearson" in ens_plot.columns else pd.DataFrame()
        if not auc_df.empty or not pearson_df.empty:
            prediction_points_by_level_combined(
                ens_plot, out / "fig3ac_source_combined.png", level="source",
                caption=("Points, mean; whiskers, 95% CI. Ensemble model. Combining "
                         "oral and gut features (oral+gut) did not improve prediction "
                         "over oral alone and is omitted."),
                title="Buccal dominates gut for symptom and diet prediction",
                dodge_span=0.16, outcome_width=0.75, stacked=False,
            )

    # Calibration: bleeding gums.
    try:
        r = cv_predict_one(cohort, "bleeding_gums", model="lgbm")
        import numpy as np
        y_true = np.concatenate([f.y_true for f in r.folds])
        y_pred = np.concatenate([f.y_pred for f in r.folds])
        calibration_plot(y_true, y_pred, out / "calibration_bleeding_gums.png")
    except Exception as e:
        log.warning("calibration skipped: %s", e)

    if mediate and len(shared):
        log.info("mediation: smoking + UPF + Med -> top shared mediators -> bleeding_gums")
        top_med = shared["feature"].head(10).tolist()
        med = run_mediation(
            cohort,
            exposures=["smoke_tobacco_now", "UPF_score", "med_score_per_day"],
            outcomes=["bleeding_gums", "abscess_or_gingivitis"],
            mediators=top_med,
            outcome_meta=outcome_meta(),
            n_boot=200,
        )
        med.to_csv(out / "mediation.csv", index=False)
        mediation_forest_plot(med, out=out / "fig4_mediation_forest.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=list(MICROBIOME_LEVELS))
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--mediate", action="store_true", default=True)
    ap.add_argument("--use-gut", action="store_true", default=True,
                    help="Also train gut-microbiome models and oral+gut ensemble")
    ap.add_argument("--ensemble", action="store_true", default=True,
                    help="Also train ridge/ols/lgbm ensemble (leakage-safe)")
    ap.add_argument("--all", action="store_true", default=True,
                    help="Deepest analysis: all levels + ensemble + gut + mediation (default)")
    args = ap.parse_args()

    if args.all:
        args.levels = list(MICROBIOME_LEVELS)
        args.mediate = True
        args.use_gut = True
        args.ensemble = True
        log.info("--all: levels=%s, mediate=True, use_gut=True, ensemble=True",
                 args.levels)

    args.out.mkdir(parents=True, exist_ok=True)
    for lvl in args.levels:
        run_level(lvl, args.out, args.data_dir, args.mediate,
                  use_gut=args.use_gut, ensemble_models=args.ensemble)


if __name__ == "__main__":
    main()
