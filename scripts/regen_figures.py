"""Redraw figures from an already-completed run. No analysis is re-run.

Reads the CSVs written by scripts.run_pipeline / scripts.run_revision and
rewrites only the figure files.

Usage:
    python -m scripts.regen_figures --run-dir /net/.../runs
    python -m scripts.regen_figures --run-dir ... --level species
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from oral_mb.config import MICROBIOME_LEVELS, OUT_DIR
from oral_mb.plots import (
    heatmap_shared, mediation_counterfactual_forest_plot, missingness_panel,
    prediction_bars, prediction_points_by_level, prediction_points_by_level_combined,
    prediction_stability_panel,
)
from oral_mb.sankey import mediation_sankey

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("regen_figures")


def regen_prediction(level_dir: Path, level: str, fig_dir: Path) -> None:
    """Per-outcome point estimate + CI, one panel per metric."""
    src = level_dir / "prediction_lgbm.csv"
    if not src.exists():
        log.warning("skip %s (no prediction_lgbm.csv)", level_dir)
        return
    pred = pd.read_csv(src)
    for metric in ("auc", "pearson"):
        if metric not in pred.columns:
            continue
        sub = pred.dropna(subset=[metric])
        if not len(sub):
            continue
        for out in (level_dir / f"{metric}_bars.png",
                    fig_dir / f"{metric}_points_{level}.png"):
            prediction_bars(sub, metric, out,
                            caption="Points, mean; whiskers, 95% CI.")
            log.info("wrote %s", out)


def regen_heatmap(level_dir: Path, level: str, fig_dir: Path) -> None:
    """Shared-signature heatmap with clustered phenotype columns + clean labels."""
    sym = level_dir / "associations_symptoms.csv"
    life = level_dir / "associations_lifestyle.csv"
    shared_p = level_dir / "shared_signature.csv"
    if not (sym.exists() and life.exists() and shared_p.exists()):
        log.warning("skip heatmap in %s (missing association/shared CSVs)", level_dir)
        return
    shared = pd.read_csv(shared_p)
    if "feature" not in shared.columns or shared.empty:
        log.warning("skip heatmap in %s (empty shared signature)", level_dir)
        return
    sym_df, life_df = pd.read_csv(sym), pd.read_csv(life)
    for out in (level_dir / "heatmap_shared.png",
                fig_dir / f"heatmap_shared_{level}.png"):
        heatmap_shared(sym_df, life_df, shared["feature"].tolist(), out,
                       cluster_columns=True, show_dendrogram=False,
                       title="Shared signature across symptom and lifestyle phenotypes")
        log.info("wrote %s", out)


def regen_sankey(level_dir: Path, level: str, fig_dir: Path) -> None:
    """Mediation Sankey (exposure -> species -> outcome) from mediation.csv."""
    med_p = level_dir / "mediation.csv"
    if not med_p.exists():
        log.warning("skip sankey in %s (no mediation.csv)", level_dir)
        return
    med = pd.read_csv(med_p)
    for out in (level_dir / "fig_mediation_sankey.png",
                fig_dir / f"fig_mediation_sankey_{level}.png"):
        try:
            mediation_sankey(med, out)
        except ValueError as err:
            log.warning("skip sankey in %s (%s)", level_dir, err)
            return
        log.info("wrote %s", out)


def regen_by_source(level_dir: Path, level: str, fig_dir: Path) -> None:
    """Fig. 3a+3c in the Fig. 3d idiom: point-and-CI per outcome, one colour
    per feature source (buccal, gut, both), ensemble model only."""
    src = None
    for name in ("prediction_with_gut.csv", "prediction_ensemble.csv"):
        if (level_dir / name).exists():
            src = level_dir / name
            break
    if src is None:
        log.warning("skip source figure in %s (no prediction_with_gut.csv)", level_dir)
        return
    pred = pd.read_csv(src)
    if "source" not in pred.columns:
        log.warning("skip source figure in %s (no source column)", level_dir)
        return
    if "model" in pred.columns and (pred["model"] == "ensemble").any():
        pred = pred[pred["model"] == "ensemble"]
    # oral+gut is computed (for the record) but not plotted: it never beats
    # oral alone, so showing it would suggest combining sources helps.
    pred_plot = pred[pred["source"] != "oral+gut"]
    for out in (level_dir / "fig3ac_source_combined.png",
                fig_dir / f"fig3ac_by_source_{level}.png"):
        prediction_points_by_level_combined(
            pred_plot, out, level="source",
            caption=("Points, mean; whiskers, 95% CI. Ensemble model. Combining "
                     "oral and gut features (oral+gut) did not improve prediction "
                     "over oral alone and is omitted."),
            title="Buccal dominates gut for symptom and diet prediction",
            dodge_span=0.16, outcome_width=0.75, stacked=False,
        )
        log.info("wrote %s", out)


def regen_by_level(pipeline: Path, levels: list[str]) -> None:
    """Fig. 3d: stack each level's prediction_lgbm.csv into one dodged panel."""
    frames = []
    for level in levels:
        src = pipeline / level / "prediction_lgbm.csv"
        if not src.exists():
            log.warning("skip level %s (no prediction_lgbm.csv)", level)
            continue
        frames.append(pd.read_csv(src).assign(level=level))
    if not frames:
        log.warning("no per-level prediction files under %s", pipeline)
        return
    stacked = pd.concat(frames, ignore_index=True)
    caption = "Points, mean; whiskers, 95% CI."
    combined = pipeline / "fig3d_by_level.png"
    prediction_points_by_level_combined(stacked, combined, caption=caption)
    log.info("wrote %s", combined)
    # Single-metric versions kept for slide/supplement use.
    for metric, name in (("auc", "fig3d_auc_by_level.png"),
                         ("pearson", "fig3d_pearson_by_level.png")):
        if metric not in stacked.columns or stacked[metric].isna().all():
            continue
        out = pipeline / name
        prediction_points_by_level(stacked, metric, out, caption=caption)
        log.info("wrote %s", out)


def regen_stability(rev_root: Path, fig_dir: Path) -> None:
    """Fig. 8: AUPRC-vs-prevalence + repeated-CV spread, from the revision
    prediction stage."""
    for pred_dir in (rev_root / "prediction", rev_root):
        auprc = pred_dir / "prediction_auprc_prevalence.csv"
        cv = pred_dir / "prediction_repeated_cv.csv"
        if auprc.exists() and cv.exists():
            break
    else:
        log.warning("skip stability figure (no prediction CSVs under %s)", rev_root)
        return
    auprc_df, cv_df = pd.read_csv(auprc), pd.read_csv(cv)
    for out in (pred_dir / "fig8_prediction_stability.png",
                fig_dir / "fig8_prediction_stability.png"):
        prediction_stability_panel(
            auprc_df, cv_df, out,
            caption="Dashed line, no-skill AUPRC = prevalence; dotted line, chance (0.5).",
        )
        log.info("wrote %s", out)


def regen_mediation_counterfactual(rev_root: Path, fig_dir: Path) -> None:
    """Fig. 7: g-formula NIE forest plot, one panel per exposure, points
    coloured by E-value. Source data is stage_mediation's
    mediation_counterfactual.csv (never plotted anywhere else in the repo)."""
    for med_dir in (rev_root / "mediation", rev_root):
        src = med_dir / "mediation_counterfactual.csv"
        if src.exists():
            break
    else:
        log.warning("skip mediation figure (no mediation_counterfactual.csv under %s)", rev_root)
        return
    med = pd.read_csv(src)
    for out in (med_dir / "fig7_mediation_counterfactual.png",
                fig_dir / "fig7_mediation_counterfactual.png"):
        mediation_counterfactual_forest_plot(
            med, out,
            caption=("Points, natural indirect effect (g-formula); whiskers, 95% bootstrap CI. "
                     "Colour, E-value for the mediator→outcome arm (higher = more robust to "
                     "unmeasured confounding). NaN E-values (continuous outcomes) shown in grey."),
        )
        log.info("wrote %s", out)


def regen_missingness(rev_dir: Path, fig_dir: Path,
                      headline: str = "bleeding_gums") -> None:
    """Non-response panel; PERMANOVA moved from inset legend to caption."""
    shift = rev_dir / f"missingness_ipw_beta_shift_{headline}.csv"
    counts = rev_dir / "missingness_counts.csv"
    if not (shift.exists() and counts.exists()):
        log.warning("skip missingness (missing %s or %s)", shift.name, counts.name)
        return
    perm_path = rev_dir / f"missingness_permanova_{headline}.csv"
    # Title carries the claim; caption carries the supporting PERMANOVA stats.
    caption = None
    if perm_path.exists():
        perm = pd.read_csv(perm_path, index_col=0).iloc[:, 0]
        caption = (
            f"PERMANOVA (Aitchison): pseudo-F = {float(perm['pseudo_F']):.2f}, "
            f"P = {float(perm['p_value']):.2f}, R² = {float(perm['R2']):.1e}."
        )
    shift_df, counts_df = pd.read_csv(shift), pd.read_csv(counts)
    for out in (rev_dir / f"fig_missingness_{headline}.png",
                fig_dir / f"fig_missingness_{headline}.png"):
        missingness_panel(shift_df, counts_df, out, headline=headline, caption=caption)
        log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=OUT_DIR,
                    help="Base dir passed to run_all (holds pipeline/ and revision/).")
    ap.add_argument("--levels", nargs="+", default=list(MICROBIOME_LEVELS))
    ap.add_argument("--headline", default="bleeding_gums")
    args = ap.parse_args()

    pipeline = args.run_dir / "pipeline"
    for level in args.levels:
        level_dir = pipeline / level
        if level_dir.is_dir():
            regen_prediction(level_dir, level, pipeline)
            regen_heatmap(level_dir, level, pipeline)
            regen_sankey(level_dir, level, pipeline)
            regen_by_source(level_dir, level, pipeline)
        else:
            log.warning("no level dir %s", level_dir)
    regen_by_level(pipeline, list(args.levels))

    # run_revision writes each stage into <out>/<stage>/; older runs put the
    # CSVs directly under <out>/, so accept both.
    revision = args.run_dir / "revision"
    for rev_dir in (revision / "missingness", revision):
        if rev_dir.is_dir() and (rev_dir / "missingness_counts.csv").exists():
            regen_missingness(rev_dir, pipeline, args.headline)
            break
    else:
        log.warning("no missingness CSVs under %s", revision)
    if revision.is_dir():
        regen_stability(revision, pipeline)
        regen_mediation_counterfactual(revision, pipeline)
    log.info("regen_figures done -> %s", args.run_dir)


if __name__ == "__main__":
    main()
