#!/usr/bin/env python
"""Dot-and-whisker figure: oral vs gut microbiome disease prediction.

Two stacked panels (all diagnoses, then GI/ICD-11 chapter 13 only) drawn with
the same primitive as Fig. 3d (``plots._level_points_panel``), so dodge,
whiskers (SD across CV folds), grid, chance line and the colourblind palette
keyed by ``SOURCE_ORDER`` are identical to the other panel figures.

Only ``oral`` and ``gut`` are shown: covariate-only and the oral+gut combo are
dropped so the two panels answer one question, can the cheap buccal swab stand
in for stool.

Input and output default to ``config.OUT_DIR``, so the usual run is bare:

    python scripts/plot_disease_prediction.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from oral_mb.config import OUT_DIR
from oral_mb.plots import SOURCE_ORDER, _level_points_panel, _levels_in

log = logging.getLogger("plot_disease_prediction")

DEFAULT_CSV = OUT_DIR / "disease_prediction.csv"
DEFAULT_OUT = OUT_DIR / "disease_prediction_oral_vs_gut.png"

KEEP_SOURCES = ("oral", "gut")

# ICD-11 code -> readable label. Unmapped codes fall back to the raw code.
ICD_NAMES = {
    "3A00": "Iron deficiency anaemia", "5A00": "Hypothyroidism",
    "5A40": "Hypopituitarism", "5A80.1": "Type 2 diabetes",
    "5B5F": "Lipoprotein disorder", "5B81": "Obesity",
    "5C80": "Hyperlipidaemia", "7A41": "Insomnia", "8A80": "Migraine",
    "9B10": "Cataract", "AB31": "Hearing loss", "AB51": "Tinnitus",
    "BA00": "Hypertension", "CA08.0": "Allergic rhinitis",
    "CA0A": "Sleep apnoea", "CA23": "Asthma", "DA01.10": "Gingivitis",
    "DA42.1": "GORD", "DB50.0": "Haemorrhoids",
    "DB60": "Diverticular disease", "DB92": "Constipation",
    "DC11": "Fatty liver", "DD91.0": "IBS", "EA80": "Acne",
    "EA90": "Psoriasis", "ED80": "Alopecia", "FB1Y": "Joint disorder",
    "FB83": "Osteoporosis", "FB83.1": "Osteopenia",
    "GA31": "Menopausal disorder", "GC08.Z": "Urinary disorder",
    "ME84": "Abnormal test", "ND56.2": "Injury",
}


def _label(code: str) -> str:
    name = ICD_NAMES.get(str(code))
    return f"{name} ({code})" if name else str(code)


def _order_by_oral(df: pd.DataFrame) -> list[str]:
    oral = df[df["source"] == "oral"].set_index("outcome")["auc"]
    return list(oral.sort_values(ascending=False).index)


def make_figure(res: pd.DataFrame, out: Path, model: str, top_n: int,
                suptitle: str, dodge_span: float = 0.22) -> None:
    df = res[res["model"] == model].copy()
    if df.empty:
        raise ValueError(f"no rows for model={model!r}")
    df = df[df["source"].isin(KEEP_SOURCES)].dropna(subset=["auc"])
    df["outcome"] = [_label(c) for c in df["disease"]]

    order_all = _order_by_oral(df)[:top_n]
    gi = df[df["gi"].astype(bool)]
    order_gi = _order_by_oral(gi)

    # Row order inside each panel drives the x order (dict.fromkeys upstream).
    top = pd.concat([df[df["outcome"] == o] for o in order_all])
    gi = pd.concat([gi[gi["outcome"] == o] for o in order_gi]) if order_gi else gi

    levels = _levels_in(df, "source")
    # Same palette recipe as Fig. 3a/3c/3d: colourblind, keyed by SOURCE_ORDER.
    full = [s for s in SOURCE_ORDER if s in set(res["source"])]
    colors = dict(zip(full, sns.color_palette("colorblind", len(full))))
    palette = {lv: colors[lv] for lv in levels}

    n_out = max(len(order_all), max(len(order_gi), 1))
    fig, axes = plt.subplots(2, 1, figsize=(max(5.5, 0.72 * n_out + 1.6), 7.2))
    for ax, d, title in (
        (axes[0], top, f"All diagnoses (top {len(order_all)} by oral AUC)"),
        (axes[1], gi, "GI diagnoses (ICD-11 chapter 13)"),
    ):
        _level_points_panel(ax, d, "auc", levels, palette, "source",
                            dodge_span=dodge_span)
        ax.set_title(title, fontsize=9)
    fig.suptitle(suptitle, fontsize=10.5)
    fig.tight_layout(h_pad=2.4)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False,
               ncol=len(levels), loc="upper center", bbox_to_anchor=(0.5, -0.02))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default="ensemble")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--dodge-span", type=float, default=0.22)
    p.add_argument("--suptitle",
                   default="Gut microbiome outpredicts the oral microbiome "
                           "across diagnoses")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    make_figure(pd.read_csv(args.csv), args.out, args.model, args.top_n,
                args.suptitle, args.dodge_span)


if __name__ == "__main__":
    main()
