"""Mediation Sankey: exposure -> mediating species -> outcome ribbons.

Ribbon width is |a x b| (the bootstrap indirect effect), node height is the
sum of its ribbons, colour is the sign of a x b and a dashed outline marks
sign-reversing paths (indirect effect opposes the total effect c). Layout is
in pixel units of a 990 x 655 canvas so the figure matches the paper panel.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

EXPOSURE_ORDER: tuple[str, ...] = ("smoke_tobacco_now", "med_score_per_day", "UPF_score")
OUTCOME_ORDER: tuple[str, ...] = ("bleeding_gums", "abscess_or_gingivitis")
NODE_LABELS: dict[str, str] = {
    "smoke_tobacco_now": "Smoking",
    "med_score_per_day": "Mediterranean diet",
    "UPF_score": "Ultra-processed food",
    "bleeding_gums": "Bleeding gums",
    "abscess_or_gingivitis": "Abscess / gingivitis",
}

RAISE_COLOR = "#D0664A"
LOWER_COLOR = "#52A683"
RIBBON_ALPHA = 0.7
REVERSE_TEXT_COLOR = "#B5452F"
NODE_FACE, NODE_EDGE = "#E6E3DC", "#9A9A9A"
HEADER_COLOR = "#555555"

# Canvas geometry (px). Columns: node left edges; ribbons run between them.
CANVAS_W, CANVAS_H = 1020, 655
NODE_W = 24
COL_X = (188, 516, 825)
TOP, SPAN, GAP = 73, 494, 32
HEADER_Y = 35
LABEL_X = (COL_X[1] + NODE_W + COL_X[2]) / 2


def short_species(name: str) -> str:
    """s__Abiotrophia_defectiva -> A. defectiva."""
    parts = name.removeprefix("s__").split("_")
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else parts[0]


def _paths(med: pd.DataFrame, q_max: float) -> pd.DataFrame:
    df = med[med["q_indirect"] < q_max].copy()
    df["w"] = df["indirect"].abs()
    df["pct"] = 100 * df["indirect"] / df["c_total"]
    df["e_rank"] = df["exposure"].map({e: i for i, e in enumerate(EXPOSURE_ORDER)})
    df["o_rank"] = df["outcome"].map({o: i for i, o in enumerate(OUTCOME_ORDER)})
    first_out = df.groupby("mediator")["o_rank"].min()
    mediators = sorted(first_out.index, key=lambda m: (first_out[m], m))
    df["m_rank"] = df["mediator"].map({m: i for i, m in enumerate(mediators)})
    return df


def _stack(df: pd.DataFrame, node_col: str, order_cols: list[str], scale: float):
    """Node (top, height) per node and per-path (y0, y1) slots in that column."""
    sums = df.groupby(node_col)["w"].sum()
    rank = df.groupby(node_col)[f"{node_col[0]}_rank"].first().sort_values()
    total = sums.sum() * scale + GAP * (len(sums) - 1)
    y = TOP + (SPAN - total) / 2
    nodes, slots = {}, {}
    for node in rank.index:
        nodes[node] = (y, sums[node] * scale)
        yy = y
        for idx, row in df[df[node_col] == node].sort_values(order_cols).iterrows():
            slots[idx] = (yy, yy + row["w"] * scale)
            yy += row["w"] * scale
        y += sums[node] * scale + GAP
    return nodes, slots


def _ribbon(ax, x0, x1, a, b, color, dashed):
    t = np.linspace(0, 1, 60)
    s = t * t * (3 - 2 * t)
    xs = x0 + (x1 - x0) * t
    top = a[0] + (b[0] - a[0]) * s
    bot = a[1] + (b[1] - a[1]) * s
    ax.fill_between(xs, top, bot, color=color, alpha=RIBBON_ALPHA, lw=0)
    if dashed:
        for edge in (top, bot):
            ax.plot(xs, edge, color="#444444", lw=1.1, ls=(0, (4, 2.5)))


def _nodes(ax, nodes, x, labels, side):
    for node, (y, h) in nodes.items():
        ax.add_patch(Rectangle((x, y), NODE_W, h, facecolor=NODE_FACE,
                               edgecolor=NODE_EDGE, lw=1, zorder=3))
        if side == "above":
            ax.text(x + NODE_W / 2, y - 6, labels(node), ha="center", va="bottom",
                    fontsize=10, style="italic", color="#333333")
        elif side == "left":
            ax.text(x - 12, y + h / 2, labels(node), ha="right", va="center", fontsize=10)
        else:
            ax.text(x + NODE_W + 12, y + h / 2, labels(node), ha="left", va="center",
                    fontsize=10)


def _effect_label(ax, row, y):
    box = dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85)
    ax.text(LABEL_X, y - 1, f"a×b = {row['indirect']:.3f}  ({row['indirect_lo']:.3f}, "
            f"{row['indirect_hi']:.3f})".replace("-", "−"), ha="center", va="bottom",
            fontsize=7.5, bbox=box, zorder=5)
    reverse = row["pct"] < 0
    text = f"{row['pct']:.1f}% mediated".replace("-", "−")
    ax.text(LABEL_X, y, text + ("  · sign-reversing" if reverse else ""), ha="center",
            va="top", fontsize=7, fontweight="bold" if reverse else "normal",
            color=REVERSE_TEXT_COLOR if reverse else "#666666", bbox=box, zorder=5)


def _legend(ax):
    y0 = CANVAS_H - 72
    for i, (color, text) in enumerate([
        (RAISE_COLOR, "Mediated effect raises the outcome (a×b > 0)"),
        (LOWER_COLOR, "Mediated effect lowers the outcome (a×b < 0)"),
    ]):
        y = y0 + 20 * i
        ax.add_patch(Rectangle((42, y - 4), 22, 9, facecolor=color, alpha=RIBBON_ALPHA, lw=0))
        ax.text(75, y, text, va="center", fontsize=8.5)
    y = y0 + 40
    ax.add_line(Line2D([42, 64], [y, y], color="#444444", lw=1.1, ls=(0, (4, 2.5))))
    ax.text(75, y, "Sign-reversing (opposes total effect c; negative % mediated)",
            va="center", fontsize=8.5)


def mediation_sankey(med: pd.DataFrame, out: Path, q_max: float = 0.05,
                     subtitle: str | None = ("Each ribbon is one exposure → mediating species "
                                             "→ outcome path; width ∝ |indirect effect a×b|")
                     ) -> None:
    """Draw the mediation Sankey from a ``mediation.csv`` frame."""
    df = _paths(med, q_max)
    if df.empty:
        raise ValueError(f"no mediation paths with q_indirect < {q_max}")
    col_sums = [df.groupby(c)["w"].sum() for c in ("exposure", "mediator", "outcome")]
    scale = min((SPAN - GAP * (len(s) - 1)) / s.sum() for s in col_sums)

    e_nodes, e_slots = _stack(df, "exposure", ["m_rank", "o_rank"], scale)
    m_nodes, m_slots = _stack(df, "mediator", ["e_rank", "o_rank"], scale)
    o_nodes, o_slots = _stack(df, "outcome", ["m_rank", "e_rank"], scale)

    fig = plt.figure(figsize=(CANVAS_W / 100, CANVAS_H / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)
    ax.set_axis_off()

    for idx, row in df.iterrows():
        color = RAISE_COLOR if row["indirect"] > 0 else LOWER_COLOR
        dashed = row["pct"] < 0
        _ribbon(ax, COL_X[0] + NODE_W, COL_X[1], e_slots[idx], m_slots[idx], color, dashed)
        _ribbon(ax, COL_X[1] + NODE_W, COL_X[2], m_slots[idx], o_slots[idx], color, dashed)
        mid = (sum(m_slots[idx]) + sum(o_slots[idx])) / 4
        _effect_label(ax, row, mid)

    _nodes(ax, e_nodes, COL_X[0], NODE_LABELS.get, "left")
    _nodes(ax, m_nodes, COL_X[1], short_species, "above")
    _nodes(ax, o_nodes, COL_X[2], NODE_LABELS.get, "right")
    for x, head in zip(COL_X, ("Exposure", "Mediating species", "Outcome")):
        ax.text(x + NODE_W / 2, HEADER_Y, head, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=HEADER_COLOR)
    if subtitle:
        ax.text(35, 8, subtitle, ha="left", va="top", fontsize=8.5, color="#666666")
    _legend(ax)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
