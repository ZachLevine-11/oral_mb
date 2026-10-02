"""Publication-quality plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

from .stats_utils import calibration_table

sns.set_context("paper")
sns.set_style("whitegrid")


def _shared_signed_matrix(
    assoc_sym: pd.DataFrame, assoc_life: pd.DataFrame, shared: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sym_w = (
        assoc_sym[assoc_sym["feature"].isin(shared)]
        .assign(signed=lambda d: -np.log10(d["pvalue"]) * np.sign(d["beta"]))
        .pivot(index="feature", columns="outcome", values="signed")
    )
    life_w = (
        assoc_life[assoc_life["feature"].isin(shared)]
        .assign(signed=lambda d: -np.log10(d["pvalue"]) * np.sign(d["beta"]))
        .pivot(index="feature", columns="outcome", values="signed")
    )
    M = pd.concat([sym_w, life_w], axis=1).fillna(0)
    return M, sym_w


def heatmap_shared(
    assoc_sym: pd.DataFrame,
    assoc_life: pd.DataFrame,
    shared: list[str],
    out: Path,
    aerobic_col: str = "bleeding_gums",
    cluster_columns: bool = True,
    title: str | None = None,
    show_dendrogram: bool = True,
) -> None:
    """Signed heatmap of the shared signature, rows sorted by aerobic-ness.

    Row order follows the sign/magnitude of association with ``aerobic_col``
    (positive = inflammatory anaerobic axis, negative = aerobic commensal
    axis, per the paper's own axis definition), with a red dashed line drawn
    at the sign boundary.

    ``cluster_columns`` hierarchically clusters the phenotype columns and draws
    the dendrogram above the map, so co-behaving symptoms/lifestyle items sit
    together while the biologically ordered rows (and the red split line) are
    preserved -- a clustermap on top, sorted rows below. ``show_dendrogram``
    False keeps the clustered column order but draws no tree.
    """
    M, sym_w = _shared_signed_matrix(assoc_sym, assoc_life, shared)
    order_score = (
        sym_w[aerobic_col].reindex(M.index).fillna(0)
        if aerobic_col in sym_w.columns
        else M.mean(axis=1)
    )
    M = M.loc[order_score.sort_values(ascending=True).index]

    dendro = link = None
    if cluster_columns and M.shape[1] > 2:
        # Correlation distance is undefined for a constant column (happens at
        # coarse levels / pathways, where a phenotype can be flat across every
        # retained feature), so fall back to Euclidean when any column has no
        # variance, and skip clustering outright if the distances still are not
        # finite.
        metric = "correlation" if (M.std(axis=0) > 0).all() else "euclidean"
        link = linkage(M.T.to_numpy(), method="average", metric=metric)
        if np.isfinite(link[:, 2]).all():
            dendro = dendrogram(link, labels=list(M.columns), no_plot=True)
            M = M[dendro["ivl"]]
        else:
            link = None

    w = 0.42 * M.shape[1] + 4.5
    h = 0.25 * M.shape[0] + 2.4
    fig = plt.figure(figsize=(w, h))
    draw_tree = dendro is not None and show_dendrogram
    if draw_tree:
        gs = fig.add_gridspec(2, 1, height_ratios=[0.12, 0.88], hspace=0.02)
        ax_d = fig.add_subplot(gs[0])
        dendrogram(link, ax=ax_d, no_labels=True, color_threshold=0,
                   above_threshold_color="0.4")
        ax_d.set_axis_off()
        ax = fig.add_subplot(gs[1])
    else:
        ax = fig.add_subplot(111)

    sns.heatmap(M, cmap="RdBu_r", center=0,
                cbar_kws={"label": "-log10(p) × sign(beta)"}, ax=ax)
    split = int((order_score.loc[M.index].to_numpy() < 0).sum())
    if 0 < split < len(M):
        ax.axhline(split, color="red", linestyle="--", linewidth=1.5)
    ax.set_xticklabels([pretty_label(c) for c in M.columns], rotation=45,
                       ha="right", fontsize=8)
    ax.set_yticklabels([pretty_feature(f) for f in M.index], fontsize=8)
    ax.set_xlabel("Symptoms | Lifestyle" if dendro is None
                  else "Symptoms | Lifestyle (clustered)")
    ax.set_ylabel("Microbiome feature (sorted aerobic commensal → inflammatory anaerobic)")
    if title:
        fig.suptitle(title, fontsize=10.5, y=1.0)
    if not draw_tree:
        # tight_layout is incompatible with the dendrogram gridspec; the
        # bbox_inches="tight" save below handles trimming in that case.
        fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def clustermap_shared(
    assoc_sym: pd.DataFrame, assoc_life: pd.DataFrame, shared: list[str], out: Path
) -> None:
    """Supplementary view of the shared signature: two-way hierarchical
    clustering, to surface any co-clumping of features/outcomes beyond the
    aerobic/anaerobic axis shown in the primary sorted heatmap."""
    M, _ = _shared_signed_matrix(assoc_sym, assoc_life, shared)
    g = sns.clustermap(
        M, cmap="RdBu_r", center=0,
        figsize=(0.4 * M.shape[1] + 4, 0.25 * M.shape[0] + 2),
        cbar_kws={"label": "-log10(p) × sign(beta)"},
    )
    g.ax_heatmap.set_xlabel("Symptoms | Lifestyle")
    g.ax_heatmap.set_ylabel("Microbiome feature")
    g.savefig(out, dpi=300)
    plt.close(g.fig)


def forest_plot(meta: pd.DataFrame, top: int = 30, out: Path | None = None) -> None:
    df = meta.nsmallest(top, "p_meta").copy()
    df["label"] = df["feature"] + " | " + df["outcome"]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(df) + 1))
    y = np.arange(len(df))
    ax.errorbar(
        df["beta_meta"], y,
        xerr=1.96 * df["se_meta"], fmt="o", color="black", capsize=2,
    )
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(df["label"], fontsize=7)
    ax.set_xlabel("Meta-analytic beta (95% CI)")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=300)
    plt.close(fig)


def mediation_forest_plot(med: pd.DataFrame, q_max: float = 0.05, out: Path | None = None) -> None:
    """Fig. 4: bootstrap mediation indirect effects (a × b, 95% CI),
    dot-and-whisker, coloured by exposure, restricted to q_indirect < q_max."""
    df = med[med["q_indirect"] < q_max].copy() if "q_indirect" in med.columns else med.copy()
    if df.empty:
        return
    df = df.sort_values("indirect")
    df["label"] = df["exposure"] + " → " + df["mediator"] + " → " + df["outcome"]
    exposures = sorted(df["exposure"].unique())
    palette = dict(zip(exposures, sns.color_palette("tab10", len(exposures))))
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(df) + 1.2))
    y = np.arange(len(df))
    for exp in exposures:
        sub = df[df["exposure"] == exp]
        idx = [df.index.get_loc(i) for i in sub.index]
        xerr = [
            (sub["indirect"] - sub["indirect_lo"]).to_numpy(),
            (sub["indirect_hi"] - sub["indirect"]).to_numpy(),
        ]
        ax.errorbar(
            sub["indirect"], y[idx], xerr=xerr, fmt="o", color=palette[exp],
            capsize=2, label=exp,
        )
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(df["label"], fontsize=7)
    ax.set_xlabel("Indirect effect (a × b, 95% bootstrap CI)")
    ax.legend(title="Exposure", fontsize=8, title_fontsize=9, loc="best", frameon=False)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=300)
    plt.close(fig)


_EXPOSURE_LABELS = {
    "smoke_tobacco_now": "Smoking",
    "UPF_score": "UPF",
    "med_score_per_day": "Mediterranean",
    "vegetarian_score_per_day": "Vegetarian",
}


def mediation_counterfactual_forest_plot(
    df: pd.DataFrame, out: Path,
    caption: str | None = None,
    title: str = ("Counterfactual mediation: natural indirect effects are small,\n"
                   "with E-values quantifying sensitivity to unmeasured confounding"),
) -> None:
    """Fig. 7: g-formula natural indirect effects (``mediation_counterfactual.csv``
    from ``stage_mediation``).

    Layout is the original dot-and-whisker forest plot (two-line
    ``exposure -> mediator -> outcome`` labels on the left, ``E=`` annotation on
    the right, one shared x-axis), split into one stacked panel per exposure per
    Tal's review: rows still sorted by indirect effect (largest at top) and
    points now coloured by E-value rather than by exposure, since the panel
    split already carries the exposure.
    """
    d = df.dropna(subset=["nie", "nie_ci_lo", "nie_ci_hi"]).copy()
    if d.empty:
        return
    d["label"] = (d["mediator"].map(pretty_feature) + "\n→ "
                  + d["outcome"].map(pretty_label))
    present = list(d["exposure"].unique())
    exposures = [e for e in _EXPOSURE_LABELS if e in present]
    exposures += [e for e in present if e not in _EXPOSURE_LABELS]

    evals = d["e_value_nie"].to_numpy(dtype=float)
    finite = evals[np.isfinite(evals)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) else (1.0, 2.0)
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    counts = [int((d["exposure"] == e).sum()) for e in exposures]
    fig, axes = plt.subplots(
        len(exposures), 1, squeeze=False, sharex=True,
        gridspec_kw={"height_ratios": counts, "hspace": 0.12},
        figsize=(7.2, 0.55 * sum(counts) + 1.8),
    )
    axes = axes[:, 0]

    # One shared, un-broken x-axis: symmetric padding around the widest CI, with
    # extra room on the right for the E-value annotations.
    lo, hi = float(d["nie_ci_lo"].min()), float(d["nie_ci_hi"].max())
    span = (hi - lo) or 1.0
    xlim = (lo - 0.08 * span, hi + 0.30 * span)
    x_ann = hi + 0.16 * span

    for ax, exp in zip(axes, exposures):
        sub = d[d["exposure"] == exp].sort_values("nie", ascending=False)
        y = np.arange(len(sub))
        colors = [cmap(norm(v)) if np.isfinite(v) else "0.75" for v in sub["e_value_nie"]]
        for yi, lo_i, hi_i, c in zip(y, sub["nie_ci_lo"], sub["nie_ci_hi"], colors):
            ax.plot([lo_i, hi_i], [yi, yi], color=c, lw=1.2, zorder=2)
            ax.plot([lo_i, lo_i], [yi - 0.16, yi + 0.16], color=c, lw=1.2, zorder=2)
            ax.plot([hi_i, hi_i], [yi - 0.16, yi + 0.16], color=c, lw=1.2, zorder=2)
        ax.scatter(sub["nie"], y, c=colors, s=34, zorder=3, edgecolor="none")
        for yi, ev in zip(y, sub["e_value_nie"]):
            txt = f"E={ev:.2f}" if np.isfinite(ev) else "E=NA"
            ax.text(x_ann, yi, txt, fontsize=7, color="0.35", va="center")
        ax.axvline(0, color="grey", lw=0.7, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"], fontsize=7.5)
        ax.set_ylim(len(sub) - 0.5, -0.5)
        ax.set_xlim(*xlim)
        ax.set_title(_EXPOSURE_LABELS.get(exp, exp), fontsize=9.5, loc="left", pad=3)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[-1].set_xlabel(
        "Natural indirect effect on symptom probability "
        "(risk difference, 95% bootstrap CI)", fontsize=9)

    sm_ = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm_.set_array([])
    cbar = fig.colorbar(sm_, ax=list(axes), fraction=0.022, pad=0.02, aspect=30)
    cbar.set_label("E-value (sensitivity to unmeasured confounding)", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    fig.suptitle(title, fontsize=10.5)
    if caption:
        fig.text(0.5, -0.03, caption, ha="center", va="top", fontsize=8)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def upf_smoking_beta_scatter(
    assoc_life: pd.DataFrame, shared: list[str] | None = None, out: Path | None = None
) -> None:
    """UPF mirrors smoking (Results): scatter of per-species beta(UPF_score)
    against beta(smoke_tobacco_now), the plot form of that finding."""
    upf = assoc_life[assoc_life["outcome"] == "UPF_score"].set_index("feature")["beta"]
    smk = assoc_life[assoc_life["outcome"] == "smoke_tobacco_now"].set_index("feature")["beta"]
    df = pd.concat([upf.rename("UPF_score"), smk.rename("smoke_tobacco_now")], axis=1).dropna()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    is_shared = df.index.isin(shared) if shared else np.zeros(len(df), dtype=bool)
    ax.scatter(df.loc[~is_shared, "smoke_tobacco_now"], df.loc[~is_shared, "UPF_score"],
               s=14, color="lightgrey", label="all species")
    if is_shared.any():
        ax.scatter(df.loc[is_shared, "smoke_tobacco_now"], df.loc[is_shared, "UPF_score"],
                   s=22, color="firebrick", label="shared signature")
    r = df["UPF_score"].corr(df["smoke_tobacco_now"])
    lim = np.nanmax(np.abs(df.to_numpy())) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "--", color="grey", lw=0.75)
    ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("beta, smoking"); ax.set_ylabel("beta, UPF score")
    ax.set_title(f"UPF mirrors smoking (r = {r:.2f})")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=300)
    plt.close(fig)


def calibration_plot(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    tab = calibration_table(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.plot(tab["bin_mean_pred"], tab["fraction_pos"], "o-")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)


def pretty_feature(name: str) -> str:
    """Taxon label without the MetaPhlAn rank prefix and underscores."""
    txt = str(name)
    if "__" in txt:
        txt = txt.split("__", 1)[1]
    return txt.replace("_", " ")


def pretty_label(name: str) -> str:
    """Human-readable outcome label: no underscores, sentence case."""
    special = {
        "med_score_per_day": "Mediterranean score",
        "Med_score_per_day": "Mediterranean score",
        "UPF_score": "Processed-food score",
        "vegetarian_score_per_day": "Vegetarian score",
        "Vegetarian_score_per_day": "Vegetarian score",
        "abscess_or_gingivitis": "Abscess/gingivitis",
        "bad_breath_lose": "Bad breath (loose)",
        "bad_breath_strict": "Bad breath (strict)",
        "smoke_tobacco_now": "Smokes tobacco now",
        "is_vegetarian": "Vegetarian",
        "is_fasting": "Fasting",
        "keto": "Keto",
    }
    if name in special:
        return special[name]
    txt = str(name).replace("_", " ").strip()
    return txt[:1].upper() + txt[1:]


def _metric_err(df: pd.DataFrame, metric: str) -> np.ndarray | None:
    """Asymmetric [lo, hi] offsets from bootstrap CI, else ±1 s.d. of folds."""
    if f"{metric}_lo" in df.columns:
        vals = df[metric].to_numpy(dtype=float)
        return np.vstack([
            vals - df[f"{metric}_lo"].to_numpy(dtype=float),
            df[f"{metric}_hi"].to_numpy(dtype=float) - vals,
        ])
    if f"{metric}_std" in df.columns:
        return np.abs(df[f"{metric}_std"].to_numpy(dtype=float))
    return None


def prediction_bars(results: pd.DataFrame, metric: str, out: Path,
                    caption: str | None = None) -> None:
    """Point estimate + CI per outcome (seaborn-pointplot style).

    Bars wasted width and forced a 0-anchored y-axis, which squashed the
    between-outcome differences; a categorical point plot keeps constant
    per-category spacing and lets the y-axis zoom on the range that matters.
    """
    df = results.copy()
    x = np.arange(len(df))
    err = _metric_err(df, metric)
    fig, ax = plt.subplots(figsize=(max(4.2, 0.42 * len(df) + 1.2), 3.6))
    if metric == "auc":
        ax.axhline(0.5, ls="--", lw=0.8, color="grey", zorder=0)
    ax.errorbar(x, df[metric].to_numpy(dtype=float), yerr=err, fmt="o",
                ms=5, lw=1.2, capsize=2.5, color="#3b6fb0", zorder=3)
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_label(o) for o in df["outcome"]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ROC-AUC" if metric == "auc" else metric.upper(), fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", visible=False)
    if caption:
        fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)


def prediction_bars_grouped(
    results: pd.DataFrame, metric: str, out: Path, group: str = "source",
) -> None:
    """Grouped bars: one cluster per outcome, one bar per group value
    (e.g. source in {oral, gut, oral+gut}). Drops rows with NaN metric."""
    df = results.dropna(subset=[metric]).copy()
    if df.empty:
        return
    groups = sorted(df[group].unique())
    outcomes = sorted(df["outcome"].unique())
    n_g = max(len(groups), 1)
    x = np.arange(len(outcomes))
    width = 0.85 / n_g
    # Scale width with #outcomes and #groups so each bar is ≥0.25in wide.
    per_cluster = max(1.2, 0.35 * n_g)
    w_in = max(8, per_cluster * len(outcomes))
    fig, ax = plt.subplots(figsize=(w_in, 5.5))
    for i, g in enumerate(groups):
        sub = df[df[group] == g].set_index("outcome").reindex(outcomes)
        vals = sub[metric].to_numpy(dtype=float)
        err = None
        if f"{metric}_lo" in sub.columns:
            lo = sub[f"{metric}_lo"].to_numpy(dtype=float)
            hi = sub[f"{metric}_hi"].to_numpy(dtype=float)
            err = [vals - lo, hi - vals]
        elif f"{metric}_std" in sub.columns:
            # Symmetric ±1 s.d. across CV folds (clip negatives from NaN rows).
            err = np.abs(sub[f"{metric}_std"].to_numpy(dtype=float))
        ax.bar(x + i * width - 0.425 + width / 2, vals, width,
               yerr=err, capsize=2, label=str(g))
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, rotation=45, ha="right", fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_xlabel("outcome", fontsize=11)
    ax.legend(title=group, fontsize=9, title_fontsize=10,
              loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _grouped_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, group: str, ylabel: str) -> None:
    groups = sorted(df[group].unique())
    outcomes = sorted(df["outcome"].unique())
    n_g = max(len(groups), 1)
    x = np.arange(len(outcomes))
    width = 0.85 / n_g
    for i, g in enumerate(groups):
        sub = df[df[group] == g].set_index("outcome").reindex(outcomes)
        vals = sub[metric].to_numpy(dtype=float)
        err = None
        if f"{metric}_lo" in sub.columns:
            lo = sub[f"{metric}_lo"].to_numpy(dtype=float)
            hi = sub[f"{metric}_hi"].to_numpy(dtype=float)
            err = [vals - lo, hi - vals]
        elif f"{metric}_std" in sub.columns:
            err = np.abs(sub[f"{metric}_std"].to_numpy(dtype=float))
        ax.bar(x + i * width - 0.425 + width / 2, vals, width, yerr=err, capsize=2, label=str(g))
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)


def prediction_bars_combined(
    auc_df: pd.DataFrame, pearson_df: pd.DataFrame, out: Path, group: str = "source",
) -> None:
    """Fig. 3a+3c combined: buccal-vs-gut prediction accuracy for binary
    (ROC-AUC) and continuous (Pearson r) outcomes side by side in one
    figure, restricted to the ensemble model only (one bar colour per
    feature source) to avoid the redundant model x source colour explosion
    of showing all four models at once."""
    auc = auc_df.dropna(subset=["auc"]).copy()
    pear = pearson_df.dropna(subset=["pearson"]).copy()
    n_panels = int(not auc.empty) + int(not pear.empty)
    if n_panels == 0:
        return
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(max(8, 0.6 * len(sorted(set(auc.get("outcome", []) ).union(set(pear.get("outcome", [])))))) , 5.5),
        squeeze=False,
    )
    axes = axes[0]
    i = 0
    if not auc.empty:
        _grouped_panel(axes[i], auc, "auc", group, "ROC-AUC")
        axes[i].set_title("Binary outcomes (symptoms, smoking, keto, ...)", fontsize=9)
        i += 1
    if not pear.empty:
        _grouped_panel(axes[i], pear, "pearson", group, "Pearson r")
        axes[i].set_title("Continuous dietary pattern scores", fontsize=9)
        i += 1
    axes[-1].legend(title=group, fontsize=8, title_fontsize=9,
                     loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Buccal vs. gut prediction accuracy by feature source (ensemble model)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def missingness_panel(
    ipw_shift: pd.DataFrame,
    counts: pd.DataFrame,
    out: Path,
    headline: str = "bleeding_gums",
    caption: str | None = None,
    title: str = "Responders match non-responders compositionally",
) -> None:
    """Two-panel non-response figure: IPW-vs-complete-case betas, and item
    response rate per symptom.

    The PERMANOVA result is reported in the caption under the figure rather
    than in an in-axes legend box, which previously overlapped the bars.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))

    ax = axes[0]
    b_cc = ipw_shift["beta_unweighted"].to_numpy(dtype=float)
    b_ipw = ipw_shift["beta_ipw"].to_numpy(dtype=float)
    lim = float(np.nanmax(np.abs(np.concatenate([b_cc, b_ipw])))) * 1.08
    ax.plot([-lim, lim], [-lim, lim], "--", lw=0.8, color="grey", zorder=0)
    ax.scatter(b_cc, b_ipw, s=14, color="#c0504d", alpha=0.7, edgecolor="none", zorder=3)
    r = float(np.corrcoef(b_cc, b_ipw)[0, 1])
    ax.text(0.03, 0.97,
            f"r = {r:.3f}\nmedian |Δβ| = {np.median(np.abs(b_ipw - b_cc)):.3f}\n"
            f"n = {len(ipw_shift)} species",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("β, complete-case", fontsize=9)
    ax.set_ylabel("β, inverse-probability-weighted", fontsize=9)
    ax.set_title(f"{pretty_label(headline)} coefficients\nsurvive non-response reweighting",
                 fontsize=9)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    cnt = counts.sort_values("frac_answered").reset_index(drop=True)
    y = np.arange(len(cnt))
    colors = ["#c0504d" if o == headline else "#8c8c8c" for o in cnt["outcome"]]
    ax.barh(y, 100 * cnt["frac_answered"].to_numpy(dtype=float), color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_label(o) for o in cnt["outcome"]], fontsize=8)
    ax.set_xlabel("% of cohort answering item", fontsize=9)
    ax.set_title("Item response varies by symptom", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", visible=False)

    if title:
        fig.suptitle(title, fontsize=10.5, y=1.02)
    fig.tight_layout()
    if caption:
        fig.text(0.5, -0.03, caption, ha="center", va="top", fontsize=8)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


LEVEL_ORDER = ("species", "genus", "family", "phylum", "pathways")
SOURCE_ORDER = ("oral", "gut", "oral+gut")

_METRIC_YLABEL = {"auc": "ROC-AUC", "pearson": "Pearson r"}


def _levels_in(df: pd.DataFrame, level: str) -> list[str]:
    """Group values present in ``df[level]``, in a meaningful order."""
    present = set(df[level])
    order = SOURCE_ORDER if level == "source" else LEVEL_ORDER
    return [lv for lv in order if lv in present] or sorted(present)


def _level_points_panel(
    ax: plt.Axes, df: pd.DataFrame, metric: str, levels: list[str],
    palette: dict, level: str = "level", dodge_span: float | None = None,
) -> None:
    """One dodged point-and-CI per (outcome, feature level) on ``ax``.

    ``dodge_span`` overrides the default spread: pass a small value (e.g.
    0.22) to keep the per-level points for one outcome clumped tightly
    together, so outcomes read as visually distinct clusters rather than a
    spread-out row of same-colour dots.
    """
    outcomes = list(dict.fromkeys(df["outcome"]))
    x = np.arange(len(outcomes))
    if dodge_span is not None:
        span = dodge_span
    else:
        # Widen the dodge when few categories share a wide axis, so the ranks
        # stay legible instead of collapsing into the middle of each slot.
        span = 0.62 if len(outcomes) >= 8 else 0.88
    offsets = (np.linspace(-span / 2, span / 2, len(levels)) if len(levels) > 1
               else np.zeros(1))
    if metric == "auc":
        ax.axhline(0.5, ls="--", lw=0.8, color="grey", zorder=0)
    for lv, dx in zip(levels, offsets):
        sub = df[df[level] == lv].set_index("outcome").reindex(outcomes)
        ax.errorbar(x + dx, sub[metric].to_numpy(dtype=float),
                    yerr=_metric_err(sub, metric), fmt="o", ms=4, lw=1.1,
                    capsize=2, color=palette[lv], label=str(lv), zorder=3)
    ax.set_xlim(-0.5, len(outcomes) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_label(o) for o in outcomes], rotation=45,
                       ha="right", fontsize=8)
    ax.set_ylabel(_METRIC_YLABEL.get(metric, metric.upper()), fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", visible=False)


def prediction_points_by_level(
    results: pd.DataFrame, metric: str, out: Path,
    level: str = "level", caption: str | None = None,
    title: str | None = None,
) -> None:
    """Fig. 3d: one point-and-CI per (outcome, feature level), dodged within
    each outcome, so taxonomic ranks are compared directly per outcome.

    Point form rather than grouped bars for the same reason as
    ``prediction_bars``: bars force a 0-anchored axis and eat the horizontal
    space that the between-rank gaps need.
    """
    df = results.dropna(subset=[metric]).copy()
    if df.empty:
        return
    levels = _levels_in(df, level)
    palette = dict(zip(levels, sns.color_palette("colorblind", len(levels))))
    n_out = df["outcome"].nunique()
    fig, ax = plt.subplots(figsize=(max(5.0, 0.72 * n_out + 1.4), 3.8))
    _level_points_panel(ax, df, metric, levels, palette, level)
    if title:
        ax.set_title(title, fontsize=10.5)
    fig.tight_layout()
    # Legend below the panel, not above it: anchoring both a legend and a
    # suptitle above the canvas in figure fractions makes them overlap once
    # bbox_inches="tight" rescales the figure.
    fig.legend(*ax.get_legend_handles_labels(), fontsize=8, frameon=False,
               ncol=len(levels), loc="upper center", bbox_to_anchor=(0.5, 0.0))
    if caption:
        # Below the legend strip; ~0.35in of legend converted to a figure
        # fraction so short figures do not put the caption on top of it.
        fig.text(0.5, -0.35 / fig.get_figheight() - 0.06, caption,
                 ha="center", va="top", fontsize=8)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def prediction_points_by_level_combined(
    results: pd.DataFrame, out: Path, level: str = "level",
    caption: str | None = None,
    title: str = "Species-level features outperform coarser ranks and pathways",
    dodge_span: float | None = None,
    outcome_width: float = 0.72,
    stacked: bool = True,
) -> None:
    """Fig. 3d as a single figure: binary outcomes (ROC-AUC) and continuous
    dietary scores (Pearson r) in one legend-sharing figure. ``stacked``
    (default) puts them in two rows (needed when many groups are dodged per
    outcome, e.g. the 5 taxonomic ranks in Fig. 3d, or the row gets too
    cramped); pass ``stacked=False`` for two side-by-side vertical/portrait
    columns instead (used for the 2-group oral-vs-gut comparison, Fig.
    3a+3c). Either way each panel's size scales with its own outcome count
    so the 3-score panel isn't stretched to match the 12-outcome panel's
    spacing.

    ``dodge_span`` (see ``_level_points_panel``) is forwarded to both
    panels so same-outcome points clump tightly instead of spreading
    across the slot. ``outcome_width`` sets the figure inches added per
    outcome column (independent of ``dodge_span``, both are in different
    units): raise it to widen the gap *between* outcomes without pulling
    the same-outcome points apart.
    """
    auc = results.dropna(subset=["auc"]) if "auc" in results.columns else results.iloc[:0]
    pear = (results.dropna(subset=["pearson"]) if "pearson" in results.columns
            else results.iloc[:0])
    panels = [(d, m) for d, m in ((auc, "auc"), (pear, "pearson")) if not d.empty]
    if not panels:
        return
    levels = _levels_in(pd.concat([d for d, _ in panels]), level)
    palette = dict(zip(levels, sns.color_palette("colorblind", len(levels))))
    titles = ("Binary outcomes", "Continuous dietary scores")
    if stacked:
        n_out = max(d["outcome"].nunique() for d, _ in panels)
        fig, axes_arr = plt.subplots(
            len(panels), 1, squeeze=False,
            figsize=(max(5.5, outcome_width * n_out + 1.6), 3.6 * len(panels)),
        )
        axes = list(axes_arr[:, 0])
    else:
        widths = [max(2.2, outcome_width * d["outcome"].nunique() + 1.0) for d, _ in panels]
        fig = plt.figure(figsize=(sum(widths), 6.6))
        gs = fig.add_gridspec(1, len(panels), width_ratios=widths, wspace=0.55)
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(panels))]
    for ax, (d, m), panel_title in zip(axes, panels, titles):
        _level_points_panel(ax, d, m, levels, palette, level, dodge_span=dodge_span)
        ax.set_title(panel_title, fontsize=9)
    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout(h_pad=2.4 if stacked else None, w_pad=2.4)
    # Legend and caption placed below the axes in figure coordinates (can go
    # negative; bbox_inches="tight" on save extends the canvas to fit them),
    # same trick as prediction_points_by_level.
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False,
               ncol=len(levels), loc="upper center", bbox_to_anchor=(0.5, -0.02))
    if caption:
        fig.text(0.5, -0.35 / fig.get_figheight() - 0.06, caption,
                 ha="center", va="top", fontsize=8)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _annotate_no_overlap(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray,
                          labels: list[str], min_gap_pts: float = 11.0,
                          dx_pts: float = 8.0) -> None:
    """Label points to the right, pushing labels apart vertically so they never
    collide, with a thin leader line back to each point."""
    order = np.argsort(ys)
    placed_y: list[float] = []
    inv = ax.transData.inverted()
    for i in order:
        # Work in display points so the gap is font-relative, not data-relative.
        y_disp = ax.transData.transform((xs[i], ys[i]))[1]
        target = y_disp
        for prev in placed_y:
            if abs(target - prev) < min_gap_pts:
                target = prev + min_gap_pts
        placed_y.append(target)
        x_disp = ax.transData.transform((xs[i], ys[i]))[0] + dx_pts
        x_lab, y_lab = inv.transform((x_disp, target))
        ax.annotate(labels[i], xy=(xs[i], ys[i]), xytext=(x_lab, y_lab),
                    fontsize=7.5, va="center", ha="left",
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.5",
                                "shrinkA": 0, "shrinkB": 2})


def prediction_stability_panel(
    auprc_tbl: pd.DataFrame,
    repeated_cv: pd.DataFrame,
    out: Path,
    headline: tuple[str, ...] = ("bleeding_gums", "smoke_tobacco_now", "keto", "is_vegetarian"),
    source: str = "oral",
    model: str = "ensemble",
    title: str = ("Held-out AUPRC relative to prevalence, and ROC-AUC variability "
                  "across repeated cross-validation"),
    caption: str | None = None,
) -> None:
    """Fig. 8: (a) held-out AUPRC vs prevalence against the no-skill line, and
    (b) the ROC-AUC spread across repeated CV folds with the held-out test
    value overlaid, for the headline outcomes."""
    a = auprc_tbl.copy()
    if "source" in a.columns:
        a = a[a["source"] == source]
    if "model" in a.columns:
        a = a[a["model"] == model]
    a = a.dropna(subset=["auprc", "prevalence"])

    cv = repeated_cv.copy()
    for col, val in (("source", source), ("model", model)):
        if col in cv.columns:
            cv = cv[cv[col] == val]
    cv = cv[cv["auc"].notna()] if "auc" in cv.columns else cv.iloc[:0]
    outcomes = [o for o in headline if o in set(cv.get("outcome", []))]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

    ax = axes[0]
    if not a.empty:
        err = None
        if "auprc_ci_lo" in a.columns:
            err = np.vstack([a["auprc"] - a["auprc_ci_lo"], a["auprc_ci_hi"] - a["auprc"]])
        lim = float(max(a["auprc"].max(), a["prevalence"].max())) * 1.15
        ax.plot([0, lim], [0, lim], "--", lw=0.8, color="grey", zorder=0)
        ax.errorbar(a["prevalence"], a["auprc"], yerr=err, fmt="o", ms=5,
                    lw=1.0, color="#c0504d", ecolor="0.6", capsize=2, zorder=3)
        lab = a[a["outcome"].isin(headline)]
        _annotate_no_overlap(ax, lab["prevalence"].to_numpy(dtype=float),
                             lab["auprc"].to_numpy(dtype=float),
                             [pretty_label(o) for o in lab["outcome"]])
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Prevalence (no-skill AUPRC)", fontsize=9)
    ax.set_ylabel("AUPRC (buccal ensemble, 95% CI)", fontsize=9)
    ax.set_title("Held-out AUPRC vs prevalence\n(dashed line: no-skill baseline)", fontsize=9)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    if outcomes:
        folds = [cv[(cv["outcome"] == o) & (cv.get("split", "cv") == "cv")]["auc"].dropna()
                 for o in outcomes]
        ax.axhline(0.5, ls=":", lw=0.9, color="grey", zorder=0)
        ax.boxplot(folds, positions=np.arange(len(outcomes)), widths=0.55,
                   showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "#cfe0f3", "edgecolor": "#3b6fb0"},
                   medianprops={"color": "#3b6fb0"},
                   whiskerprops={"color": "black"}, capprops={"color": "black"})
        for i, o in enumerate(outcomes):
            vals = cv[(cv["outcome"] == o) & (cv.get("split", "cv") == "cv")]["auc"].dropna()
            ax.scatter(np.full(len(vals), i), vals, s=9, color="0.55", alpha=0.6, zorder=2)
            test = cv[(cv["outcome"] == o) & (cv.get("split", "") == "test")]["auc"].dropna()
            if len(test):
                ax.scatter([i], [float(test.mean())], marker="D", s=34,
                           color="#c0504d", zorder=4,
                           label="held-out test" if i == 0 else None)
        ax.set_xticks(np.arange(len(outcomes)))
        ax.set_xticklabels([pretty_label(o) for o in outcomes], rotation=30,
                           ha="right", fontsize=8)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylabel("ROC-AUC", fontsize=9)
    ax.set_title("ROC-AUC across repeated CV folds\nwith held-out test value", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", visible=False)

    fig.suptitle(title, fontsize=10.5, y=1.04)
    fig.tight_layout(w_pad=2.0)
    if caption:
        fig.text(0.5, -0.03, caption, ha="center", va="top", fontsize=8)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
