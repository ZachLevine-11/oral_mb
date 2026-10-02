"""Smoke tests on synthetic data — no cluster dependency."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oral_mb.associations import associations, shared_signature
from oral_mb.data import Cohort, clr_transform, to_relative, transform_features
from oral_mb.mediation import mediation_one
from oral_mb.replication import inverse_variance_meta, meta_analysis
from oral_mb.stats_utils import bh_qvalues, bootstrap_metric


@pytest.fixture
def synth_cohort() -> Cohort:
    rng = np.random.default_rng(0)
    n, p = 400, 12
    idx = pd.Index([f"S{i}" for i in range(n)], name="RegistrationCode")
    # MetaPhlAn-style log10 abundances with sentinel -4.
    M = rng.uniform(-4, 0, size=(n, p))
    M[M < -2] = -4.0
    cols = [f"s__Sp{i}" for i in range(p)]
    micro = pd.DataFrame(M, index=idx, columns=cols)

    smoke = rng.binomial(1, 0.2, n)
    # symptom depends on Sp0 + smoking
    logit = -2 + 0.3 * smoke + 0.5 * (M[:, 0] > -2).astype(int)
    sym = pd.DataFrame(
        {"bleeding_gums": rng.binomial(1, 1 / (1 + np.exp(-logit)))}, index=idx
    )
    life = pd.DataFrame(
        {"smoke_tobacco_now": smoke, "UPF_score": rng.normal(2.5, 0.3, n),
         "med_score_per_day": rng.normal(0.45, 0.15, n),
         "is_vegetarian": rng.binomial(1, 0.15, n),
         "vegetarian_score_per_day": rng.normal(0.55, 0.1, n),
         "is_fasting": rng.binomial(1, 0.05, n), "keto": rng.binomial(1, 0.02, n)},
        index=idx,
    )
    cov = pd.DataFrame(
        {"age": rng.normal(52, 8, n), "gender": rng.binomial(1, 0.5, n),
         "bmi": rng.normal(26, 4, n)},
        index=idx,
    )
    return Cohort(microbiome=micro, symptoms=sym, lifestyle=life, covariates=cov)


def test_clr_round_trip(synth_cohort: Cohort) -> None:
    rel = to_relative(synth_cohort.microbiome)
    z = clr_transform(rel)
    assert z.shape == synth_cohort.microbiome.shape
    assert np.isfinite(z.values).all()
    # CLR rows mean ~ 0.
    assert np.abs(z.mean(axis=1)).mean() < 1e-6


def test_transform_kinds(synth_cohort: Cohort) -> None:
    for k in ("clr", "log10", "raw"):
        out = transform_features(synth_cohort.microbiome, kind=k)
        assert out.shape == synth_cohort.microbiome.shape


def test_bh_qvalues_monotone() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.20, 0.5])
    q = bh_qvalues(p)
    assert (q >= 0).all() and (q <= 1).all()
    # q sorted by p should be non-decreasing
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_associations_runs(synth_cohort: Cohort) -> None:
    a = associations(synth_cohort, feature_kind="clr")
    assert {"feature", "outcome", "beta", "pvalue", "q_within_outcome"} <= set(a.columns)


def test_shared_signature(synth_cohort: Cohort) -> None:
    a = associations(synth_cohort, feature_kind="clr")
    sym = a[a["outcome"] == "bleeding_gums"]
    life = a[a["outcome"] == "smoke_tobacco_now"]
    s = shared_signature(sym, life)
    assert isinstance(s, pd.DataFrame) and "feature" in s.columns


def test_bootstrap_metric() -> None:
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 500)
    p = y * 0.6 + rng.normal(0, 0.2, 500)
    point, lo, hi = bootstrap_metric(y, p, "auc", n_boot=100)
    assert lo <= point <= hi


def test_mediation_runs(synth_cohort: Cohort) -> None:
    feats = transform_features(synth_cohort.microbiome, kind="clr")
    r = mediation_one(
        exposure=synth_cohort.lifestyle["smoke_tobacco_now"],
        mediator=feats.iloc[:, 0],
        outcome=synth_cohort.symptoms["bleeding_gums"],
        covars=synth_cohort.covariates[["age", "gender", "bmi"]],
        outcome_kind="binary",
        n_boot=50,
    )
    assert r is not None
    assert -1 <= r.pvalue_indirect <= 1


def test_meta_analysis() -> None:
    b, s, p = inverse_variance_meta([0.2, 0.25], [0.05, 0.06])
    assert 0.15 < b < 0.3
    assert s > 0 and 0 <= p <= 1


def test_meta_table() -> None:
    a1 = pd.DataFrame({"feature": ["f1"], "outcome": ["o1"], "beta": [0.2],
                       "se": [0.05], "pvalue": [1e-4]})
    a2 = pd.DataFrame({"feature": ["f1"], "outcome": ["o1"], "beta": [0.3],
                       "se": [0.07], "pvalue": [1e-3]})
    m = meta_analysis([a1, a2])
    assert len(m) == 1
    assert "beta_meta" in m.columns and "i2" in m.columns


def test_mediation_sankey_draws_significant_paths(tmp_path) -> None:
    from oral_mb.sankey import mediation_sankey, short_species

    med = pd.DataFrame({
        "exposure": ["smoke_tobacco_now", "UPF_score", "med_score_per_day"],
        "mediator": ["s__Abiotrophia_defectiva", "s__Cardiobacterium_hominis",
                     "s__Cardiobacterium_hominis"],
        "outcome": ["bleeding_gums"] * 3,
        "indirect": [-0.05, 0.06, 0.2], "indirect_lo": [-0.09, 0.02, -0.1],
        "indirect_hi": [-0.02, 0.11, 0.4], "c_total": [-0.5, 0.46, -0.9],
        "q_indirect": [0.0, 0.0, 0.5],
    })
    out = tmp_path / "sankey.png"
    mediation_sankey(med, out)
    assert out.stat().st_size > 0
    assert short_species("s__Abiotrophia_defectiva") == "A. defectiva"
    with pytest.raises(ValueError):
        mediation_sankey(med.assign(q_indirect=1.0), out)
