"""Tests for the cross-catalogue Test-6 validation framework.

Three layers of coverage:

1. **Unit tests** for the statistical primitives (``_mad_scatter``,
   ``_coverage``, ``_bland_altman_one_cell``, ``_trend_curve``,
   ``_cell_heatmap``) on hand-built fixtures with known-answer math.
2. **Integration tests** for ``compute_cross_catalogue_report`` on a
   synthetic two-catalogue, two-mag-bin fixture: verify the long-form
   table shape, pass/fail map, and rank-summary integrity.
3. **Plot smoke tests** that exercise every plot family with a small
   fixture and assert the PDF + PNG outputs land on disk.

The plot tests use a non-interactive matplotlib backend (``Agg``) to keep
CI fast and headless.
"""

from __future__ import annotations

# isort: off
# matplotlib backend must be set before any matplotlib import; we force the
# non-interactive ``Agg`` backend so the plot smoke tests are headless. The
# ``isort: off`` block prevents the import-organiser from rearranging this
# stanza and re-introducing the early matplotlib import.
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arqueogal.xp_abundances.main.cross_catalogue import (
    DEFAULT_COVERAGE_LEVELS,
    DEFAULT_MAG_BINS,
    LABEL_SCHEMA,
    CatalogueBinding,
    _bland_altman_one_cell,
    _cell_heatmap,
    _coverage,
    _mad_scatter,
    _trend_curve,
    compute_apogee_benchmark_report,
    compute_cross_catalogue_report,
    matched_sigma_subsample,
    rank_summary,
    report_to_long_dataframe,
)
from arqueogal.xp_abundances.main.cross_catalogue_plots import render_all
# isort: on


# --- Statistical primitives ---------------------------------------------------


def test_mad_scatter_recovers_unit_sigma_on_gaussian():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20_000)
    out = _mad_scatter(x)
    assert abs(out - 1.0) < 0.05


def test_mad_scatter_robust_to_outliers():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.standard_normal(10_000), 1e6 * np.ones(50)])
    sigma_mean = float(x.std())
    sigma_mad = _mad_scatter(x)
    assert sigma_mean > 100  # mean-based estimator blown out
    assert abs(sigma_mad - 1.0) < 0.05


def test_coverage_recovers_nominal_levels_on_gaussian():
    rng = np.random.default_rng(1)
    n = 30_000
    sigma = np.full(n, 1.0)
    residual = rng.standard_normal(n)
    cov = _coverage(residual, sigma, levels=DEFAULT_COVERAGE_LEVELS)
    # Empirical fractions should land within ±1 percentage point.
    assert abs(cov["0.68"] - 0.68) < 0.01
    assert abs(cov["0.95"] - 0.95) < 0.01
    assert abs(cov["0.99"] - 0.99) < 0.01


def test_bland_altman_zero_residual_fixture():
    n = 1000
    pred = np.linspace(-1.0, 1.0, n)
    ref = pred.copy()
    sigma_p = np.full(n, 0.1)
    sigma_r = np.full(n, 0.0)
    cell = _bland_altman_one_cell(
        pred,
        ref,
        sigma_p,
        sigma_r,
        label="mh",
        catalogue="ref",
        mag_bin="bright",
        coverage_levels=DEFAULT_COVERAGE_LEVELS,
    )
    assert cell.n == n
    assert cell.bias == pytest.approx(0.0, abs=1e-12)
    assert cell.scatter == pytest.approx(0.0, abs=1e-12)
    assert cell.pearson == pytest.approx(1.0, abs=1e-12)


def test_bland_altman_constant_offset_recovered():
    n = 500
    rng = np.random.default_rng(2)
    pred = rng.standard_normal(n)
    ref = pred - 0.10  # ArqueoGal predicts 0.10 dex higher
    sigma_p = np.full(n, 0.05)
    sigma_r = np.zeros(n)
    cell = _bland_altman_one_cell(
        pred,
        ref,
        sigma_p,
        sigma_r,
        label="mh",
        catalogue="ref",
        mag_bin="bright",
        coverage_levels=DEFAULT_COVERAGE_LEVELS,
    )
    assert cell.bias == pytest.approx(0.10, abs=1e-3)
    assert cell.scatter == pytest.approx(0.0, abs=1e-3)


def test_bland_altman_handles_all_nan():
    n = 100
    pred = np.full(n, np.nan)
    ref = np.full(n, 1.0)
    cell = _bland_altman_one_cell(
        pred,
        ref,
        np.zeros(n),
        np.zeros(n),
        label="mh",
        catalogue="ref",
        mag_bin="bright",
        coverage_levels=DEFAULT_COVERAGE_LEVELS,
    )
    assert cell.n == 0
    assert np.isnan(cell.bias)


def test_trend_curve_bias_is_zero_on_unbiased_fixture():
    rng = np.random.default_rng(3)
    n = 5_000
    x = rng.uniform(-2.0, 0.5, n)
    residual = rng.standard_normal(n) * 0.05
    bins = np.linspace(-2.0, 0.5, 11)
    out = _trend_curve(x, residual, bins=bins, min_per_bin=20)
    # Median bias per bin should be ≈ 0.
    finite = np.isfinite(out["bias"])
    assert finite.sum() >= 8
    assert np.all(np.abs(out["bias"][finite]) < 0.01)


def test_cell_heatmap_recovers_known_pattern():
    rng = np.random.default_rng(4)
    n = 4_000
    teff = rng.uniform(3500, 7000, n)
    logg = rng.uniform(0.0, 5.0, n)
    # Inject a bias that depends on (Teff, log g): hot+low-g cells get +0.2, cool+high-g get -0.2.
    residual = np.where((teff > 5000) & (logg < 2.5), 0.2, -0.2)
    teff_bins = np.linspace(3500, 7000, 4)
    logg_bins = np.linspace(0.0, 5.0, 4)
    heat = _cell_heatmap(
        teff, logg, residual, teff_bins=teff_bins, logg_bins=logg_bins, min_per_cell=25
    )
    assert heat["bias"].shape == (3, 3)
    # Hot+low-g corner ought to be positive; cool+high-g ought to be negative.
    assert heat["bias"][2, 0] > 0
    assert heat["bias"][0, 2] < 0


# --- Integration: compute_cross_catalogue_report ------------------------------


@pytest.fixture
def synthetic_release_and_catalogues():
    """Two-catalogue, two-mag-bin synthetic fixture.

    1000 stars; ArqueoGal predictions are noisy versions of "truth".
    Catalogue A has zero bias; catalogue B has a +0.05 dex bias on [M/H].
    """
    rng = np.random.default_rng(42)
    n = 1000
    truth_teff = rng.uniform(4000, 6500, n)
    truth_logg = rng.uniform(1.0, 4.5, n)
    truth_mh = rng.uniform(-1.5, 0.3, n)
    truth_alpha_m = rng.uniform(-0.05, 0.30, n)
    truth_mg_h = rng.uniform(-1.5, 0.3, n)
    g_mag = rng.uniform(13.0, 17.0, n)

    release = pd.DataFrame(
        {
            "source_id": np.arange(n),
            "g_mag": g_mag,
            "teff_pred": truth_teff + rng.standard_normal(n) * 80,
            "logg_pred": truth_logg + rng.standard_normal(n) * 0.10,
            "mh_pred": truth_mh + rng.standard_normal(n) * 0.05,
            "alpha_m_pred": truth_alpha_m + rng.standard_normal(n) * 0.04,
            "mg_h_pred": truth_mg_h + rng.standard_normal(n) * 0.05,
            "teff_sigma": np.full(n, 80.0),
            "logg_sigma": np.full(n, 0.10),
            "mh_sigma": np.full(n, 0.05),
            "alpha_m_sigma": np.full(n, 0.04),
            "mg_h_sigma": np.full(n, 0.05),
            "release_tier": np.ones(n, dtype=np.int8),
        }
    )
    cat_a = pd.DataFrame(
        {
            "Teff": truth_teff,
            "logg": truth_logg,
            "mh": truth_mh,
            "alpha_m": truth_alpha_m,
        }
    )
    cat_b = pd.DataFrame(
        {
            "teff": truth_teff,
            "logg": truth_logg,
            "feh": truth_mh + 0.05,  # injected bias
        }
    )
    bindings = {
        "catA": CatalogueBinding(
            name="catA",
            column_for={"teff": "Teff", "logg": "logg", "mh": "mh", "alpha_m": "alpha_m"},
            citation="catA",
        ),
        "catB": CatalogueBinding(
            name="catB",
            column_for={"teff": "teff", "logg": "logg", "mh": "feh"},
            citation="catB",
        ),
    }
    return release, {"catA": cat_a, "catB": cat_b}, bindings


def test_compute_cross_catalogue_report_basic(synthetic_release_and_catalogues):
    release, catalogues, bindings = synthetic_release_and_catalogues
    report = compute_cross_catalogue_report(
        release, catalogues, bindings, mag_bins=DEFAULT_MAG_BINS, min_per_bin=50
    )
    long = report_to_long_dataframe(report)
    # 4 elements × catA + 3 elements × catB = 7 (label, catalogue) combinations × 3 mag bins = 21.
    assert len(long) == 21
    # Catalogue A should pass on every label-bin where n is large enough.
    catA_passing = long.query("catalogue == 'catA' and n >= 50")["passed"]
    assert catA_passing.all()
    # Catalogue B has a +0.05 dex bias on [M/H], which equals the bias_limit;
    # the ≤ comparison means the ratio is right at the boundary.
    bias_b_mh = long.query("catalogue == 'catB' and label == 'mh'")["bias"].mean()
    assert -0.06 < bias_b_mh < -0.04


def test_compute_cross_catalogue_report_emits_trend_curves(synthetic_release_and_catalogues):
    release, catalogues, bindings = synthetic_release_and_catalogues
    report = compute_cross_catalogue_report(release, catalogues, bindings)
    # We supplied [M/H] for both catalogues so two trend curves must exist.
    assert ("mh", "catA") in report.bias_vs_mh
    assert ("mh", "catB") in report.bias_vs_mh
    curve = report.bias_vs_mh[("mh", "catA")]
    for key in ("x_centre", "bias", "p16", "p84", "scatter", "n"):
        assert key in curve


def test_rank_summary_orders_catalogues_by_abs_bias(synthetic_release_and_catalogues):
    release, catalogues, bindings = synthetic_release_and_catalogues
    report = compute_cross_catalogue_report(release, catalogues, bindings)
    long = rank_summary(report)
    # On [M/H] catalogue A (zero bias) should rank below catalogue B (+0.05 bias).
    mh = long.query("label == 'mh' and mag_bin == 'intermediate'")
    if mh.shape[0] >= 2:
        rank_a = mh.query("catalogue == 'catA'")["bias_rank"].iloc[0]
        rank_b = mh.query("catalogue == 'catB'")["bias_rank"].iloc[0]
        assert rank_a < rank_b


def test_compute_rejects_misaligned_catalogue(synthetic_release_and_catalogues):
    release, catalogues, bindings = synthetic_release_and_catalogues
    catalogues["catA"] = catalogues["catA"].iloc[:10].reset_index(drop=True)
    with pytest.raises(ValueError, match="cross-match should be performed"):
        compute_cross_catalogue_report(release, catalogues, bindings)


def test_matched_sigma_subsample_returns_lower_half(synthetic_release_and_catalogues):
    release, _, _ = synthetic_release_and_catalogues
    # Inject heterogeneous σ on [M/H] so the σ-quantile filter has bite.
    release = release.copy()
    release["mh_sigma"] = np.linspace(0.01, 0.5, len(release))
    mask = matched_sigma_subsample(release, sigma_quantile=0.5)
    assert mask.sum() > 0
    assert mask.sum() < len(release)
    # All retained rows must have mh_sigma below the median.
    assert release.loc[mask, "mh_sigma"].max() <= release["mh_sigma"].median()


# --- Plot smoke ---------------------------------------------------------------


def test_render_all_writes_seven_families(tmp_path: Path, synthetic_release_and_catalogues):
    release, catalogues, bindings = synthetic_release_and_catalogues
    report = compute_cross_catalogue_report(release, catalogues, bindings)
    written = render_all(report, release, catalogues, bindings, tmp_path, apply_aa_style=False)
    expected_families = {
        "bland_altman",
        "residual_hist",
        "bias_vs_mh",
        "bias_vs_teff",
        "cell_heatmaps",
        "coverage",
        "rank_summary",
    }
    assert set(written.keys()) == expected_families
    # At least one plot per family.
    for family, paths in written.items():
        assert len(paths) > 0, f"family {family} produced no plots"
        for p in paths:
            assert p.exists(), f"missing PDF {p}"
            assert p.with_suffix(".png").exists(), f"missing PNG {p}"


def test_report_to_json_roundtrips_cells(tmp_path: Path, synthetic_release_and_catalogues):
    release, catalogues, bindings = synthetic_release_and_catalogues
    report = compute_cross_catalogue_report(release, catalogues, bindings)
    path = report.to_json(tmp_path / "report.json")
    assert path.exists()
    npz = path.with_suffix(".npz")
    assert npz.exists()
    # JSON should at least carry the cells list.
    import json

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "cells" in data
    assert len(data["cells"]) == len(report.cells)


# --- APOGEE benchmark tests (methods-paper Figure 8) ----------------------------


def test_apogee_benchmark_report_basic_fixture():
    """Synthetic fixture: ArqueoGal vs APOGEE DR19, zero residual.
    Verify basic structure and that pipelines are ranked."""
    rng = np.random.default_rng(42)
    n = 100

    # Create ArqueoGal predictions (perfect agreement with APOGEE)
    arqueogal = pd.DataFrame(
        {
            "teff_pred": 5000 + rng.normal(0, 10, n),
            "teff_sigma": np.full(n, 80.0),
            "logg_pred": 4.5 + rng.normal(0, 0.05, n),
            "logg_sigma": np.full(n, 0.10),
            "mh_pred": -0.2 + rng.normal(0, 0.05, n),
            "mh_sigma": np.full(n, 0.05),
        }
    )

    # APOGEE truth (perfect ground truth)
    apogee = pd.DataFrame(
        {
            "teff_apogee": arqueogal["teff_pred"].values,
            "e_teff_apogee": np.full(n, 50.0),
            "logg_apogee": arqueogal["logg_pred"].values,
            "e_logg_apogee": np.full(n, 0.05),
            "mh_apogee": arqueogal["mh_pred"].values,
            "e_mh_apogee": np.full(n, 0.03),
        }
    )

    # External pipeline (simulated bias)
    external = pd.DataFrame(
        {
            "teff_pred": apogee["teff_apogee"].values + 50.0,  # +50 K bias
            "teff_sigma": np.full(n, 80.0),
            "logg_pred": apogee["logg_apogee"].values + 0.1,  # +0.1 dex bias
            "logg_sigma": np.full(n, 0.10),
            "mh_pred": apogee["mh_apogee"].values,  # no bias
            "mh_sigma": np.full(n, 0.05),
        }
    )

    # Subset of LABEL_SCHEMA for this test
    label_schema_test = {
        "teff": LABEL_SCHEMA["teff"],
        "logg": LABEL_SCHEMA["logg"],
        "mh": LABEL_SCHEMA["mh"],
    }

    report = compute_apogee_benchmark_report(
        arqueogal,
        apogee,
        {"external": external},
        label_schema=label_schema_test,
    )

    # Check report structure
    assert len(report.pipelines) == 2  # ArqueoGal + external
    assert report.pipelines[0].pipeline_name == "ArqueoGal"
    assert report.pipelines[1].pipeline_name == "external"

    # ArqueoGal should have near-zero bias on all elements
    arqueogal_metrics = report.pipelines[0]
    for label in ["teff", "logg", "mh"]:
        assert abs(arqueogal_metrics.bias[label]) < 1.0, (
            f"ArqueoGal {label} bias should be near zero"
        )

    # External pipeline should show +50K bias on Teff, +0.1 dex on logg
    external_metrics = report.pipelines[1]
    assert external_metrics.bias["teff"] > 40.0, "External Teff bias should reflect +50K offset"
    assert external_metrics.bias["logg"] > 0.08, "External logg bias should reflect +0.1 dex offset"


def test_apogee_benchmark_missing_elements_handled():
    """When a pipeline doesn't publish an element, report NaN for that element."""
    rng = np.random.default_rng(43)
    n = 50

    arqueogal = pd.DataFrame(
        {
            "teff_pred": 5000 + rng.normal(0, 10, n),
            "teff_sigma": np.full(n, 80.0),
            "logg_pred": 4.5 + rng.normal(0, 0.05, n),
            "logg_sigma": np.full(n, 0.10),
        }
    )

    apogee = pd.DataFrame(
        {
            "teff_apogee": arqueogal["teff_pred"].values,
            "e_teff_apogee": np.full(n, 50.0),
            "logg_apogee": arqueogal["logg_pred"].values,
            "e_logg_apogee": np.full(n, 0.05),
        }
    )

    # External pipeline missing mh (incomplete catalogue)
    external = pd.DataFrame(
        {
            "teff_pred": apogee["teff_apogee"].values,
            "teff_sigma": np.full(n, 80.0),
            # Missing logg, mh intentionally
        }
    )

    label_schema_test = {
        "teff": LABEL_SCHEMA["teff"],
        "logg": LABEL_SCHEMA["logg"],
    }

    report = compute_apogee_benchmark_report(
        arqueogal,
        apogee,
        {"incomplete": external},
        label_schema=label_schema_test,
    )

    incomplete_metrics = [p for p in report.pipelines if p.pipeline_name == "incomplete"][0]
    assert not np.isfinite(incomplete_metrics.bias["logg"]), "Missing logg should yield NaN bias"
    assert np.isfinite(incomplete_metrics.bias["teff"]), "teff should be computed"


def test_label_schema_keys_match_release_columns():
    """Guard: every label schema entry must be a column convention used in
    release.py so the cross-catalogue framework never drifts from the
    actual release-column conventions.
    """
    from arqueogal.xp_abundances.main import release as release_mod

    for key, schema in LABEL_SCHEMA.items():
        assert release_mod._PER_ELEMENT_PRED_COL[key] == schema["pred"], (
            f"LABEL_SCHEMA[{key}].pred drifted from release._PER_ELEMENT_PRED_COL"
        )
        assert release_mod._PER_ELEMENT_SIGMA_COL[key] == schema["sigma"], (
            f"LABEL_SCHEMA[{key}].sigma drifted from release._PER_ELEMENT_SIGMA_COL"
        )
