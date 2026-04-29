"""Stage 27: Cross-catalogue Test-6 diagnostic gallery (synthetic demo).

Produces the seven Test-6 diagnostic plot families on a synthetic fixture
that mimics the Stream-1 holdout: 5000 stars covering the disc-giant region
with realistic per-element σ, then four reference catalogues injected with
known biases (zero-bias, +0.05 dex [M/H] offset, +50 K Teff offset, broad
random scatter mimicking a high-σ "loose" catalogue).

The output is a self-contained gallery rooted at
``reports/gallery/27_cross_catalogue/`` so every plot family is concretely
on disk for visual inspection. Real production Test-6 plots come out of
the same code path via ``scripts/run_cross_catalogue_validation.py`` once
external cross-matches are available.

Why a separate gallery script: the production CLI driver
(``run_cross_catalogue_validation.py``) is correctness-driven and refuses
to run without real cross-match parquets. This script provides the visual
acceptance gate the user can inspect today, and doubles as a worked example
of the diagnostic vocabulary for the methods-paper figure 7.
"""

from __future__ import annotations

# isort: off
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from arqueogal.utils.plotting import set_aa_style  # noqa: E402
from arqueogal.xp_abundances.main.cross_catalogue import (  # noqa: E402
    CatalogueBinding,
    compute_cross_catalogue_report,
    rank_summary,
    report_to_long_dataframe,
)
from arqueogal.xp_abundances.main.cross_catalogue_plots import render_all  # noqa: E402

# isort: on


OUT = REPO / "reports/gallery/27_cross_catalogue"


def _build_synthetic_fixture(n: int = 5000, seed: int = 20260428):
    """Stream-1-shaped synthetic release + four reference catalogues."""
    rng = np.random.default_rng(seed)

    # Truth distribution: disc giants (Teff 4000-5500, log g 1-3), some
    # turnoff (Teff 5000-6500, log g 3-4.5), broad metallicity tail.
    is_giant = rng.random(n) > 0.4
    truth_teff = np.where(
        is_giant,
        rng.uniform(4000, 5500, n),
        rng.uniform(5200, 6500, n),
    )
    truth_logg = np.where(
        is_giant,
        rng.uniform(1.0, 3.0, n),
        rng.uniform(3.0, 4.5, n),
    )
    # Bimodal [M/H]: thin disc peak + halo tail.
    is_thin = rng.random(n) > 0.25
    truth_mh = np.where(
        is_thin,
        rng.normal(-0.10, 0.18, n),
        rng.normal(-0.85, 0.45, n),
    )
    truth_mh = np.clip(truth_mh, -2.5, 0.5)
    # [α/M]: bimodal at high [α/M] for low [M/H], low [α/M] for high [M/H].
    truth_alpha_m = np.clip(
        np.where(truth_mh < -0.4, 0.22 + rng.normal(0, 0.04, n), 0.05 + rng.normal(0, 0.04, n)),
        -0.05,
        0.40,
    )
    truth_mg_h = truth_mh + 0.85 * (truth_alpha_m - 0.10) + rng.normal(0, 0.03, n)

    # ArqueoGal predictions: heteroscedastic noise calibrated to the
    # release-σ thresholds in release.py (Teff ~ 80 K, [M/H] ~ 0.05 dex).
    pred_sigma_teff = rng.uniform(60, 200, n)
    pred_sigma_logg = rng.uniform(0.08, 0.30, n)
    pred_sigma_mh = rng.uniform(0.03, 0.20, n)
    pred_sigma_alpha = rng.uniform(0.03, 0.08, n)
    pred_sigma_mgh = rng.uniform(0.04, 0.20, n)

    pred_teff = truth_teff + rng.standard_normal(n) * pred_sigma_teff
    pred_logg = truth_logg + rng.standard_normal(n) * pred_sigma_logg
    pred_mh = truth_mh + rng.standard_normal(n) * pred_sigma_mh
    pred_alpha = truth_alpha_m + rng.standard_normal(n) * pred_sigma_alpha
    pred_mgh = truth_mg_h + rng.standard_normal(n) * pred_sigma_mgh

    g_mag = rng.uniform(13.0, 17.0, n)
    release = pd.DataFrame(
        {
            "source_id": np.arange(n, dtype=np.int64),
            "g_mag": g_mag,
            "teff_pred": pred_teff,
            "logg_pred": pred_logg,
            "mh_pred": pred_mh,
            "alpha_m_pred": pred_alpha,
            "mg_h_pred": pred_mgh,
            "teff_sigma": pred_sigma_teff,
            "logg_sigma": pred_sigma_logg,
            "mh_sigma": pred_sigma_mh,
            "alpha_m_sigma": pred_sigma_alpha,
            "mg_h_sigma": pred_sigma_mgh,
            "release_tier": np.ones(n, dtype=np.int8),
        }
    )

    # Reference catalogue 1 (AspGap-like): well-calibrated, all 4 elements.
    aspgap = pd.DataFrame(
        {
            "Teff": truth_teff + rng.standard_normal(n) * 60,
            "logg": truth_logg + rng.standard_normal(n) * 0.08,
            "mh": truth_mh + rng.standard_normal(n) * 0.04,
            "alpha_m": truth_alpha_m + rng.standard_normal(n) * 0.03,
            "e_Teff": np.full(n, 60.0),
            "e_logg": np.full(n, 0.08),
            "e_mh": np.full(n, 0.04),
            "e_alpha_m": np.full(n, 0.03),
        }
    )

    # Reference catalogue 2 (SHBoost-like): +0.05 [M/H] systematic, no [α/M].
    shboost = pd.DataFrame(
        {
            "teff": truth_teff + rng.standard_normal(n) * 90,
            "logg": truth_logg + rng.standard_normal(n) * 0.12,
            "feh": truth_mh + 0.05 + rng.standard_normal(n) * 0.07,
            "teff_err": np.full(n, 90.0),
            "logg_err": np.full(n, 0.12),
            "feh_err": np.full(n, 0.07),
        }
    )

    # Reference catalogue 3 (Guiglion+2024-like): +50 K Teff offset, broad scatter.
    guiglion = pd.DataFrame(
        {
            "teff": truth_teff + 50.0 + rng.standard_normal(n) * 110,
            "logg": truth_logg + rng.standard_normal(n) * 0.15,
            "feh": truth_mh + rng.standard_normal(n) * 0.09,
            "alpha_m": truth_alpha_m + rng.standard_normal(n) * 0.06,
            "teff_err": np.full(n, 110.0),
            "logg_err": np.full(n, 0.15),
            "feh_err": np.full(n, 0.09),
            "alpha_m_err": np.full(n, 0.06),
        }
    )

    # Reference catalogue 4 (GALAH DR4-like): high-resolution, gold standard,
    # smaller scatter and gives [Mg/H] directly.
    galah = pd.DataFrame(
        {
            "teff": truth_teff + rng.standard_normal(n) * 50,
            "logg": truth_logg + rng.standard_normal(n) * 0.06,
            "fe_h": truth_mh + rng.standard_normal(n) * 0.03,
            "alpha_fe": truth_alpha_m + rng.standard_normal(n) * 0.025,
            "mg_fe": (truth_mg_h - truth_mh) + rng.standard_normal(n) * 0.025,
            "e_teff": np.full(n, 50.0),
            "e_logg": np.full(n, 0.06),
            "e_fe_h": np.full(n, 0.03),
            "e_alpha_fe": np.full(n, 0.025),
            "e_mg_fe": np.full(n, 0.025),
        }
    )
    # Convert [Mg/Fe] back to [Mg/H] for direct comparison.
    galah["mg_fe"] = truth_mg_h + rng.standard_normal(n) * 0.025

    catalogues = {
        "aspgap": aspgap,
        "shboost": shboost,
        "guiglion2024": guiglion,
        "galah_dr4": galah,
    }
    bindings = {
        "aspgap": CatalogueBinding(
            name="aspgap",
            column_for={"teff": "Teff", "logg": "logg", "mh": "mh", "alpha_m": "alpha_m"},
            sigma_for={
                "teff": "e_Teff",
                "logg": "e_logg",
                "mh": "e_mh",
                "alpha_m": "e_alpha_m",
            },
            citation="AspGap (Li+2024, synthetic)",
        ),
        "shboost": CatalogueBinding(
            name="shboost",
            column_for={"teff": "teff", "logg": "logg", "mh": "feh"},
            sigma_for={"teff": "teff_err", "logg": "logg_err", "mh": "feh_err"},
            citation="SHBoost (Khalatyan+2024, synthetic)",
        ),
        "guiglion2024": CatalogueBinding(
            name="guiglion2024",
            column_for={
                "teff": "teff",
                "logg": "logg",
                "mh": "feh",
                "alpha_m": "alpha_m",
            },
            sigma_for={
                "teff": "teff_err",
                "logg": "logg_err",
                "mh": "feh_err",
                "alpha_m": "alpha_m_err",
            },
            citation="Guiglion+2024 (synthetic)",
        ),
        "galah_dr4": CatalogueBinding(
            name="galah_dr4",
            column_for={
                "teff": "teff",
                "logg": "logg",
                "mh": "fe_h",
                "alpha_m": "alpha_fe",
                "mg_h": "mg_fe",
            },
            sigma_for={
                "teff": "e_teff",
                "logg": "e_logg",
                "mh": "e_fe_h",
                "alpha_m": "e_alpha_fe",
                "mg_h": "e_mg_fe",
            },
            citation="GALAH DR4 (Buder+2024, synthetic)",
        ),
    }
    return release, catalogues, bindings


def main() -> None:
    set_aa_style(usetex=False)
    print("[plot_27] Building synthetic fixture (5000 stars × 4 catalogues)")
    release, catalogues, bindings = _build_synthetic_fixture()

    print("[plot_27] Computing cross-catalogue report")
    report = compute_cross_catalogue_report(
        release,
        catalogues,
        bindings,
        min_per_bin=100,
    )
    OUT.mkdir(parents=True, exist_ok=True)

    print("[plot_27] Writing CSV / JSON sidecars")
    long = report_to_long_dataframe(report)
    long.to_csv(OUT / "cells.csv", index=False)
    rank_summary(report).to_csv(OUT / "rank_summary.csv", index=False)
    report.to_json(OUT / "report.json")

    print("[plot_27] Rendering all seven plot families")
    written = render_all(report, release, catalogues, bindings, OUT, apply_aa_style=False)
    total_pdfs = sum(len(paths) for paths in written.values())

    print("\n[plot_27] === Output summary ===")
    for family, paths in written.items():
        print(f"  {family:18s} {len(paths)} figure(s)")
    print(f"  Total: {total_pdfs} PDF + {total_pdfs} PNG companion files")
    print(f"  Output root: {OUT}")
    print()
    print("[plot_27] === Pass/fail summary on synthetic fixture ===")
    summary_cols = [
        "label",
        "catalogue",
        "mag_bin",
        "n",
        "bias",
        "scatter",
        "sigma_ratio",
        "passed",
    ]
    print(long[summary_cols].to_string(index=False, float_format=lambda x: f"{x:>+.4g}"))


if __name__ == "__main__":
    main()
