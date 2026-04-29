"""B2: Raw vs corrected parallax comparison for Stream 2.

What this shows:
- Three-panel scatter plot showing raw parallax, corrected parallax, and
  Bailer-Jones median distance (r_med_photogeo) for Stream 2 asteroseismic
  giants. The divergence between 1/parallax_raw and r_med_photogeo at low
  parallax SNR is shown explicitly.
- Each panel colour-coded by parallax over error (SNR proxy).

What it reads:
- data/interim/stream2_tess_gaia.parquet (or post-cut features if available):
  must have parallax, parallax_error, r_med_photogeo columns.

Synthetic fixture support: --synthetic flag generates realistic Stream 2
photometric and astrometric uncertainties.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/B2_parallax_comparison"


def _make_synthetic():
    """Generate synthetic Stream 2 parallax data with realistic uncertainty structure."""
    rng = np.random.default_rng(42)
    n = 3000

    # True distances: asteroseismic giants mostly local, 0.5-5 kpc
    r_true = rng.exponential(1.5, n) + 0.5

    # Gaia parallax with realistic SNR structure (better at brighter G)
    g = rng.uniform(8, 16, n)
    snr_scale = 100.0 / (10.0 ** (g / 2.5))  # crude approximation
    parallax_snr = rng.normal(snr_scale, snr_scale * 0.1)
    parallax_snr = np.maximum(parallax_snr, 1.0)  # floor at SNR=1
    parallax_true = 1000.0 / r_true  # mas
    parallax_error = parallax_true / parallax_snr
    parallax_raw = rng.normal(parallax_true, parallax_error)
    parallax_corr = parallax_raw + 0.02  # Lindegren-like correction

    # Bailer-Jones distance
    r_med = parallax_true + rng.normal(0, 0.1 * parallax_true, n)
    r_med = np.maximum(r_med, 0.1)

    return pd.DataFrame({
        "parallax": parallax_raw,
        "parallax_corr": parallax_corr,
        "parallax_error": parallax_error,
        "r_med_photogeo": r_med,
    })


def main(use_synthetic: bool = False) -> None:
    apply_style()

    if use_synthetic:
        s2 = _make_synthetic()
    else:
        candidates = [
            REPO / "data/processed/pipeline1_features_stream2.parquet",
            REPO / "data/interim/stream2_tess_gaia.parquet",
        ]
        s2 = None
        for path in candidates:
            if path.exists():
                cols = ["parallax", "parallax_error", "parallax_corr", "r_med_photogeo"]
                schema = pd.read_parquet(path).iloc[:0]
                cols_avail = [c for c in cols if c in schema.columns]
                if cols_avail:
                    s2 = pd.read_parquet(path, columns=cols_avail)
                    break
        if s2 is None:
            s2 = pd.DataFrame()

    if len(s2) == 0:
        print("No Stream 2 data available.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Compute SNR proxy
    snr = np.abs(s2["parallax"]) / s2["parallax_error"]
    snr = np.clip(snr, 0.1, 1000)

    # Panel 1: raw parallax vs Bailer-Jones
    m = np.isfinite(s2["parallax"]) & np.isfinite(s2["r_med_photogeo"])
    scatter1 = axes[0].scatter(
        s2[m]["parallax"],
        s2[m]["r_med_photogeo"],
        c=snr[m],
        s=5,
        alpha=0.5,
        cmap="viridis",
        rasterized=True,
    )
    axes[0].set_xlabel(r"$\varpi_\mathrm{raw}$ [mas]")
    axes[0].set_ylabel(r"$r_\mathrm{med,photogeo}$ [kpc]")
    axes[0].set_title(f"Raw parallax vs Bailer-Jones (n={m.sum():,})")
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label(r"SNR = $|\varpi|/\sigma_\varpi$")

    # Panel 2: corrected parallax vs Bailer-Jones
    m2 = np.isfinite(s2["parallax_corr"]) & np.isfinite(s2["r_med_photogeo"])
    scatter2 = axes[1].scatter(
        s2[m2]["parallax_corr"],
        s2[m2]["r_med_photogeo"],
        c=snr[m2],
        s=5,
        alpha=0.5,
        cmap="viridis",
        rasterized=True,
    )
    axes[1].set_xlabel(r"$\varpi_\mathrm{corr}$ [mas]")
    axes[1].set_ylabel(r"$r_\mathrm{med,photogeo}$ [kpc]")
    axes[1].set_title(f"Corrected parallax vs Bailer-Jones (n={m2.sum():,})")
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label("SNR")

    # Panel 3: Comparison of 1/parallax vs r_med_photogeo, stratified by SNR
    m3 = m & m2
    dist_from_parallax_raw = 1000.0 / np.clip(s2[m3]["parallax"], 0.01, 1000)
    dist_from_parallax_corr = 1000.0 / np.clip(s2[m3]["parallax_corr"], 0.01, 1000)
    r_med = s2[m3]["r_med_photogeo"]

    # Residuals relative to Bailer-Jones
    resid_raw = dist_from_parallax_raw - r_med
    resid_corr = dist_from_parallax_corr - r_med

    axes[2].scatter(snr[m3], resid_raw, alpha=0.3, s=4, label="raw parallax", rasterized=True)
    axes[2].scatter(snr[m3], resid_corr, alpha=0.3, s=4, label="corrected parallax", rasterized=True)
    axes[2].axhline(0, color="k", lw=0.6, ls="--")
    axes[2].set_xlabel("SNR")
    axes[2].set_ylabel(r"$r_\mathrm{parallax} - r_\mathrm{BJ21}$ [kpc]")
    axes[2].set_title(f"Distance residual vs SNR (n={m3.sum():,})")
    axes[2].set_xscale("log")
    axes[2].legend(fontsize=8, loc="best")

    fig.suptitle("B2 — Raw vs corrected parallax for Stream 2 asteroseismic giants", fontsize=11)
    save_fig(fig, OUT / "parallax_comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot B2: Parallax comparison.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic fixture.")
    args = parser.parse_args()
    main(use_synthetic=args.synthetic)
