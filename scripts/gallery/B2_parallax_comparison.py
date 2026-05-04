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

OUT = REPO / "reports/gallery/B_preprocessing"


def main() -> None:
    apply_style()

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

    # SNR diagnostics — DO NOT clip to hide pathology.
    #
    # Three regimes:
    #   ok            : 1 <= SNR <= 200 (typical Gaia DR3 disc-giant range).
    #   bright_outlier: 200 < SNR <= 1000 (Hipparcos-bright, physical but rare
    #                   in TESS asteroseismic-giant cohorts).
    #   pathological  : SNR > 1000 or parallax_error <= 0 / NaN. Likely an
    #                   upstream TAP-fetch bug where parallax_error came back
    #                   as zero or missing. These stars must be flagged and
    #                   dropped from kinematics + model inference.
    snr_full = np.where(
        np.isfinite(s2["parallax_error"]) & (s2["parallax_error"] > 0),
        np.abs(s2["parallax"]) / s2["parallax_error"],
        np.inf,
    )
    snr_quality = np.where(
        snr_full > 1000.0,
        "pathological",
        np.where(snr_full > 200.0, "bright_outlier", "ok"),
    )
    n_path = int(np.sum(snr_quality == "pathological"))
    n_bright = int(np.sum(snr_quality == "bright_outlier"))
    n_ok = int(np.sum(snr_quality == "ok"))

    finite = (
        np.isfinite(s2["parallax"])
        & np.isfinite(s2["parallax_corr"])
        & np.isfinite(s2["r_med_photogeo"])
        & np.isfinite(s2["parallax_error"])
        & (s2["parallax_error"] > 0)
    )
    s2 = s2.loc[finite].copy()
    snr_full = snr_full[finite]
    snr_quality = snr_quality[finite]
    # Plot SNR coloured on a log scale so the bright tail is visible without
    # clipping: viridis from 1 to 1000 covers the entire physical regime.
    snr_for_color = np.clip(snr_full, 1.0, 1000.0)
    n = int(finite.sum())

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    axes = axes.ravel()

    # Convert all three estimates onto the same kpc axis so 1:1 lines mean
    # something. r_inv_raw  = 1 / parallax_raw  (kpc, since parallax is mas),
    # r_med_photogeo is in PARSECS in the Gaia DR3 / Bailer-Jones+2021 schema
    # and must be divided by 1000 before comparison.
    r_inv_raw = 1.0 / s2["parallax"].clip(lower=0.001)
    r_inv_corr = 1.0 / s2["parallax_corr"].clip(lower=0.001)
    r_bj = s2["r_med_photogeo"] / 1000.0   # pc -> kpc

    diag = np.linspace(0, 6, 2)

    # Panel A: 1/parallax_raw vs Bailer-Jones, with 1:1 line.
    sc1 = axes[0].scatter(
        r_inv_raw,
        r_bj,
        c=snr_for_color,
        s=6,
        alpha=0.5,
        cmap="viridis",
        norm=plt.matplotlib.colors.LogNorm(vmin=1.0, vmax=1000.0),
        rasterized=True,
    )
    axes[0].plot(diag, diag, "k--", lw=0.9, label="1:1")
    axes[0].set_xlim(0, 6)
    axes[0].set_ylim(0, 6)
    axes[0].set_xlabel(r"$1 / \varpi_\mathrm{raw}$ [kpc]")
    axes[0].set_ylabel(r"$r_\mathrm{med,photogeo}$ [kpc] (Bailer-Jones)")
    axes[0].set_title(f"Stream 2: raw $1/\\varpi$ vs Bailer-Jones (n={n:,})")
    axes[0].legend(fontsize=8, loc="upper left")
    cb1 = plt.colorbar(sc1, ax=axes[0])
    cb1.set_label(r"SNR = $|\varpi|/\sigma_\varpi$")

    # Panel B: 1/parallax_corr vs Bailer-Jones, with 1:1 line.
    sc2 = axes[1].scatter(
        r_inv_corr,
        r_bj,
        c=snr_for_color,
        s=6,
        alpha=0.5,
        cmap="viridis",
        norm=plt.matplotlib.colors.LogNorm(vmin=1.0, vmax=1000.0),
        rasterized=True,
    )
    axes[1].plot(diag, diag, "k--", lw=0.9, label="1:1")
    axes[1].set_xlim(0, 6)
    axes[1].set_ylim(0, 6)
    axes[1].set_xlabel(r"$1 / \varpi_\mathrm{corr}$ [kpc]  (Lindegren+2021 zero-point)")
    axes[1].set_ylabel(r"$r_\mathrm{med,photogeo}$ [kpc]")
    axes[1].set_title(f"Stream 2: corrected $1/\\varpi$ vs Bailer-Jones (n={n:,})")
    axes[1].legend(fontsize=8, loc="upper left")
    cb2 = plt.colorbar(sc2, ax=axes[1])
    cb2.set_label("SNR")

    # Panel C: residual scatter (kpc) — both recipes overlaid, full SNR axis
    # with the two regime thresholds annotated.
    resid_raw = r_inv_raw - r_bj
    resid_corr = r_inv_corr - r_bj
    axes[2].scatter(
        snr_full,
        resid_raw,
        alpha=0.3,
        s=4,
        color="C3",
        label="raw $1/\\varpi$",
        rasterized=True,
    )
    axes[2].scatter(
        snr_full,
        resid_corr,
        alpha=0.3,
        s=4,
        color="C0",
        label="corrected $1/\\varpi$",
        rasterized=True,
    )
    axes[2].axhline(0, color="k", lw=0.7, ls="--")
    axes[2].axvline(200, color="0.5", lw=0.7, ls=":", label="bright-outlier threshold (200)")
    axes[2].axvline(1000, color="0.3", lw=0.7, ls=":", label="pathological threshold (1000)")
    axes[2].set_xscale("log")
    axes[2].set_ylim(-3, 3)
    axes[2].set_xlim(1.0, max(2000.0, float(np.nanmax(snr_full))) if n > 0 else 2000.0)
    axes[2].set_xlabel(r"SNR = $|\varpi|/\sigma_\varpi$  (log axis, no clipping)")
    axes[2].set_ylabel(r"$1/\varpi - r_\mathrm{BJ21}$ [kpc]")
    axes[2].set_title(
        f"Stream 2: residual vs parallax SNR  "
        f"(ok={n_ok:,}, bright={n_bright:,}, pathological={n_path:,})"
    )
    axes[2].legend(fontsize=7, loc="upper right")

    # Panel D: residual histogram — quantitative summary of the residual
    # distribution under each recipe. The user asked for a "residuals-like
    # plot"; this complements the scatter in panel C by showing the bulk
    # distribution centred on zero (or biased) per recipe.
    bins = np.linspace(-2.0, 2.0, 60)
    axes[3].hist(
        resid_raw,
        bins=bins,
        histtype="step",
        lw=1.4,
        color="C3",
        density=True,
        label=f"raw  (median {np.median(resid_raw):+.3f} kpc)",
    )
    axes[3].hist(
        resid_corr,
        bins=bins,
        histtype="step",
        lw=1.4,
        color="C0",
        density=True,
        label=f"corrected  (median {np.median(resid_corr):+.3f} kpc)",
    )
    axes[3].axvline(0.0, color="k", lw=0.7, ls="--")
    axes[3].set_xlabel(r"$1/\varpi - r_\mathrm{BJ21}$ [kpc]")
    axes[3].set_ylabel("density")
    axes[3].set_title("Stream 2: residual distribution")
    axes[3].legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "B2 — Stream 2 (Hon+2021 TESS asteroseismic giants):  "
        "raw vs corrected parallax  +  Bailer-Jones photogeometric distance.  "
        "SNR > 200 = bright-tail outlier; SNR > 1000 = pathological (zero σ_π or upstream bug, must be flagged).",
        fontsize=10,
    )
    save_fig(fig, OUT / "B2_parallax_comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot B2: Parallax comparison.")
    args = parser.parse_args()
    main()
