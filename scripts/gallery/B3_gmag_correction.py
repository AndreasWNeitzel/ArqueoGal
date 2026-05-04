"""B3: Raw vs corrected G-magnitude (Riello+2021 correction visualization).

What this shows:
- Left panel: raw G vs corrected G with 1:1 line overlay, stratified by colour bin.
- Right panel: ΔG (correction magnitude) vs raw G, stratified by BP-RP colour,
  showing the cubic color dependence of the Riello+2021 correction.

What it reads:
- data/processed/pipeline1_features_stream1_kiel.parquet (must have g_mag, bp_rp).
- Provenance audit in data/processed/pipeline1_features_stream1.provenance.json
  for detailed per-star correction history (optional).
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

    # The processed Stream-1 feature parquet only carries the corrected
    # G (column ``g_mag``). The upstream interim parquet
    # ``data/interim/stream1_apogee_gaia.parquet`` has both
    # ``phot_g_mean_mag`` (raw) and ``phot_g_mean_mag_corr`` (Riello+2021
    # corrected). Join on source_id to recover the per-star delta.
    s1_proc = pd.read_parquet(
        REPO / "data/processed/pipeline1_features_stream1_kiel.parquet",
        columns=["source_id", "bp_rp"],
    )
    s1_int = pd.read_parquet(
        REPO / "data/interim/stream1_apogee_gaia.parquet",
        columns=["source_id", "phot_g_mean_mag", "phot_g_mean_mag_corr"],
    )
    s1 = s1_proc.merge(s1_int, on="source_id", how="inner").drop_duplicates("source_id")

    g_raw = s1["phot_g_mean_mag"].values
    g_corr = s1["phot_g_mean_mag_corr"].values
    bp_rp = s1["bp_rp"].values
    m = np.isfinite(g_raw) & np.isfinite(g_corr) & np.isfinite(bp_rp)
    delta_g = g_corr - g_raw  # mag (Riello correction is sub-mmag in most cases)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: raw vs corrected with 1:1 line. Most stars sit on the diagonal
    # because the correction is sub-mmag for the bulk; the ~3% with non-zero
    # corrections deviate.
    sc0 = axes[0].scatter(
        g_raw[m],
        g_corr[m],
        c=bp_rp[m],
        s=3,
        alpha=0.4,
        cmap="coolwarm",
        rasterized=True,
    )
    plt.colorbar(sc0, ax=axes[0], label=r"BP − RP [mag]")
    g_lims = [g_raw[m].min(), g_raw[m].max()]
    axes[0].plot(g_lims, g_lims, "k--", lw=0.6, alpha=0.7, label="1:1")
    axes[0].set_xlabel(r"$G_\mathrm{raw}$ [mag]")
    axes[0].set_ylabel(r"$G_\mathrm{corrected}$ [mag] (Riello+2021)")
    axes[0].set_title(f"Raw vs Riello-corrected G (n={int(m.sum()):,})")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(True, alpha=0.25)

    # Panel 2: residual ΔG vs raw G in MAGNITUDES (not mmag). Per the user
    # request, keep the y-axis in mag; use a tight ylim to make the bulk
    # residuals visible.
    scatter = axes[1].scatter(
        g_raw[m],
        delta_g[m],
        c=bp_rp[m],
        s=3,
        alpha=0.4,
        cmap="coolwarm",
        rasterized=True,
    )
    axes[1].axhline(0, color="k", lw=0.6, ls="--", alpha=0.6)
    # Auto-tight y-range: clip to the bulk percentiles so a few extreme
    # corrections do not flatten the vertical spread.
    if m.sum() > 0:
        p05, p95 = np.nanpercentile(delta_g[m], [0.5, 99.5])
        if p95 > p05:
            axes[1].set_ylim(p05 - 0.0005, p95 + 0.0005)
    axes[1].set_xlabel(r"$G_\mathrm{raw}$ [mag]")
    axes[1].set_ylabel(r"$\Delta G = G_\mathrm{corr} - G_\mathrm{raw}$ [mag]")
    axes[1].set_title(f"Riello+2021 correction magnitude (n={int(m.sum()):,})")
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label(r"BP − RP [mag]")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        "B3 — Stream 1 (APOGEE × Gaia DR3): Riello+2021 G-mag correction "
        "(blue→red colormap = bluer→redder BP−RP)",
        fontsize=11,
    )
    save_fig(fig, OUT / "B3_gmag_correction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot B3: G-magnitude correction.")
    args = parser.parse_args()
    main()
