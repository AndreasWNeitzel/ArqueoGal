"""B4: Raw vs dereddened broadband photometry (J/H/K/W1/W2).

What this shows:
- Five-panel grid (one per band: J, H, K, W1, W2) showing raw vs dereddened
  magnitudes as a hexbin density plot plus a 1:1 reference line.
- Yuan+2013 extinction law implicit in the synthetic fixture.

What it reads:
- data/processed/pipeline1_features_stream1_kiel.parquet (must have j_mag, h_mag,
  k_mag, w1_mag, w2_mag columns and their dereddened counterparts).
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


def main() -> None:
    apply_style()

    cols = [
        "j_mag",
        "h_mag",
        "k_mag",
        "w1_mag",
        "w2_mag",
        "j_mag_dered",
        "h_mag_dered",
        "k_mag_dered",
        "w1_mag_dered",
        "w2_mag_dered",
    ]
    import pyarrow.parquet as pq
    schema_cols = {f.name for f in pq.ParquetFile(
        REPO / "data/processed/pipeline1_features_stream1_kiel.parquet").schema_arrow}
    cols = [c for c in cols if c in schema_cols]
    if "av_los" in schema_cols:
        cols = list(set(cols + ["av_los"]))
    df = pd.read_parquet(
        REPO / "data/processed/pipeline1_features_stream1_kiel.parquet", columns=cols
    )

    # Mask out pathological extinction values that produce unphysical
    # negative dereddened magnitudes (~0.3% of stars carry av_los > 10
    # from upstream dust-fusion failures). Without this filter the y-axis
    # is dominated by a thin negative-mag tail and the bulk locus is invisible.
    if "av_los" in df.columns:
        sane = (df["av_los"] >= 0) & (df["av_los"] < 5.0)
        n_dropped = int((~sane).sum())
        df = df.loc[sane].copy()
        if n_dropped:
            print(f"[B4] dropped {n_dropped:,} stars with av_los outside [0, 5) mag")

    bands = ["j", "h", "k", "w1", "w2"]
    band_names = ["J (2MASS)", "H (2MASS)", "Ks (2MASS)", "W1 (AllWISE)", "W2 (AllWISE)"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))

    for idx, (band, band_name) in enumerate(zip(bands, band_names)):
        ax = axes[idx]
        mag_col = f"{band}_mag"
        dered_col = f"{band}_mag_dered"

        if mag_col not in df.columns or dered_col not in df.columns:
            ax.text(
                0.5, 0.5, f"{band}: data missing", ha="center", va="center", transform=ax.transAxes
            )
            continue

        mag_raw = df[mag_col].values
        mag_dered = df[dered_col].values
        # 2MASS / WISE catalogues use sentinel values (99.99, -999999, etc.)
        # for missing detections. Filter to physically reasonable magnitudes
        # before hexbinning so the colour scale isn't dominated by sentinels.
        m = (
            np.isfinite(mag_raw) & np.isfinite(mag_dered)
            & (mag_raw > -5) & (mag_raw < 25)
            & (mag_dered > -5) & (mag_dered < 25)
        )

        if m.sum() > 100:
            h = ax.hexbin(
                mag_raw[m], mag_dered[m], gridsize=40, mincnt=3, cmap="viridis", bins="log"
            )
            plt.colorbar(h, ax=ax, label="log10 N")

            # 1:1 line
            lims = [mag_raw[m].min(), mag_raw[m].max()]
            ax.plot(lims, lims, "r--", lw=0.7, alpha=0.6, label="1:1")
            ax.set_xlabel(f"{band.upper()} raw [mag]")
            ax.set_ylabel(f"{band.upper()} dereddened [mag]")
            ax.set_title(f"{band_name} (n={m.sum():,})")
            ax.legend(fontsize=7)

    fig.suptitle(
        "B4 — Stream 1: Raw vs dereddened broadband IR photometry (J/H/K/W1/W2)", fontsize=11
    )
    save_fig(fig, REPO / "reports/gallery/B_preprocessing" / "B4_dereddened_broadbands", formats=("pdf", "png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot B4: Dereddened broadbands.")
    args = parser.parse_args()
    main()
