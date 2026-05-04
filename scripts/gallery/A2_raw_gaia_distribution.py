"""A2: Raw Gaia data distribution per stream (sky maps + magnitude histograms).

What this shows:
- A focused view of raw Gaia DR3 sky coverage and magnitude distributions,
  separately for Streams 1, 2, 3 (side-by-side comparison).
- Sky Mollweide projections in Galactic coordinates.
- G-magnitude histograms with per-stream statistics.

This differs from A1 in scope: A1 emphasizes the pre-cut → in-cut transition
and Kiel diagrams. A2 emphasizes raw Gaia data alone, for documentation of
the three input catalogues' geometric and photometric reach.

What it reads:
- data/processed/pipeline1_features_stream1_kiel.parquet
- data/processed/pipeline1_features_stream2.parquet (if available)
- data/processed/pipeline1_features_stream3.parquet
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

from _common import (
    PALETTE,
    apply_style,
    radec_to_galactic_mollweide,
    sample_index,
    save_fig,
    style_galactic_mollweide,
)

OUT = REPO / "reports/gallery/A_raw_data"


def main() -> None:
    apply_style()

    s1 = pd.read_parquet(
        REPO / "data/processed/pipeline1_features_stream1_kiel.parquet",
        columns=["source_id", "ra_deg", "dec_deg", "g_mag"],
    )
    s3 = pd.read_parquet(
        REPO / "data/processed/pipeline1_features_stream3.parquet",
        columns=["source_id", "ra_deg", "dec_deg", "g_mag"],
    )

    s2_path = REPO / "data/processed/pipeline1_features_stream2.parquet"
    if s2_path.exists():
        s2 = pd.read_parquet(s2_path, columns=["source_id", "ra_deg", "dec_deg", "g_mag"])
    else:
        s2 = None

    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.25)

    streams = [
        (s1, "Stream 1 — APOGEE × Gaia", PALETTE["apogee"]),
        (s2, "Stream 2 — TESS asteroseismic", "#9467bd"),
        (s3, "Stream 3 — Andrae RGB volume", PALETTE["andrae_volume"]),
    ]

    # Row 1: Sky Mollweides
    for col, (df, name, color) in enumerate(streams):
        ax = fig.add_subplot(gs[0, col], projection="mollweide")
        if df is None:
            ax.text(0, 0, "Stream 2 not available", ha="center", va="center", fontsize=10)
            ax.set_title(name + " (pending)")
            continue

        idx = sample_index(len(df), 50_000, rng)
        x, y = radec_to_galactic_mollweide(
            df["ra_deg"].iloc[idx].to_numpy(),
            df["dec_deg"].iloc[idx].to_numpy(),
        )
        ax.scatter(x, y, s=0.4, alpha=0.35, color=color, rasterized=True)
        style_galactic_mollweide(ax)
        ax.set_title(f"{name}\n(n = {len(df):,})", fontsize=9)

    # Row 2: G-magnitude histograms
    ax_g = fig.add_subplot(gs[1, :])
    bins = np.linspace(6, 18, 49)

    all_g = []
    labels_g = []
    colors_list = []

    for df, name, color in streams:
        if df is not None:
            all_g.append(df["g_mag"].dropna().values)
            labels_g.append(f"{name.split(' —')[0]}\n(n = {len(df):,})")
            colors_list.append(color)

    ax_g.hist(
        all_g,
        bins=bins,
        label=labels_g,
        color=colors_list,
        alpha=0.6,
        stacked=False,
    )
    ax_g.axvline(17.0, color="red", lw=1.2, ls="--", label=r"$G$ = 17 cap", zorder=10)
    ax_g.set_xlabel(r"Gaia $G$ [mag, corrected]", fontsize=10)
    ax_g.set_ylabel("count", fontsize=10)
    ax_g.set_title("G-magnitude distribution per stream", fontsize=10)
    ax_g.legend(fontsize=9, loc="upper left", ncol=4, frameon=True, framealpha=0.95)
    ax_g.set_yscale("log")

    fig.suptitle(
        "A2 — Raw Gaia DR3 distribution per stream: sky coverage and magnitude reach",
        fontsize=11,
    )
    save_fig(fig, OUT / "A2_raw_gaia_distribution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot A2: Raw Gaia distribution per stream.")
    args = parser.parse_args()
    main()
