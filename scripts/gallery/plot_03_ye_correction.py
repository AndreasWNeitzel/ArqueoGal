"""Stage 03: Ye+2024 NN flux correction across all three streams.

Layout 1 × 4: a sky map per stream (S1 / S2 / S3) of retained-after-Ye stars,
plus a G-mag histogram with all three streams overlaid showing the Ye-OK
distribution per stream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import (apply_style, save_fig, radec_to_galactic_mollweide,
                     style_galactic_mollweide, sample_index, PALETTE)

OUT = REPO / "reports/gallery/03_ye_correction"

STREAMS = [
    ("Stream 1", REPO / "data/processed/pipeline1_features_stream1.parquet",
     PALETTE["apogee"]),
    ("Stream 2", REPO / "data/processed/pipeline1_features_stream2.parquet",
     "#9467bd"),
    ("Stream 3", REPO / "data/processed/pipeline1_features_stream3.parquet",
     PALETTE["andrae_volume"]),
]


def main() -> None:
    apply_style()
    fig = plt.figure(figsize=(17, 5.5))
    gs = fig.add_gridspec(1, 4, wspace=0.30, width_ratios=[1, 1, 1, 1.1])

    rng = np.random.default_rng(0)
    loaded = []
    for i, (name, path, color) in enumerate(STREAMS):
        ax = fig.add_subplot(gs[0, i], projection="mollweide")
        if not path.exists():
            ax.set_title(f"{name}\n(not built)", fontsize=9)
            continue
        df = pd.read_parquet(path, columns=["ra_deg", "dec_deg", "g_mag"])
        loaded.append((name, df, color))
        idx = sample_index(len(df), 60_000, rng)
        x, y = radec_to_galactic_mollweide(df.ra_deg.iloc[idx].to_numpy(),
                                            df.dec_deg.iloc[idx].to_numpy())
        ax.scatter(x, y, s=0.4, alpha=0.35, color=color, rasterized=True)
        style_galactic_mollweide(ax)
        ax.set_title(f"{name} retained after Ye+2024\n(n={len(df):,})", fontsize=9)

    # G-mag histogram overlay
    ax = fig.add_subplot(gs[0, 3])
    bins = np.linspace(6, 18, 49)
    for name, df, color in loaded:
        ax.hist(df.g_mag.dropna(), bins=bins, density=True, histtype="step",
                color=color, lw=1.4, label=f"{name} (n={len(df):,})")
    ax.axvline(17.0, color="red", lw=0.9, ls="--", label="G = 17 cap")
    ax.set_xlabel("Gaia G (mag, corrected)")
    ax.set_ylabel("density")
    ax.set_title("Ye-OK G-mag distribution per stream")
    ax.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.4")

    fig.suptitle("Stage 03 — Ye+2024 NN flux correction outcome (post-Ye retained sample per stream)",
                  fontsize=10)
    save_fig(fig, OUT / "ye_correction.png")


if __name__ == "__main__":
    main()
