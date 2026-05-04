"""B9: Bailer-Jones+2021 distance and fused A_V distributions per stream.

What this shows (real data only):
- Top row: histogram of r_med_photogeo (BJ21 photogeometric distance) per
  stream (Stream 1, 2, 3).
- Bottom row: histogram of av_los (fused per-star A_V from
  arqueogal.data.extinction.apply_extinction_corrections) per stream,
  log-y, with vertical lines at the median.

What it reads:
- data/processed/pipeline1_features_stream1_kiel.parquet (Stream 1,
  Kiel-bounded RGB pool: logg ∈ [1.0, 3.5], Teff ∈ [4000, 5500] K)
- data/processed/pipeline1_features_stream{2,3}.parquet (columns:
  r_med_photogeo, av_los).

Stars with non-finite r_med_photogeo or av_los are dropped from the
respective panel.
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
STREAMS = (1, 2, 3)
STREAM_LABEL = {
    1: "Stream 1 (APOGEE × Gaia, Kiel-masked RGB)",
    2: "Stream 2 (TESS giants)",
    3: "Stream 3 (Andrae)",
}
STREAM_COLOR = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
# Stream 1 = Kiel-bounded training pool. Streams 2 and 3 keep their full
# inference cohorts (no Kiel mask — the bbox is a training-time decision).
STREAM_PARQUET = {
    1: "pipeline1_features_stream1_kiel.parquet",
    2: "pipeline1_features_stream2.parquet",
    3: "pipeline1_features_stream3.parquet",
}


def main() -> None:
    apply_style()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")

    for col, sid in enumerate(STREAMS):
        path = REPO / "data/processed" / STREAM_PARQUET[sid]
        df = pd.read_parquet(path, columns=["r_med_photogeo", "av_los"])

        # Top: distance distribution
        ax = axes[0, col]
        d = df["r_med_photogeo"].dropna()
        # BJ21 distances are reported in pc; convert to kpc for readable axis.
        d_kpc = d / 1000.0
        bins = np.linspace(0, np.nanpercentile(d_kpc, 99), 60)
        ax.hist(
            d_kpc, bins=bins, color=STREAM_COLOR[sid], alpha=0.85, edgecolor="black", linewidth=0.3
        )
        ax.axvline(
            d_kpc.median(), color="black", lw=1.2, ls="--", label=f"median {d_kpc.median():.2f} kpc"
        )
        ax.set_xlabel(r"$r_\mathrm{med, photogeo}$ [kpc] (Bailer-Jones+2021)")
        ax.set_ylabel("count")
        ax.set_title(f"{STREAM_LABEL[sid]}: distance (n={int(d.size):,})")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

        # Bottom: A_V distribution (log-y to show the long tail without clipping)
        ax = axes[1, col]
        av = df["av_los"].dropna()
        # Clip the upper bin edge at the 99.5th percentile so the bulk locus is
        # visible — the long pathological tail (av > 10) is still counted but
        # collapsed into the rightmost bin.
        av_p995 = float(np.nanpercentile(av, 99.5))
        bins = np.linspace(0, max(av_p995, 0.5), 60)
        ax.hist(
            av, bins=bins, color=STREAM_COLOR[sid], alpha=0.85, edgecolor="black", linewidth=0.3
        )
        ax.axvline(
            av.median(), color="black", lw=1.2, ls="--", label=f"median {av.median():.3f} mag"
        )
        ax.set_xlabel(r"$A_V^\mathrm{LOS}$ [mag] (fused dust map)")
        ax.set_ylabel("count (log)")
        ax.set_yscale("log")
        ax.set_title(f"{STREAM_LABEL[sid]}: $A_V$ (n={int(av.size):,})")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "B9 — Distance (Bailer-Jones+2021) and fused $A_V$ per stream (real data)",
        fontsize=11,
        fontweight="semibold",
    )
    save_fig(fig, OUT / "B9_distance_extinction", formats=("pdf", "png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot B9: distance and A_V per stream.")
    args = parser.parse_args()
    main()
