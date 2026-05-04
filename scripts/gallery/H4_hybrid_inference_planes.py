"""H4: Hybrid inference planes — Stream 2 and Stream 3 real output summary.

What this shows (two columns, three rows):
- Top row: Kiel diagram (Teff vs log g).
- Middle row: Chemical plane ([alpha/M] vs [M/H]).
- Bottom row: [Fe/H] histogram.

Each column represents one stream (Stream 2 / Stream 3). This is the standard
3-panel summary of spectroscopic inference quality and population differences,
using real hybrid or standard predictions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, load_real_stream, save_fig

OUT = REPO / "reports/gallery/H_hybrid_release"


def main(argv: list[str] | None = None) -> int:
    apply_style()
    print("[H4] Loading real Stream 2 and 3 predictions")

    data = {}
    # Apples-to-apples comparison: load the RAW regressor predictions for
    # all three streams (no hybrid composer mixing kNN rescue). The hybrid
    # output for Stream 3 used to inherit bimodality from the kNN-rescue
    # path's training-set truth labels, masking the underlying regressor
    # behaviour. Showing the raw regressor predictions across all streams
    # exposes whether the encoder really learned the disc-bimodality
    # geometry or whether downstream post-processing was painting it back in.
    for stream_id in (1, 2, 3):
        try:
            df = pd.read_parquet(
                REPO / f"data/processed/pipeline1_predictions_stream{stream_id}.parquet",
                columns=[
                    "source_id",
                    "teff_pred",
                    "logg_pred",
                    "mh_pred",
                    "alpha_m_pred",
                ],
            )
            # Stream 1 predictions inherit per-visit APOGEE duplicates; collapse
            # to one row per source_id for the chemistry-plane density panels.
            df = df.drop_duplicates(subset="source_id", keep="first")
            data[f"s{stream_id}"] = df
            print(f"  Stream {stream_id}: {len(df)} unique sources (raw regressor predictions)")
        except FileNotFoundError as e:
            print(f"  Error: Stream {stream_id} not available: {e}")
            return 1

    s2 = data["s2"]
    s1 = data["s1"]
    s2 = data["s2"]
    s3 = data["s3"]

    print("[H4] Rendering inference output planes (Kiel + chemical + histogram)")
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    KIEL_GRID = 70
    CHEM_GRID = 80
    KIEL_EXTENT = (3500, 6500, 0.5, 4.0)
    CHEM_EXTENT = (-2.5, 0.6, -0.20, 0.55)

    streams = [(s1, "Stream 1", "#1f77b4"),
               (s2, "Stream 2", "#d62728"),
               (s3, "Stream 3", "#ff7f0e")]

    # Row 0: Kiel diagram — hex density per stream.
    for col, (df, name, _color) in enumerate(streams):
        ax = axes[0, col]
        hb = ax.hexbin(
            df["teff_pred"], df["logg_pred"],
            gridsize=KIEL_GRID, cmap="viridis", mincnt=1, bins="log",
            extent=KIEL_EXTENT,
        )
        plt.colorbar(hb, ax=ax, label=r"log$_{10}$ N per cell")
        ax.set_xlabel(r"$T_\mathrm{eff}$ (K)")
        ax.set_ylabel(r"$\log g$ (dex)")
        ax.set_title(f"{name} - Kiel (n={len(df):,})")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(alpha=0.3)

    # Row 1: Chemical plane [alpha/M] vs [M/H] — hex density per stream.
    for col, (df, name, _color) in enumerate(streams):
        ax = axes[1, col]
        hb = ax.hexbin(
            df["mh_pred"], df["alpha_m_pred"],
            gridsize=CHEM_GRID, cmap="viridis", mincnt=1, bins="log",
            extent=CHEM_EXTENT,
        )
        plt.colorbar(hb, ax=ax, label=r"log$_{10}$ N per cell")
        ax.set_xlabel("[M/H] (dex)")
        ax.set_ylabel(r"[$\alpha$/M] (dex)")
        ax.set_title(f"{name} - Chemical (n={len(df):,})")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    save_fig(fig, OUT / "H4_hybrid_inference_planes.pdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
