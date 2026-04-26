"""Stage 05: distance fusion + A_V cascade across all three streams.

Layout 2 × 3:
- Row 1 (distance): per-stream distance histogram (S1 / S2 / S3) and A_V map
  composition cross-validation overlay.
- Row 2 (extinction): per-map A_V distribution per stream.

Stream 1: BJ21 photogeometric distance.
Stream 2: GSP-Phot distance (BJ21 not fetched for S2; gspphot is the closest
proxy for asteroseismic giants, all in the local volume).
Stream 3: BJ21 photogeometric distance from the Andrae-pool chunks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig, PALETTE

OUT = REPO / "reports/gallery/05_distance_extinction"

STREAMS = [
    ("Stream 1", REPO / "data/processed/pipeline1_features_stream1.parquet",
     PALETTE["apogee"]),
    ("Stream 2", REPO / "data/processed/pipeline1_features_stream2.parquet",
     "#9467bd"),
    ("Stream 3", REPO / "data/processed/pipeline1_features_stream3.parquet",
     PALETTE["andrae_volume"]),
]

AV_MAPS = [("av_edenhofer", "edenhofer", "Edenhofer+24"),
           ("av_lallement", "lallement", "Lallement+22"),
           ("av_sfd",       "sfd",       "SFD"),
           ("av_nbhd_median", "nbhd",     "GSP-Phot nbhd")]


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    cols = ["r_med_photogeo",
            "av_edenhofer", "av_lallement", "av_sfd", "av_nbhd_median"]
    return pd.read_parquet(path, columns=[c for c in cols])


def main() -> None:
    apply_style()

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.30)

    # Row 1: distance histograms per stream + cross-validation overlay
    loaded = []
    for i, (name, path, color) in enumerate(STREAMS):
        ax = fig.add_subplot(gs[0, i])
        df = _load(path)
        if df is None:
            ax.set_title(f"{name}\n(not built)", fontsize=9)
            continue
        loaded.append((name, color, df))
        # r is parsecs → kpc
        r = df["r_med_photogeo"].to_numpy() / 1000.0
        rf = r[np.isfinite(r) & (r > 0) & (r < 15)]
        ax.hist(rf, bins=np.linspace(0, 10, 81), color=color, alpha=0.7)
        ax.set_xlabel("photogeo distance (kpc)")
        ax.set_ylabel("counts")
        ax.set_title(f"{name} distance (n={len(rf):,})", fontsize=9)
        ax.axvline(1.25, color="k", lw=0.6, ls="--", alpha=0.7)
        ax.axvline(3.0, color="k", lw=0.6, ls=":", alpha=0.7)
        med = float(np.nanmedian(rf))
        ax.text(0.96, 0.96, f"median = {med:.2f} kpc",
                 transform=ax.transAxes, fontsize=8, ha="right", va="top",
                 bbox=dict(facecolor="white", edgecolor="0.4", alpha=0.9, pad=2))

    # Row 2: per-map A_V distributions, one panel per stream
    for i, (name, color, df) in enumerate(loaded):
        ax = fig.add_subplot(gs[1, i])
        bins = np.linspace(0, 5, 61)
        for col, ckey, lbl in AV_MAPS:
            if col not in df.columns:
                continue
            v = df[col].to_numpy()
            v = v[np.isfinite(v) & (v < 5)]
            if len(v) < 10:
                continue
            ax.hist(v, bins=bins, alpha=0.45, color=PALETTE[ckey],
                     label=f"{lbl} ({len(v):,})")
        ax.set_xlabel(r"$A_V$ (mag)")
        ax.set_ylabel("counts")
        ax.set_yscale("log")
        ax.set_title(f"{name}: per-map $A_V$", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.95,
                  facecolor="white", edgecolor="0.4")

    fig.suptitle(
        "Stage 05 — distance + A_V cascade across S1 / S2 / S3 "
        "(no Bayestar19, budget-compliant). "
        "S2 uses GSP-Phot distance; S1/S3 use BJ21 photogeo.",
        fontsize=10,
    )
    save_fig(fig, OUT / "distance_extinction.png")


if __name__ == "__main__":
    main()
