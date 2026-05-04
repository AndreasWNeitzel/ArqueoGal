"""G1: Per-star Av sources — pairwise comparison.

Stream-1 carries five independent Av estimates per star (when available):

  av_edenhofer    Edenhofer+2024 3-D dust map, valid d ≲ 1.25 kpc
  av_lallement    Lallement+2022 3-D map,        1.25 ≲ d ≲ 3 kpc
  av_sfd          SFD/SF2011 2-D map,             d ≳ 3 kpc fallback
  av_nbhd_median  per-star nearest-neighbour median (composite)
  ag_gspphot      Gaia DR3 GSP-Phot single-star fit

Pairwise hexbin scatter across these sources tells you (a) where they
agree, (b) where one map saturates, (c) where the nbhd-median composite
sits relative to its inputs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig  # noqa: E402

FEAT = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
OUT = REPO / "reports/gallery/G_extinction"
SOURCES = (
    ("av_edenhofer", "Edenhofer+2024 (d ≲ 1.25 kpc)"),
    ("av_lallement", "Lallement+2022 (1.25-3 kpc)"),
    ("av_sfd", "SFD/SF2011 (d ≳ 3 kpc)"),
    ("av_nbhd_median", "nbhd-median composite"),
    ("ag_gspphot", "Gaia GSP-Phot $A_G$"),
)


def main() -> int:
    apply_style()
    cols = ["source_id"] + [s[0] for s in SOURCES]
    df = pd.read_parquet(FEAT, columns=cols).drop_duplicates("source_id")
    n = len(df)

    n_src = len(SOURCES)
    fig, axes = plt.subplots(n_src, n_src, figsize=(20, 20))
    plt.subplots_adjust(wspace=0.30, hspace=0.30, top=0.95, bottom=0.04, left=0.05, right=0.97)

    for i in range(n_src):
        for j in range(n_src):
            ax = axes[i, j]
            xc, xn = SOURCES[j]
            yc, yn = SOURCES[i]
            x = df[xc].to_numpy()
            y = df[yc].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)

            if i == j:
                # Diagonal: histogram of this source.
                vals = x[ok]
                if vals.size:
                    ax.hist(
                        vals[vals < 6],
                        bins=80,
                        range=(0, 6),
                        color="#3a6ea5",
                        edgecolor="white",
                        linewidth=0.4,
                        alpha=0.85,
                    )
                    med = float(np.nanmedian(vals))
                    ax.axvline(med, color="#e07b00", lw=1.6, ls="--", label=f"med={med:.2f}")
                    ax.legend(fontsize=8, loc="upper right")
                ax.set_xlim(0, 6)
                ax.set_xlabel(rf"$A_V$  {xn}", fontsize=8)
                ax.set_title(xn, fontsize=9, color="#15355f")
            else:
                if ok.sum() >= 50:
                    ax.hexbin(
                        x[ok],
                        y[ok],
                        gridsize=60,
                        extent=(0, 6, 0, 6),
                        mincnt=1,
                        bins="log",
                        cmap="viridis",
                    )
                    ax.plot([0, 6], [0, 6], color="#e07b00", lw=1.2, ls="--", alpha=0.85)
                    ax.text(
                        0.05,
                        0.95,
                        f"n={int(ok.sum()):,}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="white",
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"n={int(ok.sum())}\n(too sparse)",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="gray",
                    )
                ax.set_xlim(0, 6)
                ax.set_ylim(0, 6)
                ax.set_aspect("equal")
                if i == n_src - 1:
                    ax.set_xlabel(xn, fontsize=8)
                if j == 0:
                    ax.set_ylabel(yn, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.20)

    fig.suptitle(
        "G1. Per-star $A_V$ source comparison  (Stream 1, "
        f"n = {n:,})\n"
        "Diagonal = single-source histogram with median.  "
        "Off-diagonal = pairwise hexbin (orange = 1:1 line).",
        fontsize=12,
        fontweight="semibold",
        y=0.985,
    )
    save_fig(fig, OUT / "G1_extinction_sources", tight=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
