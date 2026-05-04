"""G2: Per-star Av vs distance, by line-of-sight zone.

Three panels:

  (a) Av (each source) vs d_med — overlaid scatter / hexbin so you see
      where Edenhofer / Lallement / SFD switch over.
  (b) per-zone (≤1.25 kpc / 1.25-3 kpc / >3 kpc) Av histograms
      with the canonical zone limits annotated.
  (c) per-source Av vs the nbhd-median composite (the actual value the
      pipeline uses) — residual band tells you whether the composite is
      pulling or pushing.
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

ZONE_EDGES = (0.0, 1.25, 3.0, 12.0)  # kpc
SOURCE_COLOR = {
    "av_edenhofer": "#1f77b4",
    "av_lallement": "#2ca02c",
    "av_sfd": "#d62728",
    "av_nbhd_median": "#7d3c98",
}
SOURCE_LABEL = {
    "av_edenhofer": "Edenhofer+2024",
    "av_lallement": "Lallement+2022",
    "av_sfd": "SFD/SF2011",
    "av_nbhd_median": "nbhd-median composite",
}


def main() -> int:
    apply_style()
    cols = [
        "source_id",
        "r_med_photogeo",
        "av_edenhofer",
        "av_lallement",
        "av_sfd",
        "av_nbhd_median",
    ]
    df = pd.read_parquet(FEAT, columns=cols).drop_duplicates("source_id")
    df = df.dropna(subset=["r_med_photogeo"])
    df["d_kpc"] = df["r_med_photogeo"] / 1000.0
    df = df.loc[(df["d_kpc"] > 0) & (df["d_kpc"] < 12)]
    n = len(df)

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(
        2, 3, hspace=0.35, wspace=0.30, top=0.92, bottom=0.06, left=0.05, right=0.97
    )

    # (a) Av vs distance per source — one subpanel per source for clarity.
    ax = fig.add_subplot(gs[0, :])
    for src, color in SOURCE_COLOR.items():
        d = df["d_kpc"].to_numpy()
        v = df[src].to_numpy()
        ok = np.isfinite(d) & np.isfinite(v) & (v >= 0) & (v < 6)
        if ok.sum() < 100:
            continue
        # Median curve in distance bins.
        bins = np.linspace(0, 12, 60)
        med = np.array(
            [
                np.median(v[ok & (d >= bins[k]) & (d < bins[k + 1])])
                if int(((d >= bins[k]) & (d < bins[k + 1]) & ok).sum()) > 50
                else np.nan
                for k in range(len(bins) - 1)
            ]
        )
        x = 0.5 * (bins[:-1] + bins[1:])
        ax.plot(
            x,
            med,
            "-",
            color=color,
            lw=2.2,
            label=f"{SOURCE_LABEL[src]}  (n_star = {int(ok.sum()):,})",
        )
    for ze in ZONE_EDGES[1:-1]:
        ax.axvline(ze, color="0.4", ls=":", lw=1.2)
    ax.text(
        0.625,
        0.95,
        "Edenhofer\nzone",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="0.4",
    )
    ax.text(
        2.125 / 12,
        0.95,
        "Lallement zone",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.4",
    )
    ax.text(
        7.5 / 12,
        0.95,
        "SFD zone",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.4",
    )
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.set_xlabel("BJ21 distance (kpc)")
    ax.set_ylabel(r"median $A_V$ in 0.2-kpc bin (mag)")
    ax.set_title("(a) $A_V$ vs distance per source")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.25)

    # (b) per-zone Av histograms (using the nbhd-median composite).
    ax = fig.add_subplot(gs[1, 0])
    bins_av = np.linspace(0, 6, 60)
    for _k, (lo, hi, color, name) in enumerate(
        [
            (0.0, 1.25, "#1f77b4", "Edenhofer (≤1.25 kpc)"),
            (1.25, 3.0, "#2ca02c", "Lallement (1.25-3 kpc)"),
            (3.0, 12.0, "#d62728", "SFD (>3 kpc)"),
        ]
    ):
        m = (df["d_kpc"] >= lo) & (df["d_kpc"] < hi)
        v = df.loc[m, "av_nbhd_median"].dropna()
        if not len(v):
            continue
        ax.hist(
            v[v < 6],
            bins=bins_av,
            histtype="step",
            color=color,
            lw=2.0,
            label=f"{name}  n={len(v):,}  med={v.median():.2f}",
        )
    ax.set_xlim(0, 6)
    ax.set_xlabel("nbhd-median $A_V$ (mag)")
    ax.set_ylabel("count")
    ax.set_title("(b) Composite $A_V$ per zone")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    # (c) per-source residual against the nbhd-median composite.
    ax = fig.add_subplot(gs[1, 1])
    for src, color in SOURCE_COLOR.items():
        if src == "av_nbhd_median":
            continue
        v_src = df[src].to_numpy()
        v_nbhd = df["av_nbhd_median"].to_numpy()
        ok = np.isfinite(v_src) & np.isfinite(v_nbhd) & (v_src < 6) & (v_nbhd < 6)
        if ok.sum() < 50:
            continue
        delta = v_src[ok] - v_nbhd[ok]
        med = float(np.median(delta))
        ax.hist(
            delta,
            bins=80,
            range=(-2, 2),
            histtype="step",
            color=color,
            lw=2.0,
            label=f"{SOURCE_LABEL[src]}  med={med:+.2f}  n={int(ok.sum()):,}",
        )
    ax.axvline(0, color="0.3", ls="-", lw=0.8, alpha=0.7)
    ax.set_xlim(-2, 2)
    ax.set_xlabel(r"$A_V^{\rm src} - A_V^{\rm nbhd}$ (mag)")
    ax.set_ylabel("count")
    ax.set_title("(c) Residual vs the composite")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    # (d) Av_nbhd_std vs distance (uncertainty proxy).
    ax = fig.add_subplot(gs[1, 2])
    df2 = pd.read_parquet(FEAT, columns=["source_id", "r_med_photogeo", "av_nbhd_std"])
    df2 = df2.drop_duplicates("source_id").dropna()
    df2["d_kpc"] = df2["r_med_photogeo"] / 1000.0
    d = df2["d_kpc"].to_numpy()
    s = df2["av_nbhd_std"].to_numpy()
    ok = np.isfinite(d) & np.isfinite(s) & (d > 0) & (d < 12) & (s < 2)
    ax.hexbin(d[ok], s[ok], gridsize=80, extent=(0, 12, 0, 2), mincnt=1, bins="log", cmap="viridis")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2)
    ax.set_xlabel("BJ21 distance (kpc)")
    ax.set_ylabel(r"$\sigma(A_V)$ from nbhd-median (mag)")
    ax.set_title("(d) Composite uncertainty vs distance")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"G2. $A_V$ vs distance per source  (Stream 1, n = {n:,})\n"
        "Three dust-map zones: Edenhofer ≤ 1.25 kpc, Lallement 1.25-3 kpc, "
        "SFD > 3 kpc.  The pipeline uses the nbhd-median composite.",
        fontsize=12,
        fontweight="semibold",
        y=0.985,
    )
    save_fig(fig, OUT / "G2_extinction_vs_distance", tight=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
