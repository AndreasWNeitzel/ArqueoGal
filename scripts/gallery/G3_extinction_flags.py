"""G3: Av-source flag breakdown — what kind of Av is each star getting?

The pipeline annotates every star with metadata about how its Av was
estimated:

  av_los_source                  source name (string)
  av_is_neighborhood_fallback    True if no per-star map value, used composite
  av_distance_prior_dominated    True if parallax SNR < 5 (BJ21 prior dominates)
  av_neighbourhood_high_dispersion  True if nbhd σ exceeds 0.3 mag

This figure breaks down the Stream-1 cohort into those buckets so you
can see, at a glance, which stars carry a trustworthy per-star Av versus
which rely on a low-information fallback.
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


def main() -> int:
    apply_style()
    cols = ["source_id", "r_med_photogeo", "av_los", "av_los_source",
            "av_is_neighborhood_fallback", "av_distance_prior_dominated",
            "av_neighbourhood_high_dispersion", "av_nbhd_std"]
    df = pd.read_parquet(FEAT, columns=cols).drop_duplicates("source_id")
    n = len(df)
    df["d_kpc"] = df["r_med_photogeo"] / 1000.0

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.30,
                          top=0.92, bottom=0.07, left=0.05, right=0.97)

    # (a) Av-source bar — av_los_source is int8-coded.
    # Codebook from pipeline 1 build (data/extinction/precompute_av.py):
    # 0 = SFD, 1 = Lallement, 2 = Edenhofer, 3 = nbhd-median fallback.
    ax = fig.add_subplot(gs[0, 0])
    code_to_name = {0: "SFD/SF2011", 1: "Lallement+2022",
                    2: "Edenhofer+2024", 3: "nbhd fallback"}
    raw = df["av_los_source"].dropna().astype(int)
    counts = raw.value_counts().sort_index()
    labels = [code_to_name.get(int(k), f"code {int(k)}") for k in counts.index]
    pct = counts / counts.sum() * 100
    ax.barh(labels, counts.values, color="#3a6ea5",
            edgecolor="white", linewidth=1.4)
    for i, (v, p) in enumerate(zip(counts.values, pct.values)):
        ax.text(v + counts.max() * 0.01, i,
                f"{v:,}  ({p:.1f}%)",
                va="center", fontsize=10, color="#15355f")
    ax.set_xlim(0, counts.max() * 1.18)
    ax.set_xlabel("count")
    ax.set_title("(a) Av source per star")
    ax.grid(axis="x", alpha=0.25)

    # (b) Stacked flag prevalence.
    ax = fig.add_subplot(gs[0, 1])
    flags = [
        ("av_is_neighborhood_fallback", "fallback"),
        ("av_distance_prior_dominated", "dist prior dominated"),
        ("av_neighbourhood_high_dispersion", "high nbhd dispersion"),
    ]
    counts = [int(df[c].fillna(False).astype(bool).sum()) for c, _ in flags]
    fractions = [c / n * 100 for c in counts]
    ax.bar([n for _, n in flags], counts,
            color=["#e07b00", "#d62728", "#7d3c98"],
            edgecolor="white", linewidth=1.4)
    for i, (c, f) in enumerate(zip(counts, fractions)):
        ax.text(i, c + n * 0.003, f"{c:,}\n({f:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#15355f")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_ylabel("number of stars")
    ax.set_title("(b) Av-trust flag prevalence")
    ax.grid(axis="y", alpha=0.25)

    # (c) Av source vs distance — using the int code map.
    ax = fig.add_subplot(gs[0, 2])
    src_color = {2: ("Edenhofer", "#1f77b4"),
                 1: ("Lallement", "#2ca02c"),
                 0: ("SFD",       "#d62728"),
                 3: ("nbhd fb",   "#7d3c98")}
    for code, (name, color) in src_color.items():
        mask = df["av_los_source"] == code
        d = df.loc[mask, "d_kpc"].dropna()
        if len(d):
            ax.hist(d[d < 12], bins=60, range=(0, 12), histtype="step",
                    color=color, lw=2.0,
                    label=f"{name}  n={len(d):,}")
    ax.set_xlim(0, 12)
    ax.set_xlabel("BJ21 distance (kpc)")
    ax.set_ylabel("count")
    ax.set_title("(c) Distance distribution per Av source")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.25)

    # (d) Av distribution split by fallback flag.
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 6, 80)
    for flag, color, label in [
        (~df["av_is_neighborhood_fallback"].fillna(True), "#2ca02c", "per-star Av"),
        (df["av_is_neighborhood_fallback"].fillna(True), "#e07b00", "fallback Av"),
    ]:
        v = df.loc[flag.astype(bool), "av_los"].dropna()
        v = v[v < 6]
        if len(v):
            ax.hist(v, bins=bins, histtype="step", color=color, lw=2.0,
                    label=f"{label}  n={len(v):,}  med={v.median():.2f}")
    ax.set_xlim(0, 6)
    ax.set_xlabel(r"$A_V$ (mag)")
    ax.set_ylabel("count")
    ax.set_title("(d) Av distribution: per-star vs fallback")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.25)

    # (e) nbhd dispersion histogram.
    ax = fig.add_subplot(gs[1, 1])
    s = df["av_nbhd_std"].dropna()
    s = s[s < 1.5]
    if len(s):
        ax.hist(s, bins=80, range=(0, 1.5), color="#7d3c98",
                edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.axvline(0.3, color="#d62728", lw=1.6, ls="--",
                   label=r"high-dispersion threshold $\sigma > 0.3$")
        ax.legend(loc="upper right", fontsize=10)
    ax.set_xlabel(r"$\sigma(A_V)$ from nbhd-median (mag)")
    ax.set_ylabel("count")
    ax.set_title("(e) Composite-uncertainty distribution")
    ax.grid(True, alpha=0.25)

    # (f) confusion of two key trust flags.
    ax = fig.add_subplot(gs[1, 2])
    fb = df["av_is_neighborhood_fallback"].fillna(False).astype(bool)
    hi = df["av_neighbourhood_high_dispersion"].fillna(False).astype(bool)
    cm = np.array([
        [int((~fb & ~hi).sum()), int((~fb & hi).sum())],
        [int((fb & ~hi).sum()),  int((fb & hi).sum())],
    ])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["nbhd σ low", "nbhd σ high"])
    ax.set_yticklabels(["per-star", "fallback"])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "#15355f"
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / n * 100:.1f}%)",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color=color)
    ax.set_title("(f) Trust matrix")
    plt.colorbar(im, ax=ax, label="count")

    fig.suptitle(
        f"G3. Av-source flags & trust  (Stream 1, n = {n:,})\n"
        "Per-star Av provenance breakdown — which stars get a real "
        "per-star Av, which fall back to a composite, and where the "
        "composite is uncertain.",
        fontsize=12, fontweight="semibold", y=0.985,
    )
    save_fig(fig, OUT / "G3_extinction_flags", tight=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
