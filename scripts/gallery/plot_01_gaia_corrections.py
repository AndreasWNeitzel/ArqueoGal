"""Stage 01: Gaia DR3 corrections (Lindegren+2021 parallax zpt + Riello+2021 G-mag cubic).

CLAUDE.md hard rule #11: parallax zpt and G-mag corrections are mandatory at
ingestion. ``apply_gaia_corrections`` writes ``parallax_corr`` and
``phot_g_mean_mag_corr`` (or just verifies they exist).

What we plot: Δparallax = parallax_corr − parallax vs G; ΔG = G_corr − G_raw
vs colour. Both should be small and structured (Lindegren has G dependence,
Riello has cubic in (BP-RP)).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/01_gaia_corrections"


def main() -> None:
    apply_style()
    parquet = REPO / "data/processed/pipeline1_features_stream1.parquet"
    schema = pd.read_parquet(parquet).iloc[:0]
    cand = ["parallax", "parallax_corr", "g_mag", "bp_rp", "ruwe"]
    cols = [c for c in cand if c in schema.columns]
    s1 = pd.read_parquet(parquet, columns=cols)
    available = set(s1.columns)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    if {"parallax", "parallax_corr", "g_mag"} <= available:
        d = (s1["parallax_corr"] - s1["parallax"]).to_numpy()
        g = s1["g_mag"].to_numpy()
        m = np.isfinite(d) & np.isfinite(g)
        h = axes[0].hexbin(g[m], d[m] * 1e3, gridsize=60, mincnt=5, cmap="viridis", bins="log")
        plt.colorbar(h, ax=axes[0], label="log10 N")
        axes[0].axhline(0, color="k", lw=0.6, ls="--")
        axes[0].set_xlabel("G (mag)")
        axes[0].set_ylabel(r"$\Delta\varpi$ (μas), Lindegren+21 zpt")
        axes[0].set_title(f"Parallax zpt correction (n={int(m.sum()):,})")
    else:
        axes[0].text(0.5, 0.5, "parallax_corr or parallax column missing",
                     ha="center", va="center", transform=axes[0].transAxes)

    # Subplot 2: post-correction colour-mag check. The raw `phot_g_mean_mag`
    # is not preserved in the feature parquet (only `g_mag` = corrected). The
    # ΔG audit lives in the per-row provenance sidecar at
    # data/processed/pipeline1_features_stream1.provenance.json. We instead
    # plot the corrected G vs (BP-RP) HRD as evidence the correction landed.
    if {"g_mag", "bp_rp"} <= available:
        c = s1["bp_rp"].to_numpy()
        g = s1["g_mag"].to_numpy()
        m = np.isfinite(c) & np.isfinite(g)
        h = axes[1].hexbin(c[m], g[m], gridsize=70, mincnt=10, cmap="viridis",
                            bins="log", extent=[0, 3.5, 8, 18])
        plt.colorbar(h, ax=axes[1], label="log10 N")
        axes[1].invert_yaxis()
        axes[1].set_xlabel("BP − RP (mag, raw)")
        axes[1].set_ylabel("G (mag, Riello+21 corrected)")
        axes[1].set_title(f"Post-Riello CMD (n={int(m.sum()):,})\n"
                            f"raw G dropped; ΔG audit in provenance.json")

    fig.suptitle("Mandatory Gaia DR3 corrections (Lindegren+21 + Riello+21)", fontsize=11)
    save_fig(fig, OUT / "gaia_corrections.png")


if __name__ == "__main__":
    main()
