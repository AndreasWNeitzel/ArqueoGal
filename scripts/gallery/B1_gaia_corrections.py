"""B1: Mandatory Gaia DR3 corrections (Lindegren+2021 parallax + Riello+2021 G-mag).

What this shows:
- Parallax zero-point correction: Δparallax = parallax_corr - parallax vs G.
- Post-Riello colour-magnitude diagram: G (corrected) vs BP-RP showing
  the corrected magnitude scale is stable across the colour range.

What it reads:
- data/processed/pipeline1_features_stream1.parquet (must have parallax,
  parallax_corr, g_mag, bp_rp columns).

Synthetic fixture support: --synthetic flag generates random parallax
and G-mag data.
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

OUT = REPO / "reports/gallery/B1_gaia_corrections"


def _make_synthetic():
    """Generate synthetic Gaia corrections data."""
    rng = np.random.default_rng(42)
    n = 5000
    g = rng.uniform(6, 17, n)
    parallax_raw = rng.uniform(5, 50, n)
    # Lindegren correction is G-dependent, ~0.03 mas at G=10, ~0.05 mas at G=17
    parallax_corr = parallax_raw + 0.01 + 0.003 * (g - 10)
    bp_rp = rng.uniform(0.3, 3.0, n)
    # Riello correction is colour-dependent, ~0.05 mag at BP-RP=2
    g_mag_corr = g + 0.005 + 0.01 * (bp_rp - 1.0) ** 2

    return pd.DataFrame({
        "parallax": parallax_raw,
        "parallax_corr": parallax_corr,
        "g_mag": g_mag_corr,
        "bp_rp": bp_rp,
    })


def main(use_synthetic: bool = False) -> None:
    apply_style()

    if use_synthetic:
        s1 = _make_synthetic()
    else:
        parquet = REPO / "data/processed/pipeline1_features_stream1.parquet"
        cand = ["parallax", "parallax_corr", "g_mag", "bp_rp"]
        s1 = pd.read_parquet(parquet, columns=[c for c in cand if c in pd.read_parquet(parquet).columns])

    available = set(s1.columns)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    if {"parallax", "parallax_corr", "g_mag"} <= available:
        d = (s1["parallax_corr"] - s1["parallax"]).to_numpy()
        g = s1["g_mag"].to_numpy()
        m = np.isfinite(d) & np.isfinite(g)
        h = axes[0].hexbin(g[m], d[m] * 1e3, gridsize=60, mincnt=5, cmap="viridis", bins="log")
        plt.colorbar(h, ax=axes[0], label="log10 N")
        axes[0].axhline(0, color="k", lw=0.6, ls="--")
        axes[0].set_xlabel(r"$G$ [mag]")
        axes[0].set_ylabel(r"$\Delta\varpi$ [µas] (Lindegren+21 zpt)")
        axes[0].set_title(f"Parallax zpt correction (n={int(m.sum()):,})")
    else:
        axes[0].text(0.5, 0.5, "parallax columns missing", ha="center", va="center", transform=axes[0].transAxes)

    if {"g_mag", "bp_rp"} <= available:
        c = s1["bp_rp"].to_numpy()
        g = s1["g_mag"].to_numpy()
        m = np.isfinite(c) & np.isfinite(g)
        h = axes[1].hexbin(c[m], g[m], gridsize=70, mincnt=10, cmap="viridis", bins="log", extent=[0, 3.5, 8, 18])
        plt.colorbar(h, ax=axes[1], label="log10 N")
        axes[1].invert_yaxis()
        axes[1].set_xlabel(r"BP − RP [mag, raw]")
        axes[1].set_ylabel(r"$G$ [mag, Riello+21 corrected]")
        axes[1].set_title(f"Post-Riello CMD (n={int(m.sum()):,})")

    fig.suptitle("B1 — Mandatory Gaia DR3 corrections (Lindegren+21 + Riello+21)", fontsize=11)
    save_fig(fig, OUT / "gaia_corrections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot B1: Gaia corrections.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic fixture.")
    args = parser.parse_args()
    main(use_synthetic=args.synthetic)
