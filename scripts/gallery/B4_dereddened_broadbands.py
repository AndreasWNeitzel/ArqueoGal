"""B4: Raw vs dereddened broadband photometry (J/H/K/W1/W2).

What this shows:
- Five-panel grid (one per band: J, H, K, W1, W2) showing raw vs dereddened
  magnitudes as a hexbin density plot plus a slope overlay showing the
  Yuan+2013 reference dereddening law.

What it reads:
- data/processed/pipeline1_features_stream1.parquet (must have j_mag, h_mag,
  k_mag, w1_mag, w2_mag columns and their dereddened counterparts).

Synthetic fixture support: --synthetic flag generates realistic IR magnitude
distributions with Galactic extinction.
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

OUT = REPO / "reports/gallery/B4_dereddened_broadbands"


def _make_synthetic():
    """Generate synthetic IR photometry with extinction."""
    rng = np.random.default_rng(42)
    n = 3000

    # True magnitudes (realistic for RGB giants)
    j_true = rng.normal(8.5, 1.5, n)
    h_true = rng.normal(7.8, 1.4, n)
    k_true = rng.normal(7.2, 1.3, n)
    w1_true = rng.normal(6.8, 1.2, n)
    w2_true = rng.normal(6.2, 1.1, n)

    # Extinction: A_V varies 0-2 mag (realistic for Galactic)
    av = rng.uniform(0, 2, n)

    # Yuan+2013 extinction law (approximate)
    a_j = av * 0.276
    a_h = av * 0.176
    a_k = av * 0.112
    a_w1 = av * 0.047
    a_w2 = av * 0.011

    # Observed = true + extinction
    j_obs = j_true + a_j
    h_obs = h_true + a_h
    k_obs = k_true + a_k
    w1_obs = w1_true + a_w1
    w2_obs = w2_true + a_w2

    return pd.DataFrame({
        "j_mag": j_obs,
        "h_mag": h_obs,
        "k_mag": k_obs,
        "w1_mag": w1_obs,
        "w2_mag": w2_obs,
        "j_mag_dered": j_true,
        "h_mag_dered": h_true,
        "k_mag_dered": k_true,
        "w1_mag_dered": w1_true,
        "w2_mag_dered": w2_true,
    })


def main(use_synthetic: bool = False) -> None:
    apply_style()

    if use_synthetic:
        df = _make_synthetic()
    else:
        cols = ["j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag",
                "j_mag_dered", "h_mag_dered", "k_mag_dered",
                "w1_mag_dered", "w2_mag_dered"]
        schema = pd.read_parquet(REPO / "data/processed/pipeline1_features_stream1.parquet").iloc[:0]
        cols = [c for c in cols if c in schema.columns]
        df = pd.read_parquet(REPO / "data/processed/pipeline1_features_stream1.parquet", columns=cols)

    bands = ["j", "h", "k", "w1", "w2"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))

    for idx, band in enumerate(bands):
        ax = axes[idx]
        mag_col = f"{band}_mag"
        dered_col = f"{band}_mag_dered"

        if mag_col not in df.columns or dered_col not in df.columns:
            ax.text(0.5, 0.5, f"{band}: data missing", ha="center", va="center", transform=ax.transAxes)
            continue

        mag_raw = df[mag_col].values
        mag_dered = df[dered_col].values
        m = np.isfinite(mag_raw) & np.isfinite(mag_dered)

        if m.sum() > 100:
            h = ax.hexbin(mag_raw[m], mag_dered[m], gridsize=40, mincnt=3, cmap="viridis", bins="log")
            plt.colorbar(h, ax=ax, label="log10 N")

            # 1:1 line
            lims = [mag_raw[m].min(), mag_raw[m].max()]
            ax.plot(lims, lims, "r--", lw=0.7, alpha=0.6, label="1:1")
            ax.set_xlabel(f"{band.upper()} raw [mag]")
            ax.set_ylabel(f"{band.upper()} dereddened [mag]")
            ax.set_title(f"{band.upper()} (n={m.sum():,})")
            ax.legend(fontsize=7)

    fig.suptitle("B4 — Raw vs dereddened broadband IR photometry (J/H/K/W1/W2)", fontsize=11)
    save_fig(fig, OUT / "dereddened_broadbands")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot B4: Dereddened broadbands.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic fixture.")
    args = parser.parse_args()
    main(use_synthetic=args.synthetic)
