"""F19: Stream-2 Hayden-style (R_gal, |Z|) chemistry-plane grid.

Reproduces the Hayden+ 2015 (Fig. 4 / 5) figure layout: chemical
([M/H], [alpha/M]) plane displayed in cells of (R_gal, |Z|), so the
audience can read the radial and vertical chemical structure of the
disc directly from the catalogue.

Stream 2 Tier 1 only (TESS asteroseismic giants). Cells with fewer
than 30 stars are left blank rather than rendered with bad statistics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import Galactocentric, SkyCoord
from matplotlib.colors import LogNorm

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "gallery" / "v1_4"))
sys.path.insert(0, str(REPO / "src"))

from _loaders import FEAT_S2, load_s2_predictions  # noqa: E402

from arqueogal.style import LABELS, apply_style, save  # noqa: E402

# Hayden-style binning, 1 kpc in R per the slide reviewer to avoid the
# nearly-empty 3-5 and 11-13 kpc cells.
R_BINS = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]   # six R bins, 1 kpc wide
Z_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0)]     # three |Z| bins
CHEM = (-1.6, 0.55, -0.10, 0.45)                  # M/H x alpha/M extent
MIN_PER_CELL = 30


def _load() -> pd.DataFrame:
    pred = load_s2_predictions()
    pred = pred.loc[pred["release_tier"] == 1].reset_index(drop=True)
    feat = pd.read_parquet(
        FEAT_S2,
        columns=["source_id", "ra_deg", "dec_deg", "r_med_photogeo"],
    ).drop_duplicates("source_id")
    df = pred.merge(feat, on="source_id", how="inner")
    df = df.dropna(subset=["ra_deg", "dec_deg", "r_med_photogeo",
                            "mh_pred", "alpha_m_pred"]).reset_index(drop=True)
    df = df.loc[df["r_med_photogeo"] > 0].reset_index(drop=True)
    icrs = SkyCoord(
        ra=df["ra_deg"].to_numpy() * u.deg,
        dec=df["dec_deg"].to_numpy() * u.deg,
        distance=df["r_med_photogeo"].to_numpy() * u.pc,
        frame="icrs",
    )
    gc = icrs.transform_to(Galactocentric())
    df["X"] = gc.x.to(u.kpc).value
    df["Y"] = gc.y.to(u.kpc).value
    df["Z"] = gc.z.to(u.kpc).value
    df["Rgal"] = np.sqrt(df["X"] ** 2 + df["Y"] ** 2)
    df["absZ"] = np.abs(df["Z"])
    return df


def main() -> int:
    apply_style()
    df = _load()
    n_R = len(R_BINS) - 1
    n_Z = len(Z_BINS)
    fig, axes = plt.subplots(
        n_Z, n_R, figsize=(18.0, 8.0),
        sharex=True, sharey=True,
        layout="constrained",
    )

    for i_z, (z_lo, z_hi) in enumerate(Z_BINS[::-1]):  # high |Z| on top
        for i_r in range(n_R):
            r_lo, r_hi = R_BINS[i_r], R_BINS[i_r + 1]
            ax = axes[i_z, i_r]
            mask = (df["Rgal"] >= r_lo) & (df["Rgal"] < r_hi) \
                 & (df["absZ"] >= z_lo) & (df["absZ"] < z_hi)
            sub = df.loc[mask]
            n = len(sub)
            if n >= MIN_PER_CELL:
                ax.hexbin(
                    sub["mh_pred"].to_numpy(),
                    sub["alpha_m_pred"].to_numpy(),
                    gridsize=45, mincnt=2,
                    norm=LogNorm(),
                    cmap="viridis", linewidths=0,
                    extent=CHEM,
                )
            else:
                ax.text(0.5, 0.5, r"too few",
                         transform=ax.transAxes, ha="center", va="center",
                         fontsize=9, color="#5C6378")
            ax.axhline(0.15, color="#000000", lw=0.5, ls=":", alpha=0.6)
            ax.set_xlim(CHEM[0], CHEM[1])
            ax.set_ylim(CHEM[2], CHEM[3])
            if i_z == 0:
                ax.set_title(rf"${r_lo:.0f} \leq R < {r_hi:.0f}$ kpc",
                              fontsize=11)
            if i_r == 0:
                # |Z| label on the leftmost column.
                ax.set_ylabel(rf"${z_lo:.1f} \leq |Z| < {z_hi:.1f}$ kpc"
                              "\n" + LABELS["alpha_M"],
                              fontsize=10)
            if i_z == n_Z - 1:
                ax.set_xlabel(LABELS["Mh"])
            ax.text(
                0.05, 0.95, rf"$n$ = {n:,}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, color="#2B2D42",
                bbox=dict(facecolor="white", edgecolor="none",
                           alpha=0.85, pad=2.0),
            )
            ax.grid(False)

    save(fig, "F19_takeaways")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
