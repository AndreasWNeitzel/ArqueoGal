"""Y20: Training-cohort sky coverage vs the parent inference cohort.

Two-row Mollweide:

  Row 1 — Stream 1 (APOGEE × Gaia DR3 XP, the *training* cohort).
  Row 2 — Stream 3 (Andrae+23 RGB inference cohort, ≈ 614k stars).

Both are HEALPix NSIDE=32 number-density maps. The visual contrast makes
clear that APOGEE's footprint is sparse and biased (concentrated on the
SDSS plates, deficient at high latitudes and in the southern sky), and
that Stream 3 covers the full sky uniformly. Trained on the former,
applied to the latter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _common import (  # noqa: E402
    galactic_mollweide,
    radec_to_galactic,
    style_galactic_mollweide,
)
from _presentation import PALETTE, apply_style, headline, save  # noqa: E402

FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
FEAT_S3 = REPO / "data/processed/pipeline1_features_stream3.parquet"

NSIDE = 32


def _density(ra, dec):
    l, b = radec_to_galactic(ra, dec)
    theta = np.deg2rad(90.0 - b)
    phi = np.deg2rad(l)
    pix = hp.ang2pix(NSIDE, theta, phi)
    n_pix = hp.nside2npix(NSIDE)
    counts = np.bincount(pix, minlength=n_pix).astype(np.float64)
    counts[counts == 0] = np.nan
    return counts


def _pix_to_lonlat(idx):
    theta, phi = hp.pix2ang(NSIDE, idx)
    l = np.rad2deg(phi)
    l = np.where(l > 180.0, l - 360.0, l)
    b = 90.0 - np.rad2deg(theta)
    return galactic_mollweide(l, b)


def _draw(ax, df, title, vmin, vmax):
    counts = _density(df["ra_deg"].to_numpy(), df["dec_deg"].to_numpy())
    have = np.where(np.isfinite(counts))[0]
    lon, lat = _pix_to_lonlat(have)
    sc = ax.scatter(
        lon,
        lat,
        c=np.log10(counts[have]),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=24.0,
        marker="s",
        edgecolors="none",
        alpha=0.95,
    )
    style_galactic_mollweide(ax)
    ax.set_title(f"{title}    n = {len(df):,}", fontsize=14, color=PALETTE["navy"])
    cb = plt.colorbar(sc, ax=ax, fraction=0.030, pad=0.04, orientation="horizontal")
    cb.set_label(r"$\log_{10}$ N per HEALPix pixel", fontsize=11)
    cb.ax.tick_params(labelsize=10)


def main() -> int:
    apply_style()
    df1 = pd.read_parquet(FEAT_S1, columns=["source_id", "ra_deg", "dec_deg"])
    df1 = df1.drop_duplicates("source_id")
    df3 = pd.read_parquet(FEAT_S3, columns=["source_id", "ra_deg", "dec_deg"])
    df3 = df3.drop_duplicates("source_id")

    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_axes([0.04, 0.55, 0.92, 0.35], projection="mollweide")
    ax2 = fig.add_axes([0.04, 0.10, 0.92, 0.35], projection="mollweide")
    _draw(ax1, df1, "Stream 1 — training cohort (APOGEE × Gaia DR3)", vmin=0.0, vmax=2.5)
    _draw(ax2, df3, "Stream 3 — inference cohort (Andrae+2023 RGB)", vmin=0.0, vmax=2.5)

    headline(
        fig,
        "Trained on a sparse, biased footprint; applied to the full sky",
        "Stream 1 (top) is APOGEE's plate footprint; Stream 3 (bottom) is the all-sky "
        "Andrae+23 RGB cohort the model is then released on.",
        top=0.83,
    )
    save(fig, "Y20_training_sky_coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
