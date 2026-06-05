"""F15: Stream-2 Galactic positions, X-Y and R-Z, by chemistry (slide 16).

Now plotting Stream 2 (TESS asteroseismic giants) Tier-1 instead of
Stream 1: that puts the Galactic-geometry slide in the same data
context as the surrounding Stream-2 transfer / calibration / kinematics
slides (13, 14, 15).
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

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "gallery" / "v1_4"))
sys.path.insert(0, str(REPO / "src"))

from _loaders import FEAT_S2, load_s2_predictions  # noqa: E402

from arqueogal.style import (  # noqa: E402
    LABELS, apply_style, colorbar, median_per_cell, save,
)

# Square XY footprint around the Sun (-8, 0); ZR uses the matching
# disc-radius range so panels read as a coherent 2x2 geometry block.
XY = (-12.0, -4.0, -4.0, 4.0)
ZR = (5.0, 11.0, -2.0, 2.0)


def _load() -> pd.DataFrame:
    pred = load_s2_predictions()
    pred = pred.loc[pred["release_tier"] == 1].reset_index(drop=True)
    feat = pd.read_parquet(
        FEAT_S2,
        columns=["source_id", "ra_deg", "dec_deg", "r_med_photogeo"],
    ).drop_duplicates("source_id")
    df = pred.merge(feat, on="source_id", how="inner")
    df = df.loc[df["r_med_photogeo"] > 0].reset_index(drop=True)
    df = df.dropna(subset=["ra_deg", "dec_deg",
                            "mh_pred", "alpha_m_pred"]).reset_index(drop=True)
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
    return df


def main() -> int:
    apply_style()
    df = _load()
    mh = df["mh_pred"].to_numpy()
    am = df["alpha_m_pred"].to_numpy()
    mh_lo, mh_hi = float(np.nanpercentile(mh, 1)), float(np.nanpercentile(mh, 99))
    am_abs = max(float(np.nanpercentile(np.abs(am), 99)), 0.05)

    # Bumped gridsize from 50 to 80 for higher-resolution Galactic-position
    # panels per the v1.4 talk reviewer note.
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 6.5),
                              layout="constrained")

    ax = axes[0, 0]
    hb = median_per_cell(ax, df["X"].to_numpy(), df["Y"].to_numpy(), mh,
                          gridsize=120, mincnt=2,
                          vmin=mh_lo, vmax=mh_hi, cmap="jet", extent=XY)
    colorbar(ax, hb, LABELS["Mh"])
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(8.0 * np.cos(th), 8.0 * np.sin(th),
             color="#5C6378", lw=0.6, ls=":", alpha=0.5)
    ax.plot([-8.0], [0.0], "*", markerfacecolor="white",
             markeredgecolor="#000000", markeredgewidth=0.8, markersize=10)
    ax.set_xlim(XY[0], XY[1]); ax.set_ylim(XY[2], XY[3])
    ax.set_xlabel(LABELS["X"]); ax.set_ylabel(LABELS["Y"])
    ax.set_title(r"X-Y, coloured by [M/H]")
    ax.grid(False)

    ax = axes[0, 1]
    hb = median_per_cell(ax, df["X"].to_numpy(), df["Y"].to_numpy(), am,
                          gridsize=120, mincnt=2,
                          vmin=-am_abs, vmax=am_abs,
                          cmap="jet", extent=XY)
    colorbar(ax, hb, LABELS["alpha_M"])
    ax.plot(8.0 * np.cos(th), 8.0 * np.sin(th),
             color="#5C6378", lw=0.6, ls=":", alpha=0.5)
    ax.plot([-8.0], [0.0], "*", markerfacecolor="white",
             markeredgecolor="#000000", markeredgewidth=0.8, markersize=10)
    ax.set_xlim(XY[0], XY[1]); ax.set_ylim(XY[2], XY[3])
    ax.set_xlabel(LABELS["X"]); ax.set_ylabel(LABELS["Y"])
    ax.set_title(r"X-Y, coloured by [$\alpha$/M]")
    ax.grid(False)

    ax = axes[1, 0]
    hb = median_per_cell(ax, df["Rgal"].to_numpy(), df["Z"].to_numpy(), mh,
                          gridsize=120, mincnt=2,
                          vmin=mh_lo, vmax=mh_hi, cmap="jet", extent=ZR)
    colorbar(ax, hb, LABELS["Mh"])
    ax.axhline(0.0, color="#5C6378", lw=0.5, ls=":", alpha=0.5)
    ax.plot([8.0], [0.0], "*", markerfacecolor="white",
             markeredgecolor="#000000", markeredgewidth=0.8, markersize=10)
    ax.set_xlim(ZR[0], ZR[1]); ax.set_ylim(ZR[2], ZR[3])
    ax.set_xlabel(LABELS["Rgal"]); ax.set_ylabel(LABELS["Z"])
    ax.set_title(r"R-Z, coloured by [M/H]")
    ax.grid(False)

    ax = axes[1, 1]
    hb = median_per_cell(ax, df["Rgal"].to_numpy(), df["Z"].to_numpy(), am,
                          gridsize=120, mincnt=2,
                          vmin=-am_abs, vmax=am_abs,
                          cmap="jet", extent=ZR)
    colorbar(ax, hb, LABELS["alpha_M"])
    ax.axhline(0.0, color="#5C6378", lw=0.5, ls=":", alpha=0.5)
    ax.plot([8.0], [0.0], "*", markerfacecolor="white",
             markeredgecolor="#000000", markeredgewidth=0.8, markersize=10)
    ax.set_xlim(ZR[0], ZR[1]); ax.set_ylim(ZR[2], ZR[3])
    ax.set_xlabel(LABELS["Rgal"]); ax.set_ylabel(LABELS["Z"])
    ax.set_title(r"R-Z, coloured by [$\alpha$/M]")
    ax.grid(False)

    save(fig, "F15_galactic_geometry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
