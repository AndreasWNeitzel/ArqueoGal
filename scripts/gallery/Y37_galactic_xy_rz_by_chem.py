"""Y37: Stream-1 Tier-1 holdout, Galactocentric X-Y and R-Z, by chemistry.

2 columns x 2 rows on a slide-friendly 16:11:
  - top row:    X-Y (top-down) coloured by [M/H]; X-Y coloured by [α/M].
  - bottom row: R-Z (side view) coloured by [M/H]; R-Z coloured by [α/M].

Galactocentric frame uses astropy default with the project-canonical
solar position (R_sun = 8.122 kpc, z_sun = 20.8 pc).
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402

from arqueogal.xp_abundances.main.data import stratified_split_ids  # noqa: E402
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED_S1 = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"

XY_EXTENT = (-15, 15, -15, 15)
ZR_EXTENT = (0.0, 16.0, -3.5, 3.5)
HEX_GRID_XY = 140
HEX_GRID_ZR = (140, 60)

_TITLE_KW = dict(fontsize=11, fontweight="normal", color=PALETTE["ink"], pad=6)


def _load() -> pd.DataFrame:
    pcols = [
        "source_id", "teff_pred", "logg_pred", "mh_pred",
        "alpha_m_pred", "mg_h_pred",
        "ood_joint_flag", "label_extrapolation_flag",
    ]
    p = pd.read_parquet(PRED_S1, columns=pcols).drop_duplicates("source_id")
    fcols = ["source_id", "ra_deg", "dec_deg", "r_med_photogeo",
             "fe_h_apogee", "teff_apogee", "b_deg"]
    f = pd.read_parquet(FEAT_S1, columns=fcols).drop_duplicates("source_id")
    df = f.merge(p, on="source_id", how="inner")
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    split = stratified_split_ids(df, seed=0)
    holdout_ids = np.concatenate([split["val"], split["test"]])
    df = df.loc[df["source_id"].isin(holdout_ids)].reset_index(drop=True)
    df = df.loc[df["release_tier"] == 1].reset_index(drop=True)
    ok = np.isfinite(df["r_med_photogeo"].to_numpy()) & (df["r_med_photogeo"] > 0)
    df = df.loc[ok].reset_index(drop=True)

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


def _draw_panel(ax, x, y, c, *, gridsize, extent, vmin, vmax,
                cbar_label, title, xlabel, ylabel,
                sun=(-8.122, 0.0), gc_marker=False):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    sc = ax.hexbin(
        x[ok], y[ok], C=c[ok], reduce_C_function=np.median,
        gridsize=gridsize, extent=extent, mincnt=2,
        vmin=vmin, vmax=vmax, cmap="viridis", edgecolors="none",
    )
    ax.scatter([sun[0]], [sun[1]], marker="*", s=180, color="white",
               edgecolor=PALETTE["ink"], linewidth=1.0, zorder=5)
    if gc_marker:
        ax.scatter([0.0], [0.0], marker="x", s=80, color="white",
                   linewidth=1.5, zorder=5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, **_TITLE_KW)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label, fontsize=10)


def main() -> int:
    apply_style()
    df = _load()
    if df.empty:
        print("[Y37] no Tier-1 holdout rows with valid distance, aborting")
        return 1

    # Equal-sized 2x2 grid; same width per cell on both rows by using
    # constrained_layout=False and explicit gridspec width_ratios.  The
    # X-Y top row is square (x and y span 30 kpc each) while R-Z is
    # rectangular (16 kpc x 7 kpc).  To keep panel boxes equal, drop
    # set_aspect("equal") on X-Y so all four axes share the same panel
    # box; the data-aspect difference between rows is a property of the
    # axis ranges, not of the panel size on the slide.
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        2, 2, hspace=0.40, wspace=0.28,
        top=0.83, bottom=0.07, left=0.06, right=0.985,
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)])

    mh = df["mh_pred"].to_numpy()
    am = df["alpha_m_pred"].to_numpy()
    mh_lo = float(np.nanpercentile(mh, 2))
    mh_hi = float(np.nanpercentile(mh, 98))
    am_lo = float(np.nanpercentile(am, 2))
    am_hi = float(np.nanpercentile(am, 98))

    # Top row: X-Y top-down view.
    for r in (4, 8, 12):
        for ax in axes[0, :]:
            ax.add_patch(plt.Circle((0, 0), r, fill=False, lw=0.6, ls=":",
                                    color=PALETTE["ash"], alpha=0.7))
    _draw_panel(
        axes[0, 0], df["X"].to_numpy(), df["Y"].to_numpy(),
        df["mh_pred"].to_numpy(),
        gridsize=HEX_GRID_XY, extent=XY_EXTENT,
        vmin=mh_lo, vmax=mh_hi,
        cbar_label="median [M/H] (dex)",
        title="X-Y, coloured by [M/H]",
        xlabel="X (kpc)", ylabel="Y (kpc)",
        sun=(-8.122, 0.0), gc_marker=True,
    )
    _draw_panel(
        axes[0, 1], df["X"].to_numpy(), df["Y"].to_numpy(),
        df["alpha_m_pred"].to_numpy(),
        gridsize=HEX_GRID_XY, extent=XY_EXTENT,
        vmin=am_lo, vmax=am_hi,
        cbar_label=r"median [$\alpha$/M] (dex)",
        title=r"X-Y, coloured by [$\alpha$/M]",
        xlabel="X (kpc)", ylabel="Y (kpc)",
        sun=(-8.122, 0.0), gc_marker=True,
    )

    # Bottom row: R-Z side view.
    _draw_panel(
        axes[1, 0], df["Rgal"].to_numpy(), df["Z"].to_numpy(),
        df["mh_pred"].to_numpy(),
        gridsize=HEX_GRID_ZR, extent=ZR_EXTENT,
        vmin=mh_lo, vmax=mh_hi,
        cbar_label="median [M/H] (dex)",
        title="R-Z, coloured by [M/H]",
        xlabel=r"$R_{\rm gal}$ (kpc)", ylabel="Z (kpc)",
        sun=(8.122, 0.0208),
    )
    _draw_panel(
        axes[1, 1], df["Rgal"].to_numpy(), df["Z"].to_numpy(),
        df["alpha_m_pred"].to_numpy(),
        gridsize=HEX_GRID_ZR, extent=ZR_EXTENT,
        vmin=am_lo, vmax=am_hi,
        cbar_label=r"median [$\alpha$/M] (dex)",
        title=r"R-Z, coloured by [$\alpha$/M]",
        xlabel=r"$R_{\rm gal}$ (kpc)", ylabel="Z (kpc)",
        sun=(8.122, 0.0208),
    )

    for ax in axes.ravel():
        ax.grid(True, alpha=0.20)

    headline(
        fig,
        "Stream 1 Tier 1 holdout: Galactic positions coloured by chemistry",
        f"n = {len(df):,} (BJ21 photogeometric distances; \\u22654 stars per cell).",
        top=0.83,
    )
    save(fig, "Y37_galactic_xy_rz_by_chem")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
