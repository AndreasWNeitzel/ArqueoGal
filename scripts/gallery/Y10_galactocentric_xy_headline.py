"""Y10: Galactocentric XY top-down view, single hero panel.

Stream 3 Tier 1 stars projected to Galactocentric (X, Y) using BJ21
photogeometric distances. Hexbin median [M/H] per cell. Sun and GC marked.
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

from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED_S3 = REPO / "data/processed/pipeline1_predictions_stream3.parquet"
FEAT_S3 = REPO / "data/processed/pipeline1_features_stream3.parquet"
HYBRID_S3 = REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet"


def _load() -> pd.DataFrame:
    pcols = [
        "source_id",
        "mh_pred",
        "alpha_m_pred",
        "teff_sigma",
        "logg_sigma",
        "mh_sigma",
        "alpha_m_sigma",
        "mg_h_sigma",
        "ood_joint_flag",
        "label_extrapolation_flag",
    ]
    pred = pd.read_parquet(PRED_S3, columns=pcols).drop_duplicates("source_id")
    feat = pd.read_parquet(
        FEAT_S3,
        columns=["source_id", "ra_deg", "dec_deg", "r_med_photogeo"],
    ).drop_duplicates("source_id")
    df = feat.merge(pred, on="source_id", how="inner")
    df["kin_ood_flag"] = False
    if HYBRID_S3.exists():
        h = pd.read_parquet(HYBRID_S3, columns=["source_id", "kin_ood_flag"])
        df = df.merge(h, on="source_id", how="left", suffixes=("", "_h"))
        if "kin_ood_flag_h" in df.columns:
            df["kin_ood_flag"] = df["kin_ood_flag_h"].fillna(False).astype(bool)
            df = df.drop(columns=["kin_ood_flag_h"])
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
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


def main() -> int:
    apply_style()
    df = _load()

    fig = plt.figure(figsize=(13, 12))
    ax = fig.add_subplot(111)
    extent = (-15, 15, -15, 15)
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()
    c = df["mh_pred"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)

    vmed = np.nanmedian(c[ok])
    vsig = np.nanstd(c[ok])

    hb = ax.hexbin(
        x[ok],
        y[ok],
        C=c[ok],
        reduce_C_function=np.median,
        gridsize=70,
        extent=extent,
        mincnt=4,
        vmin=vmed - 2 * vsig,
        vmax=vmed + 2 * vsig,
        cmap="viridis",
        edgecolors="none",
    )
    # Sun and Galactic centre.
    ax.scatter(
        [-8.122],
        [0.0],
        marker="*",
        s=420,
        color="white",
        edgecolor=PALETTE["ink"],
        linewidth=1.6,
        zorder=5,
        label="Sun",
    )
    ax.scatter(
        [0.0],
        [0.0],
        marker="x",
        s=200,
        color="white",
        linewidth=2.6,
        zorder=5,
        label="Galactic centre",
    )
    # Concentric Rgal rings.
    for r in (4, 8, 12):
        circle = plt.Circle((0, 0), r, fill=False, lw=0.8, ls=":", color=PALETTE["ash"], alpha=0.7)
        ax.add_patch(circle)
        ax.text(r, -r * 0.05, f" R={r} kpc", ha="left", va="top", fontsize=9, color=PALETTE["ash"])

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("X  (kpc)")
    ax.set_ylabel("Y  (kpc)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    cb = plt.colorbar(hb, ax=ax, fraction=0.040, pad=0.03)
    cb.set_label("median [M/H]  (dex)", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    headline(
        fig,
        "Galactocentric top-down view of [M/H]",
        f"Stream 3 Tier 1, n = {int(ok.sum()):,} after BJ21-distance cut.  "
        "Hexbin median, ≥4 stars per cell.",
        top=0.86,
    )
    save(fig, "Y10_galactocentric_xy_headline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
