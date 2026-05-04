"""Y09: Mollweide [M/H] skymap, single hero panel.

Stream 3 Tier 1 sample (the largest cohort), median per HEALPix pixel,
common color limits clipped to ±2σ around the median for slide legibility.
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

from _common import galactic_mollweide, radec_to_galactic, style_galactic_mollweide  # noqa: E402
from _presentation import apply_style, headline, save  # noqa: E402

from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED_S3 = REPO / "data/processed/pipeline1_predictions_stream3.parquet"
FEAT_S3 = REPO / "data/processed/pipeline1_features_stream3.parquet"
HYBRID_S3 = REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet"

NSIDE = 64


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
    feat = pd.read_parquet(FEAT_S3, columns=["source_id", "ra_deg", "dec_deg"])
    feat = feat.drop_duplicates("source_id")
    df = feat.merge(pred, on="source_id", how="inner")

    df["kin_ood_flag"] = False
    if HYBRID_S3.exists():
        h = pd.read_parquet(HYBRID_S3, columns=["source_id", "kin_ood_flag"])
        df = df.merge(h, on="source_id", how="left", suffixes=("", "_h"))
        if "kin_ood_flag_h" in df.columns:
            df["kin_ood_flag"] = df["kin_ood_flag_h"].fillna(False).astype(bool)
            df = df.drop(columns=["kin_ood_flag_h"])

    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    return df.loc[df["release_tier"] == 1].reset_index(drop=True)


def _pix_median(ra, dec, val):
    l, b = radec_to_galactic(ra, dec)
    theta = np.deg2rad(90.0 - b)
    phi = np.deg2rad(l)
    pix = hp.ang2pix(NSIDE, theta, phi)
    n_pix = hp.nside2npix(NSIDE)
    out = np.full(n_pix, np.nan)
    g = pd.DataFrame({"pix": pix, "v": val}).groupby("pix")
    med = g["v"].median()
    out[med.index.to_numpy()] = med.to_numpy()
    return out


def _pix_to_lonlat(idx):
    theta, phi = hp.pix2ang(NSIDE, idx)
    l = np.rad2deg(phi)
    l = np.where(l > 180.0, l - 360.0, l)
    b = 90.0 - np.rad2deg(theta)
    return galactic_mollweide(l, b)


def main() -> int:
    apply_style()
    df = _load()
    n = len(df)
    fig = plt.figure(figsize=(18, 11))
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.72], projection="mollweide")

    pix_med = _pix_median(
        df["ra_deg"].to_numpy(), df["dec_deg"].to_numpy(), df["mh_pred"].to_numpy()
    )
    have = np.where(np.isfinite(pix_med))[0]
    lon, lat = _pix_to_lonlat(have)

    # Robust color limits.
    vmed = np.nanmedian(pix_med[have])
    vsig = np.nanstd(pix_med[have])
    vmin, vmax = vmed - 2 * vsig, vmed + 2 * vsig

    sc = ax.scatter(
        lon,
        lat,
        c=pix_med[have],
        cmap="cividis",
        vmin=vmin,
        vmax=vmax,
        s=12.0,
        marker="s",
        edgecolors="none",
        alpha=0.95,
    )
    style_galactic_mollweide(ax)

    cb = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.05, orientation="horizontal")
    cb.set_label("[M/H]  (dex)", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    headline(
        fig,
        "All-sky [M/H] from Gaia XP",
        f"Stream 3 Tier 1, n = {n:,}.  HEALPix NSIDE={NSIDE}, median per pixel.  "
        f"Color clipped at vmed ± 2σ for slide legibility.",
        top=0.88,
    )
    save(fig, "Y09_skymap_mh_headline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
