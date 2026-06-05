"""Y38: Stream-1 Tier-1 holdout, chemical Mollweide sky maps.

Two side-by-side Mollweide panels: median [M/H] (left) and median
[alpha/M] (right). HEALPix NSIDE = 32 (49,152 pixels). Median per pixel
reported only when at least 4 stars fall in the cell.

Slide-friendly 16:6 layout.
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
from _presentation import PALETTE, apply_style, headline, save  # noqa: E402

from arqueogal.xp_abundances.main.data import stratified_split_ids  # noqa: E402
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED_S1 = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"

NSIDE = 32
MIN_PER_PIX = 4

_TITLE_KW = dict(fontsize=11, fontweight="normal", color=PALETTE["ink"], pad=8)


def _load() -> pd.DataFrame:
    pcols = [
        "source_id", "teff_pred", "logg_pred", "mh_pred",
        "alpha_m_pred", "mg_h_pred",
        "ood_joint_flag", "label_extrapolation_flag",
    ]
    p = pd.read_parquet(PRED_S1, columns=pcols).drop_duplicates("source_id")
    fcols = ["source_id", "ra_deg", "dec_deg", "av_los", "fe_h_apogee",
             "teff_apogee", "b_deg"]
    f = pd.read_parquet(FEAT_S1, columns=fcols).drop_duplicates("source_id")
    df = f.merge(p, on="source_id", how="inner")
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    split = stratified_split_ids(df, seed=0)
    holdout_ids = np.concatenate([split["val"], split["test"]])
    df = df.loc[df["source_id"].isin(holdout_ids)].reset_index(drop=True)
    return df.loc[df["release_tier"] == 1].reset_index(drop=True)


def _pixel_aggregate(df: pd.DataFrame, value_col: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-HEALPix-pixel median (or count) over the supplied column.

    Returns (pixel_id_array_with_data, value_per_pixel).
    """
    npix = hp.nside2npix(NSIDE)
    l, b = radec_to_galactic(df["ra_deg"].to_numpy(), df["dec_deg"].to_numpy())
    pix = hp.ang2pix(NSIDE, np.radians(90.0 - b), np.radians(l), nest=False)
    if value_col is None:
        # Count map.
        bins = np.bincount(pix, minlength=npix)
        have = np.where(bins >= MIN_PER_PIX)[0]
        return have, bins[have].astype(np.float32)
    vals = df[value_col].to_numpy(dtype=np.float32)
    ok = np.isfinite(vals)
    pix = pix[ok]; vals = vals[ok]
    if pix.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    order = np.argsort(pix)
    pix_s = pix[order]; vals_s = vals[order]
    edges = np.searchsorted(pix_s, np.arange(npix + 1))
    counts = np.diff(edges)
    out = np.full(npix, np.nan, dtype=np.float32)
    for k in range(npix):
        if counts[k] >= MIN_PER_PIX:
            out[k] = float(np.median(vals_s[edges[k]:edges[k + 1]]))
    have = np.where(np.isfinite(out))[0]
    return have, out[have]


def _draw_panel(ax, df, value_col, *, vmin, vmax, label, title, cmap="viridis"):
    pix, vals = _pixel_aggregate(df, value_col)
    if pix.size == 0:
        ax.text(0.0, 0.0, "no data", ha="center", va="center")
        ax.set_title(title, **_TITLE_KW)
        return
    theta, phi = hp.pix2ang(NSIDE, pix, nest=False)
    l = np.degrees(phi); b = 90.0 - np.degrees(theta)
    lon, lat = galactic_mollweide(l, b)
    sc = ax.scatter(
        lon, lat, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
        s=20.0, marker="s", edgecolors="none", alpha=0.95,
    )
    style_galactic_mollweide(ax)
    ax.set_title(title, **_TITLE_KW)
    cb = plt.colorbar(sc, ax=ax, fraction=0.030, pad=0.04)
    cb.set_label(label, fontsize=10)


def main() -> int:
    apply_style()
    df = _load()
    if df.empty:
        print("[Y38] no Tier-1 holdout rows, aborting")
        return 1

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.20,
                          top=0.78, bottom=0.04, left=0.03, right=0.985)

    mh_lo = float(np.nanpercentile(df["mh_pred"], 2))
    mh_hi = float(np.nanpercentile(df["mh_pred"], 98))
    am_lo = float(np.nanpercentile(df["alpha_m_pred"], 2))
    am_hi = float(np.nanpercentile(df["alpha_m_pred"], 98))

    ax = fig.add_subplot(gs[0, 0], projection="mollweide")
    _draw_panel(ax, df, "mh_pred", vmin=mh_lo, vmax=mh_hi,
                label="median [M/H] (dex)",
                title="median [M/H]", cmap="viridis")
    ax = fig.add_subplot(gs[0, 1], projection="mollweide")
    _draw_panel(ax, df, "alpha_m_pred", vmin=am_lo, vmax=am_hi,
                label=r"median [$\alpha$/M] (dex)",
                title=r"median [$\alpha$/M]", cmap="viridis")

    headline(
        fig,
        "Stream 1 Tier 1 holdout, chemical Mollweide sky maps",
        f"n = {len(df):,}; HEALPix NSIDE = {NSIDE}, at least 4 stars per pixel.",
        top=0.78,
    )
    save(fig, "Y38_tier1_skymaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
