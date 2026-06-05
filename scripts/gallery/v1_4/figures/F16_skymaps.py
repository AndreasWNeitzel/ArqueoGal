"""F16: 6-panel chemical Mollweide on Stream 2.

Three columns (GSP-Spec, APOGEE DR19 cross-match, JANUS Tier 1) x
two rows ([M/H], [alpha/M]). Stream-2 source_ids drive every column;
each column reports its surviving cohort size.
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
from astropy.coordinates import ICRS, Galactic, SkyCoord
import astropy.units as u

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "gallery" / "v1_4"))
sys.path.insert(0, str(REPO / "src"))

from _loaders import (  # noqa: E402
    FEAT_S2, GAIA_RAW_S2, load_s2_apogee, load_s2_predictions,
)
from arqueogal.style import LABELS, apply_style, colorbar, save  # noqa: E402

NSIDE = 32
MIN_PER_PIX = 1


def _radec_to_galactic(ra_deg, dec_deg):
    sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame=ICRS)
    g = sc.transform_to(Galactic)
    return g.l.degree, g.b.degree


def _moll(l, b):
    lon = np.radians(np.where(l > 180, l - 360, l))
    lat = np.radians(b)
    return -lon, lat


def _per_pix(ra, dec, vals):
    npix = hp.nside2npix(NSIDE)
    l, b = _radec_to_galactic(ra, dec)
    pix = hp.ang2pix(NSIDE, np.radians(90 - b), np.radians(l), nest=False)
    vals = vals.astype(np.float32)
    ok = np.isfinite(vals)
    pix = pix[ok]; vals = vals[ok]
    out = np.full(npix, np.nan, dtype=np.float32)
    if pix.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    order = np.argsort(pix)
    pix_s = pix[order]; vals_s = vals[order]
    edges = np.searchsorted(pix_s, np.arange(npix + 1))
    counts = np.diff(edges)
    for k in range(npix):
        if counts[k] >= MIN_PER_PIX:
            out[k] = float(np.median(vals_s[edges[k]:edges[k + 1]]))
    have = np.where(np.isfinite(out))[0]
    return have, out[have]


def _draw(ax, pix, vals, vmin, vmax, label, cmap):
    if pix.size == 0:
        ax.text(0.5, 0.5, r"no data", ha="center", va="center",
                transform=ax.transAxes, color="#5C6378")
        return None
    theta, phi = hp.pix2ang(NSIDE, pix, nest=False)
    l = np.degrees(phi); b = 90.0 - np.degrees(theta)
    lon, lat = _moll(l, b)
    sc = ax.scatter(lon, lat, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=3.5, marker="s", edgecolors="none", alpha=0.95)
    ax.grid(True, lw=0.4, color="#D0D3DC", alpha=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return sc


def main() -> int:
    apply_style()

    pred = load_s2_predictions()
    pred = pred.loc[pred["release_tier"] == 1].reset_index(drop=True)
    feat = pd.read_parquet(
        FEAT_S2, columns=["source_id", "ra_deg", "dec_deg"]
    ).drop_duplicates("source_id")
    arq = pred.merge(feat, on="source_id", how="inner")
    arq = arq.dropna(subset=["ra_deg", "dec_deg",
                              "mh_pred", "alpha_m_pred"]).reset_index(drop=True)

    gsp_raw = pd.read_parquet(GAIA_RAW_S2,
                                 columns=["source_id", "mh_gspspec",
                                           "alphafe_gspspec"]
                                ).drop_duplicates("source_id")
    gsp = (feat.merge(gsp_raw, on="source_id", how="inner")
                  .dropna(subset=["ra_deg", "dec_deg",
                                   "mh_gspspec", "alphafe_gspspec"])
                  .reset_index(drop=True))

    apo = load_s2_apogee(pred["source_id"].to_numpy())
    apo = (feat.merge(apo, on="source_id", how="inner")
               .dropna(subset=["ra_deg", "dec_deg", "m_h_atm", "alpha_m_atm"])
               .reset_index(drop=True))

    fig = plt.figure(figsize=(13.5, 6.0), layout="constrained")
    gs = fig.add_gridspec(2, 3)

    columns = [
        (r"GSP-Spec ($n = $" + f"{len(gsp):,})", gsp,
         "mh_gspspec", "alphafe_gspspec", LABELS["alpha_Fe"]),
        (r"APOGEE DR19 ($n = $" + f"{len(apo):,})", apo,
         "m_h_atm", "alpha_m_atm", LABELS["alpha_M"]),
        (r"JANUS Tier 1 ($n = $" + f"{len(arq):,})", arq,
         "mh_pred", "alpha_m_pred", LABELS["alpha_M"]),
    ]

    for j, (title, frame, mh_col, am_col, am_lab) in enumerate(columns):
        # Top row: [M/H].
        ax = fig.add_subplot(gs[0, j], projection="mollweide")
        pix_mh, mh_vals = _per_pix(
            frame["ra_deg"].to_numpy(), frame["dec_deg"].to_numpy(),
            frame[mh_col].to_numpy(),
        )
        sc = _draw(ax, pix_mh, mh_vals, -0.5, 0.2,
                    LABELS["Mh"], cmap="jet")
        ax.set_title(title, fontweight="regular")
        if sc is not None:
            colorbar(ax, sc, LABELS["Mh"], fraction=0.030, pad=0.04)

        # Bottom row: [alpha/M] (or [alpha/Fe] for GSP-Spec).
        ax = fig.add_subplot(gs[1, j], projection="mollweide")
        pix_am, am_vals = _per_pix(
            frame["ra_deg"].to_numpy(), frame["dec_deg"].to_numpy(),
            frame[am_col].to_numpy(),
        )
        sc = _draw(ax, pix_am, am_vals, -0.1, 0.2,
                    am_lab, cmap="jet")
        if sc is not None:
            colorbar(ax, sc, am_lab, fraction=0.030, pad=0.04)

    save(fig, "F16_skymaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
