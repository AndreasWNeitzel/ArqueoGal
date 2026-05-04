"""F6: Side-by-side comparison: ours vs Gaia DR3 GSP-Spec vs APOGEE DR19.

Three rows showing the same Stream-1 Tier-1 held-out cohort under three
independent labellings of the same physical quantities ([M/H] and [α/Fe]):

  row 0  OURS               XP→ML predictions (mh_pred, alpha_m_pred)
  row 1  GAIA DR3 GSP-Spec  RVS-derived (mh_gspspec, alphafe_gspspec)
  row 2  APOGEE DR19        spectroscopic truth (mh_apogee, alpha_m_apogee)

Cohort: Stream-1 Tier-1 held-out (val+test, seed=0) ∩ GSP-Spec finite ∩
APOGEE finite. Three-way intersection ensures every panel shows the
same set of stars. APOGEE row is the ground truth against which both
ML predictions and GSP-Spec can be benchmarked.

Columns:
  col 0  Galactic Mollweide       median per HEALPix pixel
  col 1  Galactocentric XY        hexbin median (top-down)
  col 2  Galactocentric Z vs R    hexbin median (side view)
  col 3  Chemistry plane          [α/Fe] (or [α/M]) vs [M/H] density

Two output figures: one per chemistry quantity ([M/H] and [α/Fe] / [α/M]).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import Galactocentric, SkyCoord

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _common import (  # noqa: E402
    apply_style,
    galactic_mollweide,
    radec_to_galactic,
    save_fig,
    style_galactic_mollweide,
)

from arqueogal.xp_abundances.main.data import stratified_split_ids  # noqa: E402
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

OUT = REPO / "reports/gallery/F_kinematics"

PRED_S1 = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
GSPSPEC_S1 = REPO / "data/interim/stream1_gaia_dr3_raw.parquet"

NSIDE = 32  # smaller cohort → larger pixels for stable per-pixel medians
XY_EXTENT = (-15.0, 15.0, -15.0, 15.0)
ZR_EXTENT = (0.0, 20.0, -5.0, 5.0)


def _galactocentric(ra, dec, dist_pc):
    ok = np.isfinite(dist_pc) & (dist_pc > 0)
    out = {k: np.full(len(ra), np.nan, dtype=np.float64)
           for k in ("X", "Y", "Z", "Rgal")}
    if not ok.any():
        return out
    icrs = SkyCoord(
        ra=ra[ok] * u.deg, dec=dec[ok] * u.deg,
        distance=dist_pc[ok] * u.pc, frame="icrs")
    gc = icrs.transform_to(Galactocentric())
    X = gc.x.to(u.kpc).value
    Y = gc.y.to(u.kpc).value
    Z = gc.z.to(u.kpc).value
    out["X"][ok] = X
    out["Y"][ok] = Y
    out["Z"][ok] = Z
    out["Rgal"][ok] = np.sqrt(X * X + Y * Y)
    return out


def _load_cohort() -> pd.DataFrame:
    pred_cols = ["source_id", "mh_pred", "alpha_m_pred",
                 "teff_sigma", "logg_sigma", "mh_sigma",
                 "alpha_m_sigma", "mg_h_sigma", "ood_joint_flag",
                 "label_extrapolation_flag"]
    df_p = pd.read_parquet(PRED_S1, columns=pred_cols)
    df_p = df_p.drop_duplicates(subset="source_id", keep="first")

    feat_cols = ["source_id", "ra_deg", "dec_deg", "r_med_photogeo",
                 "fe_h_apogee", "teff_apogee", "b_deg",
                 "mh_apogee", "alpha_m_apogee"]
    df_f = pd.read_parquet(FEAT_S1, columns=feat_cols)
    df_f = df_f.drop_duplicates(subset="source_id", keep="first")
    df = df_f.merge(df_p, on="source_id", how="inner")

    # Stream 1 has no kin_ood column; assign_release_tier treats it as False.
    df["kin_ood_flag"] = False
    df["release_tier"] = assign_release_tier(df).astype(np.int8)

    # Held-out only (val + test, seed=0).
    split = stratified_split_ids(df, seed=0)
    ho = np.concatenate([split["val"], split["test"]])
    df = df.loc[df["source_id"].isin(ho)].reset_index(drop=True)
    df = df.loc[df["release_tier"] == 1].reset_index(drop=True)
    print(f"[F6] Stream-1 Tier-1 held-out: n={len(df):,}")

    # Join GSP-Spec.
    gsp = pd.read_parquet(
        GSPSPEC_S1, columns=["source_id", "mh_gspspec", "alphafe_gspspec"])
    gsp = gsp.dropna(subset=["mh_gspspec", "alphafe_gspspec"])
    df = df.merge(gsp, on="source_id", how="inner")
    print(f"[F6]  ∩ GSP-Spec(mh & αFe both finite): n={len(df):,}")

    # Restrict to APOGEE-truth-finite rows so the third row is meaningful
    # on the same source_id set.
    df = df.dropna(subset=["mh_apogee", "alpha_m_apogee"]).reset_index(drop=True)
    print(f"[F6]  ∩ APOGEE truth (mh & α/M both finite): n={len(df):,}")

    geom = _galactocentric(
        df["ra_deg"].to_numpy(), df["dec_deg"].to_numpy(),
        df["r_med_photogeo"].to_numpy())
    for k, v in geom.items():
        df[k] = v
    return df


def _healpix_median(ra, dec, value, nside=NSIDE):
    l_deg, b_deg = radec_to_galactic(ra, dec)
    theta = np.deg2rad(90.0 - b_deg)
    phi = np.deg2rad(l_deg)
    pix = hp.ang2pix(nside, theta, phi)
    n_pix = hp.nside2npix(nside)
    pix_med = np.full(n_pix, np.nan, dtype=np.float64)
    g = pd.DataFrame({"pix": pix, "v": value}).groupby("pix")
    med = g["v"].median()
    pix_med[med.index.to_numpy()] = med.to_numpy()
    return pix_med


def _pix_to_lonlat(pix_idx, nside):
    theta, phi = hp.pix2ang(nside, pix_idx)
    l_deg = np.rad2deg(phi)
    l_deg = np.where(l_deg > 180.0, l_deg - 360.0, l_deg)
    b_deg = 90.0 - np.rad2deg(theta)
    return galactic_mollweide(l_deg, b_deg)


def _draw_mollweide(ax, df, value_col, *, vmin, vmax, label, title):
    ok = np.isfinite(df[value_col].to_numpy()) & np.isfinite(df["ra_deg"].to_numpy())
    pix_med = _healpix_median(
        df.loc[ok, "ra_deg"].to_numpy(),
        df.loc[ok, "dec_deg"].to_numpy(),
        df.loc[ok, value_col].to_numpy())
    have = np.where(np.isfinite(pix_med))[0]
    if have.size == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return
    lon, lat = _pix_to_lonlat(have, NSIDE)
    sc = ax.scatter(
        lon, lat, c=pix_med[have], cmap="cividis", vmin=vmin, vmax=vmax,
        s=20.0, marker="s", edgecolors="none", alpha=0.95)
    style_galactic_mollweide(ax)
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(sc, ax=ax, fraction=0.030, pad=0.04)
    cb.set_label(label, fontsize=8)


def _draw_xy(ax, df, value_col, *, vmin, vmax, label, title):
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()
    c = df[value_col].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    sc = ax.hexbin(
        x[ok], y[ok], C=c[ok], reduce_C_function=np.median,
        gridsize=50, extent=XY_EXTENT, mincnt=2,
        vmin=vmin, vmax=vmax, cmap="viridis", edgecolors="none")
    ax.scatter([-8.122], [0.0], marker="*", s=80, color="white",
               edgecolor="black", linewidth=0.6, zorder=4)
    ax.scatter([0.0], [0.0], marker="x", s=40, color="white",
               linewidth=1.2, zorder=4)
    ax.set_xlabel("X (kpc)")
    ax.set_ylabel("Y (kpc)")
    ax.set_xlim(XY_EXTENT[0], XY_EXTENT[1])
    ax.set_ylim(XY_EXTENT[2], XY_EXTENT[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(label, fontsize=8)


def _draw_zr(ax, df, value_col, *, vmin, vmax, label, title):
    r = df["Rgal"].to_numpy()
    z = df["Z"].to_numpy()
    c = df[value_col].to_numpy()
    ok = np.isfinite(r) & np.isfinite(z) & np.isfinite(c)
    sc = ax.hexbin(
        r[ok], z[ok], C=c[ok], reduce_C_function=np.median,
        gridsize=(60, 35), extent=ZR_EXTENT, mincnt=2,
        vmin=vmin, vmax=vmax, cmap="viridis", edgecolors="none")
    ax.scatter([8.122], [0.0208], marker="*", s=80, color="white",
               edgecolor="black", linewidth=0.6, zorder=4)
    ax.set_xlabel(r"$R_{\rm gal}$ (kpc)")
    ax.set_ylabel("Z (kpc)")
    ax.set_xlim(ZR_EXTENT[0], ZR_EXTENT[1])
    ax.set_ylim(ZR_EXTENT[2], ZR_EXTENT[3])
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    cb = plt.colorbar(sc, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label(label, fontsize=8)


def _draw_chemistry(ax, df, x_col, y_col, *, x_extent, y_extent,
                    x_label, y_label, title):
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        return
    hb = ax.hexbin(
        x[ok], y[ok],
        gridsize=70, extent=(*x_extent, *y_extent),
        mincnt=1, bins="log", cmap="viridis", edgecolors="none")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_extent)
    ax.set_ylim(y_extent)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    cb = plt.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r"$\log_{10}$ N", fontsize=8)


_ROW_SPECS: list[dict] = [
    {
        "label": "OURS",
        "mh_col": "mh_pred",
        "alpha_col": "alpha_m_pred",
        "alpha_label": r"[$\alpha$/M] (dex)",
        "alpha_title": r"OURS, chemistry plane [$\alpha$/M] vs [M/H]",
    },
    {
        "label": "GAIA DR3 GSP-Spec",
        "mh_col": "mh_gspspec",
        "alpha_col": "alphafe_gspspec",
        "alpha_label": r"[$\alpha$/Fe] (dex)",
        "alpha_title": r"GAIA GSP-Spec, chemistry plane [$\alpha$/Fe] vs [M/H]",
    },
    {
        "label": "APOGEE DR19 (truth)",
        "mh_col": "mh_apogee",
        "alpha_col": "alpha_m_apogee",
        "alpha_label": r"[$\alpha$/M] (dex)",
        "alpha_title": r"APOGEE DR19, chemistry plane [$\alpha$/M] vs [M/H]",
    },
]


def _render(
    df: pd.DataFrame, *, quantity: str, value_key: str,
    label: str, suffix: str, alpha_y_extent: tuple[float, float]) -> None:
    """Render one 3x4 figure for a single chemistry quantity.

    value_key in {"mh_col", "alpha_col"}, selects which column from each
    row spec to use for the Mollweide/XY/Z-Rgal panels.
    """
    cols_used = [r[value_key] for r in _ROW_SPECS]
    vals = np.concatenate([df[c].to_numpy() for c in cols_used])
    vmin = float(np.nanpercentile(vals, 1.0))
    vmax = float(np.nanpercentile(vals, 99.0))
    print(f"[F6:{suffix}] {quantity} color limits: ({vmin:+.3f}, {vmax:+.3f})")

    n_rows = len(_ROW_SPECS)
    fig = plt.figure(figsize=(24, 5.0 * n_rows))
    gs = fig.add_gridspec(n_rows, 4, hspace=0.40, wspace=0.30,
                          width_ratios=[1.4, 1.0, 1.3, 1.0],
                          top=0.93, bottom=0.05, left=0.04, right=0.97)

    for r, spec in enumerate(_ROW_SPECS):
        col = spec[value_key]
        row_label = spec["label"]

        ax = fig.add_subplot(gs[r, 0], projection="mollweide")
        title = f"{row_label}, Mollweide {quantity}"
        if r == 0:
            title = f"{title}\nn={len(df):,}"
        _draw_mollweide(ax, df, col, vmin=vmin, vmax=vmax,
                        label=label, title=title)

        ax = fig.add_subplot(gs[r, 1])
        _draw_xy(ax, df, col, vmin=vmin, vmax=vmax,
                 label=label, title=f"{row_label}, XY {quantity}")

        ax = fig.add_subplot(gs[r, 2])
        _draw_zr(ax, df, col, vmin=vmin, vmax=vmax,
                 label=label,
                 title=rf"{row_label}, Z vs $R_{{\rm gal}}$ {quantity}")

        ax = fig.add_subplot(gs[r, 3])
        _draw_chemistry(
            ax, df, spec["mh_col"], spec["alpha_col"],
            x_extent=(-2.0, 0.6), y_extent=alpha_y_extent,
            x_label="[M/H] (dex)", y_label=spec["alpha_label"],
            title=spec["alpha_title"])

    fig.suptitle(
        "F6, OURS vs Gaia DR3 GSP-Spec vs APOGEE DR19  "
        f"({quantity}, same physical quantity on all three rows)\n"
        "Stream-1 Tier-1 held-out ∩ GSP-Spec finite ∩ APOGEE finite, "
        f"n = {len(df):,}.  Common color scale across rows.",
        fontsize=12, fontweight="semibold", y=0.985)
    save_fig(fig, OUT / f"F6_gaia_chemistry_comparison_{suffix}", tight=False)


def main() -> int:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    df = _load_cohort()

    _render(df, quantity="[M/H]", value_key="mh_col",
            label="[M/H] (dex)", suffix="mh",
            alpha_y_extent=(-0.10, 0.45))

    _render(df, quantity=r"[$\alpha$/Fe] / [$\alpha$/M]", value_key="alpha_col",
            label=r"[$\alpha$/Fe] or [$\alpha$/M] (dex)", suffix="alpha",
            alpha_y_extent=(-0.10, 0.45))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
