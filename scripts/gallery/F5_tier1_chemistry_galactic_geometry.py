"""F5: Galactocentric geometry of Tier-1 chemistry, XY top-down + Z vs Rgal.

For each of three streams (Tier 1 only; Stream 1 restricted to held-out
val+test seed=0) plus an APOGEE DR19 truth reference, draws four panels:

  col 1  X vs Y top-down,   coloured by [M/H]
  col 2  X vs Y top-down,   coloured by [α/M]
  col 3  Z vs Rgal side,    coloured by [M/H]
  col 4  Z vs Rgal side,    coloured by [α/M]

Galactocentric coordinates are computed via astropy.coordinates with
its default Galactocentric frame (Sun at galcen_distance = 8.122 kpc,
z_sun = 20.8 pc; Bovy 2015 / Gravity 2018 conventions). Distance is
the Bailer-Jones+2021 photogeometric ``r_med_photogeo``.

Aggregation: 2-D hexbin per panel with reduce_C_function = np.median so
each cell shows the median chemistry of stars at that location, not a
star count. Common color limits across rows for direct comparison;
common spatial extents across streams so the maps are co-registered.
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

from _common import apply_style, save_fig  # noqa: E402

from arqueogal.xp_abundances.main.data import stratified_split_ids  # noqa: E402
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

OUT = REPO / "reports/gallery/F_kinematics"

PRED_PATHS = {
    1: REPO / "data/processed/pipeline1_predictions_stream1.parquet",
    2: REPO / "data/processed/pipeline1_predictions_stream2.parquet",
    3: REPO / "data/processed/pipeline1_predictions_stream3.parquet",
}
FEAT_PATHS = {
    1: REPO / "data/processed/pipeline1_features_stream1_kiel.parquet",
    2: REPO / "data/processed/pipeline1_features_stream2.parquet",
    3: REPO / "data/processed/pipeline1_features_stream3.parquet",
}
HYBRID_PATHS = {
    2: REPO / "release/D-Cat-b/hybrid_pipeline_run_stream2/predictions_with_features.parquet",
    3: REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet",
}

# Spatial windows, clip extreme distance outliers so hexbins resolve the disc.
XY_EXTENT = (-15.0, 15.0, -15.0, 15.0)   # kpc
ZR_EXTENT = (0.0, 20.0, -5.0, 5.0)       # Rgal in kpc, Z in kpc
HEX_GRID_XY = 80
HEX_GRID_ZR = (90, 50)


def _galactocentric(
    ra_deg: np.ndarray, dec_deg: np.ndarray, distance_pc: np.ndarray) -> dict[str, np.ndarray]:
    """Compute Galactocentric (X, Y, Z, Rgal) in kpc from (ra, dec, distance)."""
    ok = np.isfinite(distance_pc) & (distance_pc > 0)
    out = {k: np.full(len(ra_deg), np.nan, dtype=np.float64)
           for k in ("X", "Y", "Z", "Rgal")}
    if not ok.any():
        return out

    icrs = SkyCoord(
        ra=ra_deg[ok] * u.deg,
        dec=dec_deg[ok] * u.deg,
        distance=distance_pc[ok] * u.pc,
        frame="icrs")
    gc = icrs.transform_to(Galactocentric())
    X = gc.x.to(u.kpc).value
    Y = gc.y.to(u.kpc).value
    Z = gc.z.to(u.kpc).value
    Rgal = np.sqrt(X * X + Y * Y)
    out["X"][ok] = X
    out["Y"][ok] = Y
    out["Z"][ok] = Z
    out["Rgal"][ok] = Rgal
    return out


def _load_stream(stream_id: int) -> pd.DataFrame:
    pred_cols = [
        "source_id", "mh_pred", "alpha_m_pred",
        "teff_sigma", "logg_sigma", "mh_sigma",
        "alpha_m_sigma", "mg_h_sigma", "ood_joint_flag",
        "label_extrapolation_flag",
    ]
    df_p = pd.read_parquet(PRED_PATHS[stream_id], columns=pred_cols)
    df_p = df_p.drop_duplicates(subset="source_id", keep="first")

    feat_cols = ["source_id", "ra_deg", "dec_deg", "r_med_photogeo"]
    if stream_id == 1:
        feat_cols += ["fe_h_apogee", "teff_apogee", "b_deg",
                      "mh_apogee", "alpha_m_apogee"]
    df_f = pd.read_parquet(FEAT_PATHS[stream_id], columns=feat_cols)
    df_f = df_f.drop_duplicates(subset="source_id", keep="first")

    df = df_f.merge(df_p, on="source_id", how="inner")

    df["kin_ood_flag"] = False
    hyb = HYBRID_PATHS.get(stream_id)
    if hyb is not None and hyb.exists():
        df_h = pd.read_parquet(hyb, columns=["source_id", "kin_ood_flag"])
        df = df.merge(df_h, on="source_id", how="left", suffixes=("", "_hyb"))
        if "kin_ood_flag_hyb" in df.columns:
            df["kin_ood_flag"] = df["kin_ood_flag_hyb"].fillna(False).astype(bool)
            df = df.drop(columns=["kin_ood_flag_hyb"])

    df["release_tier"] = assign_release_tier(df).astype(np.int8)

    if stream_id == 1:
        split = stratified_split_ids(df, seed=0)
        ho = np.concatenate([split["val"], split["test"]])
        df = df.loc[df["source_id"].isin(ho)].reset_index(drop=True)

    df = df.loc[df["release_tier"] == 1].reset_index(drop=True)

    geom = _galactocentric(
        df["ra_deg"].to_numpy(), df["dec_deg"].to_numpy(),
        df["r_med_photogeo"].to_numpy())
    for k, v in geom.items():
        df[k] = v
    return df


def _load_apogee_truth() -> pd.DataFrame:
    cols = ["source_id", "ra_deg", "dec_deg", "r_med_photogeo",
            "mh_apogee", "alpha_m_apogee"]
    df = pd.read_parquet(FEAT_PATHS[1], columns=cols)
    df = df.drop_duplicates(subset="source_id", keep="first")
    df = df.dropna(subset=["mh_apogee", "alpha_m_apogee",
                           "ra_deg", "dec_deg", "r_med_photogeo"])
    df = df.rename(columns={"mh_apogee": "mh_pred",
                            "alpha_m_apogee": "alpha_m_pred"})
    geom = _galactocentric(
        df["ra_deg"].to_numpy(), df["dec_deg"].to_numpy(),
        df["r_med_photogeo"].to_numpy())
    for k, v in geom.items():
        df[k] = v
    return df.reset_index(drop=True)


def _hexbin(
    ax: plt.Axes,
    x: np.ndarray, y: np.ndarray, c: np.ndarray,
    *, gridsize, extent, vmin, vmax, cmap, mincnt: int = 3):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if not ok.any():
        ax.text(0.5, 0.5, "no stars", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        return None
    return ax.hexbin(
        x[ok], y[ok], C=c[ok], reduce_C_function=np.median,
        gridsize=gridsize, extent=extent, mincnt=mincnt,
        vmin=vmin, vmax=vmax, cmap=cmap, edgecolors="none")


def _draw_xy(ax, df, value_col, *, vmin, vmax, label, title):
    sc = _hexbin(
        ax, df["X"].to_numpy(), df["Y"].to_numpy(), df[value_col].to_numpy(),
        gridsize=HEX_GRID_XY, extent=XY_EXTENT,
        vmin=vmin, vmax=vmax, cmap="viridis")
    ax.scatter([-8.122], [0.0], marker="*", s=80, color="white",
               edgecolor="black", linewidth=0.6, zorder=4, label="Sun")
    ax.scatter([0.0], [0.0], marker="x", s=40, color="white",
               linewidth=1.2, zorder=4, label="GC")
    ax.set_xlabel("X (kpc)")
    ax.set_ylabel("Y (kpc)")
    ax.set_xlim(XY_EXTENT[0], XY_EXTENT[1])
    ax.set_ylim(XY_EXTENT[2], XY_EXTENT[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    if sc is not None:
        cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(label, fontsize=8)


def _draw_zr(ax, df, value_col, *, vmin, vmax, label, title):
    sc = _hexbin(
        ax, df["Rgal"].to_numpy(), df["Z"].to_numpy(), df[value_col].to_numpy(),
        gridsize=HEX_GRID_ZR, extent=ZR_EXTENT,
        vmin=vmin, vmax=vmax, cmap="viridis")
    ax.scatter([8.122], [0.0208], marker="*", s=80, color="white",
               edgecolor="black", linewidth=0.6, zorder=4, label="Sun")
    ax.set_xlabel(r"$R_{\rm gal}$ (kpc)")
    ax.set_ylabel("Z (kpc)")
    ax.set_xlim(ZR_EXTENT[0], ZR_EXTENT[1])
    ax.set_ylim(ZR_EXTENT[2], ZR_EXTENT[3])
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    if sc is not None:
        cb = plt.colorbar(sc, ax=ax, fraction=0.030, pad=0.02)
        cb.set_label(label, fontsize=8)


def main() -> int:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    streams: dict[int, pd.DataFrame] = {}
    for sid in (1, 2, 3):
        if not PRED_PATHS[sid].exists():
            print(f"[F5] skipping Stream {sid}: predictions missing")
            continue
        df = _load_stream(sid)
        n_geom = int(np.isfinite(df["X"]).sum())
        print(f"[F5] Stream {sid}: Tier-1 n={len(df):,}; "
              f"with finite Galactocentric n={n_geom:,}")
        streams[sid] = df

    df_apogee = _load_apogee_truth()
    n_geom = int(np.isfinite(df_apogee["X"]).sum())
    print(f"[F5] APOGEE DR19 truth n={len(df_apogee):,}; finite Galactocentric n={n_geom:,}")

    rows: list[tuple[str, str, pd.DataFrame]] = []
    for sid in sorted(streams.keys()):
        scope = "held-out (val+test, seed=0)" if sid == 1 else "full release"
        rows.append((f"Stream {sid}", scope, streams[sid]))
    rows.append(("APOGEE DR19 (truth)", "Stream-1 cohort, no tier filter", df_apogee))

    # Common color limits over predictions + truth.
    mh_all = np.concatenate([d["mh_pred"].to_numpy() for _, _, d in rows])
    am_all = np.concatenate([d["alpha_m_pred"].to_numpy() for _, _, d in rows])
    mh_vmin = float(np.nanpercentile(mh_all, 1.0))
    mh_vmax = float(np.nanpercentile(mh_all, 99.0))
    am_vmin = float(np.nanpercentile(am_all, 1.0))
    am_vmax = float(np.nanpercentile(am_all, 99.0))
    print(f"[F5] color limits: [M/H] = ({mh_vmin:+.2f}, {mh_vmax:+.2f})  "
          f"[α/M] = ({am_vmin:+.2f}, {am_vmax:+.2f})")

    n_rows = len(rows)
    fig = plt.figure(figsize=(22, 5.0 * n_rows))
    gs = fig.add_gridspec(n_rows, 4, hspace=0.40, wspace=0.30,
                          width_ratios=[1.0, 1.0, 1.3, 1.3],
                          top=0.93, bottom=0.05, left=0.05, right=0.97)

    for row, (label, scope, df) in enumerate(rows):
        ax = fig.add_subplot(gs[row, 0])
        _draw_xy(ax, df, "mh_pred",
                 vmin=mh_vmin, vmax=mh_vmax,
                 label="[M/H] (dex)",
                 title=f"{label}, XY  ([M/H])\n{scope}, n={len(df):,}")

        ax = fig.add_subplot(gs[row, 1])
        _draw_xy(ax, df, "alpha_m_pred",
                 vmin=am_vmin, vmax=am_vmax,
                 label=r"[$\alpha$/M] (dex)",
                 title=rf"{label}, XY  ([$\alpha$/M])")

        ax = fig.add_subplot(gs[row, 2])
        _draw_zr(ax, df, "mh_pred",
                 vmin=mh_vmin, vmax=mh_vmax,
                 label="[M/H] (dex)",
                 title=f"{label}, Z vs $R_{{\\rm gal}}$  ([M/H])")

        ax = fig.add_subplot(gs[row, 3])
        _draw_zr(ax, df, "alpha_m_pred",
                 vmin=am_vmin, vmax=am_vmax,
                 label=r"[$\alpha$/M] (dex)",
                 title=rf"{label}, Z vs $R_{{\rm gal}}$  ([$\alpha$/M])")

    fig.suptitle(
        "F5, Tier-1 chemistry in Galactocentric geometry + APOGEE DR19 truth\n"
        "XY top-down, Z vs Rgal side view.  BJ21 distance, astropy "
        "Galactocentric (Sun at 8.122 kpc, z_sun = 20.8 pc).  Hexbin median per cell.",
        fontsize=12, fontweight="semibold", y=0.985)

    save_fig(fig, OUT / "F5_tier1_chemistry_galactic_geometry", tight=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
