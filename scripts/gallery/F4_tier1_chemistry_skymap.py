"""F4: Galactic-Mollweide skymaps of Tier-1 stars colored by [M/H] and [α/M].

Eight panels, two per row:

  Stream 1, held-out only (val + test, seed=0); Tier-1 composite gate.
  Stream 2, full hybrid release (no overlap with training); Tier-1.
  Stream 3, full hybrid release (no overlap with training); Tier-1.
  APOGEE DR19, truth labels on the Stream-1 cohort (full sample, no
                 tier filter), used as the reference distribution.

Pixel aggregation uses HEALPix NSIDE=64 (≈55,000 sky pixels of ≈55 arcmin²
each), median per pixel. Color scales fixed across rows (one common scale
per element) so the four rows are directly visually comparable. Streams
that have no coverage in a given pixel are left grey.
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
    apply_style,
    galactic_mollweide,
    radec_to_galactic,
    save_fig,
    style_galactic_mollweide,
)

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

NSIDE = 64
N_PIX = hp.nside2npix(NSIDE)


def _load_stream(stream_id: int) -> pd.DataFrame:
    pred_cols = [
        "source_id", "mh_pred", "alpha_m_pred",
        "teff_sigma", "logg_sigma", "mh_sigma",
        "alpha_m_sigma", "mg_h_sigma", "ood_joint_flag",
        "label_extrapolation_flag",
    ]
    df_p = pd.read_parquet(PRED_PATHS[stream_id], columns=pred_cols)
    df_p = df_p.drop_duplicates(subset="source_id", keep="first")

    feat_cols = ["source_id", "ra_deg", "dec_deg"]
    if stream_id == 1:
        feat_cols += ["fe_h_apogee", "teff_apogee", "b_deg"]
    df_f = pd.read_parquet(FEAT_PATHS[stream_id], columns=feat_cols)
    df_f = df_f.drop_duplicates(subset="source_id", keep="first")

    df = df_f.merge(df_p, on="source_id", how="inner")

    # kin_ood from hybrid where available; default False otherwise.
    df["kin_ood_flag"] = False
    hyb = HYBRID_PATHS.get(stream_id)
    if hyb is not None and hyb.exists():
        df_h = pd.read_parquet(hyb, columns=["source_id", "kin_ood_flag"])
        df = df.merge(df_h, on="source_id", how="left", suffixes=("", "_hyb"))
        if "kin_ood_flag_hyb" in df.columns:
            df["kin_ood_flag"] = df["kin_ood_flag_hyb"].fillna(False).astype(bool)
            df = df.drop(columns=["kin_ood_flag_hyb"])

    df["release_tier"] = assign_release_tier(df).astype(np.int8)

    # Stream 1: keep held-out only (val + test, seed=0).
    if stream_id == 1:
        split = stratified_split_ids(df, seed=0)
        ho = np.concatenate([split["val"], split["test"]])
        df = df.loc[df["source_id"].isin(ho)].reset_index(drop=True)

    # Tier-1 only.
    df = df.loc[df["release_tier"] == 1].reset_index(drop=True)
    return df


def _healpix_median(
    ra_deg: np.ndarray, dec_deg: np.ndarray, value: np.ndarray, nside: int = NSIDE) -> tuple[np.ndarray, np.ndarray]:
    """Per-HEALPix-pixel median of `value`, returned with a count map."""
    l_deg, b_deg = radec_to_galactic(ra_deg, dec_deg)
    # HEALPix conventions: theta = colatitude (= 90 - b), phi = l (radians).
    theta = np.deg2rad(90.0 - b_deg)
    phi = np.deg2rad(l_deg)
    pix = hp.ang2pix(nside, theta, phi)

    n_pix = hp.nside2npix(nside)
    pix_med = np.full(n_pix, np.nan, dtype=np.float64)
    pix_n = np.zeros(n_pix, dtype=np.int64)

    # Group via pandas, fastest reliable per-pixel median in pure Python.
    g = pd.DataFrame({"pix": pix, "v": value}).groupby("pix")
    med = g["v"].median()
    cnt = g["v"].size()
    pix_med[med.index.to_numpy()] = med.to_numpy()
    pix_n[cnt.index.to_numpy()] = cnt.to_numpy()

    return pix_med, pix_n


def _pixel_to_mollweide(pix_idx: np.ndarray, nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert pixel indices to Galactic Mollweide (lon, lat) in radians.

    HEALPix returns (theta, phi) in colatitude-and-longitude radians;
    convert to (l, b) in degrees, then through galactic_mollweide so the
    Galactic-longitude convention (right-to-left) is honoured.
    """
    theta, phi = hp.pix2ang(nside, pix_idx)
    l_deg = np.rad2deg(phi)
    l_deg = np.where(l_deg > 180.0, l_deg - 360.0, l_deg)
    b_deg = 90.0 - np.rad2deg(theta)
    return galactic_mollweide(l_deg, b_deg)


def _draw_panel(
    ax: plt.Axes, df: pd.DataFrame, value_col: str,
    *, vmin: float, vmax: float, cmap: str, label: str, title: str):
    pix_med, pix_n = _healpix_median(
        df["ra_deg"].to_numpy(), df["dec_deg"].to_numpy(),
        df[value_col].to_numpy())
    have = np.where(np.isfinite(pix_med))[0]
    if have.size == 0:
        ax.text(0.5, 0.5, "no Tier-1 stars", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        return None

    lon, lat = _pixel_to_mollweide(have, NSIDE)
    sc = ax.scatter(
        lon, lat, c=pix_med[have], cmap=cmap, vmin=vmin, vmax=vmax,
        s=6.0, marker="s", edgecolors="none", alpha=0.95)
    style_galactic_mollweide(ax)
    ax.set_title(
        f"{title}\nN={len(df):,} Tier-1 stars; "
        f"NSIDE={NSIDE} ({pix_med.shape[0]:,} pixels, "
        f"{int(pix_n.sum()):,} stars binned)",
        fontsize=10)
    cb = plt.colorbar(sc, ax=ax, fraction=0.030, pad=0.04, orientation="vertical")
    cb.set_label(label, fontsize=9)
    return sc


def main() -> int:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    streams: dict[int, pd.DataFrame] = {}
    for sid in (1, 2, 3):
        if not PRED_PATHS[sid].exists():
            print(f"[F4] skipping Stream {sid}: predictions parquet missing")
            continue
        df = _load_stream(sid)
        print(f"[F4] Stream {sid}: Tier-1 n={len(df):,}  "
              f"({'held-out only' if sid == 1 else 'full release'})")
        streams[sid] = df

    if not streams:
        print("[F4] no streams available")
        return 1

    # APOGEE DR19 truth-label reference cohort (Stream-1 features, no tier filter).
    apogee_cols = ["source_id", "ra_deg", "dec_deg",
                   "mh_apogee", "alpha_m_apogee"]
    df_apogee = pd.read_parquet(FEAT_PATHS[1], columns=apogee_cols)
    df_apogee = df_apogee.drop_duplicates(subset="source_id", keep="first")
    df_apogee = df_apogee.dropna(subset=["mh_apogee", "alpha_m_apogee",
                                         "ra_deg", "dec_deg"])
    df_apogee = df_apogee.rename(columns={
        "mh_apogee": "mh_pred",
        "alpha_m_apogee": "alpha_m_pred",
    })
    print(f"[F4] APOGEE DR19 truth: n={len(df_apogee):,} (full Stream-1, no tier filter)")

    # Common color limits across all rows (predictions + truth) for direct comparison.
    mh_all = np.concatenate(
        [d["mh_pred"].to_numpy() for d in streams.values()]
        + [df_apogee["mh_pred"].to_numpy()]
    )
    am_all = np.concatenate(
        [d["alpha_m_pred"].to_numpy() for d in streams.values()]
        + [df_apogee["alpha_m_pred"].to_numpy()]
    )
    mh_vmin = float(np.nanpercentile(mh_all, 1.0))
    mh_vmax = float(np.nanpercentile(mh_all, 99.0))
    am_vmin = float(np.nanpercentile(am_all, 1.0))
    am_vmax = float(np.nanpercentile(am_all, 99.0))
    print(f"[F4] color limits: [M/H] = ({mh_vmin:+.2f}, {mh_vmax:+.2f})  "
          f"[α/M] = ({am_vmin:+.2f}, {am_vmax:+.2f})")

    rows: list[tuple[str, str, pd.DataFrame]] = []
    for sid in sorted(streams.keys()):
        scope = "held-out (val+test, seed=0)" if sid == 1 else "full release"
        rows.append((f"Stream {sid}", scope, streams[sid]))
    rows.append(("APOGEE DR19 (truth)", "Stream-1 cohort, no tier filter", df_apogee))

    n_rows = len(rows)
    fig = plt.figure(figsize=(20, 5.5 * n_rows))
    gs = fig.add_gridspec(n_rows, 2, hspace=0.45, wspace=0.20,
                          top=0.94, bottom=0.04, left=0.05, right=0.97)

    for row, (label, scope, df) in enumerate(rows):
        ax = fig.add_subplot(gs[row, 0], projection="mollweide")
        _draw_panel(
            ax, df, "mh_pred",
            vmin=mh_vmin, vmax=mh_vmax, cmap="cividis",
            label="[M/H] (dex)",
            title=f"{label}, {scope}, [M/H] median per pixel")

        ax = fig.add_subplot(gs[row, 1], projection="mollweide")
        _draw_panel(
            ax, df, "alpha_m_pred",
            vmin=am_vmin, vmax=am_vmax, cmap="cividis",
            label=r"[$\alpha$/M] (dex)",
            title=rf"{label}, {scope}, [$\alpha$/M] median per pixel")

    fig.suptitle(
        "F4, Tier-1 chemistry sky distribution + APOGEE DR19 truth\n"
        f"Galactic Mollweide  (HEALPix NSIDE={NSIDE}, median per pixel; "
        "common color limits across rows for direct comparison)",
        fontsize=13, fontweight="semibold", y=0.985)

    save_fig(fig, OUT / "F4_tier1_chemistry_skymap", tight=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
