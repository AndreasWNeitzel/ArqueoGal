"""F2: Per-stream spatial geometry — separate subplots per stream.

Layout (3 rows × 3 cols + 1 overlay row):
  Row 1   : Galactic Mollweide  per stream (S1 / S2 / S3)
  Row 2   : Top-down Galactocentric XY  per stream
  Row 3   : Side-view Galactocentric R-Z  per stream
  Row 4   : Heliocentric distance histogram, ALL streams overlapping
            (the only panel where comparison-by-overlap is the right
             visualisation).

Distances come from Bailer-Jones+2021 ``r_med_photogeo`` (parsec) when
available; falls back to ``1/parallax_mas`` (kpc) for stars where BJ21
is missing. Galactocentric transform uses astropy's default frame
(R_sun = 8.122 kpc, z_sun = 20.8 pc).
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

from _common import (  # noqa: E402
    apply_style,
    galactic_mollweide,
    radec_to_galactic,
    save_fig,
    style_galactic_mollweide,
)

FEAT = {
    1: REPO / "data/processed/pipeline1_features_stream1_kiel.parquet",
    2: REPO / "data/processed/pipeline1_features_stream2.parquet",
    3: REPO / "data/processed/pipeline1_features_stream3.parquet",
}
STREAM_COLOR = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
STREAM_LABEL = {
    1: "Stream 1 (APOGEE × XP)",
    2: "Stream 2 (TESS × XP)",
    3: "Stream 3 (Andrae+23 × XP)",
}
N_PER_STREAM = 80_000


def _load_stream(sid: int) -> pd.DataFrame:
    cols = ["source_id", "ra_deg", "dec_deg", "r_med_photogeo"]
    try:
        df = pd.read_parquet(FEAT[sid], columns=cols)
    except (KeyError, ValueError):
        cols2 = ["source_id", "ra_deg", "dec_deg"]
        df = pd.read_parquet(FEAT[sid], columns=cols2)
        df["r_med_photogeo"] = np.nan
    df = df.drop_duplicates("source_id").reset_index(drop=True)
    if len(df) > N_PER_STREAM:
        df = df.sample(N_PER_STREAM, random_state=sid).reset_index(drop=True)
    return df


def _galactocentric_xyz(df: pd.DataFrame) -> dict[str, np.ndarray]:
    d_pc = df["r_med_photogeo"].to_numpy()
    ok = np.isfinite(d_pc) & (d_pc > 0)
    out = {k: np.full(len(df), np.nan) for k in ("X", "Y", "Z", "R")}
    if not ok.any():
        return out
    icrs = SkyCoord(
        ra=df.loc[ok, "ra_deg"].to_numpy() * u.deg,
        dec=df.loc[ok, "dec_deg"].to_numpy() * u.deg,
        distance=d_pc[ok] * u.pc,
        frame="icrs",
    )
    gc = icrs.transform_to(Galactocentric())
    out["X"][ok] = gc.x.to(u.kpc).value
    out["Y"][ok] = gc.y.to(u.kpc).value
    out["Z"][ok] = gc.z.to(u.kpc).value
    out["R"][ok] = np.sqrt(out["X"][ok] ** 2 + out["Y"][ok] ** 2)
    return out


def _mollweide_panel(ax, df, sid):
    l, b = radec_to_galactic(df["ra_deg"].to_numpy(), df["dec_deg"].to_numpy())
    lon, lat = galactic_mollweide(l, b)
    ax.scatter(
        lon, lat, s=1.5, alpha=0.30, color=STREAM_COLOR[sid], edgecolors="none", rasterized=True
    )
    style_galactic_mollweide(ax)
    ax.set_title(f"{STREAM_LABEL[sid]}  n = {len(df):,}", fontsize=10)


def _xy_panel(ax, geom, sid):
    x, y = geom["X"], geom["Y"]
    ok = np.isfinite(x) & np.isfinite(y)
    ax.hexbin(
        x[ok],
        y[ok],
        gridsize=80,
        extent=(-15, 15, -15, 15),
        mincnt=1,
        bins="log",
        cmap="viridis",
        edgecolors="none",
    )
    ax.scatter(
        [-8.122],
        [0.0],
        marker="*",
        s=120,
        color="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
    )
    ax.scatter([0.0], [0.0], marker="x", s=80, color="white", linewidth=1.4, zorder=4)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal")
    ax.set_xlabel("X (kpc)")
    ax.set_ylabel("Y (kpc)")
    ax.set_title(f"{STREAM_LABEL[sid]}  (top-down)", fontsize=10)
    ax.grid(True, alpha=0.25)


def _rz_panel(ax, geom, sid):
    R, Z = geom["R"], geom["Z"]
    ok = np.isfinite(R) & np.isfinite(Z)
    ax.hexbin(
        R[ok],
        Z[ok],
        gridsize=80,
        extent=(0, 20, -5, 5),
        mincnt=1,
        bins="log",
        cmap="viridis",
        edgecolors="none",
    )
    ax.scatter(
        [8.122],
        [0.0208],
        marker="*",
        s=120,
        color="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
    )
    ax.set_xlim(0, 20)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r"$R_{\rm gal}$ (kpc)")
    ax.set_ylabel("Z (kpc)")
    ax.set_title(f"{STREAM_LABEL[sid]}  (side view)", fontsize=10)
    ax.grid(True, alpha=0.25)


def _distance_overlay(ax, frames):
    bins = np.linspace(0, 12000, 80)
    for sid, df in frames.items():
        d = df["r_med_photogeo"].to_numpy()
        d = d[np.isfinite(d) & (d > 0)] / 1000.0  # kpc
        ax.hist(
            d,
            bins=bins / 1000.0,
            histtype="step",
            color=STREAM_COLOR[sid],
            lw=2.2,
            label=f"{STREAM_LABEL[sid]}  median={np.median(d):.2f} kpc",
        )
    ax.set_xlabel("BJ21 photogeometric distance  (kpc)")
    ax.set_ylabel("count")
    ax.set_xlim(0, 12)
    ax.set_title("Heliocentric distance distribution (overlay)", fontsize=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.25)


def main() -> int:
    apply_style()
    frames = {sid: _load_stream(sid) for sid in (1, 2, 3)}
    geoms = {sid: _galactocentric_xyz(df) for sid, df in frames.items()}

    # 4-row × 3-col layout. Row 4 spans all 3 columns for the distance overlay.
    fig = plt.figure(figsize=(20, 22))
    gs = fig.add_gridspec(
        4,
        3,
        hspace=0.40,
        wspace=0.30,
        height_ratios=[1.0, 1.2, 1.0, 0.9],
        top=0.95,
        bottom=0.05,
        left=0.05,
        right=0.97,
    )

    # Row 1: Mollweide per stream.
    for c, sid in enumerate((1, 2, 3)):
        ax = fig.add_subplot(gs[0, c], projection="mollweide")
        _mollweide_panel(ax, frames[sid], sid)
    # Row 2: top-down XY per stream.
    for c, sid in enumerate((1, 2, 3)):
        ax = fig.add_subplot(gs[1, c])
        _xy_panel(ax, geoms[sid], sid)
    # Row 3: side R-Z per stream.
    for c, sid in enumerate((1, 2, 3)):
        ax = fig.add_subplot(gs[2, c])
        _rz_panel(ax, geoms[sid], sid)
    # Row 4: distance overlay.
    ax = fig.add_subplot(gs[3, :])
    _distance_overlay(ax, frames)

    fig.suptitle(
        f"F2. Per-stream spatial geometry  ({N_PER_STREAM:,} stars per stream;"
        " BJ21 photogeometric distance + astropy Galactocentric)",
        fontsize=12,
        fontweight="semibold",
        y=0.985,
    )
    save_fig(fig, REPO / "reports/gallery/F_kinematics/F2_geometry", tight=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
