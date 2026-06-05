"""F14: Stream-2 kinematics by chemistry (slide 15)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "gallery" / "v1_4"))
sys.path.insert(0, str(REPO / "src"))

from _loaders import FEAT_S2, GAIA_RAW_S2, load_s2_predictions  # noqa: E402

from arqueogal.data.kinematics import compute_actions  # noqa: E402
from arqueogal.style import (  # noqa: E402
    LABELS, apply_style, colorbar, median_per_cell, save,
)

V_LSR = 233.1
N_MAX = 8000


def _load() -> pd.DataFrame:
    pred = load_s2_predictions()
    pred = pred.loc[pred["release_tier"] == 1]
    feat = pd.read_parquet(
        FEAT_S2,
        columns=["source_id", "ra_deg", "dec_deg", "r_med_photogeo"],
    ).drop_duplicates("source_id")
    raw = pd.read_parquet(
        GAIA_RAW_S2,
        columns=["source_id", "pmra", "pmdec", "radial_velocity"],
    ).drop_duplicates("source_id")
    df = (pred.merge(feat, on="source_id")
                .merge(raw, on="source_id"))
    df = df.rename(columns={"ra_deg": "ra", "dec_deg": "dec"})
    needed = ["source_id", "ra", "dec", "r_med_photogeo",
              "pmra", "pmdec", "radial_velocity"]
    df = df.dropna(subset=needed).reset_index(drop=True)
    df = df.loc[df["r_med_photogeo"] > 0].reset_index(drop=True)
    if len(df) > N_MAX:
        df = df.sample(n=N_MAX, random_state=0).reset_index(drop=True)
    return df


def main() -> int:
    apply_style()
    df = _load()
    kin = compute_actions(df)
    out = (df.set_index("source_id")
              .loc[kin["source_id"].to_numpy()]
              .reset_index())

    vR = kin["v_R_kms"].to_numpy()
    vT = kin["v_T_kms"].to_numpy()
    vz = kin["v_z_kms"].to_numpy()
    Lz = kin["L_z_kpc_kms"].to_numpy()
    Jr = kin["J_R_kpc_kms"].to_numpy()
    Jz = kin["J_z_kpc_kms"].to_numpy()
    E = kin["E_kms2"].to_numpy() / 1e5
    J_perp = np.sqrt(np.clip(Jr, 0, None) + np.clip(Jz, 0, None))
    Vperp = np.sqrt(vR ** 2 + vz ** 2)
    Vtl = vT - V_LSR
    mh = out["mh_pred"].to_numpy()
    am = out["alpha_m_pred"].to_numpy()

    # Wider, shorter aspect: the v1.4 talk reviewer flagged the panels as
    # too vertically stretched at 13 x 4.5.
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.0),
                              layout="constrained")
    CMAP = "jet"
    AM_VMIN, AM_VMAX = -0.1, 0.2

    ax = axes[0]
    sc = ax.scatter(Vtl, Vperp, c=am, cmap=CMAP,
                     vmin=AM_VMIN, vmax=AM_VMAX,
                     s=4.0, alpha=0.65, edgecolors="none", rasterized=True)
    colorbar(ax, sc, LABELS["alpha_M"])
    th = np.linspace(0, np.pi, 200)
    for v0 in (50, 100, 150):
        ax.plot(v0 * np.cos(th), v0 * np.sin(th),
                color="#5C6378", lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel(LABELS["VT_VLSR"])
    ax.set_ylabel(LABELS["Vperp"])
    ax.set_xlim(np.nanpercentile(Vtl, 1), np.nanpercentile(Vtl, 99))
    ax.set_ylim(0.0, np.nanpercentile(Vperp, 99))
    ax.set_title(r"Toomre diagram")
    ax.grid(True, alpha=0.20)

    ax = axes[1]
    sc = ax.scatter(Lz, J_perp, c=am, cmap=CMAP,
                     vmin=AM_VMIN, vmax=AM_VMAX,
                     s=4.0, alpha=0.65, edgecolors="none", rasterized=True)
    colorbar(ax, sc, LABELS["alpha_M"])
    ax.set_xlabel(LABELS["Lz"])
    ax.set_ylabel(LABELS["JR_Jz"])
    ax.set_xlim(np.nanpercentile(Lz, 1), np.nanpercentile(Lz, 99))
    ax.set_ylim(0.0, np.nanpercentile(J_perp, 99))
    ax.set_title(r"Action diagram")
    ax.grid(True, alpha=0.20)

    ax = axes[2]
    sc = ax.scatter(Lz, E, c=am, cmap=CMAP,
                     vmin=AM_VMIN, vmax=AM_VMAX,
                     s=4.0, alpha=0.65, edgecolors="none", rasterized=True)
    colorbar(ax, sc, LABELS["alpha_M"])
    ax.set_xlabel(LABELS["Lz"])
    ax.set_ylabel(LABELS["E"])
    ax.set_xlim(np.nanpercentile(Lz, 1), np.nanpercentile(Lz, 99))
    ax.set_ylim(np.nanpercentile(E, 1), np.nanpercentile(E, 99))
    ax.set_title(r"Lindblad diagram")
    ax.grid(True, alpha=0.20)

    save(fig, "F14_stream2_kinematics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
