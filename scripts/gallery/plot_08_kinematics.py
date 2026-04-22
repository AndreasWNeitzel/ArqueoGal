"""Stage 08: kinematics — orbital actions, E-L_z plane, eccentricity.

Outputs:
  - reports/gallery/08_kinematics/e_lz_plane.png
  - reports/gallery/08_kinematics/action_diagram.png
  - reports/gallery/08_kinematics/ecc_lz.png
  - reports/gallery/08_kinematics/orbit_families_fraction.png

Reads from data/processed/pipeline2_kinematics_stream3_volume.parquet
(galpy/agama-integrated in MWPotential14).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DATA_PROCESSED,
    GALLERY,
    apply_style,
    save_fig,
)

OUT = GALLERY / "08_kinematics"


def _have(schema, name: str) -> bool:
    return name in {f.name for f in schema}


def _load() -> "pd.DataFrame":
    import pandas as pd  # noqa: F401
    # Minimal kinematics summary (ecc, z_max) + labels (mh, alpha_m)
    path = DATA_PROCESSED / "pipeline2_kinematics_stream3_volume.parquet"
    df = pq.read_table(path).to_pandas()

    # Try to merge the 8-D downstream feature matrix for actions + energy + L_z.
    # Legacy filename prefix retained for consumer stability; active definition
    # now lives in Starfold (separate repo).
    for candidate in ("pipeline2_features_stream3_volume_v11.parquet",
                      "pipeline2_features_stream3_volume_v12.parquet",
                      "pipeline2_features_stream3_volume.parquet"):
        p = DATA_PROCESSED / candidate
        if not p.exists():
            continue
        fs = pq.read_schema(p)
        want = ["source_id", "J_R", "J_z", "L_z", "E", "ecc", "z_max",
                "jr", "jz", "lz", "energy", "eccentricity",
                "m_h", "mh", "alpha_m", "mg_m"]
        fcols = [c for c in want if _have(fs, c)]
        fdf = pq.read_table(p, columns=fcols).to_pandas()
        fdf = fdf.rename(columns={"jr": "J_R", "jz": "J_z", "lz": "L_z",
                                  "energy": "E", "eccentricity": "ecc",
                                  "m_h": "mh"})
        # drop any columns already in df except source_id
        overlap = [c for c in fdf.columns if c in df.columns and c != "source_id"]
        fdf = fdf.drop(columns=overlap)
        df = df.merge(fdf, on="source_id", how="left")
        break

    return df


def e_lz_plane() -> None:
    df = _load()
    if "L_z" not in df.columns or "E" not in df.columns:
        print(f"[gallery] skipping e_lz_plane — missing L_z or E columns; have {list(df.columns)}")
        return
    lz = df["L_z"].to_numpy(dtype=float)
    e = df["E"].to_numpy(dtype=float)
    m = np.isfinite(lz) & np.isfinite(e)
    fig, ax = plt.subplots(figsize=(9, 6))
    hb = ax.hexbin(lz[m], e[m], gridsize=100, cmap="magma", bins="log", mincnt=1)
    plt.colorbar(hb, ax=ax, shrink=0.9, pad=0.02, label="log N")
    ax.set_xlabel(r"$L_z$ [kpc km s$^{-1}$]")
    ax.set_ylabel(r"$E$ [km$^2$ s$^{-2}$]")
    ax.set_title(rf"$E$-$L_z$ plane (Stream 3 volume-limited, n={int(m.sum()):,}) "
                 r"— Gaia-Enceladus Sausage lobe expected at $L_z\approx 0$",
                 fontsize=11, fontweight="semibold")
    ax.axvline(0, color="#fff", lw=0.6, alpha=0.6, ls="--")
    save_fig(fig, OUT / "e_lz_plane.png")


def action_diagram() -> None:
    df = _load()
    if not {"J_R", "J_z"}.issubset(df.columns):
        print(f"[gallery] skipping action_diagram — missing J_R/J_z; have {list(df.columns)}")
        return
    jr = df["J_R"].to_numpy(dtype=float)
    jz = df["J_z"].to_numpy(dtype=float)
    m = np.isfinite(jr) & np.isfinite(jz) & (jr > 0) & (jz > 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(jr[m], jz[m], xscale="log", yscale="log", gridsize=90, cmap="viridis",
                   bins="log", mincnt=1)
    plt.colorbar(hb, ax=ax, shrink=0.9, pad=0.02, label="log N")
    ax.set_xlabel(r"$J_R$ [kpc km s$^{-1}$]")
    ax.set_ylabel(r"$J_z$ [kpc km s$^{-1}$]")
    ax.set_title(rf"Action diagram  (n={int(m.sum()):,})  —  thin disc lower-left, halo / thick disc upper-right",
                 fontsize=11, fontweight="semibold")
    # guide line
    xs = np.logspace(0, 4, 50)
    ax.plot(xs, xs, "w--", lw=0.5, alpha=0.5)
    save_fig(fig, OUT / "action_diagram.png")


def ecc_lz() -> None:
    df = _load()
    if not {"ecc", "L_z"}.issubset(df.columns):
        print(f"[gallery] skipping ecc_lz — missing ecc/L_z; have {list(df.columns)}")
        return
    ecc = df["ecc"].to_numpy(dtype=float)
    lz = df["L_z"].to_numpy(dtype=float)
    m = np.isfinite(ecc) & np.isfinite(lz)
    fig, ax = plt.subplots(figsize=(10, 6))
    hb = ax.hexbin(lz[m], ecc[m], gridsize=100, cmap="magma", bins="log", mincnt=1)
    plt.colorbar(hb, ax=ax, shrink=0.9, pad=0.02, label="log N")
    ax.axvline(0, color="#fff", lw=0.5, ls="--", alpha=0.6)
    ax.axhline(0.7, color="#fff", lw=0.5, ls=":", alpha=0.6)
    ax.set_xlabel(r"$L_z$ [kpc km s$^{-1}$]")
    ax.set_ylabel(r"eccentricity $\varepsilon$")
    ax.set_ylim(0, 1)
    ax.set_title(rf"Eccentricity vs $L_z$  (n={int(m.sum()):,})  —  high-$\varepsilon$ retrograde peak flags accreted candidates",
                 fontsize=11, fontweight="semibold")
    save_fig(fig, OUT / "ecc_lz.png")


def orbit_families_fraction() -> None:
    df = _load()
    if not {"ecc", "L_z"}.issubset(df.columns) or "mh" not in df.columns:
        print(f"[gallery] skipping orbit_families_fraction — missing ecc/L_z/mh; have {list(df.columns)}")
        return
    ecc = df["ecc"].to_numpy(dtype=float)
    lz = df["L_z"].to_numpy(dtype=float)
    mh = df["mh"].to_numpy(dtype=float)
    m = np.isfinite(ecc) & np.isfinite(lz) & np.isfinite(mh)
    ecc, lz, mh = ecc[m], lz[m], mh[m]

    # family assignment
    prograde_disc = (lz > 500) & (ecc < 0.4)
    retrograde    = (lz < 0) & (ecc > 0.5)
    halo          = (np.abs(lz) < 500) & (ecc > 0.6)
    thick_disc    = (lz > 500) & (ecc >= 0.4) & (ecc < 0.6)
    other         = ~(prograde_disc | retrograde | halo | thick_disc)

    bins = np.linspace(-2.0, 0.5, 11)
    centres = 0.5 * (bins[:-1] + bins[1:])
    fam = {
        "prograde disc (ε<0.4, L_z>500)": (prograde_disc, "#1f77b4"),
        "thick disc (0.4≤ε<0.6, L_z>500)": (thick_disc,   "#2ca02c"),
        "halo (ε>0.6, |L_z|<500)":        (halo,         "#d62728"),
        "retrograde (ε>0.5, L_z<0)":      (retrograde,   "#9467bd"),
        "other":                          (other,        "#7f7f7f"),
    }

    fractions = {}
    total, _ = np.histogram(mh, bins=bins)
    for lbl, (mask, _c) in fam.items():
        sub, _ = np.histogram(mh[mask], bins=bins)
        fractions[lbl] = np.where(total > 0, sub / np.maximum(total, 1), np.nan)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros_like(centres)
    for lbl, (_mask, color) in fam.items():
        ax.bar(centres, fractions[lbl], width=(bins[1] - bins[0]) * 0.9,
               bottom=bottom, color=color, edgecolor="#333", alpha=0.9, label=lbl)
        bottom = bottom + np.nan_to_num(fractions[lbl])
    ax.set_xlabel(r"$[{\rm M}/{\rm H}]$ bin")
    ax.set_ylabel("population fraction")
    ax.set_title("Orbit-family fractions vs metallicity (Stream 3 volume-limited)",
                 fontsize=12, fontweight="semibold")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=9)
    save_fig(fig, OUT / "orbit_families_fraction.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    e_lz_plane()
    action_diagram()
    ecc_lz()
    orbit_families_fraction()


if __name__ == "__main__":
    main()
