"""Y29: Disc geometry, Z vs R_gal split into chemistry tertiles.

Three columns, one row per chemistry quantity:

  Row 1: Z vs R_gal hexbin median [M/H]
  Row 2: Z vs R_gal hexbin median [α/M]
  Row 3: Z height distribution split by [α/M] tertile (low / mid / high)

Shows the canonical thin/thick disc separation in space: high-α stars are
vertically extended (thick disc), low-α stars sit close to the plane
(thin disc).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402
from _y_kinematics import load_kin_chem  # noqa: E402

R_EXTENT = (4.0, 14.0)
Z_EXTENT = (-4.0, 4.0)


def _hex(ax, R, Z, c, *, label, vmin, vmax, title):
    ok = (np.isfinite(R) & np.isfinite(Z) & np.isfinite(c)
          & (R_EXTENT[0] < R) & (R_EXTENT[1] > R)
          & (Z_EXTENT[0] < Z) & (Z_EXTENT[1] > Z))
    hb = ax.hexbin(
        R[ok], Z[ok], C=c[ok], reduce_C_function=np.median,
        gridsize=80, extent=(*R_EXTENT, *Z_EXTENT),
        mincnt=4, vmin=vmin, vmax=vmax, cmap="viridis", edgecolors="none")
    ax.scatter([8.122], [0.0208], marker="*", s=180, color="white",
               edgecolor=PALETTE["ink"], linewidth=1.6, zorder=5)
    ax.set_xlim(R_EXTENT)
    ax.set_ylim(Z_EXTENT)
    ax.set_xlabel(r"$R_{\rm gal}$  (kpc)")
    ax.set_ylabel(r"$Z$  (kpc)")
    ax.set_title(title, color=PALETTE["navy"])
    cb = plt.colorbar(hb, ax=ax, fraction=0.038, pad=0.02)
    cb.set_label(label, fontsize=10)


def _z_hist_panel(ax, df):
    am = df["alpha_m_pred"].to_numpy()
    z = df["z_galcen_kpc"].to_numpy()
    ok = np.isfinite(am) & np.isfinite(z)
    am, z = am[ok], z[ok]
    q33, q66 = np.quantile(am, [0.33, 0.66])

    bands = [
        (am < q33,                   PALETTE["navy_light"],
         rf"low α  ([α/M] < {q33:+.2f})"),
        ((am >= q33) & (am < q66),  PALETTE["accent"],
         rf"mid α  ({q33:+.2f} ≤ [α/M] < {q66:+.2f})"),
        (am >= q66,                  PALETTE["tier3"],
         rf"high α  ([α/M] ≥ {q66:+.2f})"),
    ]
    for mask, color, lab in bands:
        ax.hist(z[mask], bins=80, range=(-4, 4),
                color=color, alpha=0.55, density=True,
                label=lab, histtype="stepfilled",
                edgecolor=color, linewidth=0.6)
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"$Z$  (kpc)")
    ax.set_ylabel("density")
    ax.set_title(r"$Z$-distribution, split by [α/M] tertile",
                 color=PALETTE["navy"])
    ax.legend(loc="upper right", fontsize=10)


def main() -> int:
    apply_style()
    df = load_kin_chem()
    n = len(df)

    R = df["R_galcen_kpc"].to_numpy()
    Z = df["z_galcen_kpc"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(22, 8.0))
    plt.subplots_adjust(wspace=0.32, top=0.84, bottom=0.12,
                        left=0.05, right=0.97)
    _hex(axes[0], R, Z, df["mh_pred"].to_numpy(),
         label="median [M/H] (dex)", vmin=-1.0, vmax=0.30,
         title="[M/H] in $(R_{\\rm gal}, Z)$")
    _hex(axes[1], R, Z, df["alpha_m_pred"].to_numpy(),
         label=r"median [$\alpha$/M] (dex)", vmin=-0.05, vmax=0.32,
         title=r"[$\alpha$/M] in $(R_{\rm gal}, Z)$")
    _z_hist_panel(axes[2], df)

    headline(
        fig,
        r"Disc geometry, vertical structure by chemistry",
        f"Stream 3 Tier 1, n = {n:,}.  High-α giants extend vertically (thick disc); "
        "low-α giants concentrate near the plane (thin disc).  "
        "Star symbol marks Sun position.",
        top=0.84)
    save(fig, "Y29_disc_geometry_rz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
