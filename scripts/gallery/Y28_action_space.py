"""Y28: Action space, J_R vs L_z and J_z vs L_z, coloured by chemistry.

The Staeckel-fudge actions (galpy / McMillan+2017) are the cleanest
disc/halo separators because they are integrals of motion. Stars on
similar orbits cluster in action space even when they appear scattered in
configuration space.

Two panels:

  (top)    J_R vs L_z , radial action vs angular momentum
  (bottom) J_z vs L_z , vertical action vs angular momentum

Coloured by [α/M] in both. Hot (high-J_R, high-J_z) populations are the
thick disc and accreted halo; the thin disc clusters at low J_R, low J_z,
L_z ≈ +1700 kpc km/s.
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


def _panel(ax, x, y, c, *, x_extent, y_extent, x_lab, y_lab, c_lab, title):
    ok = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(c)
        & (x > x_extent[0])
        & (x < x_extent[1])
        & (y > y_extent[0])
        & (y < y_extent[1])
    )
    hb = ax.hexbin(
        x[ok],
        y[ok],
        C=c[ok],
        reduce_C_function=np.median,
        gridsize=80,
        extent=(*x_extent, *y_extent),
        mincnt=4,
        vmin=-0.05,
        vmax=0.32,
        cmap="viridis",
        edgecolors="none",
    )
    ax.set_xlim(x_extent)
    ax.set_ylim(y_extent)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.set_title(title, color=PALETTE["navy"])
    cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(c_lab, fontsize=10)


def main() -> int:
    apply_style()
    df = load_kin_chem()
    n = len(df)

    lz = df["L_z_kpc_kms"].to_numpy() / 1000.0
    jr = df["J_R_kpc_kms"].to_numpy() / 100.0
    jz = df["J_z_kpc_kms"].to_numpy() / 100.0
    am = df["alpha_m_pred"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    plt.subplots_adjust(wspace=0.30, top=0.84, bottom=0.10, left=0.06, right=0.97)
    _panel(
        axes[0],
        lz,
        jr,
        am,
        x_extent=(-3.0, 3.5),
        y_extent=(0.0, 8.0),
        x_lab=r"$L_z$  ($10^{3}$ kpc km s$^{-1}$)",
        y_lab=r"$J_R$  ($10^{2}$ kpc km s$^{-1}$)",
        c_lab=r"median [$\alpha$/M] (dex)",
        title=r"$J_R$ vs $L_z$ ,  radial action",
    )
    _panel(
        axes[1],
        lz,
        jz,
        am,
        x_extent=(-3.0, 3.5),
        y_extent=(0.0, 5.0),
        x_lab=r"$L_z$  ($10^{3}$ kpc km s$^{-1}$)",
        y_lab=r"$J_z$  ($10^{2}$ kpc km s$^{-1}$)",
        c_lab=r"median [$\alpha$/M] (dex)",
        title=r"$J_z$ vs $L_z$ ,  vertical action",
    )

    headline(
        fig,
        "Action space, integrals of motion versus chemistry",
        f"Stream 3 Tier 1, n = {n:,}.  Cool dynamic disc bottom-right; "
        "hot/halo populations top-left.  Colour reads off the chemo-dynamic decoupling.",
        top=0.84,
    )
    save(fig, "Y28_action_space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
