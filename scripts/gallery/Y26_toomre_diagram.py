"""Y26: Toomre diagram of Stream-3 Tier-1 stars, coloured by chemistry.

Toomre velocity: V_perp = sqrt(V_R^2 + V_z^2) plotted against (V_T - V_LSR).
Two panels, same scatter, two colour layers:

  (left)   coloured by [α/M]_pred
  (right)  coloured by [M/H]_pred

Stars to the left of (V_T - V_LSR ≈ 0) and high V_perp are on retrograde or
high-eccentricity orbits, the canonical halo / accreted-debris locus.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402
from _y_kinematics import V_LSR_KMS, load_kin_chem  # noqa: E402


def _panel(ax, df, color_col, *, label, cmap, vlim, title):
    v_perp = np.sqrt(df["v_R_kms"].to_numpy() ** 2 + df["v_z_kms"].to_numpy() ** 2)
    v_phi = df["v_T_kms"].to_numpy() - V_LSR_KMS
    c = df[color_col].to_numpy()
    ok = np.isfinite(v_perp) & np.isfinite(v_phi) & np.isfinite(c)

    hb = ax.hexbin(
        v_phi[ok],
        v_perp[ok],
        C=c[ok],
        reduce_C_function=np.median,
        gridsize=80,
        extent=(-400, 200, 0, 400),
        mincnt=4,
        vmin=vlim[0],
        vmax=vlim[1],
        cmap=cmap,
        edgecolors="none",
    )
    # Halo demarcation circle (Bonaca+2017): |V - V_LSR| > 220 km/s.
    halo = mpatches.Circle(
        (0, 0),
        220,
        fill=False,
        lw=2.0,
        edgecolor=PALETTE["accent"],
        ls="--",
        label="thin/thick disc → halo (Bonaca+2017)",
    )
    ax.add_patch(halo)
    ax.set_xlim(-400, 200)
    ax.set_ylim(0, 400)
    ax.set_xlabel(r"$V_\phi - V_{\rm LSR}$  (km s$^{-1}$)")
    ax.set_ylabel(r"$V_\perp = \sqrt{V_R^{2} + V_z^{2}}$  (km s$^{-1}$)")
    ax.set_title(title, color=PALETTE["navy"])
    ax.legend(loc="upper left")
    cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=10)


def main() -> int:
    apply_style()
    df = load_kin_chem()
    n = len(df)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    plt.subplots_adjust(wspace=0.30, top=0.86, bottom=0.10, left=0.06, right=0.97)
    _panel(
        axes[0],
        df,
        "alpha_m_pred",
        label=r"median [$\alpha$/M] (dex)",
        cmap="viridis",
        vlim=(-0.05, 0.32),
        title=r"Toomre, coloured by [$\alpha$/M]",
    )
    _panel(
        axes[1],
        df,
        "mh_pred",
        label="median [M/H] (dex)",
        cmap="viridis",
        vlim=(-1.4, 0.30),
        title="Toomre, coloured by [M/H]",
    )

    headline(
        fig,
        "Toomre diagram, chemistry across the disc/halo transition",
        f"Stream 3 Tier 1, n = {n:,}.  Halo locus outside the dashed circle "
        "(|V - V_LSR| > 220 km/s).  High-α giants concentrate in the slow-rotating tail.",
        top=0.86,
    )
    save(fig, "Y26_toomre_diagram")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
