"""Stage 10: kinematic enrichment (galpy actions, Toomre, eccentricity).

What the deploy did: ``data.enrich_kinematics`` integrates each star's
6-D phase-space initial condition (Gaia astrometry + APOGEE/Gaia RV) under
MWPotential2014 and emits actions (J_R, L_z, J_z), Galactocentric
cylindrical velocities (v_R, v_T = v_φ, v_z), eccentricity, energy E.

What we plot, 2×2:
- (0,0) **Toomre diagram**: v_T vs sqrt(v_R² + v_z²) — separates thin disc
  (v_T near v_LSR ≈ 230 km/s, low ⊥), thick disc (intermediate), halo
  (high ⊥, v_T near 0 or retrograde). Halo selection is conventionally
  |V − V_LSR| > 200 km/s.
- (0,1) **E–L_z plane**: bound population (E<0) only.
- (1,0) **Action plane**: sqrt(2 J_R) (combines non-angular momenta in
  quadrature; equivalent to a radial-action proxy) vs L_z, disc-like
  cut J_R < 10⁴ kpc km/s.
- (1,1) **Eccentricity distribution**: thin (e<0.3), thick (0.3–0.6),
  halo-like (e>0.6) regimes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/10_kinematics"
KIN = REPO / "data/processed/pipeline2_kinematics_stream3_volume.parquet"

V_LSR = 230.0  # km/s, Galactic rotation at the solar circle


def main() -> None:
    apply_style()
    if not KIN.exists():
        return
    df = pd.read_parquet(
        KIN,
        columns=[
            "E_kms2",
            "L_z_kpc_kms",
            "J_R_kpc_kms",
            "J_z_kpc_kms",
            "ecc",
            "v_R_kms",
            "v_T_kms",
            "v_z_kms",
        ],
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    Lz = df["L_z_kpc_kms"].to_numpy()
    E = df["E_kms2"].to_numpy()
    JR = df["J_R_kpc_kms"].to_numpy()
    Jz = df["J_z_kpc_kms"].to_numpy()
    ecc = df["ecc"].to_numpy()
    v_R = df["v_R_kms"].to_numpy()
    v_T = df["v_T_kms"].to_numpy()
    v_z = df["v_z_kms"].to_numpy()

    # (0,0) Toomre diagram in Galactocentric cylindrical velocities.
    v_perp = np.sqrt(v_R**2 + v_z**2)
    m = np.isfinite(v_T) & np.isfinite(v_perp)
    if m.sum() > 0:
        # Tight extent on the disc population: 0 < v_T < 400 km/s,
        # 0 < v_perp < 200 km/s — captures the bulk of Stream-3 thin/thick
        # disc stars; halo tail extends beyond and is intentionally clipped.
        x_lo, x_hi = 0.0, 400.0
        y_lo, y_hi = 0.0, 200.0
        h = axes[0, 0].hexbin(
            v_T[m],
            v_perp[m],
            gridsize=80,
            mincnt=10,
            cmap="viridis",
            bins="log",
            extent=[x_lo, x_hi, y_lo, y_hi],
        )
        plt.colorbar(h, ax=axes[0, 0], label="log10 N")
        # Halo isovelocity arcs at |V − V_LSR| = 100, 200 km/s — only the
        # portion within the new extent is drawn.
        th = np.linspace(0, 2 * np.pi, 400)
        for r_iso, color, lab in [
            (100, "#9467bd", "100 km/s from LSR"),
            (200, "#d62728", "200 km/s (halo cut)"),
        ]:
            cx = V_LSR + r_iso * np.cos(th)
            cy = r_iso * np.sin(th)
            in_box = (cx >= x_lo) & (cx <= x_hi) & (cy >= y_lo) & (cy <= y_hi)
            if in_box.any():
                axes[0, 0].plot(
                    cx[in_box], cy[in_box], color=color, lw=0.9, ls="--", alpha=0.85, label=lab
                )
        axes[0, 0].axvline(
            V_LSR, color="orange", lw=0.8, ls=":", label=f"$V_{{\\rm LSR}}$ = {V_LSR:.0f} km/s"
        )
        axes[0, 0].set_xlim(x_lo, x_hi)
        axes[0, 0].set_ylim(y_lo, y_hi)
        axes[0, 0].set_xlabel(r"$v_T$ (km/s)")
        axes[0, 0].set_ylabel(r"$\sqrt{v_R^2+v_z^2}$ (km/s)")
        # Fraction within view for honest reporting
        in_view = m & (v_T >= x_lo) & (v_T <= x_hi) & (v_perp <= y_hi)
        axes[0, 0].set_title(
            f"Toomre diagram (n={int(in_view.sum()):,} in view of {int(m.sum()):,} total)"
        )
        axes[0, 0].legend(
            fontsize=8,
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.4",
        )

    # (0,1) E vs L_z. Restrict to bound population.
    m = np.isfinite(Lz) & np.isfinite(E) & (E < 0)
    if m.sum() > 0:
        Lz_p1, Lz_p99 = np.percentile(Lz[m], [0.5, 99.5])
        E_p1, E_p99 = np.percentile(E[m], [0.5, 99.5])
        h = axes[0, 1].hexbin(
            Lz[m],
            E[m] / 1e5,
            gridsize=80,
            mincnt=10,
            cmap="viridis",
            bins="log",
            extent=[Lz_p1, Lz_p99, E_p1 / 1e5, E_p99 / 1e5],
        )
        plt.colorbar(h, ax=axes[0, 1], label="log10 N")
        axes[0, 1].set_xlabel(r"$L_z$ (kpc km/s)")
        axes[0, 1].set_ylabel(r"$E\;(10^5\,\mathrm{km^2/s^2})$")
        axes[0, 1].set_title(f"E–$L_z$ bound population (n={int(m.sum()):,})")
        axes[0, 1].axvline(0, color="0.6", lw=0.5, ls="--")

    # (1,0) Action plane: sqrt(2 J_R) vs L_z (radial-action proxy; combines
    # the two non-angular components in quadrature when v_R and v_z couple).
    # We use sqrt(2 J_R) for the radial-action axis; J_z is also non-angular
    # so we form the quadrature sqrt(2 J_R + 2 J_z) — this combined axis is
    # the action equivalent of v_perp on the Toomre.
    radial_action = np.sqrt(2.0 * (JR + Jz))
    m = np.isfinite(Lz) & np.isfinite(radial_action) & (JR < 1e4) & (Jz < 1e4)
    if m.sum() > 0:
        h = axes[1, 0].hexbin(
            Lz[m],
            radial_action[m],
            gridsize=80,
            mincnt=10,
            cmap="viridis",
            bins="log",
            extent=[
                np.percentile(Lz[m], 0.5),
                np.percentile(Lz[m], 99.5),
                0,
                np.percentile(radial_action[m], 99.5),
            ],
        )
        plt.colorbar(h, ax=axes[1, 0], label="log10 N")
        axes[1, 0].set_xlabel(r"$L_z$ (kpc km/s)")
        axes[1, 0].set_ylabel(r"$\sqrt{2(J_R+J_z)}$ (kpc km/s)$^{1/2}$")
        axes[1, 0].set_title(f"Action plane: angular vs combined non-angular (n={int(m.sum()):,})")
        axes[1, 0].axvline(0, color="0.6", lw=0.5, ls="--")

    # (1,1) Eccentricity histogram with regime labels.
    ecc_finite = ecc[np.isfinite(ecc)]
    axes[1, 1].hist(ecc_finite, bins=np.linspace(0, 1, 41), color="#1f77b4", alpha=0.75)
    axes[1, 1].axvline(0.3, color="#9467bd", lw=0.9, ls="--", label="thin/thick (e=0.3)")
    axes[1, 1].axvline(0.6, color="#d62728", lw=0.9, ls="--", label="halo-like (e=0.6)")
    axes[1, 1].set_xlabel("eccentricity")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_title(f"Stream-3 eccentricity (n={len(ecc_finite):,})")
    axes[1, 1].legend(
        fontsize=8,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
    )

    fig.suptitle(
        "Kinematic enrichment (galpy MWPotential2014): Toomre + actions + ecc", fontsize=11
    )
    save_fig(fig, OUT / "kinematics.png")


if __name__ == "__main__":
    main()
