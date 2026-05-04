"""F1: Kinematic diagnostics from real galpy outputs.

Six panels on the Stream-3 cohort (the only stream with the full
McMillan+2017 Staeckel-fudge action set computed end-to-end via
``arqueogal.data.kinematics``, parquet ``data/processed/
pipeline2_kinematics_stream3_volume.parquet``):

  (a) PMRA vs PMDEC          — Gaia DR3 proper motions
  (b) Toomre diagram         — V_perp = sqrt(V_R² + V_z²) vs (V_φ − V_LSR)
  (c) E vs L_z               — Lindblad / energy-angular-momentum plane
  (d) J_R vs L_z             — radial action vs angular momentum
  (e) J_z vs L_z             — vertical action vs angular momentum
  (f) eccentricity histogram — orbital ecc distribution
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig  # noqa: E402

KIN = REPO / "data/processed/pipeline2_kinematics_stream3_volume.parquet"
FEAT = REPO / "data/processed/pipeline1_features_stream3.parquet"

V_LSR_KMS = 233.1
N_TARGET = 200_000


def _load_pm() -> pd.DataFrame | None:
    """The Stream-3 kinematics cohort sources its astrometry from the
    *delta* enrichment parquet (449k rows, full overlap with the 249k
    kinematics subsample). The non-delta enrichment misses these
    source_ids entirely. Verified empirically.
    """
    delta = REPO / "data/interim/stream3_delta_gaia_dr3_corrected.parquet"
    if delta.exists():
        return pd.read_parquet(delta, columns=["source_id", "pmra", "pmdec"])
    return None


def main() -> int:
    apply_style()
    if not KIN.exists():
        raise SystemExit(f"missing {KIN}")
    kin = pd.read_parquet(KIN)
    pm = _load_pm()
    have_pm = pm is not None
    if have_pm:
        kin = kin.merge(pm, on="source_id", how="left")
    kin = kin.drop_duplicates("source_id").reset_index(drop=True)
    if len(kin) > N_TARGET:
        kin = kin.sample(N_TARGET, random_state=0).reset_index(drop=True)
    n = len(kin)
    print(f"[F1] cohort: {n:,} Stream-3 stars; pm available = {have_pm}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    plt.subplots_adjust(top=0.92, hspace=0.32, wspace=0.30,
                        left=0.05, right=0.97, bottom=0.07)

    # (a) PMRA vs PMDEC.
    ax = axes[0, 0]
    if have_pm:
        x = kin["pmra"].to_numpy()
        y = kin["pmdec"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        hb = ax.hexbin(x[ok], y[ok], gridsize=80,
                       extent=(-30, 30, -30, 30),
                       mincnt=1, bins="log", cmap="viridis")
        ax.set_xlim(-30, 30); ax.set_ylim(-30, 30); ax.set_aspect("equal")
        plt.colorbar(hb, ax=ax, label=r"$\log_{10}$ N",
                     fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "pmra/pmdec unavailable",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
    ax.axhline(0, color="k", lw=0.5, ls=":", alpha=0.4)
    ax.axvline(0, color="k", lw=0.5, ls=":", alpha=0.4)
    ax.set_xlabel(r"$\mu_{\alpha^\ast}$ (mas yr$^{-1}$)")
    ax.set_ylabel(r"$\mu_\delta$ (mas yr$^{-1}$)")
    ax.set_title("(a) Gaia DR3 proper motions")
    ax.grid(True, alpha=0.25)

    # (b) Toomre diagram.
    ax = axes[0, 1]
    v_perp = np.sqrt(kin["v_R_kms"].to_numpy() ** 2
                      + kin["v_z_kms"].to_numpy() ** 2)
    v_phi = kin["v_T_kms"].to_numpy() - V_LSR_KMS
    ok = np.isfinite(v_perp) & np.isfinite(v_phi)
    hb = ax.hexbin(v_phi[ok], v_perp[ok], gridsize=80,
                   extent=(-400, 200, 0, 400),
                   mincnt=1, bins="log", cmap="viridis")
    plt.colorbar(hb, ax=ax, label=r"$\log_{10}$ N",
                 fraction=0.046, pad=0.04)
    theta = np.linspace(0, np.pi, 100)
    ax.plot(220 * np.cos(theta), 220 * np.sin(theta),
            color="#e07b00", lw=1.6, ls="--",
            label=r"$|V - V_{\rm LSR}| = 220$ km s$^{-1}$ (Bonaca+2017)")
    ax.set_xlim(-400, 200); ax.set_ylim(0, 400)
    ax.set_xlabel(r"$V_\phi - V_{\rm LSR}$ (km s$^{-1}$)")
    ax.set_ylabel(r"$V_\perp = \sqrt{V_R^2 + V_z^2}$ (km s$^{-1}$)")
    ax.set_title("(b) Toomre diagram")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)

    # (c) E vs L_z.
    ax = axes[0, 2]
    lz = kin["L_z_kpc_kms"].to_numpy() / 1000.0
    E = kin["E_kms2"].to_numpy() / 1e5
    ok = np.isfinite(lz) & np.isfinite(E) & (E > -2.5) & (E < 0)
    hb = ax.hexbin(lz[ok], E[ok], gridsize=80,
                   extent=(-3, 3.5, -2.0, -0.6),
                   mincnt=1, bins="log", cmap="viridis")
    plt.colorbar(hb, ax=ax, label=r"$\log_{10}$ N",
                 fraction=0.046, pad=0.04)
    ax.axvline(1.7, color="white", lw=1.0, ls=":")
    ax.text(1.72, -0.7, "  solar circle", color="0.4", fontsize=9,
            rotation=90, va="top", fontweight="bold")
    ax.set_xlim(-3, 3.5); ax.set_ylim(-2.0, -0.6)
    ax.set_xlabel(r"$L_z$  ($10^3$ kpc km s$^{-1}$)")
    ax.set_ylabel(r"$E$  ($10^5$ km$^2$ s$^{-2}$)")
    ax.set_title("(c) Lindblad: energy vs angular momentum")
    ax.grid(True, alpha=0.25)

    # (d) J_R vs L_z.
    ax = axes[1, 0]
    jr = kin["J_R_kpc_kms"].to_numpy() / 100.0
    ok = np.isfinite(lz) & np.isfinite(jr) & (jr < 10)
    hb = ax.hexbin(lz[ok], jr[ok], gridsize=80,
                   extent=(-3, 3.5, 0, 8),
                   mincnt=1, bins="log", cmap="viridis")
    plt.colorbar(hb, ax=ax, label=r"$\log_{10}$ N",
                 fraction=0.046, pad=0.04)
    ax.set_xlim(-3, 3.5); ax.set_ylim(0, 8)
    ax.set_xlabel(r"$L_z$  ($10^3$ kpc km s$^{-1}$)")
    ax.set_ylabel(r"$J_R$  ($10^2$ kpc km s$^{-1}$)")
    ax.set_title("(d) Radial action vs angular momentum")
    ax.grid(True, alpha=0.25)

    # (e) J_z vs L_z.
    ax = axes[1, 1]
    jz = kin["J_z_kpc_kms"].to_numpy() / 100.0
    ok = np.isfinite(lz) & np.isfinite(jz) & (jz < 10)
    hb = ax.hexbin(lz[ok], jz[ok], gridsize=80,
                   extent=(-3, 3.5, 0, 5),
                   mincnt=1, bins="log", cmap="viridis")
    plt.colorbar(hb, ax=ax, label=r"$\log_{10}$ N",
                 fraction=0.046, pad=0.04)
    ax.set_xlim(-3, 3.5); ax.set_ylim(0, 5)
    ax.set_xlabel(r"$L_z$  ($10^3$ kpc km s$^{-1}$)")
    ax.set_ylabel(r"$J_z$  ($10^2$ kpc km s$^{-1}$)")
    ax.set_title("(e) Vertical action vs angular momentum")
    ax.grid(True, alpha=0.25)

    # (f) Eccentricity histogram.
    ax = axes[1, 2]
    ecc = kin["ecc"].to_numpy()
    ecc = ecc[np.isfinite(ecc)]
    ax.hist(ecc, bins=80, range=(0, 1), color="#1f77b4",
            edgecolor="white", linewidth=0.4, alpha=0.85)
    med = float(np.nanmedian(ecc))
    ax.axvline(med, color="#e07b00", lw=2.0, ls="--",
               label=f"median ecc = {med:.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("orbital eccentricity")
    ax.set_ylabel("count")
    ax.set_title("(f) Eccentricity distribution")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "F1. Stream-3 kinematics (galpy McMillan+2017 + Staeckel fudge; "
        f"n = {n:,})",
        fontsize=12, fontweight="semibold", y=0.985,
    )
    save_fig(fig, REPO / "reports/gallery/F_kinematics/F1_kinematics", tight=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
