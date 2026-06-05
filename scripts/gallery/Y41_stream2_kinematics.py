"""Y41: Stream-2 Tier-1 holdout kinematics, three scatter panels.

  - left:   Toomre,             V_perp = sqrt(V_R^2 + V_z^2) vs (V_T - V_LSR),
                                 each star coloured by [M/H].
  - middle: Action diagram,     sqrt(J_R + J_z) vs L_z, coloured by [M/H].
  - right:  Energy vs Lz,       E vs L_z, coloured by [M/H].

Galactocentric kinematics are computed on the fly with the project-canonical
McMillan+2017 potential via ``arqueogal.data.kinematics.compute_actions``.
For runtime, we subsample the Tier-1 cohort to at most 8k stars with finite
RV + parallax + photogeometric distance. Axis ranges are derived from the
data's 1-99 percentiles so the bulk of the cohort fills the panel.

Slide-friendly 18:6 layout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _presentation import OKABE_ITO, PALETTE, apply_style, headline, save  # noqa: E402

from arqueogal.data.kinematics import compute_actions  # noqa: E402
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED_S2 = REPO / "data/processed/pipeline1_predictions_stream2.parquet"
FEAT_S2 = REPO / "data/processed/pipeline1_features_stream2.parquet"
GAIA_RAW_S2 = REPO / "data/interim/stream2_gaia_dr3_raw.parquet"

V_LSR_KMS = 233.1   # McMillan+2017 native circular velocity at R_0
N_STARS_MAX = 8000
RNG_SEED = 0

_TITLE_KW = dict(fontsize=11, fontweight="normal", color=PALETTE["ink"], pad=6)


def _load_tier1_with_kin_inputs() -> pd.DataFrame:
    """Stream-2 Tier-1 cohort with the columns compute_actions needs."""
    pcols = [
        "source_id", "teff_pred", "logg_pred", "mh_pred",
        "alpha_m_pred", "mg_h_pred",
        "ood_joint_flag", "label_extrapolation_flag",
    ]
    p = pd.read_parquet(PRED_S2, columns=pcols).drop_duplicates("source_id")
    p["release_tier"] = assign_release_tier(p).astype(np.int8)
    p = p.loc[p["release_tier"] == 1].reset_index(drop=True)

    feat = pd.read_parquet(
        FEAT_S2, columns=["source_id", "ra_deg", "dec_deg", "r_med_photogeo"]
    ).drop_duplicates("source_id")

    raw = pd.read_parquet(
        GAIA_RAW_S2,
        columns=["source_id", "pmra", "pmdec", "radial_velocity"],
    ).drop_duplicates("source_id")

    df = p.merge(feat, on="source_id", how="inner").merge(raw, on="source_id", how="inner")
    # compute_actions expects column names ra, dec (degrees), distance pc.
    df = df.rename(columns={"ra_deg": "ra", "dec_deg": "dec"})
    needed = ["source_id", "ra", "dec", "r_med_photogeo",
              "pmra", "pmdec", "radial_velocity"]
    df = df.dropna(subset=needed).reset_index(drop=True)
    df = df.loc[df["r_med_photogeo"] > 0].reset_index(drop=True)
    return df


def _scatter(ax, x, y, *, c=None, cmap=None, vmin=None, vmax=None,
             xlabel: str, ylabel: str, title: str, cbar_label: str | None = None,
             color: str | None = None, s: float = 4.0):
    ok = np.isfinite(x) & np.isfinite(y)
    if c is not None:
        ok = ok & np.isfinite(c)
    sc = ax.scatter(
        x[ok], y[ok],
        c=c[ok] if c is not None else None,
        cmap=cmap, vmin=vmin, vmax=vmax,
        color=color if c is None else None,
        s=s, alpha=0.55, edgecolors="none", rasterized=True,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, **_TITLE_KW)
    ax.grid(True, alpha=0.20)
    if c is not None and cbar_label is not None:
        cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(cbar_label, fontsize=10)


def main() -> int:
    apply_style()
    df = _load_tier1_with_kin_inputs()
    if df.empty:
        print("[Y41] no Stream-2 Tier-1 stars with kinematics inputs, aborting")
        return 1

    if len(df) > N_STARS_MAX:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.choice(len(df), size=N_STARS_MAX, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
    print(f"[Y41] computing actions on n = {len(df):,} Stream-2 Tier-1 stars")

    kin = compute_actions(df)
    print(f"[Y41] surviving rows after galpy: {len(kin):,}")
    if kin.empty:
        return 1

    # Pull predicted [M/H] for the surviving sample so we can colour by metallicity.
    mh = df.set_index("source_id").loc[kin["source_id"].to_numpy(), "mh_pred"].to_numpy()
    vR = kin["v_R_kms"].to_numpy()
    vT = kin["v_T_kms"].to_numpy()
    vz = kin["v_z_kms"].to_numpy()
    Lz = kin["L_z_kpc_kms"].to_numpy()
    Jz = kin["J_z_kpc_kms"].to_numpy()
    Jr = kin["J_R_kpc_kms"].to_numpy()
    E = kin["E_kms2"].to_numpy() / 1e5   # report in 1e5 km^2/s^2
    J_perp = np.sqrt(np.clip(Jr, 0.0, None) + np.clip(Jz, 0.0, None))

    v_perp = np.sqrt(vR ** 2 + vz ** 2)
    v_phi_lsr = vT - V_LSR_KMS

    # Common metallicity colour scale across the three panels (1-99 percentile).
    mh_lo = float(np.nanpercentile(mh, 1.0))
    mh_hi = float(np.nanpercentile(mh, 99.0))

    # Data-driven axis ranges (1-99 percentile + small margin).
    def _lim(a: np.ndarray, pad: float = 0.05) -> tuple[float, float]:
        a = a[np.isfinite(a)]
        lo = float(np.nanpercentile(a, 1.0))
        hi = float(np.nanpercentile(a, 99.0))
        span = hi - lo
        return lo - pad * span, hi + pad * span

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Toomre, coloured by [M/H].
    ax = axes[0]
    _scatter(
        ax, v_phi_lsr, v_perp,
        c=mh, cmap="viridis", vmin=mh_lo, vmax=mh_hi, s=4.0,
        xlabel=r"$V_T - V_{\rm LSR}$ (km/s)",
        ylabel=r"$V_\perp = \sqrt{V_R^2 + V_z^2}$ (km/s)",
        title="Toomre diagram",
        cbar_label=r"[M/H]$_{\rm pred}$ (dex)",
    )
    th = np.linspace(0, np.pi, 200)
    for v0 in (50.0, 100.0, 150.0):
        ax.plot(v0 * np.cos(th), v0 * np.sin(th),
                lw=0.8, ls=":", color=PALETTE["ash"], alpha=0.6)
    x_lo, x_hi = _lim(v_phi_lsr, pad=0.05)
    y_hi = max(_lim(v_perp, pad=0.05)[1], 50.0)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, y_hi)

    # Panel 2: Action diagram, sqrt(J_R + J_z) vs L_z, coloured by [M/H].
    ax = axes[1]
    _scatter(
        ax, Lz, J_perp,
        c=mh, cmap="viridis", vmin=mh_lo, vmax=mh_hi, s=4.0,
        xlabel=r"$L_z$ (kpc km/s)",
        ylabel=r"$\sqrt{J_R + J_z}$ (kpc km/s)$^{1/2}$",
        title=r"Action diagram, $\sqrt{J_R + J_z}$ vs $L_z$",
        cbar_label=r"[M/H]$_{\rm pred}$ (dex)",
    )
    ax.set_xlim(*_lim(Lz, pad=0.05))
    ax.set_ylim(0.0, _lim(J_perp, pad=0.10)[1])

    # Panel 3: E vs L_z, coloured by [M/H].
    ax = axes[2]
    _scatter(
        ax, Lz, E,
        c=mh, cmap="viridis", vmin=mh_lo, vmax=mh_hi, s=4.0,
        xlabel=r"$L_z$ (kpc km/s)",
        ylabel=r"E ($10^5$ km$^2$/s$^2$)",
        title=r"Energy vs $L_z$",
        cbar_label=r"[M/H]$_{\rm pred}$ (dex)",
    )
    ax.set_xlim(*_lim(Lz, pad=0.05))
    ax.set_ylim(*_lim(E, pad=0.05))

    fig.subplots_adjust(left=0.05, right=0.97, top=0.78, bottom=0.13, wspace=0.30)
    headline(
        fig,
        "Stream 2 Tier 1 holdout: Galactic kinematics",
        f"n = {len(kin):,};  McMillan+2017 potential, V_LSR = {V_LSR_KMS:.1f} km/s.",
        top=0.78,
    )
    save(fig, "Y41_stream2_kinematics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
