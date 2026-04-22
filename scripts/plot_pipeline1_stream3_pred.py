"""Stream-3 deployment-predictions plot (truth not available).

Panels
------
Row 1: HRD (Teff vs logg) — volume | uniform, combined-support-filtered
Row 2: chemistry ([M/H], [α/M]) — volume | uniform, combined-support-filtered
Row 3: chemistry ([M/H], [Mg/H]) — volume | uniform, combined-support-filtered
Row 4: combined-OOD rate by sky pixel (Mollweide); aleatoric σ_α distribution.

"Combined support" = ``~ood_joint_flag & ~latent_support_flag`` — a star must
pass **both** the 108-D Mahalanobis/disagreement gate (ellipsoidal support)
**and** the 32-D latent kNN gate (convex-hull surrogate) to enter the science
subset. The latent gate is substantially stricter than joint (SupCon clusters
are label-tight), so combined ≈ latent where it bites.

Uses pipeline1_predictions_stream3_joint_{volume,uniform}.parquet with the
``latent_support_flag`` column merged in by
``scripts/merge_latent_support_into_predictions.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.dpi"] = 140
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.25

_REPO = Path(__file__).resolve().parents[1]
_DEF_VOL = _REPO / "data/processed/pipeline1_predictions_stream3_joint_volume.parquet"
_DEF_UNI = _REPO / "data/processed/pipeline1_predictions_stream3_joint_uniform.parquet"
_DEF_FEATURES = _REPO / "data/processed/pipeline1_features_stream3.parquet"
_DEF_OUT = _REPO / "reports/pipeline1/run_a/stream3_predictions.png"


def _attach_galactic_coords(df: pd.DataFrame, features_path: Path) -> pd.DataFrame:
    """Join (ra_deg, dec_deg, b_deg) from features on source_id and derive
    Galactic longitude via astropy. Prediction parquets don't carry l_deg;
    features carry ra/dec/b_deg but not l. Called only when the sky panel
    needs l/b.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    feats = pd.read_parquet(
        features_path, columns=["source_id", "ra_deg", "dec_deg", "b_deg"],
    )
    df = df.merge(feats, on="source_id", how="left")
    m = np.isfinite(df["ra_deg"].to_numpy()) & np.isfinite(df["dec_deg"].to_numpy())
    l_deg = np.full(len(df), np.nan, dtype=np.float64)
    if m.any():
        sc = SkyCoord(
            ra=df.loc[m, "ra_deg"].to_numpy() * u.deg,
            dec=df.loc[m, "dec_deg"].to_numpy() * u.deg,
            frame="icrs",
        ).galactic
        l_deg[m] = sc.l.deg
    df["l_deg"] = l_deg
    return df

COLS = {
    "teff": "teff_pred", "logg": "logg_pred",
    "mh": "mh_pred", "alpha_m": "alpha_m_pred", "mg_h": "mg_h_pred",
    "alpha_sig": "alpha_m_sigma",
    "ood": "ood_joint_flag", "latent": "latent_support_flag",
    "l": "l_deg", "b": "b_deg",
}


def _combined_flag(df: pd.DataFrame) -> np.ndarray:
    """True if the row fails **either** gate. Missing latent column falls back
    to ood_joint alone so the script stays usable pre-merge. NaN latent flag
    (e.g., unmerged rows) is treated as not-flagged so the combined gate
    degrades gracefully to ood_joint.
    """
    ood = df[COLS["ood"]].astype(bool).to_numpy()
    if COLS["latent"] in df.columns:
        lat = df[COLS["latent"]].fillna(False).astype(bool).to_numpy()
        return ood | lat
    return ood
RANGE = {
    "teff": (4000, 5500), "logg": (0.9, 3.7),
    "mh": (-2.0, 0.5), "alpha_m": (-0.2, 0.5), "mg_h": (-2.0, 0.6),
}
PRETTY = {
    "teff": r"$T_{\rm eff}$ [K]", "logg": r"$\log g$ [dex]",
    "mh": r"$[{\rm M/H}]$ [dex]", "alpha_m": r"$[\alpha/{\rm M}]$ [dex]",
    "mg_h": r"$[{\rm Mg/H}]$ [dex]",
}


def _hex(ax, x, y, xr, yr, xlab, ylab, title, invert_x=False, invert_y=False) -> None:
    x_lo, x_hi = sorted(xr); y_lo, y_hi = sorted(yr)
    m = np.isfinite(x) & np.isfinite(y)
    h = ax.hexbin(x[m], y[m], gridsize=60, cmap="viridis", mincnt=1,
                  extent=(x_lo, x_hi, y_lo, y_hi), linewidths=0)
    ax.set_xlim(x_hi if invert_x else x_lo, x_lo if invert_x else x_hi)
    ax.set_ylim(y_hi if invert_y else y_lo, y_lo if invert_y else y_hi)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    return h


def _sky_mollweide(ax, l_deg, b_deg, flag, title, cbar_label="flag rate") -> None:
    # Radians; Mollweide expects l ∈ [-π, π], b ∈ [-π/2, π/2]
    m = np.isfinite(l_deg) & np.isfinite(b_deg) & np.isfinite(flag)
    l = np.deg2rad(l_deg[m])
    l = np.where(l > np.pi, l - 2 * np.pi, l)  # wrap
    b = np.deg2rad(b_deg[m])
    f = flag[m].astype(float)
    # HEALPix-style pixel binning via histogram2d
    nx, ny = 64, 32
    lam_edges = np.linspace(-np.pi, np.pi, nx + 1)
    phi_edges = np.linspace(-np.pi / 2, np.pi / 2, ny + 1)
    H_sum, _, _ = np.histogram2d(l, b, bins=(lam_edges, phi_edges), weights=f)
    H_cnt, _, _ = np.histogram2d(l, b, bins=(lam_edges, phi_edges))
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(H_cnt > 0, H_sum / H_cnt, np.nan)
    # Centre coordinates
    lam = 0.5 * (lam_edges[:-1] + lam_edges[1:])
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    L, P = np.meshgrid(lam, phi, indexing="xy")
    im = ax.pcolormesh(L, P, rate.T, cmap="magma", vmin=0.0, vmax=1.0,
                       shading="nearest")
    ax.set_xticks(np.deg2rad([-120, -60, 0, 60, 120]))
    ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
    ax.set_xticklabels(["120°", "60°", "0°", "300°", "240°"], fontsize=8)
    ax.set_yticklabels(["-60°", "-30°", "0°", "30°", "60°"], fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, shrink=0.75, label=cbar_label)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--volume", type=Path, default=_DEF_VOL)
    ap.add_argument("--uniform", type=Path, default=_DEF_UNI)
    ap.add_argument("--features", type=Path, default=_DEF_FEATURES,
                    help="Stream-3 features parquet — source of (ra_deg, dec_deg, b_deg)")
    ap.add_argument("--output", type=Path, default=_DEF_OUT)
    args = ap.parse_args()

    vol = pd.read_parquet(args.volume)
    uni = pd.read_parquet(args.uniform)
    # Attach (l_deg, b_deg) for the sky panel — predictions lack them;
    # features parquet has ra/dec/b_deg, l_deg is derived via astropy.
    if args.features.exists():
        uni = _attach_galactic_coords(uni, args.features)
    # Combined-support filter for science-grade panels.
    vol_bad = _combined_flag(vol)
    uni_bad = _combined_flag(uni)
    vol_ok = vol[~vol_bad]
    uni_ok = uni[~uni_bad]

    fig = plt.figure(figsize=(22, 22))
    gs = fig.add_gridspec(4, 2, hspace=0.38, wspace=0.28)

    # Row 1: HRD
    axV = fig.add_subplot(gs[0, 0])
    _hex(axV, vol_ok[COLS["teff"]].to_numpy(), vol_ok[COLS["logg"]].to_numpy(),
         RANGE["teff"], RANGE["logg"], PRETTY["teff"], PRETTY["logg"],
         f"HRD — volume arm, support-clean (n={len(vol_ok):,})",
         invert_x=True, invert_y=True)
    axU = fig.add_subplot(gs[0, 1])
    _hex(axU, uni_ok[COLS["teff"]].to_numpy(), uni_ok[COLS["logg"]].to_numpy(),
         RANGE["teff"], RANGE["logg"], PRETTY["teff"], PRETTY["logg"],
         f"HRD — uniform arm, support-clean (n={len(uni_ok):,})",
         invert_x=True, invert_y=True)

    # Row 2: [M/H] / [α/M]
    axV = fig.add_subplot(gs[1, 0])
    _hex(axV, vol_ok[COLS["mh"]].to_numpy(), vol_ok[COLS["alpha_m"]].to_numpy(),
         RANGE["mh"], RANGE["alpha_m"], PRETTY["mh"], PRETTY["alpha_m"],
         "[M/H] vs [α/M] — volume arm")
    axU = fig.add_subplot(gs[1, 1])
    _hex(axU, uni_ok[COLS["mh"]].to_numpy(), uni_ok[COLS["alpha_m"]].to_numpy(),
         RANGE["mh"], RANGE["alpha_m"], PRETTY["mh"], PRETTY["alpha_m"],
         "[M/H] vs [α/M] — uniform arm")

    # Row 3: [M/H] / [Mg/H]
    axV = fig.add_subplot(gs[2, 0])
    _hex(axV, vol_ok[COLS["mh"]].to_numpy(), vol_ok[COLS["mg_h"]].to_numpy(),
         RANGE["mh"], RANGE["mg_h"], PRETTY["mh"], PRETTY["mg_h"],
         "[M/H] vs [Mg/H] — volume arm")
    axU = fig.add_subplot(gs[2, 1])
    _hex(axU, uni_ok[COLS["mh"]].to_numpy(), uni_ok[COLS["mg_h"]].to_numpy(),
         RANGE["mh"], RANGE["mg_h"], PRETTY["mh"], PRETTY["mg_h"],
         "[M/H] vs [Mg/H] — uniform arm")

    # Row 4: sky combined-OOD rate (uniform, all-sky) + α σ histogram
    ax_sky = fig.add_subplot(gs[3, 0], projection="mollweide")
    if COLS["l"] in uni.columns and COLS["b"] in uni.columns:
        _sky_mollweide(
            ax_sky, uni[COLS["l"]].to_numpy(), uni[COLS["b"]].to_numpy(),
            uni_bad.astype(float),
            "combined-OOD rate by sky pixel (uniform)",
            cbar_label="combined flag rate",
        )
    else:
        ax_sky.text(0.5, 0.5, "l_deg/b_deg not in predictions",
                    ha="center", va="center", transform=ax_sky.transAxes)
        ax_sky.set_title("OOD sky map — unavailable")

    ax_sig = fig.add_subplot(gs[3, 1])
    sa_v = vol_ok[COLS["alpha_sig"]].to_numpy()
    sa_v = sa_v[np.isfinite(sa_v)]
    sa_u = uni_ok[COLS["alpha_sig"]].to_numpy()
    sa_u = sa_u[np.isfinite(sa_u)]
    ax_sig.hist(sa_v, bins=60, range=(0.0, 0.25), histtype="step",
                color="tab:blue", linewidth=1.5, density=True,
                label=f"volume (n={len(sa_v):,})")
    ax_sig.hist(sa_u, bins=60, range=(0.0, 0.25), histtype="step",
                color="tab:orange", linewidth=1.5, density=True,
                label=f"uniform (n={len(sa_u):,})")
    ax_sig.set_xlabel(r"reported $\sigma_\alpha$ [dex]")
    ax_sig.set_ylabel("density")
    ax_sig.set_title(r"Stream-3 $\sigma_\alpha$ — volume vs uniform (OOD-clean)")
    ax_sig.legend(loc="upper right", fontsize=9)

    has_lat_v = COLS["latent"] in vol.columns
    has_lat_u = COLS["latent"] in uni.columns
    if has_lat_v and has_lat_u:
        lat_v = vol[COLS["latent"]].fillna(False).astype(bool).mean()
        lat_u = uni[COLS["latent"]].fillna(False).astype(bool).mean()
        suptitle = (
            f"Pipeline-1 joint — Stream-3 deployment\n"
            f"volume {len(vol):,}: OOD-joint {vol[COLS['ood']].mean():.1%}, "
            f"latent {lat_v:.1%}, combined {vol_bad.mean():.1%}    |    "
            f"uniform {len(uni):,}: OOD-joint {uni[COLS['ood']].mean():.1%}, "
            f"latent {lat_u:.1%}, combined {uni_bad.mean():.1%}"
        )
    else:
        suptitle = (
            f"Pipeline-1 joint — Stream-3 deployment "
            f"(volume {len(vol):,}, OOD-joint {vol[COLS['ood']].mean():.1%}; "
            f"uniform {len(uni):,}, OOD-joint {uni[COLS['ood']].mean():.1%})"
        )
    fig.suptitle(suptitle, fontsize=12, y=0.995)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output} ({args.output.stat().st_size/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
