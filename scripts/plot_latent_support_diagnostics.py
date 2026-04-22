"""Diagnostic plot for the latent-support gate (Option A / convex-hull surrogate).

Six panels on a 2×3 grid:

Row 1 — score distributions:
  01. val kNN-dist distribution + threshold line
  02. Stream-3 volume vs uniform kNN-dist overlay + threshold line
  03. latent_support_flag vs OOD-joint overlap (2×2 confusion counts)

Row 2 — sky + chemistry:
  04. Stream-3 latent_support_flag rate by sky pixel (Mollweide)
  05. [M/H]-[α/M] predictions coloured by flag (volume arm)
  06. [M/H]-[α/M] predictions coloured by flag (uniform arm)

Uses pipeline1_latent_support_stream3.parquet + existing joint prediction
parquets (volume + uniform). Requires ``l_deg`` / ``b_deg`` columns in the
prediction parquets for the sky map panel.
"""

from __future__ import annotations

import argparse
import json
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
_DEF_GATE = _REPO / "data/processed/pipeline1_latent_support_stream3.parquet"
_DEF_GATE_PROV = _DEF_GATE.with_suffix(_DEF_GATE.suffix + ".provenance.json")
_DEF_VOL = _REPO / "data/processed/pipeline1_predictions_stream3_joint_volume.parquet"
_DEF_UNI = _REPO / "data/processed/pipeline1_predictions_stream3_joint_uniform.parquet"
_DEF_FEATURES = _REPO / "data/processed/pipeline1_features_stream3.parquet"
_DEF_OUT = _REPO / "reports/pipeline1/run_a/latent_support_diagnostics.png"


def _attach_galactic_coords(df: pd.DataFrame, features_path: Path) -> pd.DataFrame:
    """Join ra/dec/b_deg from features on source_id and derive l_deg."""
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


def _sky_rate(ax, l_deg, b_deg, flag, title, cmap="magma") -> None:
    m = np.isfinite(l_deg) & np.isfinite(b_deg) & np.isfinite(flag)
    l = np.deg2rad(l_deg[m])
    l = np.where(l > np.pi, l - 2 * np.pi, l)
    b = np.deg2rad(b_deg[m])
    f = flag[m].astype(float)
    nx, ny = 64, 32
    lam_edges = np.linspace(-np.pi, np.pi, nx + 1)
    phi_edges = np.linspace(-np.pi / 2, np.pi / 2, ny + 1)
    Hs, _, _ = np.histogram2d(l, b, bins=(lam_edges, phi_edges), weights=f)
    Hc, _, _ = np.histogram2d(l, b, bins=(lam_edges, phi_edges))
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(Hc > 0, Hs / Hc, np.nan)
    lam = 0.5 * (lam_edges[:-1] + lam_edges[1:])
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    L, P = np.meshgrid(lam, phi, indexing="xy")
    im = ax.pcolormesh(L, P, rate.T, cmap=cmap, vmin=0.0, vmax=1.0, shading="nearest")
    ax.set_xticks(np.deg2rad([-120, -60, 0, 60, 120]))
    ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
    ax.set_xticklabels(["120°", "60°", "0°", "300°", "240°"], fontsize=8)
    ax.set_yticklabels(["-60°", "-30°", "0°", "30°", "60°"], fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, shrink=0.75, label="flag rate")


def _chem_scatter(ax, mh, am, flag, title) -> None:
    m = np.isfinite(mh) & np.isfinite(am)
    mh, am, flag = mh[m], am[m], flag[m].astype(bool)
    ax.scatter(mh[~flag], am[~flag], s=0.5, c="tab:blue", alpha=0.25,
               label=f"in-support (n={(~flag).sum():,})", rasterized=True)
    ax.scatter(mh[flag], am[flag], s=0.5, c="tab:red", alpha=0.5,
               label=f"flagged (n={flag.sum():,})", rasterized=True)
    ax.set_xlim(-2.0, 0.5); ax.set_ylim(-0.2, 0.5)
    ax.set_xlabel(r"$[{\rm M/H}]$ [dex]")
    ax.set_ylabel(r"$[\alpha/{\rm M}]$ [dex]")
    ax.set_title(title)
    ax.legend(markerscale=8, fontsize=8, loc="upper right")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", type=Path, default=_DEF_GATE)
    ap.add_argument("--gate-provenance", type=Path, default=_DEF_GATE_PROV)
    ap.add_argument("--volume", type=Path, default=_DEF_VOL)
    ap.add_argument("--uniform", type=Path, default=_DEF_UNI)
    ap.add_argument("--features", type=Path, default=_DEF_FEATURES,
                    help="Stream-3 features parquet — source of ra/dec/b_deg for sky panel")
    ap.add_argument("--output", type=Path, default=_DEF_OUT)
    args = ap.parse_args()

    gate = pd.read_parquet(args.gate)
    prov = json.loads(args.gate_provenance.read_text())
    tau = float(prov["config"]["threshold_value"])
    q = float(prov["config"]["threshold_quantile"])
    k = int(prov["config"]["k"])
    val_stats = prov["reference"]["val_knn_dist_stats"]

    vol = pd.read_parquet(args.volume)
    uni = pd.read_parquet(args.uniform)
    # Left-join gate columns only when not already merged into the prediction
    # parquets. Avoids pandas appending ``_x`` / ``_y`` suffixes on re-merge.
    _gate_cols = ["latent_knn_dist", "latent_support_flag"]
    if not all(c in vol.columns for c in _gate_cols):
        vol = vol.merge(
            gate[["source_id", *_gate_cols]],
            on="source_id", how="left",
        )
    if not all(c in uni.columns for c in _gate_cols):
        uni = uni.merge(
            gate[["source_id", *_gate_cols]],
            on="source_id", how="left",
        )

    # Attach l_deg / b_deg for the Mollweide panel.
    if args.features.exists():
        vol = _attach_galactic_coords(vol, args.features)
        uni = _attach_galactic_coords(uni, args.features)

    # Reconstruct val kNN-dist from quantile stats — approximate.
    # We don't store the full val array in provenance; instead we plot the
    # volume + uniform Stream-3 distributions and mark the threshold.
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.subplots_adjust(hspace=0.34, wspace=0.28)

    ax = axes[0, 0]
    # Stream-3 full-union distribution (both arms concatenated) — proxy for score spread.
    d_all = gate["latent_knn_dist"].to_numpy()
    d_all = d_all[np.isfinite(d_all)]
    ax.hist(d_all, bins=80, histtype="stepfilled", color="tab:gray", alpha=0.6,
            label=f"Stream-3 all (n={len(d_all):,})")
    ax.axvline(tau, color="tab:red", linestyle="--", linewidth=2.0,
               label=f"τ = val p{q*100:.0f} = {tau:.3f}")
    ax.axvline(val_stats["p50"], color="tab:blue", linewidth=1.0,
               label=f"val p50 = {val_stats['p50']:.3f}")
    ax.axvline(val_stats["p95"], color="tab:blue", linestyle=":",
               label=f"val p95 = {val_stats['p95']:.3f}")
    ax.set_xlabel(f"kNN-mean distance in latent (k={k})")
    ax.set_ylabel("count")
    ax.set_title("Latent kNN-dist — Stream-3 + val thresholds")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_yscale("log")

    ax = axes[0, 1]
    d_v = vol["latent_knn_dist"].to_numpy()
    d_u = uni["latent_knn_dist"].to_numpy()
    d_v = d_v[np.isfinite(d_v)]
    d_u = d_u[np.isfinite(d_u)]
    ax.hist(d_v, bins=80, histtype="step", color="tab:blue", linewidth=1.5,
            density=True, label=f"volume (n={len(d_v):,})")
    ax.hist(d_u, bins=80, histtype="step", color="tab:orange", linewidth=1.5,
            density=True, label=f"uniform (n={len(d_u):,})")
    ax.axvline(tau, color="tab:red", linestyle="--", linewidth=2.0, label=f"τ = {tau:.3f}")
    ax.set_xlabel(f"kNN-mean distance (k={k})")
    ax.set_ylabel("density")
    ax.set_title("Volume vs uniform arm — latent support")
    ax.legend(fontsize=9)

    ax = axes[0, 2]
    if "ood_joint_flag" in uni.columns:
        # Confusion — 2×2 counts of (latent OOD) × (joint OOD) on the union.
        full = pd.concat([vol, uni], ignore_index=True)
        lat = full["latent_support_flag"].astype(bool).to_numpy()
        joint = full["ood_joint_flag"].astype(bool).to_numpy()
        tn = int(((~lat) & (~joint)).sum())
        fp = int((lat & (~joint)).sum())  # latent flags, joint doesn't
        fn = int(((~lat) & joint).sum())  # joint flags, latent doesn't
        tp = int((lat & joint).sum())
        mat = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="black" if mat[i, j] < mat.max() * 0.5 else "white",
                    fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["OOD-joint=0", "OOD-joint=1"])
        ax.set_yticklabels(["latent=0", "latent=1"])
        ax.set_title("Overlap: latent-support × OOD-joint (union)")
        plt.colorbar(im, ax=ax, shrink=0.75)
    else:
        ax.text(0.5, 0.5, "ood_joint_flag not in prediction parquet",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    # Row 2 — sky + chemistry.
    ax_sky = fig.add_subplot(2, 3, 4, projection="mollweide")
    ax_sky.set_position(axes[1, 0].get_position())
    axes[1, 0].remove()
    if {"l_deg", "b_deg"}.issubset(uni.columns):
        full = pd.concat([vol, uni], ignore_index=True)
        _sky_rate(
            ax_sky,
            full["l_deg"].to_numpy(), full["b_deg"].to_numpy(),
            full["latent_support_flag"].to_numpy(),
            "latent_support_flag rate by sky pixel (union)",
        )
    else:
        ax_sky.text(0.5, 0.5, "l_deg/b_deg not in predictions",
                    ha="center", va="center", transform=ax_sky.transAxes)
        ax_sky.set_title("sky — unavailable")

    mh_col = "mh_pred"
    am_col = "alpha_m_pred"
    _chem_scatter(
        axes[1, 1],
        vol[mh_col].to_numpy(), vol[am_col].to_numpy(),
        vol["latent_support_flag"].to_numpy(),
        "[M/H]–[α/M] — volume arm",
    )
    _chem_scatter(
        axes[1, 2],
        uni[mh_col].to_numpy(), uni[am_col].to_numpy(),
        uni["latent_support_flag"].to_numpy(),
        "[M/H]–[α/M] — uniform arm",
    )

    flag_rate_union = float(
        pd.concat([vol, uni], ignore_index=True)["latent_support_flag"]
        .astype(bool).mean()
    )
    fig.suptitle(
        f"Latent-support gate (k={k}, τ = val-p{q*100:.0f} = {tau:.3f}): "
        f"flag rate — volume {float(vol['latent_support_flag'].astype(bool).mean()):.1%}, "
        f"uniform {float(uni['latent_support_flag'].astype(bool).mean()):.1%}, "
        f"union {flag_rate_union:.1%}",
        fontsize=13, y=0.995,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output} ({args.output.stat().st_size/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
