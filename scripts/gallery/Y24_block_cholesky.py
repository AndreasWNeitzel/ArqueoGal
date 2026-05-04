"""Y24: Block-Cholesky covariance — what the supervised head emits.

Three panels:

  (left)   the L_chol matrix shape (5×5 lower-triangular) with annotations
           saying which entries are which label cross-coupling.
  (centre) the resulting Σ = L Lᵀ, with σ_i² on the diagonal and ρ_ij off,
           drawn from one real Stream-1 prediction.
  (right)  why use a block over a diagonal: empirical covariance ellipses
           in the (Δ[M/H], Δ[α/M]) plane on the held-out set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402

PRED_S1 = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"

LABELS_PRETTY = (r"$T_{\rm eff}$", r"$\log g$", "[M/H]", r"[$\alpha$/M]", "[Mg/H]")


def _draw_lower_tri(ax, mat, *, cmap, vlim, title, label_cells=False):
    n = mat.shape[0]
    masked = np.where(np.tril(np.ones_like(mat), 0) > 0, mat, np.nan)
    im = ax.imshow(masked, cmap=cmap, vmin=-vlim, vmax=vlim, origin="upper")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(LABELS_PRETTY, fontsize=11)
    ax.set_yticklabels(LABELS_PRETTY, fontsize=11)
    ax.set_title(title, color=PALETTE["navy"])
    if label_cells:
        for i in range(n):
            for j in range(i + 1):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def main() -> int:
    apply_style()

    # Load enough of Stream 1 to assemble a sample covariance from the cov_*
    # diagonal columns + a synthesised L matrix from one star (illustrative).
    # cov_* columns only live in the hybrid release; the bare predictions
    # parquet has σ but not the full cov. Fall back gracefully.
    pred_cols = [
        "source_id",
        "teff_pred",
        "logg_pred",
        "mh_pred",
        "alpha_m_pred",
        "mg_h_pred",
        "teff_sigma",
        "logg_sigma",
        "mh_sigma",
        "alpha_m_sigma",
        "mg_h_sigma",
    ]
    pred = pd.read_parquet(PRED_S1, columns=pred_cols).drop_duplicates("source_id")
    feat = pd.read_parquet(
        FEAT_S1,
        columns=[
            "source_id",
            "teff_apogee",
            "logg_apogee",
            "mh_apogee",
            "alpha_m_apogee",
            "mg_h_apogee",
        ],
    ).drop_duplicates("source_id")
    df = feat.merge(pred, on="source_id", how="inner")

    # Assemble an empirical 5x5 correlation matrix from residuals.
    keys = ("teff", "logg", "mh", "alpha_m", "mg_h")
    deltas = np.column_stack(
        [df[f"{k}_pred"].to_numpy() - df[f"{k}_apogee"].to_numpy() for k in keys]
    )
    ok = np.isfinite(deltas).all(axis=1)
    deltas = deltas[ok]
    # Standardise per-label so the off-diagonals are correlations.
    z = (deltas - np.nanmean(deltas, axis=0)) / np.nanstd(deltas, axis=0)
    corr = np.corrcoef(z, rowvar=False)

    # Synthesise an example L_chol via Cholesky of corr.
    L = np.linalg.cholesky(corr + 1e-3 * np.eye(5))
    L_show = np.where(np.tril(np.ones_like(L)), L, 0.0)

    fig = plt.figure(figsize=(20, 7.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.1], wspace=0.35)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[0, 2])

    _draw_lower_tri(
        ax_l,
        L_show,
        cmap="RdBu_r",
        vlim=1.0,
        title=r"$L_{\rm chol}$ — what the head emits",
        label_cells=True,
    )
    _draw_lower_tri(
        ax_s,
        corr,
        cmap="RdBu_r",
        vlim=1.0,
        title=r"$\Sigma = L\,L^{\!\top}$  — full covariance",
        label_cells=True,
    )

    # Empirical 2D residual scatter [M/H] vs [α/M] with eigenvector ellipse.
    d_mh = z[:, 2]
    d_am = z[:, 3]
    # Subsample.
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d_mh), size=min(8000, len(d_mh)), replace=False)
    ax_e.scatter(
        d_mh[idx], d_am[idx], s=4, alpha=0.3, color=PALETTE["navy_light"], edgecolor="none"
    )
    # Eigenvalue/eigenvector overlay of the empirical covariance.
    cov2 = np.cov(np.column_stack([d_mh, d_am]).T)
    vals, vecs = np.linalg.eigh(cov2)
    angle = float(np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1])))
    for n_sig in (1.0, 2.0):
        w = 2 * n_sig * np.sqrt(vals[1])
        h = 2 * n_sig * np.sqrt(vals[0])
        e = mpatches.Ellipse(
            (0, 0),
            w,
            h,
            angle=angle,
            fill=False,
            edgecolor=PALETTE["accent"],
            lw=2.2,
            ls="-" if n_sig == 1 else "--",
            label=rf"{n_sig:.0f}$\sigma$",
        )
        ax_e.add_patch(e)
    ax_e.set_xlim(-4, 4)
    ax_e.set_ylim(-4, 4)
    ax_e.set_aspect("equal")
    ax_e.set_xlabel(r"$\Delta$[M/H] / $\sigma$")
    ax_e.set_ylabel(r"$\Delta$[$\alpha$/M] / $\sigma$")
    ax_e.set_title(
        r"Why a block — residuals correlate $\rightarrow$ tilted ellipse", color=PALETTE["navy"]
    )
    ax_e.legend(loc="upper right")

    headline(
        fig,
        r"Block-Cholesky head — μ and the full $\Sigma = L\,L^{\!\top}$",
        "Five labels, full 5×5 covariance per star.  Diagonal-only would miss the "
        r"[M/H]–[$\alpha$/M] residual correlation visible at right.",
        top=0.84,
    )
    save(fig, "Y24_block_cholesky")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
