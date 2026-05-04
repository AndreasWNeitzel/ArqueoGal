"""B12: Hermite coefficient redundancy diagnostic on the training pool.

The 108-D Hermite block is the dominant input to the encoder. If the
108 coefficients are highly correlated with one another, the effective
information content is much smaller than its nominal size, and the
encoder is operating on a low-rank subspace.

Layout (2x2):
  panel A (top-left)   = Pearson correlation matrix of the full 108-D
                         block (rows = BP1..BP54 ++ RP1..RP54, same on
                         columns). Diagonal blocks reveal within-band
                         structure; off-diagonal blocks reveal BP-RP
                         coupling.
  panel B (top-right)  = absolute correlation distribution: histogram
                         of |r| for all upper-triangular pairs, with
                         the |r| > 0.5 / 0.7 / 0.9 fractions reported.
  panel C (bottom-left) = PCA scree (per-component variance, log-y).
  panel D (bottom-right) = cumulative variance vs principal-component
                          rank, with the 95% / 99% / 99.9% thresholds
                          and effective-rank values labelled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _common import apply_style, save_fig

from arqueogal.xp_abundances.main.data import FeatureLayout

OUT = REPO / "reports/gallery/B_preprocessing"


def main() -> int:
    apply_style()
    layout = FeatureLayout()
    parquet = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
    if not parquet.exists():
        print(f"Error: {parquet} not found")
        return 1
    bp_cols = list(layout.bp_coef_cols)
    rp_cols = list(layout.rp_coef_cols)
    df = pd.read_parquet(parquet, columns=bp_cols + rp_cols)
    n_total = len(df)
    finite_mask = df.notna().all(axis=1).to_numpy()
    df_f = df.loc[finite_mask]
    print(f"[B12] cohort n={n_total:,}; XP-finite rows = {len(df_f):,}")

    list(layout.xp_bp_indices)
    list(layout.xp_rp_indices)
    n_bp = len(bp_cols)
    n_rp = len(rp_cols)
    n_xp = n_bp + n_rp

    stacked = df_f[bp_cols + rp_cols].to_numpy(dtype=np.float64)
    full_corr = np.corrcoef(stacked.T)

    # PCA spectrum (centred).
    centred = stacked - stacked.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centred, compute_uv=False)
    var = s**2
    var_frac = var / var.sum()
    cumvar = np.cumsum(var_frac)
    eff95 = int(np.searchsorted(cumvar, 0.95) + 1)
    eff99 = int(np.searchsorted(cumvar, 0.99) + 1)
    eff999 = int(np.searchsorted(cumvar, 0.999) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), layout="constrained")

    # Panel A: full 108x108 Pearson correlation.
    ax = axes[0, 0]
    im = ax.imshow(full_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.axhline(n_bp - 0.5, color="k", lw=0.7, ls="--")
    ax.axvline(n_bp - 0.5, color="k", lw=0.7, ls="--")
    tick_pos = [0, 27, 54, 54 + 27, 108 - 1]
    tick_lab = ["BP1", "BP28", "BP54 / RP1", "RP28", "RP54"]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=8)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab, fontsize=8)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(
        "Full 108x108 Hermite-coef correlation\n"
        "(diagonal blocks: BP-BP, RP-RP; off-diagonal: BP-RP)"
    )

    # Panel B: |r| histogram across upper-triangular pairs.
    ax = axes[0, 1]
    triu = np.triu_indices(n_xp, k=1)
    abs_r = np.abs(full_corr[triu])
    ax.hist(abs_r, bins=80, color="#444444", alpha=0.85, edgecolor="#444444", lw=0.4)

    def p_above(thr):
        return float((abs_r > thr).mean())

    ax.axvline(
        0.5, color="orange", lw=1.0, ls="--", label=f"|r|>0.5: {100 * p_above(0.5):.1f}% of pairs"
    )
    ax.axvline(
        0.7, color="red", lw=1.0, ls="--", label=f"|r|>0.7: {100 * p_above(0.7):.1f}% of pairs"
    )
    ax.axvline(
        0.9, color="darkred", lw=1.0, ls="--", label=f"|r|>0.9: {100 * p_above(0.9):.1f}% of pairs"
    )
    ax.set_xlabel("|Pearson r|  (upper-triangular pairs)")
    ax.set_ylabel("count")
    ax.set_title(f"Pairwise correlation magnitude distribution\n(n_pairs = {len(abs_r):,})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # Panel C: PCA scree (per-component variance fraction, log-y).
    ax = axes[1, 0]
    ax.plot(np.arange(1, n_xp + 1), var_frac, "-o", color="#1f77b4", ms=3, lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("PC rank (1 = largest)")
    ax.set_ylabel(r"variance fraction $\lambda_i / \sum \lambda$")
    ax.set_title("PCA scree (per-component variance, log-y)")
    ax.grid(alpha=0.3, which="both")

    # Panel D: cumulative variance vs rank.
    ax = axes[1, 1]
    ax.plot(np.arange(1, n_xp + 1), cumvar, "-", color="#1f77b4", lw=1.6)
    for thr, k in [(0.95, eff95), (0.99, eff99), (0.999, eff999)]:
        ax.axhline(thr, color="grey", lw=0.6, ls=":")
        ax.scatter([k], [cumvar[k - 1]], color="red", s=40, zorder=5)
        ax.annotate(
            f"  {int(thr * 100)}%: {k} PCs",
            (k, cumvar[k - 1]),
            textcoords="offset points",
            xytext=(4, -2),
            fontsize=9,
        )
    ax.set_xlabel("PC rank")
    ax.set_ylabel("cumulative variance fraction")
    ax.set_title(
        f"Cumulative variance vs PC rank\n"
        f"effective ranks: 95% -> {eff95},  99% -> {eff99},  "
        f"99.9% -> {eff999}  (out of {n_xp})"
    )
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"B12 - Hermite shape-coef redundancy on the Stream-1 training pool "
        f"(n={len(df_f):,} XP-finite rows).\n"
        f"108 nominal dims compress to {eff95} (95%) / {eff99} (99%) "
        "principal components.",
        fontsize=12,
        fontweight="semibold",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    save_fig(fig, OUT / "B12_xp_coef_correlation", formats=("pdf", "png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
