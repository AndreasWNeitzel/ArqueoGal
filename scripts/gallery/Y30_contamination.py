"""Y30: High-α / low-α disc contamination on Stream 1 Tier 1 held-out.

Truth-component classification: a star is high-α if [α/M]_apogee ≥ 0.15 dex,
low-α otherwise. The same threshold is applied to the prediction. The 2x2
confusion matrix is the contamination matrix; off-diagonals are stars whose
predicted disc-component is wrong.

Three panels:

  (left)   2x2 confusion matrix with absolute counts and row-normalised %
  (centre) chemistry plane with truth-correct stars in faint grey and
           contamination overlaid in red
  (right)  Δ[α/M] residual histogram split by the (truth, prediction) cell
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
from _y_holdout import load_holdout  # noqa: E402

ALPHA_DIVIDER = 0.15  # canonical thin/thick chemical separation


def main() -> int:
    apply_style()
    df = load_holdout()
    len(df)

    am_truth = df["alpha_m_apogee"].to_numpy()
    am_pred = df["alpha_m_pred"].to_numpy()
    mh_truth = df["mh_apogee"].to_numpy()
    mh_pred = df["mh_pred"].to_numpy()
    ok = np.isfinite(am_truth) & np.isfinite(am_pred) & np.isfinite(mh_truth) & np.isfinite(mh_pred)
    am_truth, am_pred = am_truth[ok], am_pred[ok]
    mh_truth, mh_pred = mh_truth[ok], mh_pred[ok]

    truth_high = am_truth >= ALPHA_DIVIDER
    pred_high = am_pred >= ALPHA_DIVIDER

    n_ll = int((~truth_high & ~pred_high).sum())  # truth low, pred low (correct)
    n_lh = int((~truth_high & pred_high).sum())  # truth low, pred high  (false high-α)
    n_hl = int((truth_high & ~pred_high).sum())  # truth high, pred low (false low-α)
    n_hh = int((truth_high & pred_high).sum())  # truth high, pred high (correct)
    cm = np.array([[n_ll, n_lh], [n_hl, n_hh]], dtype=np.int64)
    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0

    fig = plt.figure(figsize=(22, 8.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.20, 1.0], wspace=0.32, left=0.05, right=0.97)

    # --- Panel A: confusion matrix.
    ax = fig.add_subplot(gs[0, 0])
    norm = cm / cm.sum()
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred low-α", "pred high-α"], fontsize=12)
    ax.set_yticklabels(["truth low-α", "truth high-α"], fontsize=12)
    ax.set_title("Confusion matrix", color=PALETTE["navy"])
    for i in range(2):
        for j in range(2):
            color = "white" if norm[i, j] > 0.4 else PALETTE["ink"]
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n({row_pct[i, j]:.1f}%)",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=color,
            )
    purity = (cm[0, 0] + cm[1, 1]) / cm.sum()
    ax.text(
        0.5,
        -0.18,
        f"overall purity = {purity * 100:.1f}%   (α-cut at {ALPHA_DIVIDER:.2f} dex)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["accent"],
    )

    # --- Panel B: chemistry plane, contamination highlighted.
    ax = fig.add_subplot(gs[0, 1])
    correct = truth_high == pred_high
    ax.hexbin(
        mh_truth[correct],
        am_truth[correct],
        gridsize=70,
        extent=(-1.6, 0.55, -0.10, 0.42),
        mincnt=1,
        bins="log",
        cmap="Greys",
        alpha=0.85,
    )
    ax.scatter(
        mh_truth[~correct],
        am_truth[~correct],
        s=4,
        color=PALETTE["tier3"],
        alpha=0.45,
        edgecolor="none",
        label=f"contaminated  n={int((~correct).sum()):,}",
    )
    ax.axhline(
        ALPHA_DIVIDER,
        color=PALETTE["accent"],
        lw=1.6,
        ls="--",
        label=rf"α divider = {ALPHA_DIVIDER:.2f} dex",
    )
    ax.set_xlim(-1.6, 0.55)
    ax.set_ylim(-0.10, 0.42)
    ax.set_xlabel("[M/H]  (truth)")
    ax.set_ylabel(r"[$\alpha$/M]  (truth)")
    ax.set_title("Chemistry plane — contaminated stars in red", color=PALETTE["navy"])
    ax.legend(loc="upper right", fontsize=10)

    # --- Panel C: Δα histograms by truth-pred cell.
    ax = fig.add_subplot(gs[0, 2])
    delta = am_pred - am_truth
    cells = [
        (~truth_high & ~pred_high, PALETTE["navy_light"], "T low → P low"),
        (~truth_high & pred_high, PALETTE["accent"], "T low → P high"),
        (truth_high & ~pred_high, PALETTE["tier3"], "T high → P low"),
        (truth_high & pred_high, PALETTE["tier1"], "T high → P high"),
    ]
    for mask, color, lab in cells:
        if int(mask.sum()) == 0:
            continue
        ax.hist(
            delta[mask],
            bins=80,
            range=(-0.20, 0.20),
            histtype="stepfilled",
            alpha=0.55,
            color=color,
            density=True,
            edgecolor=color,
            linewidth=0.8,
            label=f"{lab}  n={int(mask.sum()):,}",
        )
    ax.axvline(0.0, color=PALETTE["ink"], lw=1.0, alpha=0.6)
    ax.set_xlim(-0.20, 0.20)
    ax.set_xlabel(r"$\Delta$[$\alpha$/M] = pred − truth (dex)")
    ax.set_ylabel("density")
    ax.set_title("Residual by truth/pred component", color=PALETTE["navy"])
    ax.legend(loc="upper right", fontsize=9.5)

    headline(
        fig,
        "Disc-bimodality contamination — does the model preserve the two sequences?",
        f"Stream 1 Tier 1 held-out, n = {len(am_truth):,}.  "
        f"Hard α-cut at {ALPHA_DIVIDER:.2f} dex; off-diagonals are misassigned stars.",
        top=0.84,
    )
    save(fig, "Y30_contamination")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
