"""Y44: sample-flow cascade reconciling the n-counts across the deck.

A horizontal stage diagram showing how the cohort shrinks from the
parent Gaia DR3 XP catalogue down to the Tier-1 holdout used to quote
RMSE numbers. Each stage carries the surviving count and the cut that
discards stars upstream.

Stages (left to right):
  Gaia DR3 XP    : 219,197,643 (Gaia DR3 release count, BP/RP available)
  S1 raw join    : 326,724     (XP x APOGEE DR19, Lindegren+Riello-corrected)
  S1 dedup       : 292,948     (one row per source_id, post quality cut)
  Holdout        : 87,882      (val + test, stratified seed=0)
  Tier 1         : 82,548      (label-Mahalanobis-clean + ood_joint=0)

Slide-friendly 14:5 layout.
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

from _presentation import OKABE_ITO, PALETTE, apply_style, headline, save  # noqa: E402

STAGES = [
    ("Gaia DR3 XP",        219_197_643, "BP/RP coefficients"),
    ("S1 raw join",        326_724,     "x APOGEE DR19 (Lindegren+Riello)"),
    ("S1 dedup",           292_948,     "one row per source_id"),
    ("Holdout",            87_882,      "val + test, stratified seed = 0"),
    ("Tier 1",             82_548,      r"$\mathrm{ood\_joint} = 0$ + label-Mahalanobis"),
]


def main() -> int:
    apply_style()
    fig, ax = plt.subplots(figsize=(13.5, 5.0))

    n = len(STAGES)
    box_h = 1.2
    box_w = 2.05
    x_centres = np.linspace(0.5, 12.5, n)
    y_centre = 1.6

    for i, (label, count, sub) in enumerate(STAGES):
        x = x_centres[i]
        # Box.
        rect = plt.Rectangle(
            (x - box_w / 2, y_centre - box_h / 2),
            box_w, box_h,
            facecolor="white", edgecolor=OKABE_ITO[0], linewidth=1.4,
        )
        ax.add_patch(rect)
        ax.text(x, y_centre + 0.32, label,
                ha="center", va="center", fontsize=12,
                color=PALETTE["ink"], fontweight="semibold")
        ax.text(x, y_centre + 0.02, f"{count:,}",
                ha="center", va="center", fontsize=15,
                color=OKABE_ITO[0], fontweight="semibold")
        ax.text(x, y_centre - 0.32, sub,
                ha="center", va="center", fontsize=9.5,
                color=PALETTE["ash"])

        # Arrow + survival fraction.
        if i < n - 1:
            x_next = x_centres[i + 1]
            arrow_lo = x + box_w / 2 + 0.05
            arrow_hi = x_next - box_w / 2 - 0.05
            ax.annotate(
                "",
                xy=(arrow_hi, y_centre), xytext=(arrow_lo, y_centre),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=PALETTE["ash"], lw=1.4,
                    shrinkA=2, shrinkB=2, mutation_scale=16,
                ),
            )
            survival = STAGES[i + 1][1] / count
            ax.text(
                (arrow_lo + arrow_hi) / 2, y_centre + 0.55,
                f"survives {survival * 100:.2f}%",
                ha="center", va="bottom", fontsize=9.5,
                color=PALETTE["ash"],
            )

    ax.set_xlim(-0.4, 13.4)
    ax.set_ylim(0.2, 2.9)
    ax.set_aspect("equal")
    ax.axis("off")

    headline(
        fig,
        "Sample flow, parent catalogue to Tier-1 holdout",
        "Each n in the deck (87,882 / 82,548 / 15,000 UMAP subsample) "
        "lives at one of these stages.",
        top=0.84,
    )
    save(fig, "Y44_sample_flow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
