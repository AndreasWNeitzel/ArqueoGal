"""Y23: Trunk + projection head + supervised head — the full model graph.

A more honest version of Y04. Shows that the encoder has TWO outputs:

  z = projection head output (L2-normalised)  →  SupCon contrastive loss
  h = trunk embedding                         →  supervised head (μ, L)

The supervised head emits μ (5 means) AND L (lower-triangular Cholesky
factor of the 5×5 covariance). The full predictive distribution is
N(μ, L L^T).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402


def _box(ax, x, y, w, h, *, title, body, fc, edge=PALETTE["ash"], tc=None):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=2.0, facecolor=fc, edgecolor=edge,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.70, title,
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=tc or PALETTE["ink"])
    if body:
        ax.text(x + w / 2, y + h * 0.32, body,
                ha="center", va="center", fontsize=10.5,
                color=tc or PALETTE["ash"])


def _arrow(ax, x0, y0, x1, y1, color, label=None, label_offset=(0, 0.08)):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                shrinkA=4, shrinkB=4, mutation_scale=18))
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                ha="center", va="center", fontsize=10,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.95))


def main() -> int:
    apply_style()
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # Input.
    _box(ax, 0.4, 4.0, 2.2, 1.2,
         title="INPUT", body="x ∈ ℝ¹⁴⁰\n(113 XP + 27 aux)",
         fc=PALETTE["paper"], tc=PALETTE["ink"])

    # Trunk.
    _box(ax, 3.4, 4.0, 2.4, 1.2,
         title="TRUNK", body="MLP 256 → 128\nReLU + dropout 0.05",
         fc="#dfe7f1", edge=PALETTE["navy"])

    # Branching point.
    ax.plot([6.1, 6.4], [4.6, 4.6], color=PALETTE["navy"], lw=2.0)

    # Projection head (top).
    _box(ax, 6.6, 6.6, 3.0, 1.4,
         title="PROJECTION HEAD",
         body="2-layer MLP\n→ L2-normalised z",
         fc="#fff4e0", edge=PALETTE["accent"])

    # Supervised head (bottom).
    _box(ax, 6.6, 1.6, 3.0, 1.4,
         title="SUPERVISED HEAD",
         body="single MLP\n→ (μ, L_chol)",
         fc="#cce3d4", edge=PALETTE["tier1"])

    # Outputs and losses.
    _box(ax, 10.4, 6.6, 3.2, 1.4,
         title="z  (contrastive)",
         body="SupCon-soft loss\n(pretraining only)",
         fc="white", edge=PALETTE["accent"], tc=PALETTE["accent"])
    _box(ax, 10.4, 1.6, 3.2, 1.4,
         title="(μ, L)  →  N(μ, L Lᵀ)",
         body="β-NLL loss\n(every star, every epoch)",
         fc="white", edge=PALETTE["tier1"], tc=PALETTE["tier1"])

    # Arrows.
    _arrow(ax, 2.6, 4.6, 3.4, 4.6, PALETTE["navy"])
    _arrow(ax, 5.8, 4.6, 6.4, 4.6, PALETTE["navy"])
    # Branch up.
    ax.plot([6.4, 6.4], [4.6, 7.3], color=PALETTE["navy"], lw=2.0)
    _arrow(ax, 6.4, 7.3, 6.6, 7.3, PALETTE["accent"])
    _arrow(ax, 9.6, 7.3, 10.4, 7.3, PALETTE["accent"], label="L2-norm")
    # Branch down.
    ax.plot([6.4, 6.4], [4.6, 2.3], color=PALETTE["navy"], lw=2.0)
    _arrow(ax, 6.4, 2.3, 6.6, 2.3, PALETTE["tier1"])
    _arrow(ax, 9.6, 2.3, 10.4, 2.3, PALETTE["tier1"], label="μ + L_chol")

    # Annotation strip at bottom.
    ax.text(7.0, 0.3,
            "Pretraining: trunk + projection head learn under SupCon (encoder unlocks).  "
            "Fine-tune: supervised head trains under β-NLL with the trunk warm-started.",
            ha="center", va="center", fontsize=11, fontstyle="italic",
            color=PALETTE["ash"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["paper"],
                      edgecolor=PALETTE["mist"]))

    # h-vs-z annotation between trunk and branch.
    ax.text(6.4, 5.4, "h", color=PALETTE["navy"], fontsize=14,
            fontweight="bold")

    headline(
        fig,
        "Trunk + two heads",
        "The encoder is shared. The projection head supplies a contrastive view; "
        "the supervised head supplies the catalog (μ, Σ).",
        top=0.88,
    )
    save(fig, "Y23_trunk_heads_detail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
