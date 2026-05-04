"""Y22: SupCon-soft-positive contrastive loss explained.

The contrastive pretraining loss used in Pipeline 1 (and inherited from the
TESS_ML prototype). Two panels:

  (left)   schematic — anchor in label space surrounded by neighbours
           weighted by a Gaussian kernel exp(-||y_a - y_k||²/(2σ²)). Each
           neighbour is a "soft positive": its weight in the InfoNCE
           sum is the kernel value, not a binary same-class flag.

  (right)  why it helps regression — the same scatter coloured by the
           soft-positive weight from a single anchor; the kernel decides
           which stars are "neighbours in label space" without binning.

This loss runs only during *pretraining* (encoder only); the supervised
β-NLL loss takes over once the trunk weights are unlocked.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402


def _schematic_panel(ax):
    rng = np.random.default_rng(0)
    # Mock label cloud in [M/H], [α/M].
    n = 220
    mh = rng.normal(loc=-0.10, scale=0.55, size=n)
    am = rng.normal(loc=0.05, scale=0.10, size=n)
    # Add a high-α arm.
    n2 = 80
    mh2 = rng.normal(loc=-0.55, scale=0.30, size=n2)
    am2 = rng.normal(loc=0.25, scale=0.04, size=n2)
    mh = np.concatenate([mh, mh2])
    am = np.concatenate([am, am2])

    anchor = np.array([0.0, 0.05])
    sigma = 0.10
    d2 = (mh - anchor[0]) ** 2 + (am - anchor[1]) ** 2
    weight = np.exp(-d2 / (2 * sigma**2))

    # Background scatter.
    ax.scatter(
        mh,
        am,
        s=18,
        c=weight,
        cmap="viridis",
        vmin=0,
        vmax=1,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    # Concentric kernel rings at 1σ and 2σ.
    for r, ls, lab in [(sigma, "-", r"$1\sigma$"), (2 * sigma, "--", r"$2\sigma$")]:
        ring = mpatches.Circle(
            anchor, r, fill=False, lw=1.6, ls=ls, edgecolor=PALETTE["accent"], zorder=3
        )
        ax.add_patch(ring)
        ax.text(
            anchor[0] + r,
            anchor[1] + 0.005,
            lab,
            color=PALETTE["accent"],
            fontsize=11,
            fontweight="bold",
        )
    # Anchor marker.
    ax.scatter(
        *anchor,
        marker="*",
        s=520,
        color="white",
        edgecolor=PALETTE["accent"],
        linewidth=2.4,
        zorder=4,
        label="anchor star",
    )
    ax.set_xlabel("[M/H]  (label space)")
    ax.set_ylabel(r"[$\alpha$/M]")
    ax.set_xlim(-2.0, 0.6)
    ax.set_ylim(-0.10, 0.42)
    ax.set_title("Anchor + Gaussian-kernel soft positives", color=PALETTE["navy"])
    ax.legend(loc="upper right")


def _formula_panel(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rect = mpatches.FancyBboxPatch(
        (0.04, 0.04),
        0.92,
        0.92,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6,
        facecolor=PALETTE["paper"],
        edgecolor=PALETTE["mist"],
    )
    ax.add_patch(rect)
    ax.text(
        0.5,
        0.92,
        "SupCon-soft-positive loss",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["navy"],
    )
    ax.text(
        0.5,
        0.78,
        r"$w_{ak} \;=\; \exp(-\,||y_a - y_k||^{2}\,/\,(2\sigma^{2}))$",
        ha="center",
        va="center",
        fontsize=15,
        color=PALETTE["ink"],
    )
    ax.text(
        0.5,
        0.62,
        r"$L_{\rm SupCon} = -\frac{1}{B}\sum_{a}\,"
        r"\frac{\sum_k w_{ak}\,\log[s_{ak}/\tau]}{\sum_k w_{ak}}$",
        ha="center",
        va="center",
        fontsize=15,
        color=PALETTE["ink"],
    )
    ax.text(
        0.5,
        0.45,
        r"$s_{ak} = z_a \cdot z_k\,/\,(||z_a||\,||z_k||)$",
        ha="center",
        va="center",
        fontsize=13,
        color=PALETTE["ash"],
        fontstyle="italic",
    )
    ax.text(
        0.5,
        0.34,
        "trunk projects to L2-normalised $z$;\n"
        "label-space kernel weights neighbours\n"
        "without arbitrary class binning",
        ha="center",
        va="center",
        fontsize=11.5,
        color=PALETTE["ash"],
    )
    ax.text(
        0.5,
        0.17,
        "Only runs during pretraining\n"
        r"$\rightarrow$ β-NLL takes over once trunk unlocked",
        ha="center",
        va="center",
        fontsize=11,
        color=PALETTE["accent"],
        fontweight="bold",
    )


def main() -> int:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    plt.subplots_adjust(wspace=0.30, left=0.06, right=0.97, bottom=0.10)
    _schematic_panel(axes[0])
    _formula_panel(axes[1])
    headline(
        fig,
        "Pretraining — SupCon with soft positives",
        "Khosla+2020 SupCon adapted to regression: continuous label kernel "
        "instead of binary same-class match (TESS_ML prototype, ported in losses.supcon_soft_positive).",
        top=0.85,
    )
    save(fig, "Y22_supcon_loss")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
