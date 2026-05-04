"""Y03: Preprocessing flow — what the input vector to the MLP actually is.

Stages, left → right:

  1. Raw Gaia DR3 XP coefficients (110 each, BP and RP)
  2. Ye+2024 NN flux correction (per-coefficient, learned)
  3. Reproject onto frozen Hermite basis (SHA-256 fingerprinted)
  4. Per-coefficient frozen z-score (training stats, never recomputed)
  5. Concatenate XP block + auxiliary scalars → 140-D feature vector

The point of the figure is to make clear that the MLP never sees raw flux
and that the training-time normalisation contract is frozen.
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


def _stage(ax, x, y, w, h, idx, title, body, color):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6,
        facecolor="white",
        edgecolor=color,
    )
    ax.add_patch(rect)
    # Numbered chip top-left.
    chip = mpatches.Circle(
        (x + 0.18, y + h - 0.18),
        0.14,
        facecolor=color,
        edgecolor="none",
    )
    ax.add_patch(chip)
    ax.text(
        x + 0.18,
        y + h - 0.18,
        str(idx),
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="white",
    )
    ax.text(
        x + w / 2,
        y + h - 0.45,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        x + w / 2,
        y + h * 0.30,
        body,
        ha="center",
        va="center",
        fontsize=10.5,
        color=PALETTE["ash"],
    )


def _arrow(ax, x0, y0, x1, y1, color):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=2.2, shrinkA=4, shrinkB=4, mutation_scale=18
        ),
    )


def main() -> int:
    apply_style()
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")

    bw, bh, by = 2.4, 1.8, 1.2
    gap = 0.2
    xs = [0.20 + i * (bw + gap) for i in range(5)]
    cols = [
        PALETTE["ash"],
        PALETTE["navy_light"],
        PALETTE["navy"],
        PALETTE["accent"],
        PALETTE["tier1"],
    ]
    bodies = [
        "Gaia DR3 XP\n110 BP + 110 RP\nspectral coefficients",
        "Per-coefficient\nflux correction\n(7-layer MLP, frozen)",
        "Project onto frozen\nHermite basis;\nSHA-256 fingerprinted",
        "Subtract μ, divide by σ\nfrom Stream 1 training\n(never recomputed)",
        "113 XP + 27 aux\n= 140-D vector\nfed to the model",
    ]
    titles = [
        "Raw XP",
        "Ye+2024",
        "Hermite",
        "Frozen z-score",
        "Feature vector",
    ]
    for i, (x, c, t, b) in enumerate(zip(xs, cols, titles, bodies)):
        _stage(ax, x, by, bw, bh, i + 1, t, b, c)

    for i in range(4):
        _arrow(ax, xs[i] + bw, by + bh / 2, xs[i + 1], by + bh / 2, color=PALETTE["ash"])

    # Annotation block under the chain.
    ax.text(
        7.0,
        0.35,
        "Frozen contract: Stages 3 + 4 commit to a single bit-exact transform across all streams. "
        "Streams 2 and 3 inherit the Stream 1 fit; nothing is re-estimated at inference time.",
        ha="center",
        va="center",
        fontsize=11,
        fontstyle="italic",
        color=PALETTE["navy"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["paper"], edgecolor=PALETTE["mist"]),
    )

    headline(
        fig,
        "Preprocessing — five stages from photons to features",
        "The MLP never sees flux. It sees a 140-D vector under a transform that is frozen at train time.",
        top=0.84,
    )
    save(fig, "Y03_preprocessing_flow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
